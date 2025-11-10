import os
import json
import argparse
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer, BertModel, BertTokenizerFast,
    get_linear_schedule_with_warmup
)
from peft import PeftModel
from torch.optim import AdamW
from sklearn.metrics import precision_score, recall_score, f1_score
from utils import gen_report
import numpy as np
import re
import wandb
import math
from peft import get_peft_model, LoraConfig, TaskType


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_json(response, pattern = r'\{.*?\}'):
    matches = re.findall(pattern, response, re.DOTALL)

    json_data = matches[0] if len(matches) == 1 else matches[-1]
    
    try:
        # Load the JSON data
        data = json.loads(json_data)
        return None, data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        # print(response)
        # print(json_data)
        return e, json_data

class EMSQADataset(Dataset):
    """
    Multi-task dataset: binary retrieval + multi-label category classification
    """
    def __init__(self, file_path, tokenizer, max_length=128, all_categories=None, all_levels=None):
        with open(file_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # build category mapping
        if all_categories is None:
            cats = {c for s in samples for c in s['category']}
            self.cat2idx = {c: i for i, c in enumerate(sorted(cats))}
        else:
            self.cat2idx = {c: i for i, c in enumerate(all_categories)}

        if all_levels is None:
            levels = sorted({l for s in samples for l in s['level']})
            self.level2idx = {lvl:i for i,lvl in enumerate(levels)}
        else:
            self.level2idx = {lvl:i for i,lvl in enumerate(all_levels)}
        
        self.num_levels = len(self.level2idx)
        self.num_cats = len(self.cat2idx)
        self.examples = []
        for s in samples:
            text = s['question'] + ' ' + ' '.join(s.get('choices', [])) + '<classify>'
            enc = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # lvl = s['level'][0]
            # lvl_idx = self.level2idx[lvl]

            lvl_label = torch.zeros(self.num_levels, dtype=torch.float)
            for l in s['level']:
                if l in self.level2idx:
                    lvl_label[self.level2idx[l]] = 1.0

            ml_label = torch.zeros(self.num_cats, dtype=torch.float)
            for c in s['category']:
                if c in self.cat2idx:
                    ml_label[self.cat2idx[c]] = 1.0
            self.examples.append({
                'input_ids':      enc.input_ids.squeeze(0),
                'attention_mask': enc.attention_mask.squeeze(0),
                'level':      lvl_label,
                'category':       ml_label
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class PromptDataset(EMSQADataset):
    """
    Prompt-based LM dataset for category + certification prediction, with generation labels
    """
    def __init__(self, file_path, tokenizer, max_length=512, all_categories=None, all_levels=None):
        # reuse base mappings
        super().__init__(file_path, tokenizer, max_length, all_levels, all_categories)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        with open("/scratch/zar8jw/EMS-MCQA/code/EMSRAG/categories.json", "r") as f:
            categories = json.load(f)

        with open("/scratch/zar8jw/EMS-MCQA/code/EMSRAG/levels.json", "r") as f:
            levels = json.load(f)

        for s in samples:
            # build the two-part prompt
            base_prompt = (
                "What is the category and certification of this question? "
                "Category: 0.airway_respiration_and_ventilation; 1.anatomy; 2.assessment; "
                "3.cardiology_and_resuscitation; 4.ems_operations; 5.medical_and_obstetrics_gynecology; "
                "6.others; 7.pediatrics; 8.pharmacology; 9.trauma. "
                "Certification: 0.aemt; 1.emr; 2.emt; 3.paramedic"
                "Return the results in JSON: {\"category\": [\"1\", \"2\"], \"certification\": [\"1\"]}."
                f"QUESTION: {s['question']} + {';'.join(s['choices'])}"
                "ANSWER:"
            )

            answer_cats = [categories.index(c) for c in s["category"]]
            answer_lvls = [levels.index(l) for l in s["level"]]
            answer_prompt = {
                "category": answer_cats,
                "certification": answer_lvls
            }
            # The full sequence (prompt + expected answer placeholder)
            full_sequence = base_prompt + str(answer_prompt)
            # Tokenize full sequence as labels
            tok_full = tokenizer(
                full_sequence,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            # Tokenize just the prompt for input to generate
            tok_pr = tokenizer(
                base_prompt,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            # prepare category multi-label
            ml_label = torch.zeros(self.num_cats, dtype=torch.float)
            for c in s['category']:
                if c in self.cat2idx:
                    ml_label[self.cat2idx[c]] = 1.0
            # prepare certification level
            lvl_idx = self.level2idx.get(s['level'], -1)
            # append example with ground truth for generation
            self.examples.append({
                'input_ids':            tok_pr.input_ids.squeeze(0),
                'attention_mask':       tok_pr.attention_mask.squeeze(0),
                'labels':               tok_full.input_ids.squeeze(0),
                'input_ids_prompt':     tok_pr.input_ids.squeeze(0),
                'attention_mask_prompt':tok_pr.attention_mask.squeeze(0),
                'category':             ml_label,
                'certification':        torch.tensor(lvl_idx, dtype=torch.long)
            })


class BertRetrievalClassifier(nn.Module):
    """
    BERT backbone + two heads (retrieval + category)
    """
    def __init__(self, pretrained_model, num_levels, num_categories, dropout_prob=0.1, lambda_lvl=0.5, lambda_cat=1.5):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.lambda_lvl, self.lambda_cat = lambda_lvl, lambda_cat
        # self.retriever = nn.Linear(hidden, 1)
        self.level_classifier = nn.Linear(hidden, num_levels)  # CHANGED
        self.categorizer = nn.Linear(hidden, num_categories)

    def forward(self, input_ids, attention_mask,
                level=None, category=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output
        dropped = self.dropout(pooled)
        # ret_logit = self.retriever(dropped).squeeze(-1)
        lvl_logits = self.level_classifier(dropped)
        cat_logits = self.categorizer(dropped)
        loss = None
        if level is not None and category is not None:
            # loss_ret = nn.BCEWithLogitsLoss()(ret_logit, retrieval)
            # loss_lvl = nn.CrossEntropyLoss()(lvl_logits, level)
            loss_lvl = nn.BCEWithLogitsLoss()(lvl_logits, level)
            loss_cat = nn.BCEWithLogitsLoss()(cat_logits, category)
            # loss = loss_ret + loss_cat
            loss = self.lambda_lvl * loss_lvl + self.lambda_cat * loss_cat
        # return {'loss': loss, 'retrieval_logits': ret_logit, 'category_logits': cat_logits}
        return {'loss': loss,                 
                'loss_lvl': loss_lvl,
                'loss_cat': loss_cat,
                'level_logits': lvl_logits, 
                'category_logits': cat_logits}


class QwenRetrievalClassifier(nn.Module):
    """
    Qwen backbone + LoRA + two heads
    """
    def __init__(self,  backbone, num_levels, num_categories, lora_dropout=0.05, lambda_lvl=0.5, lambda_cat=1.5):
        super().__init__()
        self.model = backbone
        self.lambda_lvl, self.lambda_cat = lambda_lvl, lambda_cat
        self.dropout = nn.Dropout(lora_dropout)
        hidden = self.model.config.hidden_size
        # self.retriever = nn.Linear(hidden, 1)
        self.level_classifier = nn.Linear(hidden, num_levels)
        self.categorizer = nn.Linear(hidden, num_categories)

    def forward(self, input_ids, attention_mask,
                level=None, category=None):
        out = self.model(input_ids=input_ids, 
                         attention_mask=attention_mask, 
                         return_dict=True,
                         output_hidden_states=True)
        last_hidden = out.hidden_states[-1][:, -1, :]
        # last_hidden = out.last_hidden_state[:, -1, :]
        dropped = self.dropout(last_hidden)
        # ret_logit = self.retriever(dropped).squeeze(-1)
        lvl_logits = self.level_classifier(dropped)   # shape (B, num_levels)
        cat_logits = self.categorizer(dropped)
        loss = None
        loss_lvl = None
        loss_cat = None
        if level is not None and category is not None:
            # loss_ret = nn.BCEWithLogitsLoss()(ret_logit, retrieval)
            # loss_lvl = nn.CrossEntropyLoss()(lvl_logits, level)
            loss_lvl = nn.BCEWithLogitsLoss()(lvl_logits, level)
            loss_cat = nn.BCEWithLogitsLoss()(cat_logits, category)
            loss = self.lambda_lvl * loss_lvl + self.lambda_cat * loss_cat
            # loss = loss_ret + loss_cat
        return {'loss': loss, 
                'loss_lvl': loss_lvl,
                'loss_cat': loss_cat,
                'level_logits': lvl_logits, 
                'category_logits': cat_logits
                }
    
    def generate(self, **gen_kwargs):
        # just delegate to the underlying LM
        # this will incur more latency, see https://arxiv.org/abs/2405.17741
        with self.model.disable_adapter():
            return self.model.generate(**gen_kwargs)

class QwenPromptRetrieval(nn.Module):
    """Qwen causal LM + LoRA, training generate ' Yes'/' No'"""
    def __init__(self, pretrained_model, lora_r, lora_alpha, lora_dropout):
        super().__init__()
        base = AutoModelForCausalLM.from_pretrained(pretrained_model)
        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=["qkv_proj","o_proj"]
        )
        self.model = get_peft_model(base, cfg)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

    def generate(self, **gen_kwargs):
        # disable LoRA adapters during generation for speed
        with self.model.disable_adapter():
            return self.model.generate(**gen_kwargs)

def evaluate(args, model, tokenizer, loader, device, val_or_test):
    sigmoid = nn.Sigmoid()
    # validation
    model.eval()
    if args.prompt:
        val_loss = 0.0
        # compute LM loss + generation evaluation
        all_preds_lvl, all_labels_lvl = [], []
        all_preds_cat, all_labels_cat = [], []
        with torch.no_grad():
            for batch in loader:
                # LM loss on full
                lm_out = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=batch["labels"].to(device)
                )
                val_loss += lm_out['loss'].item()
                # generate
                gen = model.model.generate(
                    batch['input_ids_prompt'].to(device),
                    attention_mask=batch['attention_mask_prompt'].to(device),
                    max_new_tokens=10
                )

                for g, r_lbl, c_lbl in zip(gen, batch['level'], batch['category']):
                    txt = tokenizer.decode(g, skip_special_tokens=True)
                    error, jsonfile = extract_json(txt)
                    if error:
                        all_preds_lvl.append([0])
                        all_preds_cat.append([0])
                    else:
                        all_preds_lvl.append(jsonfile["certification"])
                        all_preds_cat.append(jsonfile["category"])
                    all_labels_lvl.append(r_lbl.item())
                    all_labels_cat.append(c_lbl.item())

        # metrics prompt
        all_levels_true = np.array(all_labels_lvl)
        all_levels_pred = np.array(all_preds_lvl)
        all_cat_true = np.array(all_labels_cat)
        all_cat_pred = np.array(all_preds_cat)
        avg_val_loss = val_loss / len(loader)
    else:
        val_loss = 0.0
        all_levels_true, all_levels_pred = [], []
        all_cat_true, all_cat_pred = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=val_or_test):
                inputs = {
                    'input_ids':      batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                    'level':          batch['level'].to(device),
                    'category':       batch['category'].to(device)
                }
                out = model(**inputs)
                val_loss += out['loss'].item()
                lvl_logits = out['level_logits']
                # preds_lvl = lvl_logits.argmax(dim=-1).cpu().numpy()
                probs_lvl = sigmoid(lvl_logits).cpu().numpy()
                preds_lvl = (probs_lvl > args.lvl_threshold).astype(int)
                for i in range(preds_lvl.shape[0]):
                    if preds_lvl[i].sum() == 0:
                        preds_lvl[i, probs_lvl[i].argmax()] = 1
                all_levels_pred.extend(preds_lvl.tolist())
                all_levels_true.extend(batch['level'].cpu().numpy().tolist())


                cat_logits = out['category_logits']
                probs_cat = sigmoid(cat_logits).cpu().numpy()
                preds_cat = (probs_cat > args.cat_threshold).astype(int)
                for i in range(preds_cat.shape[0]):
                    if preds_cat[i].sum() == 0:
                        preds_cat[i, probs_cat[i].argmax()] = 1
                all_cat_pred.extend(preds_cat.tolist())
                all_cat_true.extend(batch['category'].cpu().numpy().tolist())
        avg_val_loss = val_loss / len(loader)

    # Level metrics
    lvl_acc = (np.array(all_levels_pred) == np.array(all_levels_true)).mean()
    lvl_prec_micro = precision_score(all_levels_true, all_levels_pred, average='micro')
    lvl_rec_micro  = recall_score(all_levels_true, all_levels_pred, average='micro')
    lvl_f1_micro   = f1_score(all_levels_true, all_levels_pred, average='micro')
    lvl_prec_macro = precision_score(all_levels_true, all_levels_pred, average='macro')
    lvl_rec_macro  = recall_score(all_levels_true, all_levels_pred, average='macro')
    lvl_f1_macro   = f1_score(all_levels_true, all_levels_pred, average='macro')
    
    # Category metrics
    cat_prec_micro = precision_score(all_cat_true, all_cat_pred, average='micro')
    cat_rec_micro  = recall_score(all_cat_true, all_cat_pred, average='micro')
    cat_f1_micro   = f1_score(all_cat_true, all_cat_pred, average='micro')
    cat_prec_macro = precision_score(all_cat_true, all_cat_pred, average='macro')
    cat_rec_macro  = recall_score(all_cat_true, all_cat_pred, average='macro')
    cat_f1_macro   = f1_score(all_cat_true, all_cat_pred, average='macro')

    return {
        'loss':              avg_val_loss,
        'lvl_acc':           lvl_acc,
        'lvl_prec_micro':    lvl_prec_micro,
        'lvl_rec_micro':     lvl_rec_micro,
        'lvl_f1_micro':      lvl_f1_micro,
        'lvl_prec_macro':    lvl_prec_macro,
        'lvl_rec_macro':     lvl_rec_macro,
        'lvl_f1_macro':      lvl_f1_macro,
        'cat_prec_micro':    cat_prec_micro,
        'cat_rec_micro':     cat_rec_micro,
        'cat_f1_micro':      cat_f1_micro,
        'cat_prec_macro':    cat_prec_macro,
        'cat_rec_macro':     cat_rec_macro,
        'cat_f1_macro':      cat_f1_macro
    }


def train(args):
    set_seed(args.seed)
    if args.prompt:
        if args.tau > 0:
            run_name = f"{args.pretrained_model}_bs{args.batch_size}_lr{args.lr}_ret:{args.lambda_lvl}_cat:{args.lambda_cat}_tau:{args.tau}_len:{args.max_length}_prompt"
        else:
            run_name = f"{args.pretrained_model}_bs{args.batch_size}_lr{args.lr}_ret:{args.lambda_lvl}_cat:{args.lambda_cat}_len:{args.max_length}_prompt"
    else:
        if args.tau > 0:
            run_name = f"{args.pretrained_model}_bs{args.batch_size}_lr{args.lr}_ret:{args.lambda_lvl}_cat:{args.lambda_cat}_tau:{args.tau}_len:{args.max_length}"
        else:
            run_name = f"{args.pretrained_model}_bs{args.batch_size}_lr{args.lr}_ret:{args.lambda_lvl}_cat:{args.lambda_cat}_len:{args.max_length}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args)
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # tokenizer
    if args.prompt:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        tokenizer.padding_side = 'left'
        tokenizer.truncation_side = 'left'
    else:
        if 'Qwen' in args.pretrained_model:
            tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
            tokenizer.padding_side = 'left'
            tokenizer.truncation_side = 'left'
        else:
            tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_model)
    # datasets
    ds_train = (PromptDataset if args.prompt else EMSQADataset)(
        args.train_file, tokenizer, args.max_length
    )
    ds_val   = (PromptDataset if args.prompt else EMSQADataset)(
        args.val_file,   tokenizer, args.max_length,
        all_categories=list(ds_train.cat2idx.keys())
    )
    ds_test = (PromptDataset if args.prompt else EMSQADataset)(
        args.test_file,   tokenizer, args.max_length,
        all_categories=list(ds_train.cat2idx.keys())
    )
    # loaders
    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    loader_val   = DataLoader(ds_val,   batch_size=args.batch_size)
    loader_test  = DataLoader(ds_test, batch_size=args.batch_size)


    with open("/scratch/zar8jw/EMS-MCQA/code/EMSRAG/categories.json", "w") as f:
        json.dump(list(ds_train.cat2idx.keys()), f, indent=4)
    
    with open("/scratch/zar8jw/EMS-MCQA/code/EMSRAG/levels.json", "w") as f:
        json.dump(list(ds_train.level2idx.keys()), f, indent=4)

    # model init
    if args.prompt:
        model = QwenPromptRetrieval(
            args.pretrained_model, args.lora_r, args.lora_alpha, args.lora_dropout
        ).to(device)
    else:
        num_cats = ds_train.num_cats
        num_levels = ds_train.num_levels
        if 'Qwen' in args.pretrained_model:
            base = AutoModelForCausalLM.from_pretrained(args.pretrained_model)
            peft_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["qkv_proj","o_proj"],
                )
            backbone = get_peft_model(base, peft_cfg)
            model = QwenRetrievalClassifier(
                backbone,
                num_levels,
                num_cats,
                lora_dropout=args.lora_dropout,
                lambda_lvl=args.lambda_lvl,
                lambda_cat=args.lambda_cat
            ).to(device)
        else:
            model = BertRetrievalClassifier(
                args.pretrained_model, 
                num_levels, 
                num_cats,
                lambda_lvl=args.lambda_lvl,
                lambda_cat=args.lambda_cat
            ).to(device)
    # optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(loader_train) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    T = args.tau
    lvl_loss_hist = []   # stores avg retrieval loss per epoch
    cat_loss_hist = []   # stores avg categorization loss per epoch

    # training loop
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        train_loss = 0.0

        if args.tau > 0:
            assert args.prompt != True, "tau cannot be specified when using prompt training"
            if epoch <= 2:
                # first two epochs: equal weights
                w_lvl, w_cat = args.lambda_lvl, args.lambda_cat
            else:
                # “rates of change” from last two epochs
                ω_lvl = lvl_loss_hist[-1] / lvl_loss_hist[-2]
                ω_cat = cat_loss_hist[-1] / cat_loss_hist[-2]
                exp_lvl = math.exp(ω_lvl / T)
                exp_cat = math.exp(ω_cat / T)
                denom  = exp_lvl + exp_cat
                w_lvl  = 2 * exp_lvl / denom
                w_cat  = 2 * exp_cat / denom

            epoch_lvl_loss = 0.0
            epoch_cat_loss = 0.0
        
        for batch in tqdm(loader_train, desc=f"Train ep{epoch}"):
            optimizer.zero_grad()

            if args.prompt:
                out = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=batch['labels'].to(device)
                )

            else:
                inputs = {
                    'input_ids':      batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                    'level':          batch['level'].to(device),
                    'category':       batch['category'].to(device)
                }
                out = model(**inputs)
            

            if args.tau > 0:
                loss_lvl, loss_cat = out["loss_lvl"], out["loss_cat"]
                loss = w_lvl * loss_lvl + w_cat * loss_cat
            else:
                loss = out.loss if args.prompt else out['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            if args.tau > 0:
                epoch_lvl_loss += loss_lvl.item()
                epoch_cat_loss += loss_cat.item()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(loader_train)

        if args.tau > 0:
            avg_lvl = epoch_lvl_loss / len(loader_train)
            avg_cat = epoch_cat_loss / len(loader_train)
            lvl_loss_hist.append(avg_lvl)
            cat_loss_hist.append(avg_cat)
        

        report = evaluate(args, model, tokenizer, loader_val, device, "Val")
        # log
        wandb.log({
            'train_loss':        avg_train_loss,
            'val_loss':          report["loss"],
            'lvl_acc':           report["lvl_acc"],
            'lvl_prec_micro':    report["lvl_prec_micro"],
            'lvl_rec_micro':     report["lvl_rec_micro"],
            'lvl_f1_micro':      report["lvl_f1_micro"],
            'lvl_prec_macro':    report["lvl_prec_macro"],
            'lvl_rec_macro':     report["lvl_rec_macro"],
            'lvl_f1_macro':      report["lvl_f1_macro"],
            'cat_prec_micro':    report["cat_prec_micro"],
            'cat_rec_micro':     report["cat_rec_micro"],
            'cat_f1_micro':      report["cat_f1_micro"],
            'cat_prec_macro':    report["cat_prec_macro"],
            'cat_rec_macro':     report["cat_rec_macro"],
            'cat_f1_macro':      report["cat_f1_macro"],
        })

        if args.tau > 0:
            wandb.log({
            'level_train_loss': avg_lvl,
            'categorizer_train_loss': avg_cat,
            })
            

        print(
            f"Epoch {epoch} | train_loss={avg_train_loss:.4f} | val_loss={report['loss']:.4f}\n"
            f"Level → acc={report['lvl_acc']:.4f}, micro F1={report['lvl_f1_micro']:.4f}, macro F1={report['lvl_f1_macro']:.4f}\n"
            f"Category  → micro F1={report['cat_f1_micro']:.4f}, macro F1={report['cat_f1_macro']:.4f}"
        )

        # Save checkpoint
        if args.use_lora:
            if not args.prompt:
                if args.tau > 0:
                    ckpt = os.path.join(args.output_dir, f"{args.pretrained_model}_len:{args.max_length}_lr:{args.lr}_bs:{args.batch_size}_lvl:{args.lambda_lvl}_cat:{args.lambda_cat}_tau:{args.tau}", f"epoch{epoch}_lora")
                else:
                    ckpt = os.path.join(args.output_dir, f"{args.pretrained_model}_len:{args.max_length}_lr:{args.lr}_bs:{args.batch_size}_lvl:{args.lambda_lvl}_cat:{args.lambda_cat}", f"epoch{epoch}_lora")
            else:
                if args.tau > 0:
                    ckpt = os.path.join(args.output_dir, f"{args.pretrained_model}_len:{args.max_length}_lr:{args.lr}_bs:{args.batch_size}_lvl:{args.lambda_lvl}_cat:{args.lambda_cat}_tau:{args.tau}", f"epoch{epoch}_lora_prompt")
                else:
                    ckpt = os.path.join(args.output_dir, f"{args.pretrained_model}_len:{args.max_length}_lr:{args.lr}_bs:{args.batch_size}_lvl:{args.lambda_lvl}_cat:{args.lambda_cat}", f"epoch{epoch}_lora_prompt")
        else:
            ckpt = os.path.join(args.output_dir, f"{args.pretrained_model}_len:{args.max_length}_lr:{args.lr}_bs:{args.batch_size}", f'epoch:{epoch}')
        os.makedirs(ckpt, exist_ok=True)

        if args.prompt or ("Qwen" in args.pretrained_model and args.use_lora):
            backbone.save_pretrained(ckpt)
        else:
            model.bert.save_pretrained(ckpt)
        
        if not args.prompt:
            torch.save(model.level_classifier.state_dict(), os.path.join(ckpt, "level_head.pt"))
            torch.save(model.categorizer.state_dict(), os.path.join(ckpt, "categorizer_head.pt"))
        tokenizer.save_pretrained(ckpt)

        args.epoch = epoch
        for src in ["open", "close"]:
            args.src = src
            test(args)



def test(args):
    set_seed(args.seed)
    if args.use_lora:
        if not args.prompt:
            if args.tau > 0:
                ckpt = os.path.join(args.output_dir, f"{args.pretrained_model}_len:{args.max_length}_lr:{args.lr}_bs:{args.batch_size}_lvl:{args.lambda_lvl}_cat:{args.lambda_cat}_tau:{args.tau}", f"epoch{args.epoch}_lora")
            else:
                ckpt = os.path.join(args.output_dir, f"{args.pretrained_model}_len:{args.max_length}_lr:{args.lr}_bs:{args.batch_size}_lvl:{args.lambda_lvl}_cat:{args.lambda_cat}", f"epoch{args.epoch}_lora")
        else:
            if args.tau > 0:
                ckpt = os.path.join(args.output_dir, f"{args.pretrained_model}_len:{args.max_length}_lr:{args.lr}_bs:{args.batch_size}_lvl:{args.lambda_lvl}_cat:{args.lambda_cat}_tau:{args.tau}", f"epoch{args.epoch}_lora_prompt")
            else:
                ckpt = os.path.join(args.output_dir, f"{args.pretrained_model}_len:{args.max_length}_lr:{args.lr}_bs:{args.batch_size}_lvl:{args.lambda_lvl}_cat:{args.lambda_cat}", f"epoch{args.epoch}_lora_prompt")
    else:
        ckpt = os.path.join(args.output_dir, f"{args.pretrained_model}_len:{args.max_length}_lr:{args.lr}_bs:{args.batch_size}", f'epoch:{args.epoch}')

    print(ckpt)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load tokenizer
    tokenizer=AutoTokenizer.from_pretrained(ckpt)
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'    

    with open("/scratch/zar8jw/EMS-MCQA/code/EMSRAG/categories.json", "r") as f:
        cats = json.load(f)
    num_cats = len(cats)

    with open("/scratch/zar8jw/EMS-MCQA/code/EMSRAG/levels.json", "r") as f:
        lvls = json.load(f)
    num_levels = len(lvls)

    if 'Qwen' in args.pretrained_model:
        # base = AutoModel.from_pretrained(args.pretrained_model)
        base = AutoModelForCausalLM.from_pretrained(args.pretrained_model)
        backbone = PeftModel.from_pretrained(
            base,
            ckpt,
        ).to(device)

        model = QwenRetrievalClassifier(
            backbone, 
            num_levels,
            num_cats,
            lora_dropout=args.lora_dropout,
            lambda_lvl=args.lambda_lvl,
            lambda_cat=args.lambda_cat,
        ).to(device)

        # 3) Load the two heads
        model.level_classifier.load_state_dict(
            torch.load(os.path.join(ckpt, "level_head.pt"),
                    weights_only=True,
                    map_location=device)
        )
        model.categorizer.load_state_dict(
            torch.load(os.path.join(ckpt, "categorizer_head.pt"),
                    weights_only=True,
                    map_location=device)
        )

    else:
        model = BertRetrievalClassifier(ckpt, num_levels, num_cats).to(device)
        level_path = os.path.join(ckpt, 'level_head.pt')
        model.level_classifier.load_state_dict(
            torch.load(level_path, weights_only=True, map_location=device)
            )
        categorizer_path = os.path.join(ckpt, 'categorizer_head.pt')
        model.categorizer.load_state_dict(
            torch.load(categorizer_path, weights_only=True, map_location=device)
            )

    model.eval()
    # data
    if args.src == "open":
        test_file = "/scratch/zar8jw/EMS-MCQA/log/retrieval annotation/open/test.json"
    else:
        test_file = "/scratch/zar8jw/EMS-MCQA/log/retrieval annotation/close/test.json"
    test_ds=EMSQADataset(test_file, tokenizer, args.max_length, all_categories=cats, all_levels=lvls)
    loader=DataLoader(test_ds, batch_size=args.batch_size)
    
    # metrics
    metrics = evaluate(args, model, tokenizer, loader, device, "Test")
    with open(os.path.join(ckpt, f'test_report_{args.src}.json'),'w') as f: json.dump(metrics,f,indent=2)
    # print('Test metrics:',metrics)
    print(f"Epoch {args.epoch} | test_loss={metrics['loss']:.4f}\n"
    f"Level → acc={metrics['lvl_acc']:.4f}, micro F1={metrics['lvl_f1_micro']:.4f}, macro F1={metrics['lvl_f1_macro']:.4f}\n"
    f"Category  → micro F1={metrics['cat_f1_micro']:.4f}, macro F1={metrics['cat_f1_macro']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train or test retrieval classifier")
    subparsers = parser.add_subparsers(dest='mode', required=True)

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--train_file', type=str, default="/scratch/zar8jw/EMS-MCQA/data/final/train_open_woNA.json")
    train_parser.add_argument('--val_file', type=str, default="/scratch/zar8jw/EMS-MCQA/data/final/val_open.json")
    train_parser.add_argument('--test_file', type=str, default="/scratch/zar8jw/EMS-MCQA/data/final/test_open.json")
    train_parser.add_argument('--pretrained_model', type=str, default='Qwen/Qwen3-4B')
    train_parser.add_argument('--use_lora', type=str2bool, default=True)
    train_parser.add_argument('--prompt', type=str2bool, default=False)
    train_parser.add_argument('--max_length', type=int, default=128)
    train_parser.add_argument('--batch_size', type=int, default=8)
    train_parser.add_argument('--lr', type=float, default=1e-5)
    train_parser.add_argument('--weight_decay', type=float, default=0.01)
    train_parser.add_argument('--num_epochs', type=int, default=20)
    train_parser.add_argument('--lvl_threshold', type=float, default=0.5)
    train_parser.add_argument('--cat_threshold', type=float, default=0.5)
    train_parser.add_argument('--max_grad_norm', type=float, default=1.0)
    train_parser.add_argument('--lora_r', type=int, default=8)
    train_parser.add_argument('--lora_alpha', type=int, default=16)
    train_parser.add_argument('--lora_dropout', type=float, default=0.05)
    train_parser.add_argument('--tau', type=float, default=0.0)
    train_parser.add_argument('--lambda_lvl', type=float, default=1.0)
    train_parser.add_argument('--lambda_cat', type=float, default=1.0)
    train_parser.add_argument('--output_dir', type=str, default="/scratch/zar8jw/EMS-MCQA/log/multi-task")
    train_parser.add_argument('--wandb_project', type=str, default='EMSRAG-router')
    train_parser.add_argument('--wandb_entity', type=str, default='gexueren')
    train_parser.add_argument('--wandb_run_name', type=str, default='multi-task')
    train_parser.add_argument('--seed', type=int, default=42)

    test_parser = subparsers.add_parser('test')
    test_parser.add_argument('--test_file', type=str, default="/scratch/zar8jw/EMS-MCQA/data/final/test_open.json")
    test_parser.add_argument('--src', type=str, default="open")
    test_parser.add_argument('--pretrained_model', type=str, default='UFNLP/gatortron-base')
    test_parser.add_argument('--epoch', type=int, required=True)
    test_parser.add_argument('--use_lora', type=str2bool, default=True)
    test_parser.add_argument('--prompt', type=str2bool, default=False)
    test_parser.add_argument('--batch_size', type=int, default=8)
    test_parser.add_argument('--lr', type=float, default=1e-5)
    test_parser.add_argument('--max_length', type=int, default=128)
    test_parser.add_argument('--cat_threshold', type=float, default=0.5)
    test_parser.add_argument('--lora_r', type=int, default=8)
    test_parser.add_argument('--lora_alpha', type=int, default=16)
    test_parser.add_argument('--lora_dropout', type=float, default=0.05)
    test_parser.add_argument('--tau', type=float, default=0.0)
    test_parser.add_argument('--lambda_lvl', type=float, default=1.0)
    test_parser.add_argument('--lambda_cat', type=float, default=1.0)
    test_parser.add_argument('--output_dir', type=str, default="/scratch/zar8jw/EMS-MCQA/log/multi-task")
    test_parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)


if __name__ == '__main__':
    main()
