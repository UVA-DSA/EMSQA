import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BertModel, BertTokenizerFast
from torch.optim import AdamW
from tqdm import tqdm
import wandb
from peft import get_peft_model, LoraConfig, TaskType
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import re

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
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RetrievalDataset(Dataset):
    """
    Binary retrieval dataset: each JSON sample must include:
      - "question": str
      - "choices": List[str]
      - "retrieval-or-not": "retrieval" or other

    Returns tokenized text + binary label.
    """
    def __init__(self, file_path, tokenizer, max_length=128):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['question'] + ' ' + ' '.join(sample.get('choices', []))
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        label = 1 if sample['retrieval or not'] == 'retrieval' else 0
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.float)
        return item

class PromptDataset(Dataset):
    """
    Prompt-based LM dataset: model generates ' Yes' or ' No'.
    """
    def __init__(self, file_path, tokenizer, max_length=512):
        samples = json.load(open(file_path, 'r', encoding='utf-8'))
        self.examples = []
        for s in samples:
            prompt = (
                f"Do you need to retrieve documents to answer the following question? Directly answer 'Yes' or 'No' without other explanation in your response, for example, 'Yes'\n"
                f"QUESTION: {s['question']} + {';'.join(s['choices'])}\n"
                "ANSWER:"
            )
            suffix = " Yes" if s['retrieval or not']=='retrieval' else " No"
            full = prompt + suffix
            tok = tokenizer(full, truncation=True, padding='max_length',
                             max_length=max_length, return_tensors='pt')
            # tokenize prompt only for generation
            prompt_enc = tokenizer(prompt, truncation=True, padding='max_length',
                                   max_length=max_length, return_tensors='pt')
            binary_label = 1 if s['retrieval or not'] == 'retrieval' else 0
            self.examples.append({
                'input_ids': tok.input_ids.squeeze(0),
                'attention_mask': tok.attention_mask.squeeze(0),
                'labels': tok.input_ids.squeeze(0),
                'input_ids_prompt': prompt_enc.input_ids.squeeze(0),
                'attention_mask_prompt': prompt_enc.attention_mask.squeeze(0),
                'binary_label': binary_label

            })

    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]

class BertRetrievalClassifier(nn.Module):
    """
    BERT backbone + single-logit head for binary classification.
    """
    def __init__(self, pretrained_model, dropout_prob=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled = outputs.pooler_output
        dropped = self.dropout(pooled)
        logit = self.classifier(dropped).squeeze(-1)
        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logit, labels)
        return {'loss': loss, 'logits': logit}

class QwenRetrievalClassifier(nn.Module):
    """
    Qwen (causal LM) backbone + single-logit head on the last token embedding.
    """
    def __init__(self, pretrained_model, dropout_prob=0.1, lora_r=8, lora_alpha=16, lora_dropout=0.05):
        super().__init__()

        if lora_r>0:
            # apply LoRA
            base_model = AutoModel.from_pretrained(pretrained_model)
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["qkv_proj", "o_proj"]
            )
            self.model = get_peft_model(base_model, peft_config)
        else:
            self.model = AutoModel.from_pretrained(pretrained_model)


        hidden_size = self.model.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        # causal LM: returns last_hidden_state [batch, seq_len, hidden]
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # take hidden state of last token
        last_hidden = outputs.last_hidden_state[:, -1, :]  # [batch, hidden]
        dropped = self.dropout(last_hidden)
        logit = self.classifier(dropped).squeeze(-1)  # [batch]
        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logit, labels)
        return {'loss': loss, 'logits': logit}

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


def train(args):
    set_seed(args.seed)
    if args.prompt:
        run_name = f"{args.pretrained_model}_bs{args.batch_size}_lr{args.lr}_len:{args.max_length}_prompt"
    else:
        run_name = f"{args.pretrained_model}_bs{args.batch_size}_lr{args.lr}_len:{args.max_length}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args)
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.prompt:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        tokenizer.padding_side='left'; tokenizer.truncation_side='left'
        ds_train = PromptDataset(args.train_file, tokenizer, args.max_length)
        ds_val   = PromptDataset(args.val_file,   tokenizer, args.max_length)
    else:
        if "Qwen" in args.pretrained_model:
            tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
            tokenizer.padding_side = 'left'
            tokenizer.truncation_side = 'left'
        else:
            tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_model)

        ds_train = RetrievalDataset(args.train_file, tokenizer, args.max_length)
        ds_val = RetrievalDataset(args.val_file, tokenizer, args.max_length)
    
    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    loader_val   = DataLoader(ds_val,   batch_size=args.batch_size)


    # model init
    if args.prompt:
        model = QwenPromptRetrieval(
            args.pretrained_model, args.lora_r, args.lora_alpha, args.lora_dropout
        ).to(device)
    else:
        if "Qwen" in args.pretrained_model:
            model = QwenRetrievalClassifier(
                args.pretrained_model, lora_r=args.use_lora and args.lora_r or 0,
                lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout
            ).to(device)
        else:
            model = BertRetrievalClassifier(args.pretrained_model).to(device)


    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = len(loader_train)*args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*steps), num_training_steps=steps)

    sigmoid = nn.Sigmoid()
    for ep in range(1, args.num_epochs + 1):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(loader_train, desc=f"Train ep{ep}"):
            optimizer.zero_grad()
            labels = batch['labels'].to(device)
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            out = model(**inputs, labels=labels)
            loss = out.loss if args.prompt else out['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        avg_train = train_loss / len(loader_train)


        # validation
        model.eval()
        acc = precision_micro = recall_micro = f1_micro = precision_macro = recall_macro = f1_macro = None
        val_loss = 0
        if args.prompt:
            all_preds, all_labels = [], []
            positive = {'yes', 'y', '1', 'true', 't'}
            count = 0
            with torch.no_grad():
                for batch in tqdm(loader_val, desc='Val gen'):
                    ins = {'input_ids': batch['input_ids'].to(device), 
                              'attention_mask': batch['attention_mask'].to(device)}
                    lbl = batch['labels'].to(device)
                    # compute LM loss
                    lm_out = model(**ins, labels=lbl)
                    batch_loss = lm_out.loss
                    val_loss += batch_loss.item()
                    count += 1

                    # generation uses prompt only
                    gen = model.model.generate(
                        batch['input_ids_prompt'].to(device),
                        attention_mask=batch['attention_mask_prompt'].to(device),
                        max_new_tokens=2
                    )

                    # 3) compare against the stored binary_label
                    for pred_ids, gold_label in zip(gen, batch["binary_label"]):
                        raw_pred = tokenizer.decode(pred_ids[-2:], skip_special_tokens=True).strip().lower()
                        pred = re.sub(r"[^\w\s]", "", raw_pred).strip().lower()
                        # print("pred", pred)
                        # print("gt", gold_label)

                        all_preds.append(1 if pred in positive else 0)
                        all_labels.append(gold_label.item())    # <— this is where binary_label is used


            y_pred = np.array(all_preds)
            y_true = np.array(all_labels)
            avg_val = val_loss / count if count>0 else 0
        else:
            all_scores, all_labels = [], []
            with torch.no_grad():
                for batch in tqdm(loader_val, desc='Validation'):
                    labels = batch['labels'].to(device)
                    inputs = {k: v.to(device) for k, v in batch.items() if k!='labels'}
                    out = model(**inputs, labels=labels)
                    val_loss = (val_loss or 0) + out['loss'].item()
                    all_scores.append(out['logits'].cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
            avg_val = val_loss / len(loader_val)
            y_scores = np.concatenate(all_scores)
            y_true = np.concatenate(all_labels)
            y_pred = (sigmoid(torch.from_numpy(y_scores)) > args.threshold).int().numpy()

        acc = (y_pred==y_true).mean()
        precision_micro = precision_score(y_true, y_pred, average='micro')
        recall_micro = recall_score(y_true, y_pred, average='micro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        f1_macro = f1_score(y_true, y_pred, average='macro')


        logs = {
            'train_loss': avg_train,
            'val_loss': avg_val,
            'accuracy': acc,
            'precision_micro': precision_micro,
            'recall_micro':    recall_micro,
            'f1_micro':        f1_micro,
            'precision_macro': precision_macro,
            'recall_macro':    recall_macro,
            'f1_macro':        f1_macro
        }
        wandb.log(logs)
        print(f"Epoch {ep}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}\n"
              f"Accuracy: {acc}\n"
              f"MICRO precision={precision_micro:.4f}, recall={recall_micro:.4f}, f1={f1_micro:.4f}\n"
              f"MACRO precision={precision_macro:.4f}, recall={recall_macro:.4f}, f1={f1_macro:.4f}")

        # Save checkpoint

        if args.use_lora:
            if not args.prompt:
                ckpt = os.path.join(args.output_dir, f"{args.pretrained_model}_len:{args.max_length}_lr:{args.lr}_bs:{args.batch_size}", f"epoch{ep}_lora")
            else:
                ckpt = os.path.join(args.output_dir, f"{args.pretrained_model}_len:{args.max_length}_lr:{args.lr}_bs:{args.batch_size}", f"epoch{ep}_lora_prompt")
        else:
            ckpt = os.path.join(args.output_dir, f"{args.pretrained_model}_len:{args.max_length}_lr:{args.lr}_bs:{args.batch_size}", f'epoch:{ep}')
        os.makedirs(ckpt, exist_ok=True)

        if args.prompt or ("Qwen" in args.pretrained_model and args.use_lora):
            model.model.save_pretrained(ckpt)
        elif "Qwen" in args.pretrained_model:
            model.backbone.save_pretrained(ckpt)
        else:
            model.bert.save_pretrained(ckpt)
        if not args.prompt:
            torch.save(model.classifier.state_dict(), os.path.join(ckpt,'classifier.pt'))
        tokenizer.save_pretrained(ckpt)


def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir = os.path.join(args.output_dir, f"epoch_{args.model_epoch}")
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model = BertRetrievalClassifier(model_dir).to(device)
    state = torch.load(os.path.join(model_dir, 'classifier.pt'), map_location=device)
    model.classifier.load_state_dict(state)
    sigmoid = nn.Sigmoid()

    test_ds = RetrievalDataset(args.test_file, tokenizer, args.max_length)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            labels = batch['labels'].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k!='labels'}
            out = model(**inputs)
            all_scores.append(out['logits'].cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_scores = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels)
    y_pred = (sigmoid(torch.from_numpy(y_scores)) > args.threshold).int().numpy()



    acc = (y_pred==y_true).mean()
    precision_micro = precision_score(y_true, y_pred, average='micro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    metrics = {
        "accuracy": acc,
        'micro precision': precision_micro,
        'micro recall': recall_micro,
        'micro f1': f1_micro,
        "macro precision": precision_macro,
        "macro recall": recall_macro,
        "macro f1": f1_macro
    }

    with open(os.path.join(args.output_dir, 'test_report.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Test metrics: {metrics}")


def model_inference(model_dir: str, question: str, choices: list, threshold: float = 0.5, device: str = 'cpu') -> dict:
    """
    Load a trained model from `model_dir` and perform inference on a single example.

    Returns:
      {
        'probability': float,
        'prediction': int  # 1 for retrieval, 0 for not
      }
    """
    device = torch.device(device)
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model = BertRetrievalClassifier(model_dir).to(device)
    state = torch.load(os.path.join(model_dir, 'classifier.pt'), map_location=device)
    model.classifier.load_state_dict(state)
    model.eval()

    # prepare text
    text = question + ' ' + ' '.join(choices)
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=tokenizer.model_max_length,
        return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        out = model(**inputs)
        logit = out['logits'].cpu().item()
        prob = torch.sigmoid(torch.tensor(logit)).item()
        pred = 1 if prob > threshold else 0
    return {'probability': prob, 'prediction': pred}


def main():
    parser = argparse.ArgumentParser(description="Train or test retrieval classifier")
    subparsers = parser.add_subparsers(dest='mode', required=True)

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--train_file', type=str, default="/scratch/zar8jw/EMS-MCQA/log/retrieval annotation/open/train.json")
    train_parser.add_argument('--val_file', type=str, default="/scratch/zar8jw/EMS-MCQA/log/retrieval annotation/open/val.json")
    train_parser.add_argument('--pretrained_model', type=str, default='UFNLP/gatortron-base')
    train_parser.add_argument('--use_lora', type=str2bool, default=True)
    train_parser.add_argument('--prompt', type=str2bool, default=False)
    train_parser.add_argument('--max_length', type=int, default=512)
    train_parser.add_argument('--batch_size', type=int, default=16)
    train_parser.add_argument('--lr', type=float, default=1e-5)
    train_parser.add_argument('--weight_decay', type=float, default=0.01)
    train_parser.add_argument('--num_epochs', type=int, default=20)
    train_parser.add_argument('--threshold', type=float, default=0.5)
    train_parser.add_argument('--max_grad_norm', type=float, default=1.0)
    train_parser.add_argument('--lora_r', type=int, default=8)
    train_parser.add_argument('--lora_alpha', type=int, default=16)
    train_parser.add_argument('--lora_dropout', type=float, default=0.05)
    train_parser.add_argument('--output_dir', type=str, default="/scratch/zar8jw/EMS-MCQA/log/retrieval or not")
    train_parser.add_argument('--wandb_project', type=str, default='EMSQA')
    train_parser.add_argument('--wandb_entity', type=str, default='gexueren')
    train_parser.add_argument('--wandb_run_name', type=str, default='retrieval_train')
    train_parser.add_argument('--seed', type=int, default=42)

    test_parser = subparsers.add_parser('test')
    test_parser.add_argument('--test_file', type=str, default="/scratch/zar8jw/EMS-MCQA/log/retrieval annotation/open/test.json")
    test_parser.add_argument('--model_epoch', type=int, required=True)
    test_parser.add_argument('--max_length', type=int, default=128)
    test_parser.add_argument('--batch_size', type=int, default=16)
    test_parser.add_argument('--threshold', type=float, default=0.5)
    test_parser.add_argument('--output_dir', type=str, default="/scratch/zar8jw/EMS-MCQA/log/retrieval or not")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)


if __name__ == '__main__':
    main()
