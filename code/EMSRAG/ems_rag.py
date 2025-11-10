
import os
import re
import json
import torch
from tqdm import tqdm
import time
import transformers
import argparse
import openai
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from retrieval import EMSRetriever
from transformers import BertModel, BertTokenizerFast
from retrieval_model_train import BertRetrievalClassifier
from multi_task_model_train import QwenRetrievalClassifier
from pathlib import Path
import torch.nn as nn
# Set seed for reproducibility
torch.manual_seed(42)

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

parser = argparse.ArgumentParser(description="Evaluate EMS MCQA with the LLM of your choice")
parser.add_argument(
    "--model_name_or_path",
    type=str,
    default="Qwen/Qwen3-4B",
    help="Name or path of the LLM (e.g. Qwen/Qwen3-4B or o4-mini-2025-04-16)"
)

parser.add_argument(
    "--src",
    type=str,
    default="open",
    help="source: open or close"
)

parser.add_argument(
    "--mode",
    type=str,
    default="infer",
    help="eval or infer"
)

parser.add_argument(
    "--start",
    type=int,
    default=0,
    help="Start index of the data to process"
)
parser.add_argument(
    "--end",
    type=int,
    default=-1,
    help="End index of the data to process (exclusive). Use -1 to indicate processing until the end"
)

parser.add_argument(
    "--prompt",
    choices=["zeroshot", "zeroshot_attr", "fewshot", "cot", "cot_attr"],
    type=str,
    default="zeroshot",
    help="zeroshot or fewshot or cot"
)

parser.add_argument(
    "--use_kb",
    type=str2bool,
    default=True,
    help="if use KB or not (True or False)"
)

parser.add_argument(
    "--kb_k",
    type=int,
    default=32,
    help="top-k KB documents"
)

parser.add_argument(
    "--use_pr",
    type=str2bool,
    default=True,
    help="if use PR or not (True or False)"
)

parser.add_argument(
    "--pr_k",
    type=int,
    default=8,
    help="top-k Patient Records"
)

parser.add_argument(
    "--filter_mode",
    choices=["retrieve_then_filter", "filter_then_retrieve", "global"],
    type=str,
    default="retrieve_then_filter",
    help="filtering mode"
)

parser.add_argument(
    "--use_adapter",
    type=str2bool,
    default=False,
    help="if use adapter or not"
)

parser.add_argument(
    "--enable_think",
    type=str2bool,
    nargs="?",            # so --enable_think alone also works
    const=True,           # if no value given, assume True
    default=True,         # default if flag not provided
    help="enable Qwen model to think or not (true/false)"
)

args = parser.parse_args()
model_name_or_path = args.model_name_or_path
start = args.start
end = args.end
mode = args.mode
filter_mode = args.filter_mode
think = args.enable_think
use_kb = args.use_kb
kb_k = args.kb_k
use_pr = args.use_pr
pr_k = args.pr_k
use_adapter = args.use_adapter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if mode == "infer":
    print("initializing retriever...")
    retriever = EMSRetriever(
        kb_data_dir="/scratch/zar8jw/EMS-MCQA/knowledge",
        pr_data_dir="/scratch/zar8jw/NEMSIS/csv",
        use_kb=use_kb,
        use_pr=use_pr,
        article_model_name="ncbi/MedCPT-Article-Encoder",
        query_model_name="ncbi/MedCPT-Query-Encoder",
        chunk_strategy="token",
        token_chunk_size=512,
        token_overlap=128,
        index_type="flat"
    )

    if use_adapter:
        print("initializing adapter...")
        # ckpt = "/scratch/zar8jw/EMS-MCQA/log/multi-task/Qwen/Qwen3-4B_len:128_lr:0.0001_bs:8_ret:1.0_cat:1.0/epoch20_lora"
        # ckpt = "/scratch/zar8jw/EMS-MCQA/log/multi-task/Qwen/Qwen3-4B_len:128_lr:0.0001_bs:8_ret:0.2_cat:1.0_tau:2.0/epoch20_lora"
        ckpt = "/scratch/zar8jw/EMS-MCQA/log/multi-task/Qwen/Qwen3-4B_len:128_lr:0.0001_bs:8_lvl:0.2_cat:1.0_tau:2.0/epoch20_lora"

        adapter_info = os.path.join(*Path(ckpt).parts[-2:])
        tokenizer=AutoTokenizer.from_pretrained(ckpt)
        tokenizer.padding_side = 'left'
        tokenizer.truncation_side = 'left'
        # base = AutoModel.from_pretrained(args.model_name_or_path)
        base = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        backbone = PeftModel.from_pretrained(
            base,
            ckpt,
        ).to(device)

        with open("/scratch/zar8jw/EMS-MCQA/code/EMSRAG/categories.json", "r") as f:
            all_categories = json.load(f)
        
        with open("/scratch/zar8jw/EMS-MCQA/code/EMSRAG/levels.json", "r") as f:
            all_levels = json.load(f)

        num_cats = len(all_categories)
        num_levels = len(all_levels)
    
        adapter = QwenRetrievalClassifier(
            backbone, 
            num_levels,
            num_cats,
            lora_dropout=0.05,
            lambda_lvl=1.0,
            lambda_cat=1.0,
        ).to(device)

        # 3) Load the two heads
        adapter.level_classifier.load_state_dict(
            torch.load(os.path.join(ckpt, "level_head.pt"),
                    weights_only=True,
                    map_location=device)
        )
        adapter.categorizer.load_state_dict(
            torch.load(os.path.join(ckpt, "categorizer_head.pt"),
            weights_only=True,
            map_location=device)
        )


    print(f"initializing {model_name_or_path}...")
    if "Qwen3" in model_name_or_path:
        # load the tokenizer and the model
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype="auto", device_map="auto")
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map=None)
        model.to(device)

        print("✓ Model is on", next(model.parameters()).device)
    # for any of the OpenAI-hosted models:
    elif any(tag in model_name_or_path for tag in ["o3-2025","o4-mini","gpt"]):
        # 1) load your env file
        load_dotenv("./api_key/openai.env")
        # 2) correctly assign the key
        openai.api_key = os.getenv("OPENAI_API_KEY")
    elif "gemini" in model_name_or_path:
        #gemini-2.5-pro-preview-05-06
        load_dotenv("./api_key/gemini.env")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found in environment")
        client = genai.Client(api_key=api_key)
    elif "selfbiorag":
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype="auto", device_map="auto")
    else:
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
else:
    ckpt = "/scratch/zar8jw/EMS-MCQA/log/multi-task/Qwen/Qwen3-4B_len:128_lr:0.0001_bs:8_lvl:0.2_cat:1.0_tau:2.0/epoch20_lora"
    adapter_info = os.path.join(*Path(ckpt).parts[-2:])


def apply_selfbiorag(messages):
    prompt = messages[0]["content"]
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    input_len    = model_inputs.input_ids.shape[1]
    start_time = time.time()
    outputs = model.generate(
        **model_inputs,
        eos_token_id     = tokenizer.eos_token_id,
        pad_token_id     = tokenizer.eos_token_id,
    )
    end_time = time.time()
    t_infer = end_time - start_time
    gen_ids = outputs[0, input_len:].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text, t_infer

def apply_gemini(messages, model=model_name_or_path, temperature=0.3):
    start_time = time.time()
    response = client.models.generate_content(
        model=model, 
        contents=messages[0]["content"],
        config=types.GenerateContentConfig(
            temperature=temperature
        )
    )
    end_time = time.time()

    t_infer = end_time - start_time
    time.sleep(5)
    return response.text, t_infer

def apply_chatgpt(messages, model=model_name_or_path, temperature=0.3, max_tokens=8192):
    if "o4-mini" in model_name_or_path or "o3-2025" in model_name_or_path: 
        start_time = time.time()
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            # max_completion_tokens=max_tokens,
        )
        end_time = time.time()
    elif "gpt" in model_name_or_path:
        start_time = time.time()
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            # max_tokens=max_tokens,
            # top_p=top_p,
        )
        end_time = time.time()
    # return response.choices[0].message["content"]
    t_infer = end_time - start_time
    return response.choices[0].message.content, t_infer

def apply_medllama3(messages, temperature=0.3, max_tokens=-1, top_k=150, top_p=0.75):
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if max_tokens != -1:
        start_time = time.time()
        outputs = pipeline(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        end_time = time.time()
        response = outputs[0]["generated_text"][len(prompt) :]
    else:
        start_time = time.time()
        outputs = pipeline(
            prompt,
            max_new_tokens=8192,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        end_time = time.time()
        response = outputs[0]["generated_text"][len(prompt) :]
    t_infer = end_time - start_time
    return response, t_infer

def apply_qwen(messages, enable_think):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_think # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    # conduct text completion
    
    start_time = time.time()
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    end_time = time.time()
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    t_infer = end_time - start_time
    return content, t_infer

def apply_openbiollm(messages, max_tokens=-1, temperature=0.3, top_p=0.9):
    terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if max_tokens != -1:
        start_time = time.time()
        outputs = pipeline(
            prompt,
            max_new_tokens=max_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        end_time = time.time()
        response = outputs[0]["generated_text"][len(prompt) :]
    else:
        start_time = time.time()
        outputs = pipeline(
            prompt,
            max_new_tokens=8192,
            do_sample=True,
            eos_token_id=terminators,
            temperature=temperature,
            top_p=top_p,
        )
        end_time = time.time()
        response = outputs[0]["generated_text"][len(prompt) :]
    t_infer = end_time - start_time
    return response, t_infer

def extract_json(response, pattern = r'\[.*?\]'):
    def extract_answer(text):
        # Try to extract option letter and optional explanation
        # match = re.search(r"(?:The\s+correct\s+answer\s+is|Answer|Correct\s+option)[:\s]*([a-dA-D])(?:[.)]?\s*([^\n.]+)?)?", text, re.IGNORECASE)
        # if match:
        #     option = match.group(1).lower()
        #     if option in ["a", "b", "c", "d", "e", "f", "g"]:
        #         return [option]
        # return None

        match = re.search(r"(?:The\s+correct\s+answer\s+is|Answer|Correct\s+option)[:\s]*([a-zA-Z])(?:[.)]?\s*([^\n.]+)?)?", text, re.IGNORECASE)

        if match:
            option = match.group(1).lower()
            if option in ["a", "b", "c", "d", "e", "f", "g"]:
                return [option]
        return None

    # Regular expression pattern to match JSON content

    # Search for the pattern in the text
    # match = re.search(pattern, response, re.DOTALL)
    matches = re.findall(pattern, response, re.DOTALL)

    if not matches:
        print("No JSON object found in the text.")
        print(response)
        json_data = extract_answer(response)
        if json_data: 
            return None, json_data
        else:
            # 2) LaTeX boxed style:  …\boxed{c}…
            boxed = re.search(r'\\boxed\{\s*([A-Ga-g])\s*\}', response)
            if boxed:
                return None, [boxed.group(1).lower()]
        
        return "no json", None

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

def handleError(messages, next_response, prompt_method, enable_think):
    if prompt_method == "cot":
        pattern = r'\{.*?\}'
    else:
        pattern = r'\[.*?\]'

    t_infer = None
    error, next_response_dict = extract_json(next_response, pattern)
    print(error)
    print(next_response)
    ################################ no json file, regenerate ################################
    cnt = 1
    while error == "no json" and next_response_dict == None and next_response:
        print(f"No json, repeat generating for the {cnt} time")
        raw_prompt = messages[1]["content"]
        if prompt_method != "cot":
            prompt = "Plsease return the correct answer in a list, for example [\"a\"]. If no answer return [\"none\"]\n\n" + raw_prompt
        else:
            prompt = """Plsease formatted as (only Letters e.g.: A/B/C/D) with explanation as Dict{{"step_by_step_thinking": Str(explanation), "answer": List(A, B) or List([C])}}. If no answer return Dict{{"step_by_step_thinking": Str(explanation), "answer": List(["none"])}}""" + raw_prompt
        messages[1]["content"] = prompt

        # pull out the two “chatGPT-style” tags once
        is_chatgpt = any(tag in model_name_or_path for tag in ("o4-mini", "o3-2025", "gpt"))
        if is_chatgpt:
            next_response, t_infer = apply_chatgpt(messages, temperature=0.3)
        elif "gemini" in model_name_or_path:
            next_response, t_infer = apply_gemini(messages, temperature=0.3)
        elif "Qwen3" in model_name_or_path:
            next_response, t_infer = apply_qwen(messages, enable_think)
        elif "OpenBioLLM" in model_name_or_path:
            next_response, t_infer = apply_openbiollm(messages, temperature=0.3)
        elif "selfbiorag" in model_name_or_path:
            next_response, t_infer = apply_selfbiorag(messages)
        else:
            next_response, t_infer = apply_medllama3(messages, temperature=0.3)

        error, next_response_dict = extract_json(next_response, pattern)
        cnt += 1

    ################################ json file incorrect ################################
    cnt = 1
    while error and cnt < 10:
        t_infer = None
        print(f"fix error for the {cnt} time")
        prompt = f"""There is an Error decoding JSON: {error} in the following json data
        {next_response_dict}, Can you fix the error and return the correct json format. Directly return the json without explanation.
        """
        messages = [{"role": "user", "content": prompt}]


        # pull out the two “chatGPT-style” tags once
        is_chatgpt = any(tag in model_name_or_path for tag in ("o4-mini", "o3-2025", "gpt"))
        if is_chatgpt:
            new_response, _ = apply_chatgpt(messages, temperature=0.3)
        elif "gemini" in model_name_or_path:
            new_response, _ = apply_gemini(messages, temperature=0.3)
        elif "Qwen3" in model_name_or_path:
            new_response, _ = apply_qwen(messages, enable_think=True)
        elif "OpenBioLLM" in model_name_or_path:
            new_response, _ = apply_openbiollm(messages, temperature=0.3)
        elif "selfbiorag" in model_name_or_path:
            new_response, _ = apply_selfbiorag(messages)
        else:
            new_response, _ = apply_medllama3(messages, temperature=0.3)

        print(new_response)
        error, next_response_dict = extract_json(new_response, pattern)
        cnt += 1
    
    if error:
        print("Error decoding JSON: ", error)
        print("Response: ", next_response)
        print("Next response dict: ", next_response_dict)
        # raise Exception("Error decoding JSON after handling error")
        
    return next_response_dict, t_infer

def call_llm(question, choices, prompt_method, kb_topk, pr_topk, category, level, filter_mode, enable_think):
#     prompt = f"""Well organize the textbook. The overall text is all about {sec}. 
#     1. Ignore the figure and its captions.
#     2. If you think there are subtitles in the text, well organize it like "subtitle": "paragraph". But the paragraph must be the exact raw text. no summarization.
#     3. Return a json format {{"your content"}}
# """
    use_think = enable_think
    t_retrieve = 0

    assert (not use_kb) or (kb_topk > 0), "When use_kb=True, kb_topk must be > 0"
    assert (not use_pr) or (pr_topk > 0), "When use_pr=True, pr_topk must be > 0"
    if filter_mode == "retrieve_then_filter":
        start_time = time.time()
        docs = retriever.retrieve(query=question, kb_k=kb_topk, pr_k=pr_topk, categories=category, mode=filter_mode)
        end_time = time.time()
    elif filter_mode == "filter_then_retrieve":
        start_time = time.time()
        docs = retriever.retrieve(query=question, kb_k=kb_topk, pr_k=pr_topk, categories=category, mode=filter_mode)
        end_time = time.time()
    elif filter_mode == "global":
        start_time = time.time()
        docs = retriever.retrieve(query=question, kb_k=kb_topk, pr_k=pr_topk, mode=filter_mode)
        end_time = time.time()
    else:
        raise Exception("check filter_mode, it should be retrieve_then_filter or filter_then_retrieve")
    t_retrieve = end_time - start_time

    kb_docs, pt_records = "", ""
    if use_kb:
        kb_docs = "EMS Knowledge:\n" + "\n".join(f"- {doc['title']}: {doc['content']}" for doc in docs["kb"])
    
    if use_pr:
        pt_records = "Patient Records:\n" + "\n".join(f"- {doc['text']}" for doc in docs["pr"])

    # Combine whichever parts are not empty, joined by double newlines
    docs_text = "\n\n".join(part for part in [kb_docs, pt_records] if part)

    # # if there is no retrieved documents
    # # and the mode is first filter category then retrieve per category,
    # # then we do want think (this only appears in PR retrieve because KB has all categories)
    # if not docs_text and filter_mode == "filter_then_retrieve":
    #     use_think=False

    if prompt_method == "zeroshot":
        sys_prompt = "You are an expert in EMS. Choose the correct answer to the following multiple-choice question. Respond ONLY in the JSON schema provided."
        if use_think:
            raw_prompt = open("prompts/rag_zeroshot.txt", "r").read()
            prompt = raw_prompt.format(
                documents=docs_text,
                question=question, 
                choices=choices
            )
        else:
            raw_prompt = open("prompts/zeroshot.txt", "r").read()
            prompt = raw_prompt.format(
                question=question, 
                choices=choices
            )

        pattern = r'\[.*?\]'
    elif prompt_method == "zeroshot_attr":
        sys_prompt = "You are an expert in EMS. You will be given a certification level, a question category, and a multiple-choice question. Choose the correct answer to the following multiple-choice question. Respond ONLY in the JSON schema provided."
        raw_prompt = open("prompts/rag_zeroshot_attr.txt", "r").read()
        prompt = raw_prompt.format(
                documents=docs_text,
                question=question, 
                choices=choices,
                level=level,
                category=category
            )
        pattern = r'\[.*?\]'
    elif prompt_method == "cot":
        sys_prompt = """You are a helpful EMS expert, and your task is to answer a multi-choice ems question using the relevant documents. Please first think step-by-step and then choose the answer from the provided options. Your responses will be used for research purposes only, so please have a definite answer. Respond ONLY in the JSON schema provided."""
        raw_prompt = open("prompts/rag_cot.txt", "r").read()
        prompt = raw_prompt.format(
            documents=docs_text,
            question=question,
            choices=choices
        )
        pattern = r'\{.*?\}'
    elif prompt_method == "cot_attr":
        sys_prompt = f"""You are a helpful EMS expert, and your task is to answer a multi-choice ems question at the requested certification depth using the relevant documents. Please first think step-by-step from the standpoint of {category} and then choose the answer from the provided options. Your responses will be used for research purposes only, so please have a definite answer. Respond ONLY in the JSON schema provided."""
        raw_prompt = open("prompts/rag_cot_attr.txt", "r").read()
        prompt = raw_prompt.format(
            documents=docs_text,
            question=question,
            choices=choices,
            level=level,
            category=category
        )
        pattern = r'\[.*?\]'
    else:
        raise Exception("check prompt_method, it should be zeroshot")


    messages = [{"role": "system", "content": sys_prompt}]
    messages.append({"role": "user", "content": prompt})

    # pull out the two “chatGPT-style” tags once
    is_chatgpt = any(tag in model_name_or_path for tag in ("o4-mini", "o3-2025", "gpt"))
    if is_chatgpt:
        response, t_infer = apply_chatgpt(messages, temperature=0.3)
    elif "gemini" in model_name_or_path:
        response, t_infer = apply_gemini(messages, temperature=0.3)
    elif "Qwen3" in model_name_or_path:
        response, t_infer = apply_qwen(messages, enable_think=use_think)
    elif "OpenBioLLM" in model_name_or_path:
        response, t_infer = apply_openbiollm(messages, temperature=0.3)
    elif "selfbiorag" in model_name_or_path:
        response, t_infer = apply_selfbiorag(messages)
    else:
        response, t_infer = apply_medllama3(messages, temperature=0.3)


    error, jsonfile = extract_json(response, pattern)
    
    if error:
        prev_t_infer = t_infer
        jsonfile, error_t_infer = handleError(messages, response, prompt_method, enable_think=think)

        # Update t_infer only if error_t_infer is valid
        t_infer = error_t_infer if error_t_infer else prev_t_infer

        if not jsonfile:
            print("After handling error, there is still no json file.")
            return response, None, t_retrieve
            # raise Exception("after handling error, there is still no json file")
    
    if not jsonfile:
        print(prompt)
        print(response)
        print("No json file found in the response.")
        return response, None, t_retrieve
        # raise Exception("no json, rerun the code")

    return jsonfile, t_infer, t_retrieve

def call_adapter(question: str, choices: list, threshold: float = 0.5) -> dict:
    """
    Load a trained model from `model_dir` and perform inference on a single example.

    Returns:
      {
        'probability': float,
        'prediction': int  # 1 for retrieval, 0 for not
      }
    """
    adapter.eval()

    # prepare text
    text = question + ' ' + ' '.join(choices)
    enc = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    ).to(device)

    sigmoid = nn.Sigmoid()
    start = time.time()
    with torch.no_grad():
        start = time.time()
        out = adapter(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        end = time.time()
    t_adapter = end-start

    lvl_logits = out["level_logits"]               # shape (1, num_levels)
    lvl_idx    = int(lvl_logits.argmax(dim=-1).item())
    lvl_pred   = all_levels[lvl_idx]

    # category prediction (multi-label)
    cat_logits = out["category_logits"]             # shape (1, num_cats)
    probs      = sigmoid(cat_logits).squeeze(0).cpu().numpy()
    mask       = probs > threshold
    if not mask.any():
        # if nothing crosses threshold, force the max-prob class
        mask[probs.argmax()] = True
    cat_pred = [all_categories[i] for i, m in enumerate(mask) if m]

    return {'level': lvl_pred, 
            'category': cat_pred,
            "t_adapter": t_adapter}

def average_f1(preds, golds):
    def f1_per_sample(pred, gold):
        pred_set = set(pred)
        gold_set = set(gold)
        if not pred_set and not gold_set:
            return 1.0  # both empty: perfect match
        if not pred_set or not gold_set:
            return 0.0  # one is empty: 0 score

        tp = len(pred_set & gold_set)
        precision = tp / len(pred_set)
        recall = tp / len(gold_set)

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    return sum(f1_per_sample(p, g) for p, g in zip(preds, golds)) / len(golds)

def exact_match_accuracy(preds, golds):
    correct = sum(set(p) == set(g) for p, g in zip(preds, golds))
    return correct / len(golds)

def emsQA_evaluate(src, prompt_method, enable_think):

    use_think = enable_think
    if src == "close":
        with open(f"../../data/final/MCQA_{src}_final.json", "r") as f:
            data = json.load(f)
    elif src == "open":
        with open(f"../../data/final/test_open.json", "r") as f:
            data = json.load(f)
    else:
        raise Exception("check src")

    if use_kb and not use_pr:
        log_dir = f"../../log/benchmark/{model_name_or_path}/{src}/rag_kb({kb_k})_{prompt_method}_{filter_mode}_think:{use_think}_adapter:{use_adapter}"
        if use_adapter:
            log_dir = f"../../log/benchmark/{model_name_or_path}/{src}/rag_kb({kb_k})_{prompt_method}_{filter_mode}_think:{use_think}_adapter:{adapter_info}"
    
    if use_pr and not use_kb:
        log_dir = f"../../log/benchmark/{model_name_or_path}/{src}/rag_pr({pr_k})_{prompt_method}_{filter_mode}_think:{use_think}_adapter:{use_adapter}"
        if use_adapter:
            log_dir = f"../../log/benchmark/{model_name_or_path}/{src}/rag_kb({kb_k})_{prompt_method}_{filter_mode}_think:{use_think}_adapter:{adapter_info}"
    
    if use_kb and use_pr:
        log_dir = f"../../log/benchmark/{model_name_or_path}/{src}/rag_kb({kb_k})&pr({pr_k})_{prompt_method}_{filter_mode}_think:{use_think}_adapter:{use_adapter}"
        if use_adapter:
            log_dir = f"../../log/benchmark/{model_name_or_path}/{src}/rag_kb({kb_k})_{prompt_method}_{filter_mode}_think:{use_think}_adapter:{adapter_info}"
    

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    res_dir = f"results/{model_name_or_path}"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # result = []


    # copy the CLI args into local variables
    start_idx, end_idx = start, end

    # if user passed end=-1, run to the end of the list
    if end_idx == -1:
        end_idx = len(data)
    
    
    for i, each in tqdm(enumerate(data[start_idx:end_idx])):
        question = each["question"]
        choices = "\n".join(each["choices"])
        category = each["category"]
        level = each["level"]

        if f"{start_idx+i}.json" in os.listdir(log_dir):
            # print(f"already exist {start_idx+i}.json")
            continue
        
        t_adapter = 0
        if use_adapter:
            adapter_out = call_adapter(question, each["choices"])
            pred_level, pred_category, t_adapter = adapter_out["level"], adapter_out["category"], adapter_out["t_adapter"]
            
            print(pred_level, level, pred_category, category)
            output, t_infer, t_retrieve = call_llm(question, 
                                                choices, 
                                                prompt_method, 
                                                kb_k,
                                                pr_k,
                                                pred_category,
                                                pred_level,
                                                filter_mode=filter_mode,
                                                enable_think=use_think
                                                )
        else:
            output, t_infer, t_retrieve = call_llm(question, 
                                        choices, 
                                        prompt_method, 
                                        kb_k,
                                        pr_k,
                                        category,
                                        level,
                                        filter_mode=filter_mode,
                                        enable_think=use_think
                                        )
        if not t_infer:
            with open(f"{log_dir}/{start_idx+i}.txt", "w") as f:
                f.write(output)

        if prompt_method != "cot":
            with open(f"{log_dir}/{start_idx+i}.json", "w") as f:
                json.dump({"pred": output, "t_retrieve": t_retrieve, "t_adapter": t_adapter, "t_infer": t_infer, 
                           "t_total": t_infer+t_retrieve+t_adapter}, f, indent=4)
        else:

            # build base filename
            base = os.path.join(log_dir, f"{start_idx + i}")

            # start with the known fields
            data = {
                "pred": output.get("answer"),
                "t_retrieve": t_retrieve,
                "t_adapter": t_adapter,
                "t_infer": t_infer,
                "t_total": t_retrieve + t_infer + t_adapter
            }

            # then add every other key/value pair in `output`
            for k, v in output.items():
                if k != "answer":
                    data["step by step thinking"] = v

            with open(base + ".json", "w") as f:
                json.dump(data, f, indent=4)

def evaluation_report(src, dir="", save_dir=""):
    if not dir:
        dir = f"/scratch/zar8jw/EMS-MCQA/log/benchmark/{model_name_or_path}/{src}"

    if src == "close":
        bad = [f for f in os.listdir(dir) if not f.endswith('.json')]
        print("Bad files:", bad)
        assert len(os.listdir(dir)) == 5669, "Close inference not finished"
    elif src == "open":
        print(len(os.listdir(dir)), dir)
        bad = [f for f in os.listdir(dir) if not f.endswith('.json')]
        print("Bad files:", bad)
        assert len(os.listdir(dir)) == 3721, "Open inference not finished"
    else:
        raise Exception("check src")

    if not save_dir:
        save_dir = f"/scratch/zar8jw/EMS-MCQA/code/benchmark/results/{model_name_or_path}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if src == "close":
        data_path = f"../../data/final/MCQA_close_final.json"
    elif src == "open":
        data_path = f"../../data/final/test_open.json"
    else:
        raise Exception("check src")
    
    with open(data_path, "r") as f:
        data = json.load(f)
    
    res = {
        "all": [],
        "emr": [],
        "emt": [],
        "aemt": [],
        "paramedic": [],
        "NA": []
    }

    preds = {
        "all": [],
        "emr": [],
        "emt": [],
        "aemt": [],
        "paramedic": [],
        "NA": []
    }

    gts = {
        "all": [],
        "emr": [],
        "emt": [],
        "aemt": [],
        "paramedic": [],
        "NA": []
    }

    times = {
        "all": [],
        "emr": [],
        "emt": [],
        "aemt": [],
        "paramedic": [],
        "NA": []
    }

    for i, item in enumerate(data):
        with open(os.path.join(dir, f"{i}.json"), "r") as f:
            pred = json.load(f)
        levels = item["level"]

        pred_ans = [p.lower().strip() for p in pred["pred"]]

        #sanity check
        for j in range(len(pred_ans)):
            if len(pred_ans[j]) != 1:
                if "none" in pred_ans[j] or "None" in pred_ans[j] or "" in pred_ans[j]:
                    continue
                raise Exception(f"{item}\n\ncheck {i}.json")

        for level in levels:
            preds[level].append(pred_ans)
            times[level].append(pred["t_total"])
            gts[level].append(item["answer"])
            res[level].append({
                "question": item["question"],
                "choices": item["choices"],
                "answer": item["answer"],
                "pred": pred,
                "explanation": item["explanation"],
                "link": item["link"],
                "level": item["level"],
                "category": item["category"]
        })

        preds["all"].append(pred_ans)
        times["all"].append(pred["t_total"])
        gts["all"].append(item["answer"])
        res["all"].append({
            "question": item["question"],
            "choices": item["choices"],
            "answer": item["answer"],
            "pred": pred_ans,
            "t_total time": pred["t_total"],
            "explanation": item["explanation"],
            "link": item["link"],
            "level": item["level"],
            "category": item["category"]
        })
    
    print(f"{model_name_or_path}")
    for key in ["emr", "emt", "aemt", "paramedic", "all"]:
        em_acc = exact_match_accuracy(preds[key], gts[key])
        f1_samped = average_f1(preds[key], gts[key])
        avg_t = np.mean(times[key])
        print(f"{key} | exact match acc: {em_acc} | f1_sampled: {f1_samped} | t_total: {avg_t}")
    
    with open(os.path.join(save_dir, f"{src}_think:{think}_adapter:{use_adapter}.json"), "w") as f:
        json.dump(res, f, indent=4)


if __name__ == "__main__":

    if mode == "infer":
        emsQA_evaluate(args.src, args.prompt, args.enable_think)
    if mode == "eval":
        if model_name_or_path == "Qwen/Qwen3-4B":

            if use_kb and not use_pr:
                print("**"*20 + f"EMSRAG: {filter_mode} - KB topk: {kb_k}" + "**"*20)
                dir = f"/scratch/zar8jw/EMS-MCQA/log/benchmark/Qwen/Qwen3-4B/{args.src}/rag_kb({kb_k})_{args.prompt}_{filter_mode}_think:{think}_adapter:{use_adapter}"
                if use_adapter:
                    dir = f"/scratch/zar8jw/EMS-MCQA/log/benchmark/Qwen/Qwen3-4B/{args.src}/rag_kb({kb_k})_{args.prompt}_{filter_mode}_think:{think}_adapter:{adapter_info}"
                save_dir = f"/scratch/zar8jw/EMS-MCQA/code/benchmark/results/Qwen/Qwen3-4B/rag_kb({kb_k})_{args.prompt}_{filter_mode}"
            
            if use_pr and not use_kb:
                print("**"*20 + f"EMSRAG: {filter_mode} - PR topk: {pr_k}" + "**"*20)
                dir = f"/scratch/zar8jw/EMS-MCQA/log/benchmark/Qwen/Qwen3-4B/{args.src}/rag_pr({pr_k})_{args.prompt}_{filter_mode}_think:{think}_adapter:{use_adapter}"
                if use_adapter:
                    dir = f"/scratch/zar8jw/EMS-MCQA/log/benchmark/Qwen/Qwen3-4B/{args.src}/rag_kb({kb_k})_{args.prompt}_{filter_mode}_think:{think}_adapter:{adapter_info}"
                save_dir = f"/scratch/zar8jw/EMS-MCQA/code/benchmark/results/Qwen/Qwen3-4B/rag_pr({pr_k})_{args.prompt}_{filter_mode}"
            
            if use_kb and use_pr:
                print("**"*20 + f"EMSRAG: {filter_mode} - KB topk: {kb_k} - PR topk: {pr_k}" + "**"*20)
                dir = f"/scratch/zar8jw/EMS-MCQA/log/benchmark/Qwen/Qwen3-4B/{args.src}/rag_kb({kb_k})&pr({pr_k})_{args.prompt}_{filter_mode}_think:{think}_adapter:{use_adapter}"
                if use_adapter:
                    dir = f"/scratch/zar8jw/EMS-MCQA/log/benchmark/Qwen/Qwen3-4B/{args.src}/rag_kb({kb_k})_{args.prompt}_{filter_mode}_think:{think}_adapter:{adapter_info}"
                save_dir = f"/scratch/zar8jw/EMS-MCQA/code/benchmark/results/Qwen/Qwen3-4B/rag_kb({kb_k})&pr({pr_k})_{args.prompt}_{filter_mode}"

            evaluation_report(args.src, 
                              dir=dir,
                              save_dir=save_dir)

        else:
            evaluation_report(args.src)