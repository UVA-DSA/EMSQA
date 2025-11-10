
import os
import re
import json
import torch
from tqdm import tqdm
import time
import transformers
import argparse
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
import csv
# import tiktoken

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

# model_name_or_path = "m42-health/Llama3-Med42-70B"
# model_name_or_path = "ProbeMedicalYonseiMAILab/medllama3-v20"
# model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3-70B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3.1-70B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3.1-405B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8"
# model_name_or_path = "meta-llama/Llama-3.3-70B-Instruct"
# model_name_or_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

parser = argparse.ArgumentParser(description="Evaluate EMS MCQA with the LLM of your choice")
parser.add_argument(
    "--model_name_or_path",
    type=str,
    default="meta-llama/Llama-3.3-70B-Instruct",
    help="Name or path of the LLM (e.g. meta-llama/Llama-3.3-70B-Instruct or o4-mini-2025-04-16)"
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
    choices=["zeroshot", "zeroshot_attr", "4-shot", "8-shot", "16-shot", "32-shot", "64-shot", "cot", "cot_attr"],
    type=str,
    default="zeroshot",
    help="zeroshot or fewshot or cot"
)

parser.add_argument(
    "--enable_think",
    type=str2bool,
    default=None,
    help="if use think in Qwen3"
)


args = parser.parse_args()
model_name_or_path = args.model_name_or_path
start = args.start
end = args.end
mode = args.mode
think = args.enable_think





if mode == "infer":
    print(model_name_or_path)
    if "Qwen3" in model_name_or_path:
        # load the tokenizer and the model
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype="auto", device_map="auto")
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
    
def apply_selfbiorag(messages):
    prompt = messages[0]["content"]
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
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


def apply_qwen(messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=think # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
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

def handleError(messages, next_response, pattern):
    t_infer = None
    error, next_response_dict = extract_json(next_response, pattern)
    print(error)
    print(next_response)
    ################################ no json file, regenerate ################################
    cnt = 1
    while error == "no json" and next_response_dict == None and next_response:
        print(f"No json, repeat generating for the {cnt} time")
        raw_prompt = messages[0]["content"]
        prompt = "Plseas return the correct answer in a list, for example [\"a\"].\n\n" + raw_prompt
        messages[0]["content"] = prompt

        # pull out the two “chatGPT-style” tags once
        is_chatgpt = any(tag in model_name_or_path for tag in ("o4-mini", "o3-2025", "gpt"))
        if is_chatgpt:
            next_response, t_infer = apply_chatgpt(messages, temperature=0.3)
        elif "gemini" in model_name_or_path:
            next_response, t_infer = apply_gemini(messages, temperature=0.3)
        elif "Qwen3" in model_name_or_path:
            next_response, t_infer = apply_qwen(messages)
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
            new_response, _ = apply_qwen(messages)
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

def call_llm(question, choices, level, category, prompt_method):
#     prompt = f"""Well organize the textbook. The overall text is all about {sec}. 
#     1. Ignore the figure and its captions.
#     2. If you think there are subtitles in the text, well organize it like "subtitle": "paragraph". But the paragraph must be the exact raw text. no summarization.
#     3. Return a json format {{"your content"}}
# """
    if prompt_method == "zeroshot":
        sys_prompt = "You are an expert in EMS. Choose the correct answer to the following multiple-choice question. Respond ONLY in the JSON schema provided."
        raw_prompt = open("prompts/zeroshot.txt", "r").read()
        prompt = raw_prompt.format(question=question, choices=choices)
        pattern = r'\[.*?\]'
    elif prompt_method == "zeroshot_attr":
        sys_prompt = "You are an expert in EMS. You will be given a certification level, a question category, and a multiple-choice question. Choose the correct answer to the following multiple-choice question. Respond ONLY in the JSON schema provided."
        raw_prompt = open("prompts/zeroshot_attr.txt", "r").read()
        prompt = raw_prompt.format(level=level, category=category, question=question, choices=choices)
        pattern = r'\[.*?\]'

    elif prompt_method == "4-shot":
        sys_prompt = "You are an expert in EMS. Choose the correct answer to the following multiple-choice question. Respond ONLY in the JSON schema provided."
        raw_prompt = open("prompts/4-shot.txt", "r").read()
        question_text = question + "\n" + choices
        prompt = raw_prompt.format(question=question_text)
        pattern = r'\[.*?\]'
    elif prompt_method == "8-shot":
        sys_prompt = "You are an expert in EMS. Choose the correct answer to the following multiple-choice question. Respond ONLY in the JSON schema provided."
        raw_prompt = open("prompts/8-shot.txt", "r").read()
        question_text = question + "\n" + choices
        prompt = raw_prompt.format(question=question_text)
        pattern = r'\[.*?\]'
    elif prompt_method == "16-shot":
        sys_prompt = "You are an expert in EMS. Choose the correct answer to the following multiple-choice question. Respond ONLY in the JSON schema provided."
        raw_prompt = open("prompts/16-shot.txt", "r").read()
        question_text = question + "\n" + choices
        prompt = raw_prompt.format(question=question_text)
        pattern = r'\[.*?\]'
    elif prompt_method == "32-shot":
        sys_prompt = "You are an expert in EMS. Choose the correct answer to the following multiple-choice question. Respond ONLY in the JSON schema provided."
        raw_prompt = open("prompts/32-shot.txt", "r").read()
        question_text = question + "\n" + choices
        prompt = raw_prompt.format(question=question_text)
        pattern = r'\[.*?\]'
    elif prompt_method == "64-shot":
        sys_prompt = "You are an expert in EMS. Choose the correct answer to the following multiple-choice question. Respond ONLY in the JSON schema provided."
        raw_prompt = open("prompts/64-shot.txt", "r").read()
        question_text = question + "\n" + choices
        prompt = raw_prompt.format(question=question_text)
        pattern = r'\[.*?\]'
    elif prompt_method == "cot":
        sys_prompt = """You are an expert in EMS. Choose the correct answer to the following multiple-choice question. Please first think step-by-step and then choose the answer from the provided options. Respond ONLY in the JSON schema provided."""
        raw_prompt = open("prompts/cot.txt", "r").read()
        question_text = question + "\n" + choices
        prompt = raw_prompt.format(question=question_text)
        pattern = r'\{.*?\}'
    elif prompt_method == "cot_attr":
        sys_prompt = "You are an expert Emergency Medical Services (EMS) educator. Answer multiple-choice questions at the requested certification depth, using evidence-based reasoning. Respond ONLY in the JSON schema provided."
        raw_prompt = open("prompts/cot_attr.txt", "r").read()
        question_text = question + "\n" + choices
        prompt = raw_prompt.format(level=level, category=category, question=question_text)
        pattern = r'\{.*?\}'
    else:
        raise Exception("check prompt_method, it should be zeroshot, fewshot or cot")


    messages = [{"role": "system", "content": sys_prompt}]
    messages.append({"role": "user", "content": prompt})

    # pull out the two “chatGPT-style” tags once
    is_chatgpt = any(tag in model_name_or_path for tag in ("o4-mini", "o3-2025", "gpt"))
    if is_chatgpt:
        response, t_infer = apply_chatgpt(messages, temperature=0.3)
    elif "gemini" in model_name_or_path:
        response, t_infer = apply_gemini(messages, temperature=0.3)
    elif "Qwen3" in model_name_or_path:
        response, t_infer = apply_qwen(messages)
    elif "OpenBioLLM" in model_name_or_path:
        response, t_infer = apply_openbiollm(messages, temperature=0.3)
    elif "selfbiorag" in model_name_or_path:
        response, t_infer = apply_selfbiorag(messages)
    else:
        response, t_infer = apply_medllama3(messages, temperature=0.3)

    error, jsonfile = extract_json(response, pattern)
    
    if error:
        prev_t_infer = t_infer
        jsonfile, error_t_infer = handleError(messages, response, pattern)

        # Update t_infer only if error_t_infer is valid
        t_infer = error_t_infer if error_t_infer else prev_t_infer

        if not jsonfile:
            print("After handling error, there is still no json file.")
            return response, None
            # raise Exception("after handling error, there is still no json file")
    
    if not jsonfile:
        print(prompt)
        print(response)
        print("No json file found in the response.")
        return response, None
        # raise Exception("no json, rerun the code")

    return jsonfile, t_infer

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


def json_exists_and_nonempty(path: str) -> bool:
    """
    Returns True if `path` exists, is a valid JSON file,
    and the loaded object is non‐empty (bool(data) == True).
    """
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return bool(data)
    except (json.JSONDecodeError, OSError):
        # either not valid JSON or I/O error
        return False

def emsQA_evaluate(src, prompt_method):

    if src == "close":
        with open(f"../../data/final/MCQA_{src}_final.json", "r") as f:
            data = json.load(f)
    elif src == "open":
        with open(f"../../data/final/test_open.json", "r") as f:
            data = json.load(f)
    elif src == "open-test":
        with open(f"/scratch/zar8jw/EMS-MCQA/code/benchmark/test.json", "r") as f:
            data = json.load(f)
    else:
        raise Exception("check src")

    
    log_dir = f"../../log/benchmark/{model_name_or_path}/{src}/{prompt_method}"
    if "Qwen3" in model_name_or_path and prompt_method in ["zeroshot", "zeroshot_attr"]:
        if think:
            log_dir = os.path.join(log_dir, "think")
        else:
            log_dir = os.path.join(log_dir, "no_think")
        
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    res_dir = f"results/{model_name_or_path}"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # copy the CLI args into local variables
    start_idx, end_idx = start, end

    # if user passed end=-1, run to the end of the list
    if end_idx == -1:
        end_idx = len(data)
    
    
    for i, each in tqdm(enumerate(data[start_idx:end_idx])):
        question = each["question"]
        choices = "\n".join(each["choices"])
        level = each["level"][0]

        if src == "open-test":
            category = ";".join(each["gpt category"])
        else:
            category = ";".join(each["category"])


        if f"{start_idx+i}.json" in os.listdir(log_dir):
            if json_exists_and_nonempty(f"{log_dir}/{start_idx+i}.json"):
                print(f"already exist {start_idx+i}.json")
                continue
        
        output, t_infer = call_llm(question, choices, level, category, prompt_method)

        if not t_infer:
            with open(f"{log_dir}/{start_idx+i}.txt", "w") as f:
                f.write(output)

        if prompt_method not in ["cot", "cot_attr"]:
            with open(f"{log_dir}/{start_idx+i}.json", "w") as f:
                json.dump({"pred": output, "time": t_infer}, f, indent=4)
        else:
            # build base filename
            base = os.path.join(log_dir, f"{start_idx + i}")

            # start with the known fields
            data = {
                "pred": output.get("answer"),
                "time": t_infer
            }

            # then add every other key/value pair in `output`
            for k, v in output.items():
                if k != "answer":
                    data["step by step thinking"] = v

            with open(base + ".json", "w") as f:
                json.dump(data, f, indent=4)


def evaluation_report(src, dir="", save_dir=""):
    if not dir:
        dir = f"../../log/benchmark/{model_name_or_path}/{src}"

    if src == "close":
        assert len(os.listdir(dir)) == 5669, "Close inference not finished"
    elif src in ["open", "open-test"]:
        print(len(os.listdir(dir)), dir)
        bad = [f for f in os.listdir(dir) if not f.endswith('.json')]
        print("Bad files:", bad)
        assert len(os.listdir(dir)) == 3721, "Open inference not finished"
    else:
        raise Exception("check src")

    if not save_dir:
        save_dir = f"./results/{model_name_or_path}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if src == "close":
        data_path = f"../../data/final/MCQA_close_final.json"
    elif src == "open":
        data_path = f"../../data/final/test_open.json"
    elif src == "open-test":
        data_path = "/scratch/zar8jw/EMS-MCQA/code/benchmark/test.json"
    else:
        raise Exception("check src")
    
    with open(data_path, "r") as f:
        data = json.load(f)
    
    categories = [
        "airway_respiration_and_ventilation",
        "anatomy",
        "assessment",
        "cardiology_and_resuscitation",
        "ems_operations",
        "medical_and_obstetrics_gynecology",
        "others",
        "pediatrics",
        "pharmacology",
        "trauma",
    ]

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
    
    preds_cat_all = {cat: [] for cat in categories}
    gts_cat_all   = {cat: [] for cat in categories}
    times_cat_all = {cat: [] for cat in categories}

    preds_cat = {
        lvl: {cat: [] for cat in categories}
        for lvl in ["emr", "emt", "aemt", "paramedic"]
    }
    gts_cat = {
        lvl: {cat: [] for cat in categories}
        for lvl in ["emr", "emt", "aemt", "paramedic"]
    }
    times_cat = {
        lvl: {cat: [] for cat in categories}
        for lvl in ["emr", "emt", "aemt", "paramedic"]
    }

    for i, item in enumerate(data):
        # print(i)
        with open(os.path.join(dir, f"{i}.json"), "r") as f:
            pred = json.load(f)
        levels = item["level"]
        raw_cats = item["category"]

        if isinstance(raw_cats, list):
            cat_list = raw_cats
        else:
            cat_list = [raw_cats]

        if isinstance(pred["pred"], list):
            pred_lst = [p.lower().strip() for p in pred["pred"]]
        elif isinstance(pred["pred"], str):
            pred_lst = [pred["pred"].lower().strip()]

        for cat in cat_list:
            preds_cat_all[cat].append(pred_lst)
            gts_cat_all[cat].append(item["answer"])
            times_cat_all[cat].append(pred["time"])

        for level in levels:
            preds[level].append(pred_lst)
            times[level].append(pred["time"])
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
            for cat in cat_list:
                preds_cat[level][cat].append(pred_lst)
                gts_cat[level][cat].append(item["answer"])
                times_cat[level][cat].append(pred["time"])

        preds["all"].append(pred_lst)
        times["all"].append(pred["time"])
        gts["all"].append(item["answer"])
        res["all"].append({
            "question": item["question"],
            "choices": item["choices"],
            "answer": item["answer"],
            "pred": pred["pred"],
            "infer time": pred["time"],
            "explanation": item["explanation"],
            "link": item["link"],
            "level": item["level"],
            "category": item["category"]
        })
    
    print(f"{model_name_or_path}")
    per_level_stats = []
    for key in ["emr", "emt", "aemt", "paramedic", "all"]:
        em_acc = exact_match_accuracy(preds[key], gts[key])
        f1_samped = average_f1(preds[key], gts[key])
        avg_t_infer = np.mean(times[key])
        per_level_stats.append({              # <--- collect for CSV
            "level": key,
            "acc":  f"{em_acc}",
            "f1":   f"{f1_samped}",
            "t_avg":f"{avg_t_infer}",
        })
        print(f"{key} | exact match acc: {em_acc} | f1_sampled: {f1_samped} | t_infer: {avg_t_infer}")

    print("\nOverall per-category performance:")
    metrics_all = []
    for cat in categories:
        y_true = gts_cat_all[cat]
        if not y_true:
            print(f"  {cat:40} | no samples")
            continue
        y_pred = preds_cat_all[cat]
        t_avg  = np.mean(times_cat_all[cat])
        acc    = exact_match_accuracy(y_pred, y_true)
        f1     = average_f1(y_pred, y_true)
        metrics_all.append({
            "category": cat,
            "acc":      f"{exact_match_accuracy(preds_cat_all[cat], y_true)}",
            "f1":       f"{average_f1(preds_cat_all[cat], y_true)}",
            "t_avg":    f"{np.mean(times_cat_all[cat])}"
        })
        print(f"  {cat:40} | acc: {acc:.3f} | f1: {f1:.3f} | t: {t_avg:.3f}")

    print("\nPer-category performance by certification level:")
    metrics_list = []
    for lvl in ["emr", "emt", "aemt", "paramedic"]:
        print(f"\n{lvl.upper()}:")
        for cat in categories:
            y_true = gts_cat[lvl][cat]
            if not y_true:
                print(f"  {cat}:  no samples")
                continue
            y_pred = preds_cat[lvl][cat]
            t_times = times_cat[lvl][cat]

            acc    = exact_match_accuracy(y_pred, y_true)
            f1     = average_f1(y_pred, y_true)
            t_avg  = np.mean(t_times)
            print(f"  {cat:40} | acc: {acc:.3f} | f1: {f1:.3f} | t: {t_avg:.3f}")
            metrics_list.append({
                "level":    lvl,
                "category": cat,
                "acc":      f"{acc}",
                "f1":       f"{f1}",
                "t_avg":    f"{t_avg}"
            })

    csv_path = os.path.join(save_dir, f"{src}_metrics.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["level","category","acc","f1","t_avg"])
        writer.writeheader()
        writer.writerows(metrics_list)

    csv_path = os.path.join(save_dir, f"{src}_overall_per_category.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["category","acc","f1","t_avg"])
        writer.writeheader()
        writer.writerows(metrics_all)

    csv_path = os.path.join(save_dir, f"{src}_overall_per_level.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["level", "acc", "f1", "t_avg"])
        writer.writeheader()
        writer.writerows(per_level_stats)

    with open(os.path.join(save_dir, f"{src}.json"), "w") as f:
        json.dump(res, f, indent=4)


if __name__ == "__main__":

    if mode == "infer":
        emsQA_evaluate(args.src, args.prompt)
    if mode == "eval":
        if model_name_or_path in ["Qwen/Qwen3-32B", "Qwen/Qwen3-4B"]:
            if args.prompt in ["zeroshot", "zeroshot_attr"]:
                print("**"*30 + f"{args.prompt} evaluation" + "**"*30)
                if not think:
                    print("=="*20 + "no think" + "=="*20)
                    evaluation_report(args.src, 
                                    dir=f"../../log/benchmark/{model_name_or_path}/{args.src}/{args.prompt}/no_think",
                                    save_dir=f"./results/{model_name_or_path}/{args.prompt}/no_think")
                else:
                    print("=="*20 + "think" + "=="*20)
                    evaluation_report(args.src, 
                                    dir=f"../../log/benchmark/{model_name_or_path}/{args.src}/{args.prompt}/think",
                                    save_dir=f"./results/{model_name_or_path}/{args.prompt}/think")
            elif args.prompt in ["4-shot", "8-shot", "16-shot", "32-shot", "64-shot"]:
                print("**"*30 + f"{args.prompt} evaluation" + "**"*30)
                evaluation_report(args.src, 
                                dir=f"../../log/benchmark/{model_name_or_path}/{args.src}/{args.prompt}",
                                save_dir=f"./results/{model_name_or_path}/{args.prompt}")
            elif args.prompt == "cot":
                print("**"*30 + "cot evaluation" + "**"*30)
                evaluation_report(args.src, 
                                dir=f"../../log/benchmark/{model_name_or_path}/{args.src}/cot",
                                save_dir=f"./results/{model_name_or_path}/cot")
            elif args.prompt == "cot_attr":
                print("**"*30 + "cot attribute evaluation" + "**"*30)
                evaluation_report(args.src, 
                                dir=f"../../log/benchmark/{model_name_or_path}/{args.src}/cot_attr",
                                save_dir=f"./results/{model_name_or_path}/cot_attr")
            else:
                raise Exception("check prompt_method, it should be zeroshot, fewshot or cot")

        else:
            evaluation_report(args.src)