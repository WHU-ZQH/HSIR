import os
import torch
import json
import time
import gc
from tqdm import tqdm
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import sys
random.seed(1234)
import re

def get_result(response):
    if "##Answer:\n" not in response or response.count("##Answer:\n")>1:
        return "no"
    else:
        extract_result=response.split("##Answer:\n")[-1]
        if len(extract_result)>0:
            if "A" == extract_result[0]:
                return "A"
            elif "B" == extract_result[0]:
                return "B"
            elif "C" == extract_result[0]:
                return "C"
            elif "D" == extract_result[0]:
                return "D"
            else:
                return "no"
        else:
            return "no"

def get_hint(response, answer):
    sentences = re.split(r'[.!?]+', response)
    
    answer_re = re.compile(r'\banswer\b', flags=re.I)
    label_re      = re.compile(r'\b {}\b'.format(re.escape(answer)))
    not_re = re.compile(r'\bnot\b', flags=re.I)
    no_re = re.compile(r'\bno\b', flags=re.I)
    
    for sent in sentences:
        if answer_re.search(sent) and label_re.search(sent) and not not_re.search(sent) and not no_re.search(sent):
            hint=response.split(sent)[0]+sent
            return f"{hint}.\n\n</think>\n<answer>\n##Explanation:\n"
    return None

def inference(llm, tokenizer, sampling_params, t_item, sampling_size):
    results=[]
    pass_res=list(t_item["rejection_sampling"])
    prompts = []
    hints=[]

    if "label" not in t_item:
        t_item["label"]=get_result(t_item["output"])

    if "system" in t_item:
        messages = [{"role": "system", "content": t_item["system"]}]
    else:
        messages = []
    messages.append({"role": "user", "content": t_item["instruction"] + ("\n" + t_item["input"] if t_item.get("input") else "")})
    
    for res in t_item["rejection_sampling"]:
        pred=get_result(res)
        if pred == t_item["label"]:
            pass  # Original candidates are retained above for DPO.
        else:
            hint=get_hint(res, t_item["label"])
            if hint is not None:
                candidate_messages = messages + [{"role": "assistant", "content": hint}]
                text = tokenizer.apply_chat_template(candidate_messages, tokenize=False, add_generation_prompt=False)
                if text[-11:] == "<|im_end|>\n":
                    text=text[:-11]
                elif text[-8:] =="<|end|>\n":
                    text=text[:-8]
                elif text[-7:] =="<|end|>":
                    text=text[:-7]
                elif text[-5:] =="</s>\n":
                    text=text[:-5]
                elif text[-4:] =="</s>":
                    text=text[:-4]
                elif text[-14:] =="<|endoftext|>\n":
                    text=text[:-14]
                elif text[-13:] =="<|endoftext|>":
                    text=text[:-13]
                elif text[-16:] =="<|end_of_text|>\n":
                    text=text[:-16]
                elif text[-15:] =="<|end_of_text|>":
                    text=text[:-15]
                prompts.append(text)
                hints.append(hint)

    if len(prompts) >0:
        with torch.no_grad():
            responses = llm.generate(prompts, sampling_params, use_tqdm=False)
            for hint, result in zip(hints, responses):
                for respone in result.outputs:
                    res=f"{hint}{respone.text}"
                    pred=get_result(res)
                    if pred == t_item["label"]:
                        results.append(res)
        pass_res.append(results)
    return pass_res

if __name__ == "__main__":
    gpus=1
    llm_model_path = sys.argv[1]
    test_path = sys.argv[2]
    save_path = sys.argv[3]

    temp=float(sys.argv[4])

    part=int(sys.argv[5])
    all_part=int(sys.argv[6])
    
    sampling_size=int(sys.argv[7])
    
    start_index=0

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(test_path, "r", encoding="utf-8") as f:
        test_list = json.load(f)

    start_id = int(len(test_list)/all_part*part)
    end_id = int(len(test_list)/all_part*(part+1))
    test_list=test_list[start_id:end_id]

    if os.path.exists(f"{save_path}/part_{part}.json"):
        raise FileExistsError("Use a fresh output directory; partial outputs are not resumed.")
    
    sampling_params = SamplingParams(n=sampling_size, temperature=temp, max_tokens=2048)

    llm = LLM(model=llm_model_path,trust_remote_code=True, gpu_memory_utilization=0.6, max_model_len=16384,enforce_eager=True, tensor_parallel_size=gpus) 
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path)

    for idx in tqdm(range(len(test_list))):
        results = inference(llm, tokenizer, sampling_params, test_list[idx], sampling_size)
        test_list[idx]["rejection_sampling"] = results

        if (idx+1)%2==0:
            with open(f"{save_path}/part_{part}.json", "w", encoding="utf-8") as save_f:
                json.dump(test_list[:idx+1], save_f, ensure_ascii=False, indent=4)

    with open(f"{save_path}/part_{part}.json", "w", encoding="utf-8") as save_f:
        json.dump(test_list, save_f, ensure_ascii=False, indent=4)
