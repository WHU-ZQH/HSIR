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

def inference(llm, tokenizer, sampling_params, batch_items, sampling_size):
    results=[]
        
    prompts = []
    for t_item in batch_items:
        if "system" in t_item:
            messages = [{"role": "system", "content": t_item["system"]}]
        else:
            messages = []
        messages.append({"role": "user", "content": t_item["instruction"] + ("\n" + t_item["input"] if t_item.get("input") else "")})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    with torch.no_grad():
        responses = llm.generate(prompts, sampling_params, use_tqdm=False)
        for id, result in enumerate(responses):
            for respone in result.outputs:
                results.append(respone.text)

    final_results=[]
    for i in range(0, len(results), sampling_size):
        final_results.append(results[i:i+sampling_size])

    return final_results

if __name__ == "__main__":
    gpus=1
    llm_model_path = sys.argv[1]
    test_path = sys.argv[2]
    save_path = sys.argv[3]

    temp=float(sys.argv[4])

    part=int(sys.argv[5])
    all_part=int(sys.argv[6])
    
    sampling_size=int(sys.argv[7])

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(test_path, "r", encoding="utf-8") as f:
        test_list = json.load(f)

    start_id = int(len(test_list)/all_part*part)
    end_id = int(len(test_list)/all_part*(part+1))
    test_list=test_list[start_id:end_id]

    sampling_params = SamplingParams(n=sampling_size, temperature=temp, max_tokens=2048)

    llm = LLM(model=llm_model_path,trust_remote_code=True, gpu_memory_utilization=0.6, max_model_len=16384,enforce_eager=True, tensor_parallel_size=gpus) 
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path)

    batch_size = 4
    idx=0
    for batch_start in tqdm(range(0, len(test_list), batch_size)):
        batch_end = min(batch_start + batch_size, len(test_list))
        batch_items = test_list[batch_start:batch_end]

        results = inference(llm, tokenizer, sampling_params, batch_items, sampling_size)
        for i in range(batch_end-batch_start):
            test_list[i+batch_start]["rejection_sampling"] = results[i]

        if (idx+1)%2==0:
            with open(f"{save_path}/part_{part}.json", "w", encoding="utf-8") as save_f:
                json.dump(test_list[:batch_end], save_f, ensure_ascii=False, indent=4)
        idx+=1

    with open(f"{save_path}/part_{part}.json", "w", encoding="utf-8") as save_f:
        json.dump(test_list, save_f, ensure_ascii=False, indent=4)
