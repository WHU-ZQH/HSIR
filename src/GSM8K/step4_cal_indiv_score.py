import os
import json
import torch
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch.nn as nn
log_softmax = nn.LogSoftmax(dim=-1)
nll_loss = nn.NLLLoss(reduction='none')

import random
random.seed(1234)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    args = parser.parse_args()
    return args

def centered_svd_val(Z, alpha=0.001, atten_score=None):
    # assumes Z is in full precision
    J = (torch.eye(Z.shape[0]) - (1 / Z.shape[0]) * torch.ones(Z.shape[0], Z.shape[0])).to(Z.dtype).to(device)
    Sigma = torch.matmul(torch.matmul(Z.t(), J), Z).to(device)
    Sigma = Sigma + alpha * torch.eye(Sigma.shape[0]).to(device)
    svdvals = torch.linalg.svdvals(Sigma).to(device)
    if atten_score==None:
        eigscore = torch.log(svdvals).mean().to(device)
    else:
        eigscore=(torch.log(svdvals)*atten_score).sum().to(device)
    return eigscore

def get_svd_eval(hidden_acts, atten_score):
    svd_scores = []
    Z = torch.transpose(hidden_acts, 0, 1).to(device)
    svd_scores.append(centered_svd_val(Z, 0.001, atten_score).item())
    # print("Sigma matrix shape:",Z.shape[1])
    return svd_scores

# Used to get the ppl and emb for the whole input
def get_reasoning_score(tokenizer, model, text, max_length):
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    
    labels=torch.clone(input_ids).contiguous()
    
    with torch.no_grad(): 
        outputs = model(input_ids, output_attentions=True, output_hidden_states=True, labels=labels)

        attentions = outputs.attentions
        hidden_states = outputs.hidden_states

        atten=attentions[int(len(attentions)/2)+1][0]
        atten=atten.detach().cpu()
        max_atten, _ = torch.max(atten, dim=1)
        max_atten = torch.mean(max_atten, dim=0)

        max_atten=(max_atten/max_atten.sum()).to(device)

        prob_hidden = hidden_states[int(len(attentions)/2)+1]
        svd_scores=get_svd_eval(prob_hidden[0], max_atten)
        
        torch.cuda.empty_cache()

    return svd_scores[0]

def main():

    args = parse_args()
    print(args)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", cache_dir='../cache', output_hidden_states=True, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir='../cache', trust_remote_code=True)

    model.eval()

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    random.shuffle(data)

    start_id = int(len(data)/args.end_idx*args.start_idx)
    end_id = int(len(data)/args.end_idx*(args.start_idx+1))
    sampled_data=data[start_id:end_id]

    start_index=0

    import time
    strat_time = time.time()
    for i in tqdm(range(start_index, len(sampled_data))):

        data_i = sampled_data[i]
        instruct_i = data_i['problem']
        system=data_i["system"]
        content_i = instruct_i
        indiv_scores=[]
        for response in data_i["right_response"]:
            messages = [{"role": "system", "content": system}]
            messages.append({"role": "user", "content": f'{content_i}'})
            messages.append({"role": "assistant", "content": f'{response}'})
            whole_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            indiv_score = get_reasoning_score(tokenizer, model, whole_text, args.max_length)
            indiv_scores.append(indiv_score)

        sampled_data[i]['indiv_score'] = indiv_scores

        if (i+1)%2==0:
            with open(args.save_path, "w", encoding="utf-8") as w:
                json.dump(sampled_data[:i], w, indent=4, ensure_ascii=False)

    with open(args.save_path, "w", encoding="utf-8") as w:
        json.dump(sampled_data, w, indent=4, ensure_ascii=False)

    print('Time Used:',(time.time()-strat_time)/60,'(min)')

if __name__ == "__main__":
    main()
