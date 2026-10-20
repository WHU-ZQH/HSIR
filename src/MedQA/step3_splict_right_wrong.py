import json
from tqdm import tqdm
import random
random.seed(1234)
import re
import sys

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

def contains_chinese(s):
    pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(pattern.search(s))

data_path=sys.argv[1]
prefix_name=sys.argv[2]
save_path=sys.argv[3]

data=json.load(open(f"{data_path}/{prefix_name}.json"))

no_right=0
new_data=[]
all_right_data=[]
for item in tqdm(data):
    right_response=[]
    wrong_response=[]
    
    originals = [r for r in item["rejection_sampling"] if isinstance(r, str)][:10]
    recovered = [r for group in item["rejection_sampling"] if isinstance(group, list) for r in group]
    for response in originals:
        if get_result(response)==item["label"]:
            right_response.append(response)
        elif get_result(response) in ["A", "B", "C", "D"] and not contains_chinese(response):
            wrong_response.append(response)
    wrong_response = random.sample(wrong_response, min(5, len(wrong_response)))
    budget = max(0, 10 - len(right_response) - len(wrong_response))
    recovered = [response for response in recovered if get_result(response)==item["label"]]
    right_response.extend(random.sample(recovered, min(budget, len(recovered))))

    assert len(right_response)<=10, "Error."
    assert len(wrong_response)<=5, "Error."
    assert len(wrong_response+right_response)<=10, "Error."

    if len(right_response)>0:
        if "system" in item:
            new_data.append({
                "instruction": item["instruction"],
                "input": item.get("input", ""),
                "system": item["system"],
                "right_response": right_response,
                "wrong_response": wrong_response
            })
        else:
            new_data.append({
                "instruction": item["instruction"],
                "input": item.get("input", ""),
                "right_response": right_response,
                "wrong_response": wrong_response
            })
    else:
        no_right+=1

with open(save_path, "w", encoding="utf-8") as save_f:
    json.dump(new_data, save_f, ensure_ascii=False, indent=4)

print(no_right)
print(len(new_data))
