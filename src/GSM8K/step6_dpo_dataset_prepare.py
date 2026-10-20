import json
from tqdm import tqdm
import random
import re
import sys
import pickle
random.seed(1234)

data_path=sys.argv[1]
prefix_name=sys.argv[2]
save_name=sys.argv[3]

data=json.load(open(f"{data_path}/{prefix_name}.json"))

no_reject=0
new_data=[]
for item in tqdm(data):
    right_response=item["right_response"]
    wrong_response=item["wrong_response"]

    if len(wrong_response)>0:
        for res in right_response:
            new_data.append({
                "instruction": item["problem"],
                "system": item.get("system", ""),
                "chosen": res,
                "rejected": random.choice(wrong_response),
            })
    else:
        no_reject+=1
                
with open(f"{save_name}", "w", encoding="utf-8") as save_f:
    json.dump(new_data, save_f, ensure_ascii=False, indent=4)

print(no_reject)
print(len(new_data))
