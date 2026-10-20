import json
from tqdm import tqdm
import random
random.seed(1234)
import re
import sys

data_path=sys.argv[1]
prefix_name=sys.argv[2]
save_name=sys.argv[3]

data=json.load(open(f"{data_path}/{prefix_name}.json"))

no_reject=0
new_data=[]
for item in tqdm(data):
    for response in item["right_response"]:
        new_data.append({
            "instruction": item["instruction"],
            "input": item.get("input", ""),
            "system": item.get("system", ""),
            "output": response,
        })

with open(f"{save_name}", "w", encoding="utf-8") as save_f:
    json.dump(new_data, save_f, ensure_ascii=False, indent=4)

print(no_reject)
print(len(new_data))
