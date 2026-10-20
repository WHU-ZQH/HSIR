import json
from tqdm import tqdm
import random
random.seed(1234)
import re
import sys
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig

def get_result(response, answer):
    gold_parsed = parse(
        answer,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    if len(gold_parsed) != 0:
        # We require the answer to be provided in correct latex (no malformed operators)
        answer_parsed = parse(
            response,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
    else:
        return False

    if verify(answer_parsed, gold_parsed):
        return True
    else:
        return False

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
        if get_result(response, item["groud_truth_solution"]):
            right_response.append(response)
        elif True:
            wrong_response.append(response)
    wrong_response = random.sample(wrong_response, min(5, len(wrong_response)))
    budget = max(0, 10 - len(right_response) - len(wrong_response))
    recovered = [response for response in recovered if get_result(response, item["groud_truth_solution"])]
    right_response.extend(random.sample(recovered, min(budget, len(recovered))))

    assert len(right_response)<=10, "Error."
    assert len(wrong_response)<=5, "Error."
    assert len(wrong_response+right_response)<=10, "Error."

    if len(right_response)>0:
        if "system" in item:
            new_data.append({
                "problem": item["problem"],
                "system": item["system"],
                "right_response": right_response,
                "wrong_response": wrong_response
            })
        else:
            new_data.append({
                "problem": item["problem"],
                "right_response": right_response,
                "wrong_response": wrong_response
            })
    else:
        no_right+=1

with open(save_path, "w", encoding="utf-8") as save_f:
    json.dump(new_data, save_f, ensure_ascii=False, indent=4)

print(no_right)
print(len(new_data))
