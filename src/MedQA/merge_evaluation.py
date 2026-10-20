import os
import json
import sys
import re

def get_acc(test_list):
    refs=[i["output"] for i in test_list]
    preds=[i["extract_answer"] for i in test_list]

    correct=0
    for ref, pred in zip(refs, preds):
        if ref==pred:
            correct+=1
    return correct/len(refs)

if __name__ == "__main__":

    data_path = sys.argv[1]
    task=sys.argv[2] if sys.argv[2] != "" else "all_ppl"
    all_data=[]
    save_path = f"{data_path}/{task}.json"
    for path in sorted(os.listdir(data_path), key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else -1):
        if re.fullmatch(r"part_?\d+\.json", path):
            data=json.load(open(f"{data_path}/{path}", "r", encoding="utf-8"))
            all_data.extend(data)

    if not all_data:
        raise ValueError("No predictions found in part JSON files")

    with open(save_path, "w", encoding="utf-8") as save_f:
        json.dump(all_data, save_f, ensure_ascii=False, indent=4)

    correct_num = 0
    for item in all_data:
        if item["extract_answer"] == item["output"]:
            correct_num += 1

    print(f"Accuracy: {correct_num / len(all_data)}")
    scores = 100*(correct_num / len(all_data))

    with open(f"{data_path}/test_results.json", "w", encoding="utf-8") as save_f:
        json.dump({"correct_num": correct_num, "accuracy": scores}, save_f, ensure_ascii=False, indent=4)
