import os
import json
import sys
import re

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
