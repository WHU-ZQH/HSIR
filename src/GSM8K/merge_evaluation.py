import os
import json
import sys
import re
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig


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
        gold_parsed = parse(
            item["answer"],
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        answer_parsed = []
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                item["model_output"],
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
        if gold_parsed and answer_parsed and verify(answer_parsed, gold_parsed):
            correct_num += 1

    print(f"Accuracy: {correct_num / len(all_data)}")
    scores = 100*(correct_num / len(all_data))

    with open(f"{data_path}/test_results.json", "w", encoding="utf-8") as save_f:
        json.dump({"correct_num": correct_num, "accuracy": scores}, save_f, ensure_ascii=False, indent=4)
