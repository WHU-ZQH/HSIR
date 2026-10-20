import os
import json
import argparse
from tqdm import tqdm
import numpy as np

import random
random.seed(1234)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--indiv_score_thre", type=float, default=-0.5)
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    print(args)

    data = json.load(open(args.data_path))

    for item in tqdm(data):
        
        solutions, indiv_scores = [], []
        if len(item["right_response"]) != len(item["indiv_score"]):
            raise ValueError("Response/score length mismatch")
        if not item["indiv_score"]:
            continue
        indiv_score_mean, indiv_score_std = np.array(item["indiv_score"]).mean(), np.array(item["indiv_score"]).std()
        for res, indiv_score in zip(item["right_response"], item["indiv_score"]):
            # filter the data with lower indiv_score
            if (0.0 if indiv_score_std == 0 else (indiv_score-indiv_score_mean)/indiv_score_std)>args.indiv_score_thre:
                solutions.append(res)
                indiv_scores.append(indiv_score)
        del item["right_response"], item["indiv_score"]
        item["right_response"], item["indiv_score"] = solutions, indiv_scores
            
                
    with open(args.save_path, "w", encoding="utf-8") as w:
        json.dump(data, w, indent=4, ensure_ascii=False)

    print("Finished!")

if __name__ == "__main__":
    main()
