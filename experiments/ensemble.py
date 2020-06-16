"""
Elliot Schumacher, Johns Hopkins University
Created 5/17/19
"""

import pandas as pd
import re
import os
import logging
from sklearn.preprocessing import *
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
def main():
    root = "/Users/elliotschumacher/Dropbox/git/concept-linker/experiments/results"
    timestamp = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    directory = os.path.join(root, timestamp)
    os.makedirs(directory)
    log = logging.getLogger()
    log.handlers.clear()
    formatter = logging.Formatter('%(message)s')
    fh = logging.FileHandler(os.path.join(directory, "log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.setLevel(logging.INFO)
    log.addHandler(ch)

    neural_results = "/Users/elliotschumacher/Dropbox/git/concept-linker/experiments/thres/run_2019_05_20_13_50_05/results.csv"
    dnorm_results = "/Users/elliotschumacher/Dropbox/git/DNorm-0.0.7 copy/umls_data_test/output_full.csv"

    neural_df = pd.read_csv(neural_results)
    dnorm_df = pd.read_csv(dnorm_results)

    comb_op = "max"
    log.info("Comb op:{0}".format(comb_op))
    neural_n = 10
    dnorm_df.dropna(subset=['score'], inplace=True)
    top_dnorm_scores = dnorm_df.loc[dnorm_df['rank'] < 10.0]["score"].values

    f, axes = plt.subplots(2, figsize=(7, 7), sharex=True)
    sns.distplot(top_dnorm_scores, color="skyblue", ax=axes[0], label="raw")
    sns.distplot(np.tanh(top_dnorm_scores), color="red", ax=axes[1], label="tanh")
    f.legend()
    plt.show()

    output_list = []

    regex = re.compile(r'\(([0-9.-]+)\)')
    sources = []
    ranks = []
    cui_ranks = []
    for i, neu_row in neural_df.iterrows():
        dnorm_scores_df = dnorm_df.loc[dnorm_df['mention_uuid'] == neu_row['_mention_uuid']]

        dnorm_scores_dict = defaultdict(lambda: float("-inf"))
        for j, (_, d_row) in enumerate(dnorm_scores_df.iterrows()):
            if j <= neural_n:
                this_dnorm_score = np.array(d_row['score']).reshape(-1, 1)
                pred_cui = d_row['pred_cui']
                if pred_cui == "cui-less":
                    pred_cui = "CUI-less"
                dnorm_scores_dict[pred_cui] = np.tanh(this_dnorm_score)
            else:
                break
        neural_scores_dict = defaultdict(lambda: float("-inf"))
        for k in range(neural_n):
            column_name = "pred_{0}".format(k)
            #"C0151699=Intracranial hemorrhage, NOS (0.86)"
            cui, rest = neu_row[column_name].split(":")
            score = float(rest)
            neural_scores_dict[cui] = score
        correct_cui = neu_row["_gold_cui"]

        combined_scores = []
        for cui in set(dnorm_scores_dict.keys()).union(set(neural_scores_dict.keys())):
            if comb_op == "sum":
                max_score = dnorm_scores_dict[cui] + neural_scores_dict[cui]
            else:
                max_score = max(dnorm_scores_dict[cui], neural_scores_dict[cui])
            source = "neural"
            if dnorm_scores_dict[cui] > neural_scores_dict[cui]:
                source = "dnorm"
            combined_scores.append((cui, max_score, source))

        out_dict = {
            "_gold_cui" : correct_cui,
            "_mention_uuid" : neu_row['_mention_uuid'],
            "_gold_name" : neu_row["_gold_name"],
            "_sentence" : neu_row["_sentence"],
            "_text" : neu_row["_text"],
        }
        combined_scores = sorted(combined_scores, key=lambda x : x[1], reverse=True)
        this_rank = 0
        this_source = ""
        for i, (cui, score, source) in enumerate(combined_scores):
            if cui == correct_cui:
                this_rank = i+1
                this_source = source
            out_dict["pred_{0}".format(i)] = "{0}:{1}:{2}".format(cui, score, source)
        ranks.append(this_rank)
        sources.append(this_source)

        if correct_cui != "CUI-less":
            cui_ranks.append(this_rank)

        if this_rank == 0:
            out_dict["_mrr"] = 0
        else:
            out_dict["_mrr"] = 1. / float(this_rank)

        if this_rank == 1:
            out_dict["acc"] = 1
        else:
            out_dict["acc"] = 0


        output_list.append(out_dict)

    recip_ranks = []
    for r in ranks:
        if r == 0:
            recip_ranks.append(0)
        else:
            recip_ranks.append(1./r)
    mrr = sum(recip_ranks) / len(recip_ranks)
    log.info("MRR:{0}".format(mrr))

    accuracy = sum(1 for x in ranks if x == 1) / len(ranks)
    log.info("acc:{0}".format(accuracy))

    num_dnorm = sum(1 for x in sources if x == "dnorm")
    log.info("Num dnorm:{0}".format(num_dnorm))
    num_neural = sum(1 for x in sources if x == "neural")
    log.info("Num neural:{0}".format(num_neural))
    log.info("Num none:{0}".format(sum(1 for x in sources if x == "")))


    cui_recip_ranks = []
    for r in cui_ranks:
        if r == 0:
            cui_recip_ranks.append(0)
        else:
            cui_recip_ranks.append(1./r)
    mrr = sum(cui_recip_ranks) / len(cui_recip_ranks)
    log.info("CUI MRR:{0}".format(mrr))

    accuracy = sum(1 for x in cui_ranks if x == 1) / len(cui_ranks)
    log.info("CUI acc:{0}".format(accuracy))

    output_df = pd.DataFrame.from_records(output_list)
    output_df.to_csv(os.path.join(directory, "results.csv"))


if __name__ == "__main__":
    main()