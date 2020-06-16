"""
Elliot Schumacher, Johns Hopkins University
Created 5/19/19
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
from scipy.stats import describe
def main():
    root = "/Users/elliotschumacher/Dropbox/git/concept-linker/experiments/thres"
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

    neural_results = "/Users/elliotschumacher/Dropbox/git/concept-linker/results/run_2019_05_20_23_38_43_c20/eval_dev.csv"

    neural_df = pd.read_csv(neural_results)

    output_list = []

    regex = re.compile(r'\(([0-9.-]+)\)')
    scores = defaultdict(lambda :[])
    for i, neu_row in neural_df.iterrows():

        column_name = "~pred_cuis_0"
        #"C0151699=Intracranial hemorrhage, NOS (0.86)"
        cui, rest = neu_row[column_name].split("=")
        score = float(regex.findall(rest)[-1])
        correct_cui = neu_row["_gold_cui"]
        if correct_cui == "CUI-less":
            scores["CUI-less"].append(score)
        else:
            scores["CUI"].append(score)

    sns.distplot(scores["CUI-less"], color="skyblue", label="cuiless", kde=False, rug=True, bins=20)
    plt.show()

    sns.distplot(np.tanh(scores["CUI"]), color="red", label="cui", kde=False, rug=True, bins=20)
    plt.show()
    cuiless_desc = describe(scores["CUI-less"])
    cui_desc = describe(scores["CUI"])
    print("CUI-less:{0}".format(cuiless_desc))
    print("CUI:{0}".format(cui_desc))

    threshold = cuiless_desc.mean# + cuiless_desc.variance
    neural_n = 10
    ranks = []
    ranks_cui = []
    cui_ranks = []

    neural_results = "/Users/elliotschumacher/Dropbox/git/concept-linker/results/run_2019_05_20_23_38_43_c20/eval_test.csv"

    neural_df = pd.read_csv(neural_results)

    for i, neu_row in neural_df.iterrows():

        neural_scores_list = []
        cuiless_added = False
        for k in range(neural_n):
            column_name = "~pred_cuis_{0}".format(k)
            #"C0151699=Intracranial hemorrhage, NOS (0.86)"
            cui, rest = neu_row[column_name].split("=")
            score = float(regex.findall(rest)[-1])
            if score <= threshold and not cuiless_added:
                neural_scores_list.append(("CUI-less", threshold))
                cuiless_added = True
            #neural_scores_dict[cui] = score
            neural_scores_list.append((cui, score))
        correct_cui = neu_row["_gold_cui"]
        correct_score = neu_row["_gold_cui_score"]
        correct_rank = neu_row["_gold_cui_rank"]


        out_dict = {
            "_gold_cui" : correct_cui,
            "_mention_uuid" : neu_row['~~mention_uuid'],
            "_gold_name" : neu_row["_gold_name"],
            "_sentence" : neu_row["_sentence"],
            "_text" : neu_row["_text"],
        }


        combined_scores = sorted(neural_scores_list, key=lambda x : x[1], reverse=True)
        this_rank = 0
        this_source = ""
        for i, (cui, score) in enumerate(neural_scores_list):
            if cui == correct_cui:
                this_rank = i+1
            out_dict["pred_{0}".format(i)] = "{0}:{1}".format(cui, score)

        if this_rank == 0 and correct_score != float("-inf"):
            if cuiless_added or correct_score <= threshold:
                this_rank = correct_rank + 1
            else:
                this_rank = correct_rank

        ranks.append(this_rank)
        if correct_cui != "CUI-less":
            ranks_cui.append(this_rank)

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

    cui_recip_ranks = []
    for r in ranks_cui:
        if r == 0:
            cui_recip_ranks.append(0)
        else:
            cui_recip_ranks.append(1./r)

    mrr = sum(cui_recip_ranks) / len(cui_recip_ranks)
    log.info("CUI MRR:{0}".format(mrr))

    accuracy = sum(1 for x in ranks_cui if x == 1) / len(ranks_cui)
    log.info("CUI acc:{0}".format(accuracy))


    output_df = pd.DataFrame.from_records(output_list)
    output_df.to_csv(os.path.join(directory, "results.csv"))


if __name__ == "__main__":
    main()