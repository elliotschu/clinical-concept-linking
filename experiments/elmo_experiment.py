"""
Elliot Schumacher, Johns Hopkins University
Created 3/19/19
"""
import random
import os
import pickle
from concrete.util import file_io
import numpy as np
from scipy.stats import describe
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as st
from scipy.spatial.distance import cdist
def rank(concept_mentions, mention_rep):
    cos_sim = []
    #for i in range(concept_mentions.shape[0]):
    #    cos_sim.append(-cosine_similarity([concept_mentions[i, :]], [mention_rep])[0][0])
    #rep_men = np.broadcast_to(mention_rep, (concept_mentions.shape[0], mention_rep.shape[0] ))
    #cos_sim = -cosine_similarity(concept_mentions.tolist(),rep_men.tolist())
    cos_sim = cdist(concept_mentions, mention_rep.reshape(1, -1), metric='cosine')
    return cos_sim


def main():
    loc_mention_embeddings = "/Users/elliotschumacher/Dropbox/git/synonym_detection/resources/bilm/out_max/mention_embeddings"
    loc_concept_embeddings = "/Users/elliotschumacher/Dropbox/git/synonym_detection/resources/bilm/out_max/embedding_output"

    dev_file = "/Users/elliotschumacher/Dropbox/concept/share_clef/SPLIT_2017-12-08-13-38-01/train/dev_fix_concrete.tar"

    test_dict = {}
    for (comm, filename) in file_io.CommunicationReader(dev_file):
        for menset in comm.entityMentionSetList[0].mentionList:
            test_dict[menset.uuid.uuidString] = menset


    with open(os.path.join(loc_mention_embeddings, 'mention_representations.npy'),
              'rb') as mention_representations_npy, \
            open(os.path.join(loc_mention_embeddings, 'mention_to_info.pkl'), 'rb') as mention_to_info_pkl, \
            open(os.path.join(loc_mention_embeddings, 'id_to_mention_info.pkl'), 'rb') as id_to_mention_info_pkl:
        mention_representations = np.load(mention_representations_npy)
        id_to_mention_info = pickle.load(id_to_mention_info_pkl)
        mention_to_info = pickle.load(mention_to_info_pkl)

    with open(os.path.join(loc_concept_embeddings, 'concept_representations.npy'),
              'rb') as concept_representations_npy, \
            open(os.path.join(loc_concept_embeddings, 'id_to_concept_name_alt.pkl'),
                 'rb') as id_to_concept_name_alt_pkl, \
            open(os.path.join(loc_concept_embeddings, 'concept_to_id_name_alt.pkl'),
                 'rb') as concept_to_id_name_alt_pkl:
        concept_representations = np.load(concept_representations_npy)
        id_to_concept_info = pickle.load(id_to_concept_name_alt_pkl)
        cui_to_concept_info = pickle.load(concept_to_id_name_alt_pkl)

    output_file = "elmo_exp.csv"
    result_list = []
    input_csv = "/Users/elliotschumacher/Dropbox/git/concept-linker/results/run_2019_03_06_11_01_30_b13/eval_759.csv"
    eval_csv = pd.DataFrame.from_csv(input_csv)

    cos_sims = {}
    shuffled_keys = list(mention_to_info.keys())
    for mention_uuid1 in list(mention_to_info.keys()):
        if mention_uuid1 in test_dict:
            menset = test_dict[mention_uuid1]
            if menset.entityType in cui_to_concept_info:
                random.shuffle(shuffled_keys)
                for i in range(10):
                    mention_uuid2 = shuffled_keys[i]
                    if mention_uuid1 != mention_uuid2:
                        m_indx1 = mention_to_info[mention_uuid1]["index"]
                        m_indx2 = mention_to_info[mention_uuid2]["index"]
                        cos_sim = cosine_similarity([mention_representations[m_indx1, :]], [mention_representations[m_indx2, :]])[0][0]
                        min_uuid = min(mention_uuid1, mention_uuid2)
                        max_uuid = max(mention_uuid1, mention_uuid2)
                        cos_sims[min_uuid, max_uuid] = cos_sim

    print("Stats for mention cos similarity")
    print(describe(list(cos_sims.values())))

    outer_concept_list = list(cui_to_concept_info)
    inner_concept_list = list(cui_to_concept_info)
    random.shuffle(outer_concept_list)
    cos_sims_cui = {}

    for cui1 in outer_concept_list[:1000]:
        c_indx1 = cui_to_concept_info[cui1][0]["index"]
        c_indexes = random.sample(range(0, len(inner_concept_list)), 10)
        for cui2_indx in c_indexes:
            cui2 = inner_concept_list[cui2_indx]
            c_indx2 = cui_to_concept_info[cui2][0]["index"]
            if c_indx1 != c_indx2:
                cos_sim = \
                cosine_similarity([concept_representations[c_indx1, :]], [concept_representations[c_indx2, :]])[0][0]
                min_uuid = min(cui1, cui2)
                max_uuid = max(cui1, cui2)
                cos_sims_cui[min_uuid, max_uuid] = cos_sim

    print("Stats for concept cos similarity")
    print(describe(list(cos_sims_cui.values())))
    #df = pd.DataFrame([list(cos_sims.keys()), list(cos_sims_cui.keys())], columns=['Mention', 'Concept'])
    plt.hist([list(cos_sims.values()), list(cos_sims_cui.values())], color=['r', 'b'], alpha=0.5)
    plt.gca().legend(('Mentions', 'Concepts'))

    plt.show()

    for _, row in eval_csv.iterrows():
        menset = test_dict[row["~~mention_uuid"]]
        if menset.entityType in cui_to_concept_info:
            mention_info = mention_to_info[menset.uuid.uuidString]
            concept_info = cui_to_concept_info[menset.entityType][0]

            m_indx = mention_info["index"]
            c_indx = mention_info["index"]
            sentence = " ".join([w.text.strip() for w in menset.tokens.tokenization.tokenList.tokenList])

            m_rep = mention_representations[m_indx, :]
            c_rep = concept_representations[c_indx, :]
            cos_dist = cdist(concept_representations, m_rep.reshape(1, -1), metric='cosine')
            ranking = st.rankdata(cos_dist)

            gold_rank = ranking[c_indx]

            cos_sim = cosine_similarity([m_rep], [c_rep])[0][0]

            print("Cosine dist:{0}, sim:{1}".format(cos_dist[c_indx], cos_sim))
            row["cos_dist"] = cos_sim
            row["sentence"] = sentence
            row["cos_rank"] = gold_rank
            result_list.append(row)


    dataframe = pd.DataFrame.from_records(result_list)
    dataframe.to_csv(output_file, index=False)
if __name__ == "__main__":
    main()