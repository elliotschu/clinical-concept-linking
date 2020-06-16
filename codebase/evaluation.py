import numpy as np
import logging
import csv
import scipy.stats as st
from codebase import mention_links
from collections import defaultdict
import pandas as pd
"""
 from Spotlight codebase (see https://github.com/maciejkula/spotlight)
"""

FLOAT_MAX = np.finfo(np.float32).max


def score(mention_links, predictions, test_dict, outpath=None, output_top=10):
    """
    Given a set of mention ids with an attached devevelopment set, scores all instances in the development set and outputs
    appropriate logging information to a sub-directory of the result root directory.
    :param mention_links: a mention links object containing a development (test) set of mentions
    :param predictions: predictions for each mention
    :param test_dict: a dictionary containing the original documents and annotations for the testing partition.
    :param outpath:
    :param output_top: top n scores to return.
    :return:
    """
    log = logging.getLogger()
    score_list = defaultdict(lambda: [])
    result_list = []
    for indx in range(mention_links.test_mention_ids.shape[0]):
        mention_indx = mention_links.test_mention_ids[indx]
        if hasattr(mention_links, 'test_id_to_mention_info'):

            mention_info = mention_links.test_id_to_mention_info[int(mention_indx)]
        else:
            mention_info = mention_links.id_to_mention_info[mention_indx]

        comm = test_dict[mention_info["comm_uuid"]]

        mention = [mention for mention in comm['concepts'] if mention['index'] == mention_info['mention_uuid']][0]


        # mention = [x for x in comm.entityMentionSetList[0].mentionList
        #            if x.uuid.uuidString == mention_info["mention_uuid"]][0]

        concept_indx = [x["index"] for x in mention_links.cui_to_concept_info[mention['concept']]]


        ranking = st.rankdata(-predictions[indx, :])
        max_ranking = st.rankdata(-predictions[indx, :], method='max')

        if mention['concept'].lower() != "cui-less":

            gold_rank_indx = concept_indx[ranking[concept_indx].argmin()]
            gold_rank = ranking[gold_rank_indx]
            gold_rank_max_indx = concept_indx[max_ranking[concept_indx].argmin()]
            gold_rank_max = max_ranking[gold_rank_max_indx]
            score = predictions[indx, concept_indx].max()

            score_list["mrr"].append(1.0 / gold_rank)
            score_list["max_mrr"].append((1.0 / gold_rank_max))

            score_list["accuracy"].append(1. if gold_rank == np.float64(1) else 0.)
            score_list["max_accuracy"].append(1. if gold_rank_max == np.float64(1) else 0.)
        else:
            gold_rank = ""
            score = float("-inf")

        if outpath:
            top_ind = np.argpartition(predictions[indx, :],-output_top)[-output_top:]
            pred_list = []
            for q in top_ind:
                pred_list.append((mention_links.id_to_concept_info[q]["concept_id"],
                                  mention_links.id_to_concept_info[q]["name"],
                                  predictions[indx, q],
                                  ranking[q]))
            pred_list = sorted(pred_list, key=lambda x: x[-1])
            try:
                sentence = str(mention['mention'][0].sent)
            except:
                log.warning("Cannot print sentence for mention:{0}".format(mention['index']))
                sentence = ""

            row = {
                "_text" : " ".join([str(w) for span in mention['mention'] for w in span]),
                "_sentence" : sentence,
                "_gold_cui": mention['concept'],
                "_gold_name": mention_links.cui_to_concept_info[mention['concept']][0]["name"],
                "_gold_cui_rank" : gold_rank,
                "_gold_cui_score" : score,
                "~~mention_uuid": mention['index'],
                "~~comm": comm['id']
            }
            for ip, pred in enumerate(pred_list):
                row["~pred_cuis_{0}".format(ip)]  = "{0}={1} ({2:.2f})".format(pred[0], pred[1], pred[2])
            if mention['concept'].lower() != "cui-less":

                for fn in score_list:
                    row[fn] = score_list[fn][-1]
            result_list.append(row)
    results = {
        "mrr": np.mean(score_list["mrr"]),
        "max_mrr" : np.mean(score_list["max_mrr"]),
        "accuracy" : np.mean(score_list["accuracy"]),
        "max_accuracy": np.mean(score_list["max_accuracy"]),
    }
    if outpath:
        dataframe = pd.DataFrame.from_dict(result_list)
        dataframe.to_csv(outpath, index=False)

    return results


def sequence_mrr_score(model, test, exclude_preceding=False):
    """
    Compute mean reciprocal rank (MRR) scores. Each sequence
    in test is split into two parts: the first part, containing
    all but the last elements, is used to predict the last element.

    The reciprocal rank of the last element is returned for each
    sequence.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.SequenceInteractions`
        Test interactions.
    exclude_preceding: boolean, optional
        When true, items already present in the sequence will
        be excluded from evaluation.

    Returns
    -------

    mrr scores: numpy array of shape (num_users,)
        Array of MRR scores for each sequence in test.
    """

    sequences = test.sequences[:, :-1]
    targets = test.sequences[:, -1:]

    mrrs = []

    for i in range(len(sequences)):

        predictions = -model.predict(sequences[i])

        if exclude_preceding:
            predictions[sequences[i]] = FLOAT_MAX

        mrr = (1.0 / st.rankdata(predictions)[targets[i]]).mean()

        mrrs.append(mrr)

    return np.array(mrrs)


def sequence_precision_recall_score(model, test, k=10, exclude_preceding=False):
    """
    Compute sequence precision and recall scores. Each sequence
    in test is split into two parts: the first part, containing
    all but the last k elements, is used to predict the last k
    elements.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.SequenceInteractions`
        Test interactions.
    exclude_preceding: boolean, optional
        When true, items already present in the sequence will
        be excluded from evaluation.

    Returns
    -------

    mrr scores: numpy array of shape (num_users,)
        Array of MRR scores for each sequence in test.
    """
    sequences = test.sequences[:, :-k]
    targets = test.sequences[:, -k:]
    precision_recalls = []
    for i in range(len(sequences)):
        predictions = -model.predict(sequences[i])
        if exclude_preceding:
            predictions[sequences[i]] = FLOAT_MAX

        predictions = predictions.argsort()[:k]
        precision_recall = _get_precision_recall(predictions, targets[i], k)
        precision_recalls.append(precision_recall)

    precision = np.array(precision_recalls)[:, 0]
    recall = np.array(precision_recalls)[:, 1]
    return precision, recall


def _get_precision_recall(predictions, targets, k):

    predictions = predictions[:k]
    num_hit = len(set(predictions).intersection(set(targets)))

    return float(num_hit) / len(predictions), float(num_hit) / len(targets)


def precision_recall_score(model, test, train=None, k=10):
    """
    Compute Precision@k and Recall@k scores. One score
    is given for every user with interactions in the test
    set, representing the Precision@k and Recall@k of all their
    test items.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.
    train: :class:`spotlight.interactions.Interactions`, optional
        Train interactions. If supplied, scores of known
        interactions will not affect the computed metrics.
    k: int or array of int,
        The maximum number of predicted items
    Returns
    -------

    (Precision@k, Recall@k): numpy array of shape (num_users, len(k))
        A tuple of Precisions@k and Recalls@k for each user in test.
        If k is a scalar, will return a tuple of vectors. If k is an
        array, will return a tuple of arrays, where each row corresponds
        to a user and each column corresponds to a value of k.
    """

    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    if np.isscalar(k):
        k = np.array([k])

    precision = []
    recall = []

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        predictions = -model.predict(user_id)

        if train is not None:
            rated = train[user_id].indices
            predictions[rated] = FLOAT_MAX

        predictions = predictions.argsort()

        targets = row.indices

        user_precision, user_recall = zip(*[
            _get_precision_recall(predictions, targets, x)
            for x in k
        ])

        precision.append(user_precision)
        recall.append(user_recall)

    precision = np.array(precision).squeeze()
    recall = np.array(recall).squeeze()

    return precision, recall


def rmse_score(model, test):
    """
    Compute RMSE score for test interactions.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.

    Returns
    -------

    rmse_score: float
        The RMSE score.
    """

    predictions = model.predict(test.user_ids, test.item_ids)

    return np.sqrt(((test.ratings - predictions) ** 2).mean())
