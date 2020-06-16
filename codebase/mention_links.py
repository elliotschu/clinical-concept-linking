"""
 from Spotlight codebase (see https://github.com/maciejkula/spotlight)


Classes describing datasets of mention-concept interactions. Instances of these
are returned by dataset-fetching and dataset-processing functions.
"""
import math
import csv
import logging
import os
import pickle
import torch.nn.functional as F
from random import random, shuffle, sample
import nltk
from itertools import cycle, islice
from spacy.tokens import Span
import numpy as np
import torch
from allennlp.modules.elmo import batch_to_ids
from gensim.models import KeyedVectors
from pytorch_pretrained_bert import BertTokenizer, BertModel
from scipy.sparse import *
from sklearn.preprocessing import RobustScaler
from codebase.ElmoAllLayers import ElmoAllLayers
from collections import defaultdict
from codebase import torch_utils
#from medacy.pipeline_components.metamap.metamap import MetaMap
#from n2c2_2019.clinical_concept_linker import load_share_clef_2013

def _sliding_window(tensor, window_size, step_size=1):

    for i in range(len(tensor), 0, -step_size):
        yield tensor[max(i - window_size, 0):i]


def _generate_sequences(mention_ids, concept_ids,
                        indices,
                        max_sequence_length,
                        step_size):

    for i in range(len(indices)):

        start_idx = indices[i]

        if i >= len(indices) - 1:
            stop_idx = None
        else:
            stop_idx = indices[i + 1]

        for seq in _sliding_window(concept_ids[start_idx:stop_idx],
                                   max_sequence_length,
                                   step_size):

            yield (mention_ids[i], seq)

def mention_iterator(comm_dict):
    for comm_fn in comm_dict:
        comm = comm_dict[comm_fn]
        for menset in comm.entityMentionSetList[0].mentionList:
            yield (comm_fn, menset)

def mention_iterator_obj(comm_dict):
    for comm_fn in comm_dict:
        comm = comm_dict[comm_fn]
        for menset in comm.entityMentionSetList[0].mentionList:
            yield (comm, menset)

class MentionLinks(object):
    """
    This object encapsulates the storage of representations (embeddings) for text mentions and an ontology of concepts.
    Given a document with span annotations, this object provides reference to labeled concepts during training alongside
    access to a validation set of mentions for use in prediction.
    """

    default_arguments = {
        "embedding": "elmo",
        "emb_layer" : 0,
        "bert_model": "bert-base-cased",
        "bert_path": "",
        "include_cuiless": False,
        "test_include_cuiless":True,
        "online": False,
        "test_limit": 10000,
        "ont_emb" : False,
        "ont_w2v_filename": "",
        "ont_id_mapping": "",
        "ont_name_mapping": "",
        "dnorm_feats" : "",
        "concept_context": "",
        "attention" : False,
        "neg_samps": "",
        "max_neg_samples" : 100,
        "metamap": False,
        "syn_file": "",
        "limit_syns" : -1,
        "max_concept_length": 50,
    }

    bert_models = {
        "bert-large-uncased" : 1024,
        "bert-base-uncased" : 768,
        "bert-large-cased" : 1024,
        "bert-base-cased" : 768
    }




    def __init__(self, comm_dict, args, test_comm_dict = None):
        """

        :param comm_dict: a dictionary mapping training document ids to document text and mention annotations.
        :param args: the arguments of the training/prediction run
        :param test_comm_dict: a dictionary mapping validation document ids to document text and mention annotations.
        """
        self.log = logging.getLogger()
        self.dnorm_features = False

        if args.metamap:
            self.load_elmo_attention_mm(comm_dict, args, test_comm_dict)


        elif args.embedding == "bert" and args.online is False:
            self.load_bert(comm_dict, args, test_comm_dict)
        elif args.embedding == "bert" and args.online is True and args.attention is True:
            self.load_bert_online_att(comm_dict, args, test_comm_dict)
        elif args.embedding == "bert" and args.online is True:
            self.load_bert_online(comm_dict, args, test_comm_dict)
        elif args.embedding == "elmo" and args.attention is True:
            self.load_elmo_attention(comm_dict, args, test_comm_dict)
        elif args.embedding == "elmo" and args.online is False and args.attention is False:
            self.load_elmo_old(comm_dict, args, test_comm_dict)
        elif args.embedding == "elmo" and args.online is True:
            self.load_elmo_online(comm_dict, args, test_comm_dict)

    def get_embedding(self, text, model, tokenizer, args, layer):
        tok_men = tokenizer.tokenize(text)
        indx_toks = tokenizer.convert_tokens_to_ids(tok_men)
        segment_tensor = torch_utils.gpu(torch.zeros(len(indx_toks), dtype=torch.int64), args.use_cuda)
        tok_tensor = torch_utils.gpu(torch.tensor([indx_toks]), args.use_cuda)

        with torch.no_grad():
            encoded_layers, _ = model(tok_tensor, segment_tensor)
            selection = encoded_layers[layer].squeeze().mean(dim=0)
            return selection

    def get_embeddings_batch(self, concept_chars, model, args, layer, out_dim_size, concept_att, concept_mask):
        max_len = 0
        padding_id = self.tokenizer.convert_tokens_to_ids(["[PAD]"])

        concept_chars = torch_utils.gpu(concept_chars,  args.use_cuda)
        concept_att = torch_utils.gpu(concept_att,  args.use_cuda)
        representations = torch_utils.gpu(torch.zeros(size=(len(concept_chars),out_dim_size)),  args.use_cuda)
        concept_mask = torch_utils.gpu(concept_mask,  args.use_cuda)
        for i in range(0, len(concept_chars), args.batch_size):
            upper = min(i+args.batch_size, len(concept_chars))

            with torch.no_grad():
                encoded_layers, pooled = model(concept_chars[i:upper , :])#, attention_mask=concept_att[i:upper , :])

                if args.comb_op.lower() != "cls":

                    embedding_layer = encoded_layers[layer]
                    masked_concept = (embedding_layer + concept_mask[i:upper , :]
                                      .view(embedding_layer.size()[0], embedding_layer.size()[1], 1)
                                      .expand(embedding_layer.size()[0], embedding_layer.size()[1],
                                              embedding_layer.size()[2]))

                    concept_embedding = masked_concept.max(dim=1)[0]

                    representations[i:upper , :] = concept_embedding
                else:
                    representations[i:upper , :] = pooled


        return representations

    def load_bert(self, comm_dict, args, test_comm_dict):
        self.only_annotated_concepts = args.only_annotated_concepts


        self.id_to_concept_info = {}
        self.cui_to_concept_info = {}

        mask_true_val = 0.
        mask_false_val =  torch.Tensor([float("-inf")]).float()

        att_true_val = 1.
        att_false_val = 0.


        self.tokenizer = BertTokenizer.from_pretrained(args.bert_path,
                                                  do_lower_case="uncased" in args.bert_model)
        self.test_data = test_comm_dict is not None

        #concept_names = []

        self.included_concepts = set()
        if self.only_annotated_concepts:
            # for (comm, m) in mention_iterator_obj(comm_dict):
            #     self.included_concepts.add(m.entityType)
            for file in comm_dict:
                for mention in comm_dict[file]['concepts']:
                    self.included_concepts.add(mention['concept'])

        concept_parts = []
        longest_seq = 0
        max_num_syns = 4
        if args.concept_context != "":
            with open (args.concept_context, 'rb') as confile:
                concept_context_dict = pickle.load(confile)
        with open(args.lexicon, encoding='utf-8') as lex_file:
            tsv_reader = csv.reader(lex_file, delimiter="\t")
            index = 0
            for  row in tsv_reader:
                name = row[0]
                conceptId = row[1]
                if self.only_annotated_concepts and conceptId not in self.included_concepts:
                    continue
                concept_map = {"name": name,
                               "concept_id": conceptId,
                               "alternate": False,
                               "index": index
                               }
                #tokenized_name = nltk.word_tokenize(name)
                #concept_names.append(tokenized_name)
                name = "[CLS] {0} [SEP]".format(name)

                if args.concept_context and conceptId in concept_context_dict:
                    for ni, syn in enumerate(concept_context_dict[conceptId]["names"]):
                        name += " {0} [SEP]".format(syn)

                        if (ni+1) == max_num_syns:
                            break

                tokenized_name = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(name))

                concept_parts.append(tokenized_name)
                longest_seq = max(len(tokenized_name), longest_seq)
                self.cui_to_concept_info[conceptId] = [concept_map]
                self.id_to_concept_info[index] = concept_map
                index += 1
                pass

        padding_id = self.tokenizer.convert_tokens_to_ids(["[PAD]"])

        self.concept_chars = torch.zeros((len(concept_parts), longest_seq), dtype=torch.long) + padding_id[0]
        self.concept_mask = torch.zeros((self.concept_chars.shape[0], self.concept_chars.shape[1])) + mask_false_val
        self.concept_att = torch.zeros((self.concept_chars.shape[0], self.concept_chars.shape[1])) + att_false_val

        for i, cp in enumerate(concept_parts):
            self.concept_chars[i, :len(cp)] = torch.tensor(cp)
            self.concept_mask[i, :len(cp)] = mask_true_val
            self.concept_att[i, :len(cp)] = att_true_val
        bert_model = BertModel.from_pretrained(args.bert_path)
        bert_model = torch_utils.gpu(bert_model, args.use_cuda)



        self.concept_representations = self.get_embeddings_batch(concept_chars=self.concept_chars,
                                                          model=bert_model,
                                                          args=args,
                                                          layer=args.emb_layer,
                                                          out_dim_size=768,
                                                          concept_att=self.concept_att,
                                                          concept_mask=self.concept_mask)

        self.num_concepts = len(self.concept_representations)

        #self.num_examples = sum(1 for _, m in mention_iterator(comm_dict) if m.entityType in self.cui_to_concept_info)
        self.num_examples = sum([1 for file_key in comm_dict for mention in comm_dict[file_key]['concepts'] if mention['concept'] in self.cui_to_concept_info])

        self.mention_ids = np.zeros(self.num_examples)
        self.concept_ids = np.zeros(self.num_examples)

        mention_sentences = []
        self.id_to_mention_info = {}
        self.test_id_to_mention_info = {}
        self.mention_to_info = {}

        self.concepts_used = set()
        self.concepts_ids_used = set()

        indx = 0
        self.mention_indexes = []
        men_longest_seq = 0

        for file in comm_dict:
            for mention in comm_dict[file]['concepts']:
                if mention['concept'] in self.cui_to_concept_info:
                    mention_map = {"comm_uuid": comm_dict[file]["id"],
                                   "mention_uuid": mention["index"],
                                   "index": indx
                                   }
                    self.mention_to_info[mention["index"]] = mention_map
                    self.id_to_mention_info[indx] = mention_map
                    #self.mention_representations[indx, :] =  self.get_embedding(m.text, model, tokenizer, args, args.emb_layer)
                    sent = mention['mention'][0].sent
                    self.mention_indexes.append([1 + token.i - sent.start for span in mention['mention'] for token in span])
                    #self.mention_indexes.append([x+1 for x in m.tokens.tokenIndexList])
                    self.mention_ids[indx] = mention_map['index']
                    self.concept_ids[indx] = self.cui_to_concept_info[mention['concept']][0]['index']

                    self.concepts_used.add(mention['concept'])
                    self.concepts_ids_used.add(self.cui_to_concept_info[mention['concept']][0]['index'])

                    sentence = [w.text.strip() for w in sent]
                    sentence.insert(0, "[CLS]")
                    sentence.append("[SEP]")

                    mention_sentences.append(sentence)
                    indx += 1


        length = len(sorted(self.mention_indexes, key=len, reverse=True)[0])
        self.mention_indexes = np.array([xi+[-1]*(length-len(xi)) for xi in self.mention_indexes])
        self.mention_chars, self.mention_mask, self.mention_att, _ = self.bert_tokenize(mention_sentences,
                                                                             self.mention_indexes,
                                                                             self.tokenizer,
                                                                             mask_true_val,
                                                                             mask_false_val,
                                                                             padding_id,
                                                                             att_true_val,
                                                                             att_false_val)

        self.mention_representations = self.get_embeddings_batch(concept_chars=self.mention_chars,
                                                          model=bert_model,
                                                          args=args,
                                                          layer=args.emb_layer,
                                                          out_dim_size=768,
                                                          concept_att=self.mention_att,
                                                          concept_mask=self.mention_mask)

        # test stuff
        if self.test_data:
            test_mention_indexes_list = []
            test_mention_sentences = []
            # self.test_num_examples = sum(1 for _, m in mention_iterator(test_comm_dict)
            #                              if m.entityType in self.cui_to_concept_info)

            self.test_num_examples = sum([1 for file_key in test_comm_dict for mention in test_comm_dict[file_key]['concepts'] if mention['concept'] in self.cui_to_concept_info])

            self.test_concept_ids = np.zeros(self.test_num_examples)
            self.test_mention_ids = np.zeros(self.test_num_examples)
            other_indx = 0
            for file_key in test_comm_dict:
                for mention in test_comm_dict[file_key]['concepts']:
                    if mention['concept'] in self.cui_to_concept_info:
                        mention_map = {"comm_uuid": test_comm_dict[file_key]["id"],
                                       "mention_uuid": mention['index'],
                                       "index": indx
                                       }
                        self.test_id_to_mention_info[indx] = mention_map
                        # retrieves the first, possibly disjoint, Span object corresponding to the mention and accesses it's parent sentence
                        sent = mention['mention'][0].sent
                        test_mention_indexes_list.append(
                            [1 + token.i - sent.start for span in mention['mention'] for token in span])

                        sentence = [w.text.strip() for w in sent]

                        sentence.insert(0, "[CLS]")
                        sentence.append("[SEP]")
                        test_mention_sentences.append(sentence)
                        self.test_mention_ids[other_indx] = mention_map['index']

                        self.test_concept_ids[other_indx] = self.cui_to_concept_info[mention["concept"]][0]['index']
                        self.concepts_used.add(mention["concept"])
                        self.concepts_ids_used.add(self.cui_to_concept_info[mention["concept"]][0]['index'])

                        indx += 1
                        other_indx += 1
                        if args.test_limit is not None and i == args.test_limit -1 :
                            self.log.warning("Limiting test dataset to {0}".format(args.test_limit))
                            self.test_mention_ids = self.test_mention_ids[:args.test_limit]
                            self.test_concept_ids = self.test_concept_ids[:args.test_limit]
                            self.test_num_examples = args.test_limit
                            break

            length = len(sorted(test_mention_indexes_list, key=len, reverse=True)[0])
            self.test_mention_indexes = np.array([xi + [-1] * (length - len(xi)) for xi in test_mention_indexes_list])

            self.test_mention_char, self.test_mention_mask, self.test_mention_att, _ = self.bert_tokenize(test_mention_sentences,
                                                                                 self.test_mention_indexes,
                                                                                 self.tokenizer,
                                                                                 mask_true_val,
                                                                                 mask_false_val,
                                                                                 padding_id,
                                                                                 att_true_val,
                                                                                 att_false_val)
            self.test_mention_representations = self.get_embeddings_batch(concept_chars=self.test_mention_char,
                                                                     model=bert_model,
                                                                     args=args,
                                                                     layer=args.emb_layer,
                                                                     out_dim_size=768,
                                                                     concept_att=self.test_mention_att,
                                                                     concept_mask=self.test_mention_mask)



        self.mention_representations = torch.cat((self.mention_representations, self.test_mention_representations), 0)

        #self.mention_indexes = np.array(self.mention_indexes)

        self.log.info("Loaded bert characters")


    def bert_tokenize(self, sentences, mention_indexes, tokenizer, mask_true_val, mask_false_val, padding_id, att_true_val, att_false_val):
        mention_parts = []
        longest_seq = 0
        longest_men = 0

        mention_mask_list = []
        mention_att_list = []
        mention_size_list = []
        for sent, mi in zip(sentences, mention_indexes):
            this_sentence  = []
            this_mask = []
            this_att = []
            this_mention_size = 0
            for i_tok, tok in enumerate(sent):
                this_tok = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tok))
                this_sentence.extend(this_tok)
                if i_tok in mi:
                    this_mask.extend([mask_true_val for x in this_tok])
                    this_att.extend([att_true_val for x in this_tok])
                    this_mention_size += len(this_tok)
                else:
                    this_mask.extend([mask_false_val for x in this_tok])
                    this_att.extend([att_false_val for x in this_tok])

            mention_parts.append(this_sentence)
            mention_mask_list.append(this_mask)
            mention_att_list.append(this_att)
            mention_size_list.append(this_mention_size)
            longest_seq = max(longest_seq, len(this_sentence))
            longest_men = max(longest_men, sum(int(x) for x in this_mask))

        mention_representations = torch.zeros((len(mention_parts), longest_seq + longest_men), dtype=torch.long) + padding_id[0]
        mention_reduced_mask = torch.zeros((len(mention_parts), longest_men)) + mask_false_val

        for i, cp in enumerate(mention_parts):
            mention_representations[i, :len(cp)] = torch.tensor(cp)
            mention_reduced_mask[i, :mention_size_list[i]] = mask_true_val



        mention_mask = torch.zeros((mention_representations.shape[0], mention_representations.shape[1])) + mask_false_val
        mention_att = torch.zeros((mention_representations.shape[0], mention_representations.shape[1])) + att_false_val

        for k in range (len(mention_representations)):
            mention_mask[k, :len(mention_mask_list[k])] = torch.tensor(mention_mask_list[k])
            mention_att[k, :len(mention_att_list[k])] = torch.tensor(mention_att_list[k])

            upper = mention_mask.shape[1]
            lower = upper - (longest_men -  sum(int(x) for x in mention_mask_list[k]))
            mention_mask[k, lower:upper] = self.mask_true_val

            upper = mention_att.shape[1]
            lower = upper - (longest_men -  sum(int(x) for x in mention_att_list[k]))
            mention_att[k, lower:upper] = self.mask_true_val

        return mention_representations, mention_mask, mention_att, mention_reduced_mask

    def load_bert_online(self, doc_dict, args, test_doc_dict):
        self.only_annotated_concepts = args.only_annotated_concepts


        self.id_to_concept_info = {}
        self.cui_to_concept_info = {}

        mask_true_val = 1.
        mask_false_val =  0.

        att_true_val = 1.
        att_false_val = 0.


        self.tokenizer = BertTokenizer.from_pretrained(args.bert_path,
                                                  do_lower_case="uncased" in args.bert_model)
        self.test_data = test_doc_dict is not None

        #concept_names = []

        self.included_concepts = set()
        if self.only_annotated_concepts:
            for file in doc_dict:
                for mention in doc_dict[file]['concepts']:
                    self.included_concepts.add(mention['concept'])

        concept_parts = []
        longest_seq = 0
        with open(args.lexicon, encoding='utf-8') as lex_file:
            tsv_reader = csv.reader(lex_file, delimiter="\t")
            index = 0
            for  row in tsv_reader:
                name = row[0]
                conceptId = row[1]
                if self.only_annotated_concepts and conceptId not in self.included_concepts:
                    continue
                concept_map = {"name": name,
                               "concept_id": conceptId,
                               "alternate": False,
                               "index": index
                               }
                #tokenized_name = nltk.word_tokenize(name)
                #concept_names.append(tokenized_name)
                name = "[CLS] {0} [SEP]".format(name)
                tokenized_name = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(name))
                concept_parts.append(tokenized_name)
                longest_seq = max(len(tokenized_name), longest_seq)
                self.cui_to_concept_info[conceptId] = [concept_map]
                self.id_to_concept_info[index] = concept_map
                index += 1
                pass
                """
                if row[7].strip() != "":
                    alt_names = row[7].split("|")
                    for an in alt_names:"""

        padding_id = self.tokenizer.convert_tokens_to_ids(["[PAD]"])

        self.concept_representations = torch.zeros((len(concept_parts), longest_seq), dtype=torch.long) + padding_id[0]
        self.concept_mask = torch.zeros((self.concept_representations.shape[0], self.concept_representations.shape[1])) + mask_false_val
        self.concept_att = torch.zeros((self.concept_representations.shape[0], self.concept_representations.shape[1])) + att_false_val

        for i, cp in enumerate(concept_parts):
            self.concept_representations[i, :len(cp)] = torch.tensor(cp)
            self.concept_mask[i, :len(cp)] = mask_true_val
            self.concept_att[i, :len(cp)] = att_true_val

        self.num_concepts = len(self.concept_representations)

        self.num_examples = sum([1 for file_key in doc_dict for mention in doc_dict[file_key]['concepts'] if mention['concept'] in self.cui_to_concept_info])

        self.mention_ids = np.zeros(self.num_examples)
        self.concept_ids = np.zeros(self.num_examples)

        mention_sentences = []
        self.id_to_mention_info = {}
        self.test_id_to_mention_info = {}
        self.mention_to_info = {}

        self.concepts_used = set()
        self.concepts_ids_used = set()

        indx = 0
        self.mention_indexes = []
        men_longest_seq = 0
        for file in doc_dict:
            for mention in doc_dict[file]['concepts']:
                if mention['concept'] in self.cui_to_concept_info:
                    mention_map = {"comm_uuid": doc_dict[file]["id"],
                                   "mention_uuid": mention["index"],
                                   "index": indx
                                   }
                    self.mention_to_info[mention["index"]] = mention_map
                    self.id_to_mention_info[indx] = mention_map
                    #self.mention_representations[indx, :] =  self.get_embedding(m.text, model, tokenizer, args, args.emb_layer)
                    sent = mention['mention'][0].sent
                    self.mention_indexes.append([1 + token.i - sent.start for span in mention['mention'] for token in span])
                    #self.mention_ids[indx] = self.mention_to_info[m.uuid.uuidString]['index']
                    self.concept_ids[indx] = self.cui_to_concept_info[mention['concept']][0]['index']

                    self.concepts_used.add(mention['concept'])
                    self.concepts_ids_used.add(self.cui_to_concept_info[mention['concept']][0]['index'])

                    sentence = [w.text.strip() for w in sent]
                    sentence.insert(0, "[CLS]")
                    sentence.append("[SEP]")

                    mention_sentences.append(sentence)
                    indx += 1
        length = len(sorted(self.mention_indexes, key=len, reverse=True)[0])
        self.mention_indexes = np.array([xi+[-1]*(length-len(xi)) for xi in self.mention_indexes])
        self.mention_representations, self.mention_mask, self.mention_att = self.bert_tokenize(mention_sentences,
                                                                             self.mention_indexes,
                                                                             self.tokenizer,
                                                                             mask_true_val,
                                                                             mask_false_val,
                                                                             padding_id,
                                                                             att_true_val,
                                                                             att_false_val)

        # test stuff
        if self.test_data:
            test_mention_indexes_list = []
            test_mention_sentences = []
            self.test_num_examples = sum([1 for file_key in test_doc_dict for mention in test_doc_dict[file_key]['concepts'] if mention['concept'] in self.cui_to_concept_info])

            self.test_concept_ids = np.zeros(self.test_num_examples)
            self.test_mention_ids = np.zeros(self.test_num_examples)
            indx = 0
            for file_key in test_doc_dict:
                for mention in test_doc_dict[file_key]['concepts']:
                    if mention['concept'] in self.cui_to_concept_info:
                        mention_map = {"comm_uuid": test_doc_dict[file_key]["id"],
                                       "mention_uuid": mention['index'],
                                       "index": indx
                                       }
                        self.test_id_to_mention_info[indx] = mention_map
                        sent = mention['mention'][0].sent
                        test_mention_indexes_list.append(
                            [1 + token.i - sent.start for span in mention['mention'] for token in span])

                        sentence = [w.text.strip() for w in sent]

                        sentence.insert(0, "[CLS]")
                        sentence.append("[SEP]")
                        test_mention_sentences.append(sentence)

                        self.test_mention_ids[indx] = mention_map['index']
                        self.test_concept_ids[indx] = self.cui_to_concept_info[mention["concept"]][0]['index']
                        self.concepts_used.add(mention["concept"])
                        self.concepts_ids_used.add(self.cui_to_concept_info[mention["concept"]][0]['index'])

                        if args.test_limit is not None and indx == args.test_limit -1 :
                            self.log.warning("Limiting test dataset to {0}".format(args.test_limit))
                            self.test_mention_ids = self.test_mention_ids[:args.test_limit]
                            self.test_concept_ids = self.test_concept_ids[:args.test_limit]
                            self.test_num_examples = args.test_limit
                            break
                        indx += 1
            length = len(sorted(test_mention_indexes_list, key=len, reverse=True)[0])
            self.test_mention_indexes = np.array([xi + [-1] * (length - len(xi)) for xi in test_mention_indexes_list])

            self.test_mention_representations, self.test_mention_mask, self.test_mention_att = self.bert_tokenize(test_mention_sentences,
                                                                                 self.test_mention_indexes,
                                                                                 self.tokenizer,
                                                                                 mask_true_val,
                                                                                 mask_false_val,
                                                                                 padding_id,
                                                                                 att_true_val,
                                                                                 att_false_val)



        #self.mention_indexes = np.array(self.mention_indexes)

        self.log.info("Loaded bert characters")

    def load_bert_online_att(self, doc_dict, args, test_doc_dict):
        self.only_annotated_concepts = args.only_annotated_concepts


        self.id_to_concept_info = {}
        self.cui_to_concept_info = {}

        self.mask_true_val = 1.
        self.mask_false_val =  0.#torch.Tensor([float("-inf")]).float()


        self.tokenizer = BertTokenizer.from_pretrained(args.bert_path,
                                                  do_lower_case="uncased" in args.bert_model)
        self.test_data = test_doc_dict is not None

        #concept_names = []

        self.included_concepts = set()
        if self.only_annotated_concepts:
            for file in doc_dict:
                for mention in doc_dict[file]['concepts']:
                    self.included_concepts.add(mention['concept'])

        concept_parts = []
        longest_seq = 0
        with open(args.lexicon, encoding='utf-8') as lex_file:
            tsv_reader = csv.reader(lex_file, delimiter="\t")
            index = 0
            for  row in tsv_reader:
                name = row[0]
                conceptId = row[1]
                if self.only_annotated_concepts and conceptId not in self.included_concepts:
                    continue
                concept_map = {"name": name,
                               "concept_id": conceptId,
                               "alternate": False,
                               "index": index
                               }
                #tokenized_name = nltk.word_tokenize(name)
                #concept_names.append(tokenized_name)
                name = "[CLS] {0} [SEP]".format(name)
                tokenized_name = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(name))
                concept_parts.append(tokenized_name)
                longest_seq = max(len(tokenized_name), longest_seq)
                self.cui_to_concept_info[conceptId] = [concept_map]
                self.id_to_concept_info[index] = concept_map
                index += 1
                pass
                """
                if row[7].strip() != "":
                    alt_names = row[7].split("|")
                    for an in alt_names:"""

        padding_id = self.tokenizer.convert_tokens_to_ids(["[PAD]"])

        self.concept_representations = torch.zeros((len(concept_parts), longest_seq), dtype=torch.long) + padding_id[0]
        self.concept_mask = torch.zeros((self.concept_representations.shape[0], self.concept_representations.shape[1])) + self.mask_false_val
        self.concept_att = torch.zeros((self.concept_representations.shape[0], self.concept_representations.shape[1])) + self.mask_false_val

        for i, cp in enumerate(concept_parts):
            self.concept_representations[i, :len(cp)] = torch.tensor(cp)
            self.concept_mask[i, :len(cp)] = self.mask_true_val
            self.concept_att[i, :len(cp)] = self.mask_true_val

        self.num_concepts = len(self.concept_representations)

        self.num_examples = sum([1 for file_key in doc_dict for mention in doc_dict[file_key]['concepts'] if mention['concept'] in self.cui_to_concept_info])

        self.mention_ids = np.zeros(self.num_examples)
        self.concept_ids = np.zeros(self.num_examples)

        mention_sentences = []
        self.id_to_mention_info = {}
        self.test_id_to_mention_info = {}
        self.mention_to_info = {}

        self.concepts_used = set()
        self.concepts_ids_used = set()

        indx = 0
        self.mention_indexes = []
        men_longest_seq = 0

        if args.syn_file:
            with open(args.syn_file, 'rb') as sf:
                syn_dict = pickle.load(sf)
            self.num_examples = sum(len(y) for y in syn_dict.values())
            self.mention_ids = np.zeros(self.num_examples)
            self.concept_ids = np.zeros(self.num_examples)

            for cui, syn_list in syn_dict.items():
                for syn in syn_list:
                    mention_map = {"comm_uuid": "{0}_{1}".format(cui, syn),
                                   "mention_uuid": syn,
                                   "index": indx
                                   }
                    # self.mention_to_info[mention["index"]] = mention_map
                    self.id_to_mention_info[indx] = mention_map
                    # self.mention_representations[indx, :] =  self.get_embedding(m.text, model, tokenizer, args, args.emb_layer)
                    sentence = nltk.word_tokenize(syn)

                    self.mention_indexes.append([i for i in range(len(sentence))])
                    # self.mention_ids[indx] = self.mention_to_info[m.uuid.uuidString]['index']
                    self.concept_ids[indx] = self.cui_to_concept_info[cui][0]['index']

                    self.concepts_used.add(cui)
                    self.concepts_ids_used.add(self.cui_to_concept_info[cui][0]['index'])

                    sentence.insert(0, "[CLS]")
                    sentence.append("[SEP]")

                    mention_sentences.append(sentence)
                    indx += 1
        else:
            for file in doc_dict:
                for mention in doc_dict[file]['concepts']:
                    if mention['concept'] in self.cui_to_concept_info:
                        mention_map = {"comm_uuid": doc_dict[file]["id"],
                                       "mention_uuid": mention["index"],
                                       "index": indx
                                       }
                        self.mention_to_info[mention["index"]] = mention_map
                        self.id_to_mention_info[indx] = mention_map
                        #self.mention_representations[indx, :] =  self.get_embedding(m.text, model, tokenizer, args, args.emb_layer)
                        sent = mention['mention'][0].sent
                        self.mention_indexes.append([1 + token.i - sent.start for span in mention['mention'] for token in span])
                        #self.mention_ids[indx] = self.mention_to_info[m.uuid.uuidString]['index']
                        self.concept_ids[indx] = self.cui_to_concept_info[mention['concept']][0]['index']

                        self.concepts_used.add(mention['concept'])
                        self.concepts_ids_used.add(self.cui_to_concept_info[mention['concept']][0]['index'])

                        sentence = [w.text.strip() for w in sent]
                        sentence.insert(0, "[CLS]")
                        sentence.append("[SEP]")

                        mention_sentences.append(sentence)
                        indx += 1
        length = len(sorted(self.mention_indexes, key=len, reverse=True)[0])
        self.mention_indexes = np.array([xi+[-1]*(length-len(xi)) for xi in self.mention_indexes])
        self.mention_representations, self.mention_mask, self.mention_att, self.mention_reduced_mask = self.bert_tokenize(mention_sentences,
                                                                             self.mention_indexes,
                                                                             self.tokenizer,
                                                                             self.mask_true_val,
                                                                             self.mask_false_val,
                                                                             padding_id,
                                                                           self.mask_true_val,
                                                                           self.mask_false_val)
        self.max_men_length = self.mention_reduced_mask.shape[1]

        if self.test_data:
            test_mention_indexes_list = []
            test_mention_sentences = []
            self.test_num_examples = sum([1 for file_key in test_doc_dict for mention in test_doc_dict[file_key]['concepts'] if mention['concept'] in self.cui_to_concept_info])

            self.test_concept_ids = np.zeros(self.test_num_examples)
            self.test_mention_ids = np.zeros(self.test_num_examples)
            indx = 0
            for file_key in test_doc_dict:
                for mention in test_doc_dict[file_key]['concepts']:
                    if mention['concept'] in self.cui_to_concept_info:
                        mention_map = {"comm_uuid": test_doc_dict[file_key]["id"],
                                       "mention_uuid": mention['index'],
                                       "index": indx
                                       }
                        self.test_id_to_mention_info[indx] = mention_map
                        sent = mention['mention'][0].sent
                        test_mention_indexes_list.append(
                            [1 + token.i - sent.start for span in mention['mention'] for token in span])

                        sentence = [w.text.strip() for w in sent]

                        sentence.insert(0, "[CLS]")
                        sentence.append("[SEP]")
                        test_mention_sentences.append(sentence)

                        self.test_mention_ids[indx] = mention_map['index']
                        self.test_concept_ids[indx] = self.cui_to_concept_info[mention["concept"]][0]['index']
                        self.concepts_used.add(mention["concept"])
                        self.concepts_ids_used.add(self.cui_to_concept_info[mention["concept"]][0]['index'])

                        if args.test_limit is not None and indx == args.test_limit -1 :
                            self.log.warning("Limiting test dataset to {0}".format(args.test_limit))
                            self.test_mention_ids = self.test_mention_ids[:args.test_limit]
                            self.test_concept_ids = self.test_concept_ids[:args.test_limit]
                            self.test_num_examples = args.test_limit
                            break
                        indx += 1
            length = len(sorted(test_mention_indexes_list, key=len, reverse=True)[0])
            self.test_mention_indexes = np.array([xi + [-1] * (length - len(xi)) for xi in test_mention_indexes_list])

            self.test_mention_representations, self.test_mention_mask, self.test_mention_att, self.test_mention_reduced_mask = self.bert_tokenize(test_mention_sentences,
                                                                                 self.test_mention_indexes,
                                                                                 self.tokenizer,
                                                                                 self.mask_true_val,
                                                                                 self.mask_false_val,
                                                                                 padding_id,
                                                                              self.mask_true_val,
                                                                              self.mask_false_val)



        #self.mention_indexes = np.array(self.mention_indexes)

        self.log.info("Loaded bert characters")


    def load_bert_online_att_pretraining(self, args, test_doc_dict):
        self.only_annotated_concepts = args.only_annotated_concepts


        self.id_to_concept_info = {}
        self.cui_to_concept_info = {}

        self.mask_true_val = 1.
        self.mask_false_val =  0.#torch.Tensor([float("-inf")]).float()

        with open(args.syn_file, 'rb') as sf:
            syn_dict = pickle.load(sf)

        self.tokenizer = BertTokenizer.from_pretrained(args.bert_path,
                                                  do_lower_case="uncased" in args.bert_model)
        self.test_data = test_doc_dict is not None

        #concept_names = []

        concept_parts = []
        longest_seq = 0
        with open(args.lexicon, encoding='utf-8') as lex_file:
            tsv_reader = csv.reader(lex_file, delimiter="\t")
            index = 0
            for  row in tsv_reader:
                name = row[0]
                conceptId = row[1]

                concept_map = {"name": name,
                               "concept_id": conceptId,
                               "alternate": False,
                               "index": index
                               }
                #tokenized_name = nltk.word_tokenize(name)
                #concept_names.append(tokenized_name)
                name = "[CLS] {0} [SEP]".format(name)
                tokenized_name = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(name))
                concept_parts.append(tokenized_name)
                longest_seq = max(len(tokenized_name), longest_seq)
                self.cui_to_concept_info[conceptId] = [concept_map]
                self.id_to_concept_info[index] = concept_map
                index += 1
                pass
                """
                if row[7].strip() != "":
                    alt_names = row[7].split("|")
                    for an in alt_names:"""

        padding_id = self.tokenizer.convert_tokens_to_ids(["[PAD]"])

        self.concept_representations = torch.zeros((len(concept_parts), longest_seq), dtype=torch.long) + padding_id[0]
        self.concept_mask = torch.zeros((self.concept_representations.shape[0], self.concept_representations.shape[1])) + self.mask_false_val
        self.concept_att = torch.zeros((self.concept_representations.shape[0], self.concept_representations.shape[1])) + self.mask_false_val

        for i, cp in enumerate(concept_parts):
            self.concept_representations[i, :len(cp)] = torch.tensor(cp)
            self.concept_mask[i, :len(cp)] = self.mask_true_val
            self.concept_att[i, :len(cp)] = self.mask_true_val

        self.num_concepts = len(self.concept_representations)

        self.num_examples = sum(len(y) for y in syn_dict.values())

        self.mention_ids = np.zeros(self.num_examples)
        self.concept_ids = np.zeros(self.num_examples)

        mention_sentences = []
        self.id_to_mention_info = {}
        self.test_id_to_mention_info = {}
        self.mention_to_info = {}

        self.concepts_used = set()
        self.concepts_ids_used = set()

        indx = 0
        self.mention_indexes = []
        men_longest_seq = 0
        for cui, syn_list in syn_dict.items():
            for syn in syn_list:
                mention_map = {"comm_uuid": "{0}_{1}".format(cui, syn),
                               "mention_uuid": syn,
                               "index": indx
                               }
                #self.mention_to_info[mention["index"]] = mention_map
                self.id_to_mention_info[indx] = mention_map
                #self.mention_representations[indx, :] =  self.get_embedding(m.text, model, tokenizer, args, args.emb_layer)
                sentence = nltk.word_tokenize(syn)

                self.mention_indexes.append([i for i in range(len(sentence))])
                #self.mention_ids[indx] = self.mention_to_info[m.uuid.uuidString]['index']
                self.concept_ids[indx] = self.cui_to_concept_info[cui][0]['index']

                self.concepts_used.add(cui)
                self.concepts_ids_used.add(self.cui_to_concept_info[cui][0]['index'])

                sentence.insert(0, "[CLS]")
                sentence.append("[SEP]")

                mention_sentences.append(sentence)
                indx += 1


        length = len(sorted(self.mention_indexes, key=len, reverse=True)[0])
        self.mention_indexes = np.array([xi+[-1]*(length-len(xi)) for xi in self.mention_indexes])
        self.mention_representations, self.mention_mask, self.mention_att, self.mention_reduced_mask = self.bert_tokenize(mention_sentences,
                                                                             self.mention_indexes,
                                                                             self.tokenizer,
                                                                             self.mask_true_val,
                                                                             self.mask_false_val,
                                                                             padding_id,
                                                                           self.mask_true_val,
                                                                           self.mask_false_val)
        self.max_men_length = self.mention_reduced_mask.shape[1]

        if self.test_data:
            test_mention_indexes_list = []
            test_mention_sentences = []
            self.test_num_examples = sum([1 for file_key in test_doc_dict for mention in test_doc_dict[file_key]['concepts'] if mention['concept'] in self.cui_to_concept_info])

            self.test_concept_ids = np.zeros(self.test_num_examples)
            self.test_mention_ids = np.zeros(self.test_num_examples)
            indx = 0
            for file_key in test_doc_dict:
                for mention in test_doc_dict[file_key]['concepts']:
                    if mention['concept'] in self.cui_to_concept_info:
                        mention_map = {"comm_uuid": test_doc_dict[file_key]["id"],
                                       "mention_uuid": mention['index'],
                                       "index": indx
                                       }
                        self.test_id_to_mention_info[indx] = mention_map
                        sent = mention['mention'][0].sent
                        test_mention_indexes_list.append(
                            [1 + token.i - sent.start for span in mention['mention'] for token in span])

                        sentence = [w.text.strip() for w in sent]

                        sentence.insert(0, "[CLS]")
                        sentence.append("[SEP]")
                        test_mention_sentences.append(sentence)

                        self.test_mention_ids[indx] = mention_map['index']
                        self.test_concept_ids[indx] = self.cui_to_concept_info[mention["concept"]][0]['index']
                        self.concepts_used.add(mention["concept"])
                        self.concepts_ids_used.add(self.cui_to_concept_info[mention["concept"]][0]['index'])

                        if args.test_limit is not None and indx == args.test_limit -1 :
                            self.log.warning("Limiting test dataset to {0}".format(args.test_limit))
                            self.test_mention_ids = self.test_mention_ids[:args.test_limit]
                            self.test_concept_ids = self.test_concept_ids[:args.test_limit]
                            self.test_num_examples = args.test_limit
                            break
                        indx += 1
            length = len(sorted(test_mention_indexes_list, key=len, reverse=True)[0])
            self.test_mention_indexes = np.array([xi + [-1] * (length - len(xi)) for xi in test_mention_indexes_list])

            self.test_mention_representations, self.test_mention_mask, self.test_mention_att, self.test_mention_reduced_mask = self.bert_tokenize(test_mention_sentences,
                                                                                 self.test_mention_indexes,
                                                                                 self.tokenizer,
                                                                                 self.mask_true_val,
                                                                                 self.mask_false_val,
                                                                                 padding_id,
                                                                              self.mask_true_val,
                                                                              self.mask_false_val)



        #self.mention_indexes = np.array(self.mention_indexes)

        self.log.info("Loaded bert characters")

    def load_elmo_online(self, doc_dict, args, test_doc_dict):
        self.only_annotated_concepts = args.only_annotated_concepts

        mask_true_val = 0.
        mask_false_val =  torch.Tensor([float("-inf")]).float()

        self.id_to_concept_info = {}
        self.cui_to_concept_info = {}

        self.test_data = test_doc_dict is not None

        concept_names = []

        self.included_concepts = set()
        if self.only_annotated_concepts:
            for file in doc_dict:
                for mention in doc_dict[file]['concepts']:
                    self.included_concepts.add(mention['concept'])

        if args.dataset != 'n2c2':
            with open(args.lexicon, encoding='utf-8') as lex_file:
                tsv_reader = csv.reader(lex_file, delimiter="\t")
                index = 0
                for  row in tsv_reader:
                    name = row[0]
                    conceptId = row[1]
                    if self.only_annotated_concepts and conceptId not in self.included_concepts:
                        continue
                    concept_map = {"name": name,
                                   "concept_id": conceptId,
                                   "alternate": False,
                                   "index": index
                                   }

                    concept_names.append(nltk.word_tokenize(name))
                    self.cui_to_concept_info[conceptId] = [concept_map]
                    self.id_to_concept_info[index] = concept_map
                    index += 1
        elif args.dataset == 'n2c2':
            with open(args.lexicon, 'rb') as lex_pickle_file:
                lex_pickle = pickle.load(lex_pickle_file)
                index = 0
            for id, stuff in lex_pickle.items():

                concept_map = stuff
                concept_map['index'] = index
                if self.only_annotated_concepts and concept_map['concept_id'] not in self.included_concepts:
                    continue
                concept_names.append(nltk.word_tokenize( concept_map['name']))
                self.cui_to_concept_info[ concept_map['concept_id']] = [concept_map]
                self.id_to_concept_info[index] = concept_map
                index += 1

        self.concept_representations = batch_to_ids(concept_names)
        self.num_concepts = len(self.concept_representations)

        self.concept_mask = torch.zeros((self.concept_representations.shape[0], self.concept_representations.shape[1])) + mask_false_val
        for k in range (len(self.concept_representations)):
            self.concept_mask[k, :len(concept_names[k])] = mask_true_val
        if self.test_data:
            combined_dict = {**doc_dict, **test_doc_dict}
        else:
            combined_dict = doc_dict

        self.num_examples = sum([1 for file_key in doc_dict for mention in doc_dict[file_key]['concepts']
                                 if mention['concept'] in self.cui_to_concept_info])

        self.mention_ids = np.zeros(self.num_examples)
        self.concept_ids = np.zeros(self.num_examples)

        self.mention_representations = torch.zeros(size=(self.num_examples, self.bert_models[args.bert_model]))
        self.id_to_mention_info = {}
        self.test_id_to_mention_info = {}
        self.mention_to_info = {}

        self.concepts_used = set()
        self.concepts_ids_used = set()

        indx = 0
        mention_sentences = []
        self.mention_indexes = []
        for file in doc_dict:
            for mention in doc_dict[file]['concepts']:
                if mention['concept'] in self.cui_to_concept_info:
                    mention_map = {"comm_uuid": doc_dict[file]["id"],
                                   "mention_uuid": mention["index"],
                                   "index": indx
                                   }
                    self.mention_to_info[mention["index"]] = mention_map
                    self.id_to_mention_info[indx] = mention_map

                    sent = mention['mention'][0].sent

                    sentence = [w.text.strip() for w in sent]
                    sentence.insert(0, "[CLS]")
                    sentence.append("[SEP]")
                    mention_sentences.append(sentence)

                    self.mention_indexes.append(
                        [1 + token.i - sent.start for span in mention['mention'] for token in span])
                    self.mention_ids[indx] = mention_map['index']
                    self.concept_ids[indx] = self.cui_to_concept_info[mention['concept']][0]['index']

                    self.concepts_used.add(mention['concept'])
                    self.concepts_ids_used.add(self.cui_to_concept_info[mention['concept']][0]['index'])

                    indx += 1
        self.mention_representations = batch_to_ids(mention_sentences)
        self.mention_mask = torch.zeros((self.mention_representations.shape[0], self.mention_representations.shape[1])) + mask_false_val
        for k in range (len(self.mention_representations)):
            for l in self.mention_indexes[k]:
                self.mention_mask[k, l] = mask_true_val


        # test stuff
        if self.test_data:
            if args.test_include_cuiless:
                c_map = {"name": "CUI-less",
                         "concept_id": "CUI-less",
                         "alternate": False,
                         "index": self.concept_representations.shape[0]
                         }
                self.id_to_concept_info[self.concept_representations.shape[0]] = c_map
                self.cui_to_concept_info["CUI-less"] = [c_map]
                self.log.info("Adding cuiless")
            test_mention_sentences = []
            test_mention_indexes_list = []
            self.test_num_examples = sum([1 for file_key in test_doc_dict for mention in test_doc_dict[file_key]['concepts'] if mention['concept'] in self.cui_to_concept_info])

            self.test_mention_ids = np.zeros(self.test_num_examples)

            #self.test_mention_ids = np.zeros(self.test_num_examples)
            self.test_concept_ids = np.zeros(self.test_num_examples)
            indx = 0
            for file_key in test_doc_dict:
                for mention in test_doc_dict[file_key]['concepts']:
                    if mention['concept'] in self.cui_to_concept_info:
                        mention_map = {"comm_uuid": test_doc_dict[file_key]["id"],
                                       "mention_uuid": mention['index'],
                                       "index": indx
                                       }
                        self.mention_to_info[mention["index"]] = mention_map
                        self.test_id_to_mention_info[indx] = mention_map

                        sent = mention['mention'][0].sent
                        test_mention_indexes_list.append(
                            [1 + token.i - sent.start for span in mention['mention'] for token in span])

                        sentence = [w.text.strip() for w in sent]

                        sentence.insert(0, "[CLS]")
                        sentence.append("[SEP]")
                        test_mention_sentences.append(sentence)

                        self.test_mention_ids[indx] = mention_map['index']

                        self.test_concept_ids[indx] = self.cui_to_concept_info[mention["concept"]][0]['index']
                        self.concepts_used.add(mention["concept"])
                        self.concepts_ids_used.add(self.cui_to_concept_info[mention["concept"]][0]['index'])

                        indx += 1
                        if args.test_limit is not None and indx == args.test_limit -1 :
                            self.log.warning("Limiting test dataset to {0}".format(args.test_limit))
                            self.test_mention_ids = self.test_mention_ids[:args.test_limit]
                            self.test_concept_ids = self.test_concept_ids[:args.test_limit]
                            self.test_num_examples = args.test_limit
                            break
            self.test_mention_representations = batch_to_ids(test_mention_sentences)
            length = len(sorted(test_mention_indexes_list, key=len, reverse=True)[0])
            self.test_mention_indexes = np.array([xi + [-1] * (length - len(xi)) for xi in test_mention_indexes_list])

            self.test_mention_mask = torch.zeros(
                (self.test_mention_representations.shape[0], self.test_mention_representations.shape[1])) + mask_false_val
            for k in range(len(self.test_mention_representations)):
                for l in test_mention_indexes_list[k]:
                    self.test_mention_mask[k, l] = mask_true_val

        length = len(sorted(self.mention_indexes, key=len, reverse=True)[0])
        self.mention_indexes = np.array([xi+[-1]*(length-len(xi)) for xi in self.mention_indexes])
        #self.mention_indexes = np.array(self.mention_indexes)

        self.log.info("Loaded elmo characters")


    def elmo_representations(self, args, characters, indexes, elmo=None, mention=False):
        if not args.syn_file and not args.concept_context and args.dataset != 'n2c2':

            characters = torch_utils.gpu(characters, gpu=args.use_cuda)
        if mention:
            max_men_length = max(len(x) for x in indexes)
            representations = torch.zeros((characters.shape[0], max_men_length,
                                           elmo._elmo_lstm._elmo_lstm.input_size * 2))
        else:

            representations = torch.zeros((characters.shape[0], characters.shape[1],
                                                 elmo._elmo_lstm._elmo_lstm.input_size*2))

        if not args.online:
            for i in range(0, len(characters), args.batch_size):
                upper = min(i+args.batch_size, len(characters))


                if mention:
                    intermediate = \
                        elmo(characters[i:upper, :,:])["elmo_representations"][args.emb_layer].detach()

                    for k in range(i, upper):
                        representations[k, :len(indexes[k]), :] = intermediate[k-i, indexes[k], :]
                else:
                    representations[i:upper, :,  :] = \
                        elmo(characters[i:upper, :,:])["elmo_representations"][args.emb_layer].detach()

        mask = torch.zeros((characters.shape[0], characters.shape[1])) + self.mask_false_val

        for k in range(len(representations)):
            for index_e, l in enumerate(indexes[k]):
                # if mention:
                #    mask[k, index_e] = self.mask_true_val
                # else:
                mask[k, l] = self.mask_true_val
            if mention:
                upper = mask.shape[1]
                lower = upper - (max_men_length - len(indexes[k]))
                mask[k, lower:upper] = self.mask_true_val


        max_rep = None
        if not args.online:

            masked_rep = (representations + mask
                              .view(representations.size()[0], representations.size()[1], 1)
                              .expand(representations.size()[0], representations.size()[1],
                                      representations.size()[2]))

            max_rep = masked_rep.max(dim=1)[0]

        if mention:
            reduced_mask = torch.zeros((characters.shape[0], max_men_length)) + self.mask_false_val
            for k in range(len(representations)):
                for index_e, l in enumerate(indexes[k]):
                    reduced_mask[k, index_e] = self.mask_true_val
            return representations, mask, max_rep, reduced_mask
        else:
            return representations, mask, max_rep


    def remove_whitespace(self, mention):
        sent_list = []
        orig_token_index_list = [token.i - mention['mention'][0].sent.start for span in mention['mention'] for token in
                                 span]
        new_tok_index_list = []
        for orig_i, token in enumerate(mention['mention'][0].sent):
            if token.text.strip() != "":
                sent_list.append(token.text.strip())
                if orig_i in orig_token_index_list:
                    new_tok_index_list.append(len(sent_list) - 1)

        return sent_list, new_tok_index_list

    def remove_whitespace_mm(self, men_span):
        sent_list = []
        orig_token_index_list = [token.i - men_span.sent.start for token in men_span]
        new_tok_index_list = []
        for orig_i, token in enumerate(men_span.sent):
            if token.text.strip() != "":
                sent_list.append(token.text.strip())
                if orig_i in orig_token_index_list:
                    new_tok_index_list.append(len(sent_list) - 1)

        return sent_list, new_tok_index_list


    def remove_whitespace_tokens(self, token_list):
        sent_list = []
        orig_token_index_list = [token.i - token_list[0].i for token in token_list]
        new_tok_index_list = []
        for orig_i, token in enumerate(token_list):
            if token.text.strip() != "":
                sent_list.append(token.text.strip())
                if orig_i in orig_token_index_list:
                    new_tok_index_list.append(len(sent_list) - 1)

        return sent_list, new_tok_index_list


    def load_elmo_attention(self, doc_dict, args, test_doc_dict):

        elmo = torch_utils.gpu(ElmoAllLayers(args.elmo_options_file, args.elmo_weight_file, 1,
                                         requires_grad=False),
                                    gpu=args.use_cuda)
        elmo.eval()

        self.only_annotated_concepts = args.only_annotated_concepts

        self.mask_true_val = 1.
        self.mask_false_val =  0.#torch.Tensor([float("-inf")]).float()

        self.id_to_concept_info = {}
        self.cui_to_concept_info = {}

        self.test_data = test_doc_dict is not None

        concept_names = []

        self.included_concepts = set()
        if self.only_annotated_concepts:
            for file in doc_dict:
                for mention in doc_dict[file]['concepts']:
                    self.included_concepts.add(mention['concept'])

        self.concept_indexes = []


        max_num_syns = 4
        min_num_concept_toks = 20
        concept_context_dict = {}
        if args.concept_context != "":
            with open (args.concept_context, 'rb') as confile:
                concept_context_dict = pickle.load(confile)

        if args.dataset != 'n2c2':

            with open(args.lexicon, encoding='utf-8') as lex_file:
                tsv_reader = csv.reader(lex_file, delimiter="\t")
                index = 0
                for  row in tsv_reader:
                    name = row[0]
                    conceptId = row[1]
                    if self.only_annotated_concepts and conceptId not in self.included_concepts:
                        continue
                    concept_map = {"name": name,
                                   "concept_id": conceptId,
                                   "alternate": False,
                                   "index": index
                                   }
                    tok_name = nltk.word_tokenize(name)

                    if args.concept_context and conceptId in concept_context_dict:
                        for ni, syn in enumerate(concept_context_dict[conceptId]["def"]):
                            if len(tok_name) >= min_num_concept_toks:
                                break

                            tok_name += nltk.word_tokenize(syn)
                        for ni, syn in enumerate(concept_context_dict[conceptId]["names"]):
                            if len(tok_name) >= min_num_concept_toks:
                                break

                            tok_name += nltk.word_tokenize(syn)

                    concept_names.append(tok_name)
                    self.cui_to_concept_info[conceptId] = [concept_map]
                    self.id_to_concept_info[index] = concept_map

                    self.concept_indexes.append(list(range(len(tok_name))))

                    index += 1

        elif args.dataset == 'n2c2':
            with open(args.lexicon, 'rb') as lex_pickle_file:
                lex_pickle = pickle.load(lex_pickle_file)
                index = 0
            for id, stuff in lex_pickle.items():

                concept_map = stuff
                concept_map['index'] = index
                if self.only_annotated_concepts and concept_map['concept_id'] not in self.included_concepts:
                    continue

                self.cui_to_concept_info[ concept_map['concept_id']] = [concept_map]
                self.id_to_concept_info[index] = concept_map



                tok_name = nltk.word_tokenize( concept_map['name'])

                if len(tok_name) >= 50:
                    tok_name = tok_name[:50]

                if args.concept_context and  concept_map['concept_id'] in concept_context_dict:
                    for ni, syn in enumerate(concept_context_dict[ concept_map['concept_id']]["def"]):
                        if len(tok_name) >= min_num_concept_toks:
                            break

                        tok_name += nltk.word_tokenize(syn)
                    for ni, syn in enumerate(concept_context_dict[ concept_map['concept_id']]["names"]):
                        if len(tok_name) >= min_num_concept_toks:
                            break

                        tok_name += nltk.word_tokenize(syn)

                concept_names.append(tok_name)

                self.concept_indexes.append(list(range(len(tok_name))))

                index += 1
            self.log.info("Num of concepts: {0}".format(len(concept_names)))
        concept_chars = batch_to_ids(concept_names)

        # concept_lengths = [len(x) for x in concept_names]
        # num_bins = len(set(concept_lengths))
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # sns.distplot(concept_lengths, color="skyblue", label="cuiless", kde=False, rug=True, bins=num_bins)
        # plt.show()

        if not args.online:
            self.concept_representations, self.concept_mask, rep_masked = self.elmo_representations(args, concept_chars, self.concept_indexes, elmo)

        else:
            _, self.concept_mask, _ = self.elmo_representations(args, concept_chars, self.concept_indexes, elmo)

            self.concept_representations = concept_chars

        self.num_concepts = len(self.concept_representations)

        if self.test_data:
            combined_dict = {**doc_dict, **test_doc_dict}
        else:
            combined_dict = doc_dict

        self.num_examples = sum([1 for file_key in doc_dict for mention in doc_dict[file_key]['concepts']
                                 if mention['concept'] in self.cui_to_concept_info])

        self.mention_ids = np.zeros(self.num_examples)
        self.concept_ids = np.zeros(self.num_examples)

        self.id_to_mention_info = {}
        self.test_id_to_mention_info = {}
        self.mention_to_info = {}

        self.concepts_used = set()
        self.concepts_ids_used = set()

        if args.neg_samps:
            concept_neighbors = defaultdict(set)
            concept_neighbor_ids = defaultdict(list)

            with open(args.neg_samps, 'rb') as neg_samp_pickle:
                negative_samples = pickle.load(neg_samp_pickle)
            self.mention_neighbors = torch.zeros(size=(self.num_examples, args.max_neg_samples))
            for cui in negative_samples:
                concept_neighbors[cui].update(negative_samples[cui]["parents"].keys())
                concept_neighbors[cui].update(negative_samples[cui]["siblings"].keys())
                these_children = negative_samples[cui]["children"].keys()
                max_children = args.max_neg_samples - len(concept_neighbors[cui])
                if len(negative_samples[cui]["children"]) < max_children:
                    concept_neighbors[cui].update(these_children)

                else:
                    concept_neighbors[cui].update(sample(these_children, max_children))

                these_others = set(x for x in negative_samples[cui]["others"].keys() if x not in concept_neighbors[cui])
                max_others = args.max_neg_samples - len(concept_neighbors[cui])
                if len(these_others) < max_others:
                    concept_neighbors[cui].update(these_others)

                else:
                    concept_neighbors[cui].update(sample(these_others, max_others))

                concept_neighbor_ids[cui] = [self.cui_to_concept_info[q][0]["index"] for q in concept_neighbors[cui] if q in self.cui_to_concept_info]


        indx = 0
        mention_sentences = []
        self.mention_indexes = []
        mention_uuids = []
        mention_neighbor_list = []

        if args.syn_file:

            with open(args.syn_file, 'rb') as sf:
                syn_dict = pickle.load(sf)
            self.num_examples = sum(len(y) for y in syn_dict.values())
            self.mention_ids = np.zeros(self.num_examples)
            self.concept_ids = np.zeros(self.num_examples)
            skip = False
            for cui, syn_list in syn_dict.items():
                for syn in syn_list:
                    if not skip:
                        mention_map = {"comm_uuid": "{0}_{1}".format(cui, syn),
                                       "mention_uuid": syn,
                                       "index": indx
                                       }
                        # self.mention_to_info[mention["index"]] = mention_map
                        self.id_to_mention_info[indx] = mention_map
                        # self.mention_representations[indx, :] =  self.get_embedding(m.text, model, tokenizer, args, args.emb_layer)
                        sentence = nltk.word_tokenize(syn)
                        if len(sentence) >= 50:
                            sentence = sentence[:50]

                        self.mention_indexes.append([i for i in range(len(sentence))])
                        # self.mention_ids[indx] = self.mention_to_info[m.uuid.uuidString]['index']
                        self.concept_ids[indx] = self.cui_to_concept_info[cui][0]['index']

                        self.concepts_used.add(cui)
                        self.concepts_ids_used.add(self.cui_to_concept_info[cui][0]['index'])

                        sentence.insert(0, "[CLS]")
                        sentence.append("[SEP]")

                        mention_sentences.append(sentence)
                        indx += 1

                    if args.limit_syns != -1 and args.limit_syns <= indx:
                        self.log.info("Limit synonyms to {0}".format(indx))
                        skip = True

        else:
            for file in doc_dict:
                doc = doc_dict[file]['note']
                for mention in doc_dict[file]['concepts']:
                    if mention['concept'] in self.cui_to_concept_info:
                        mention_map = {"comm_uuid": doc_dict[file]["id"],
                                       "mention_uuid": mention["index"],
                                       "index": indx
                                       }
                        self.mention_to_info[mention["index"]] = mention_map
                        self.id_to_mention_info[indx] = mention_map
                        mention_uuids.append(mention["index"])
                        sent = mention['mention'][0].sent
                        toks = list(sent)

                        start_tok = toks[0]
                        end_tok = toks[-1]

                        if sum(1 for x in toks if not x.is_space) > 20:

                            for j in range (1, max(start_tok.i, doc[-1].i - end_tok.i)):

                                if start_tok.i - j >= 0:
                                    toks.insert(0, doc[start_tok.i - j])
                                if end_tok.i + j <= doc[-1].i:
                                    toks.append(doc[end_tok.i + j])
                                if sum(1 for x in toks if not x.is_space) > 20:
                                    break



                        sent_list, new_tok_index_list = self.remove_whitespace(mention)
                        #sent_list, new_tok_index_list = self.remove_whitespace_tokens(toks)

                        mention_sentences.append(sent_list)
                        self.mention_indexes.append(new_tok_index_list)

                        self.mention_ids[indx] = mention_map['index']
                        this_concept = self.cui_to_concept_info[mention['concept']]
                        this_cui = this_concept[0]["concept_id"]
                        self.concept_ids[indx] = this_concept[0]['index']
                        if args.neg_samps:
                            repeated_list = list(islice(cycle(concept_neighbor_ids[this_cui]), args.max_neg_samples))
                            if len(repeated_list) == 0:
                                repeated_list = np.random.randint(0, self.num_concepts, size=args.max_neg_samples)

                            self.mention_neighbors[indx, :] = torch.tensor(repeated_list)
                            mention_neighbor_list.append(repeated_list)

                        self.concepts_used.add(mention['concept'])
                        self.concepts_ids_used.add(self.cui_to_concept_info[mention['concept']][0]['index'])

                        indx += 1
        mention_characters = batch_to_ids(mention_sentences)

        #self.mention_neighbors.gather(1, torch.LongTensor(self.num_examples, 10).random_(0, args.max_neg_samples))
        # concept_lengths = [len(x) for x in mention_sentences if len(x) < 20]
        # num_bins = len(set(concept_lengths))
        #
        # sns.distplot(concept_lengths, color="red", label="cuiless", kde=False, rug=True, bins=num_bins)
        # plt.show()
        max_men_length = max(len(x) for x in self.mention_indexes)

        mention_characters = F.pad(mention_characters, (0,0,0,max_men_length-1), "constant", 0)
        if not args.online:
            self.mention_representations, self.mention_mask, men_mask_rep, self.mention_reduced_mask = \
                self.elmo_representations(args, mention_characters, self.mention_indexes, elmo, mention=True)
        else:
            _, self.mention_mask, _,  self.mention_reduced_mask\
                = self.elmo_representations(args, mention_characters, self.mention_indexes, elmo, mention=True)
            self.mention_representations = mention_characters
            self.max_men_length = max(len(x) for x in self.mention_indexes)
            self.mention_indexes = np.array([xi + [-1] * (self.max_men_length - len(xi)) for xi in self.mention_indexes])
        # test stuff
        if self.test_data:
            if args.test_include_cuiless:
                c_map = {"name": "CUI-less",
                         "concept_id": "CUI-less",
                         "alternate": False,
                         "index": self.concept_representations.shape[0]
                         }
                self.id_to_concept_info[self.concept_representations.shape[0]] = c_map
                self.cui_to_concept_info["CUI-less"] = [c_map]
                self.log.info("Adding cuiless")
            test_mention_sentences = []
            self.test_mention_indexes = []
            self.test_num_examples = sum([1 for file_key in test_doc_dict for mention in test_doc_dict[file_key]['concepts'] if mention['concept'] in self.cui_to_concept_info])

            self.test_mention_ids = np.zeros(self.test_num_examples)

            #self.test_mention_ids = np.zeros(self.test_num_examples)
            self.test_concept_ids = np.zeros(self.test_num_examples)
            test_men_uuids = []
            #indx = 0
            inner_indx = 0
            for file_key in test_doc_dict:
                for mention in test_doc_dict[file_key]['concepts']:
                    if mention['concept'] in self.cui_to_concept_info:
                        mention_map = {"comm_uuid": test_doc_dict[file_key]["id"],
                                       "mention_uuid": mention['index'],
                                       "index": indx
                                       }
                        self.mention_to_info[mention["index"]] = mention_map
                        self.test_id_to_mention_info[indx] = mention_map

                        sent = mention['mention'][0].sent

                        sent_list, new_tok_index_list = self.remove_whitespace(mention)

                        self.test_mention_indexes.append(new_tok_index_list)

                        test_mention_sentences.append(sent_list)

                        self.test_mention_ids[inner_indx] = mention_map['index']

                        self.test_concept_ids[inner_indx] = self.cui_to_concept_info[mention["concept"]][0]['index']
                        self.concepts_used.add(mention["concept"])
                        self.concepts_ids_used.add(self.cui_to_concept_info[mention["concept"]][0]['index'])
                        test_men_uuids.append(mention["index"])

                        indx += 1
                        inner_indx += 1
                        if args.test_limit is not None and indx == args.test_limit -1 :
                            self.log.warning("Limiting test dataset to {0}".format(args.test_limit))
                            self.test_mention_ids = self.test_mention_ids[:args.test_limit]
                            self.test_concept_ids = self.test_concept_ids[:args.test_limit]
                            self.test_num_examples = args.test_limit
                            break

            test_mention_characters = batch_to_ids(test_mention_sentences)
            test_max_men_length = max(len(x) for x in self.test_mention_indexes)

            test_mention_characters = F.pad(test_mention_characters, (0, 0, 0, test_max_men_length - 1), "constant", 0)
            if not args.online:
                self.test_mention_representations, self.test_mention_mask, test_mention_max, self.test_mention_reduced_mask = \
                    self.elmo_representations(args, test_mention_characters, self.test_mention_indexes, elmo, mention=True)

                padded_test_mention_representations = torch.zeros((self.test_mention_representations.shape[0],
                                                                   self.mention_representations.shape[1],
                                                                   self.mention_representations.shape[2]))
                padded_test_mention_representations[:, :self.test_mention_representations.shape[1], :] = self.test_mention_representations
                self.mention_representations = torch.cat((self.mention_representations, padded_test_mention_representations), 0)
            else:
                _, self.test_mention_mask, _, self.test_mention_reduced_mask = \
                    self.elmo_representations(args, test_mention_characters, self.test_mention_indexes, elmo, mention=True)
                self.test_mention_representations = test_mention_characters
                length = max(len(x) for x in self.test_mention_indexes)
                self.test_mention_indexes = np.array([xi + [-1] * (length - len(xi)) for xi in self.test_mention_indexes])
                #self.max_men_length = max(self.max_men_length, length)
                if length > self.max_men_length:
                    #self.mention_reduced_mask  = F.pad(self.mention_reduced_mask, (0, length - self.max_men_length, 0, 0), "constant", 0)
                    self.max_men_length = length
        #self.mention_indexes = np.array(self.mention_indexes)

        self.log.info("Loaded elmo characters")

    def load_elmo_attention_mm(self, doc_dict, args, test_doc_dict):

        elmo = torch_utils.gpu(ElmoAllLayers(args.elmo_options_file, args.elmo_weight_file, 1,
                                         requires_grad=False),
                                    gpu=args.use_cuda)
        elmo.eval()

        self.only_annotated_concepts = args.only_annotated_concepts

        self.mask_true_val = 1.
        self.mask_false_val =  0.#torch.Tensor([float("-inf")]).float()

        self.id_to_concept_info = {}
        self.cui_to_concept_info = {}

        self.test_data = test_doc_dict is not None

        concept_names = []

        self.included_concepts = set()
        if self.only_annotated_concepts:
            for file in doc_dict:
                for mention in doc_dict[file]['concepts']:
                    self.included_concepts.add(mention['concept'])

        self.concept_indexes = []


        max_num_syns = 4
        min_num_concept_toks = 20
        concept_context_dict = {}
        if args.concept_context != "":
            with open (args.concept_context, 'rb') as confile:
                concept_context_dict = pickle.load(confile)



        with open(args.lexicon, encoding='utf-8') as lex_file:
            tsv_reader = csv.reader(lex_file, delimiter="\t")
            index = 0
            for  row in tsv_reader:
                name = row[0]
                conceptId = row[1]
                if self.only_annotated_concepts and conceptId not in self.included_concepts:
                    continue
                concept_map = {"name": name,
                               "concept_id": conceptId,
                               "alternate": False,
                               "index": index
                               }
                tok_name = nltk.word_tokenize(name)

                if args.concept_context and conceptId in concept_context_dict:
                    for ni, syn in enumerate(concept_context_dict[conceptId]["def"]):
                        if len(tok_name) >= min_num_concept_toks:
                            break

                        tok_name += nltk.word_tokenize(syn)
                    for ni, syn in enumerate(concept_context_dict[conceptId]["names"]):
                        if len(tok_name) >= min_num_concept_toks:
                            break

                        tok_name += nltk.word_tokenize(syn)

                concept_names.append(tok_name)
                self.cui_to_concept_info[conceptId] = [concept_map]
                self.id_to_concept_info[index] = concept_map

                self.concept_indexes.append(list(range(len(tok_name))))

                index += 1
                """
                if row[7].strip() != "":
                    alt_names = row[7].split("|")
                    for an in alt_names:"""
        concept_chars = batch_to_ids(concept_names)

        # concept_lengths = [len(x) for x in concept_names]
        # num_bins = len(set(concept_lengths))
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # sns.distplot(concept_lengths, color="skyblue", label="cuiless", kde=False, rug=True, bins=num_bins)
        # plt.show()

        if not args.online:
            self.concept_representations, self.concept_mask, rep_masked = self.elmo_representations(args, concept_chars, self.concept_indexes, elmo)

        else:
            _, self.concept_mask, _ = self.elmo_representations(args, concept_chars, self.concept_indexes, elmo)

            self.concept_representations = concept_chars

        self.num_concepts = len(self.concept_representations)

        if self.test_data:
            combined_dict = {**doc_dict, **test_doc_dict}
        else:
            combined_dict = doc_dict

        self.num_examples = sum([1 for file_key in doc_dict for mention in doc_dict[file_key]['concepts']
                                 if mention['concept'] in self.cui_to_concept_info])

        self.mention_ids = np.zeros(self.num_examples)
        self.concept_ids = np.zeros(self.num_examples)

        self.id_to_mention_info = {}
        self.test_id_to_mention_info = {}
        self.mention_to_info = {}

        self.concepts_used = set()
        self.concepts_ids_used = set()




        if args.neg_samps:
            concept_neighbors = defaultdict(set)
            concept_neighbor_ids = defaultdict(list)

            with open(args.neg_samps, 'rb') as neg_samp_pickle:
                negative_samples = pickle.load(neg_samp_pickle)
            self.mention_neighbors = torch.zeros(size=(self.num_examples, args.max_neg_samples))
            for cui in negative_samples:
                concept_neighbors[cui].update(negative_samples[cui]["parents"].keys())
                concept_neighbors[cui].update(negative_samples[cui]["siblings"].keys())
                these_children = negative_samples[cui]["children"].keys()
                max_children = args.max_neg_samples - len(concept_neighbors[cui])
                if len(negative_samples[cui]["children"]) < max_children:
                    concept_neighbors[cui].update(these_children)

                else:
                    concept_neighbors[cui].update(sample(these_children, max_children))

                these_others = set(x for x in negative_samples[cui]["others"].keys() if x not in concept_neighbors[cui])
                max_others = args.max_neg_samples - len(concept_neighbors[cui])
                if len(these_others) < max_others:
                    concept_neighbors[cui].update(these_others)

                else:
                    concept_neighbors[cui].update(sample(these_others, max_others))

                concept_neighbor_ids[cui] = [self.cui_to_concept_info[q][0]["index"] for q in concept_neighbors[cui] if q in self.cui_to_concept_info]

        """
        indx = 0
        mention_sentences = []
        self.mention_indexes = []
        mention_uuids = []
        mention_neighbor_list = []
        for file in doc_dict:
            doc = doc_dict[file]['note']
            for mention in doc_dict[file]['concepts']:
                if mention['concept'] in self.cui_to_concept_info:
                    mention_map = {"comm_uuid": doc_dict[file]["id"],
                                   "mention_uuid": mention["index"],
                                   "index": indx
                                   }
                    self.mention_to_info[mention["index"]] = mention_map
                    self.id_to_mention_info[indx] = mention_map
                    mention_uuids.append(mention["index"])
                    sent = mention['mention'][0].sent
                    toks = list(sent)

                    start_tok = toks[0]
                    end_tok = toks[-1]

                    if sum(1 for x in toks if not x.is_space) > 20:

                        for j in range (1, max(start_tok.i, doc[-1].i - end_tok.i)):

                            if start_tok.i - j >= 0:
                                toks.insert(0, doc[start_tok.i - j])
                            if end_tok.i + j <= doc[-1].i:
                                toks.append(doc[end_tok.i + j])
                            if sum(1 for x in toks if not x.is_space) > 20:
                                break



                    sent_list, new_tok_index_list = self.remove_whitespace(mention)
                    #sent_list, new_tok_index_list = self.remove_whitespace_tokens(toks)

                    mention_sentences.append(sent_list)
                    self.mention_indexes.append(new_tok_index_list)

                    self.mention_ids[indx] = mention_map['index']
                    this_concept = self.cui_to_concept_info[mention['concept']]
                    this_cui = this_concept[0]["concept_id"]
                    self.concept_ids[indx] = this_concept[0]['index']
                    if args.neg_samps:
                        repeated_list = list(islice(cycle(concept_neighbor_ids[this_cui]), args.max_neg_samples))
                        if len(repeated_list) == 0:
                            repeated_list = np.random.randint(0, self.num_concepts, size=args.max_neg_samples)

                        self.mention_neighbors[indx, :] = torch.tensor(repeated_list)
                        mention_neighbor_list.append(repeated_list)

                    self.concepts_used.add(mention['concept'])
                    self.concepts_ids_used.add(self.cui_to_concept_info[mention['concept']][0]['index'])

                    indx += 1
        mention_characters = batch_to_ids(mention_sentences)

        #self.mention_neighbors.gather(1, torch.LongTensor(self.num_examples, 10).random_(0, args.max_neg_samples))
        # concept_lengths = [len(x) for x in mention_sentences if len(x) < 20]
        # num_bins = len(set(concept_lengths))
        #
        # sns.distplot(concept_lengths, color="red", label="cuiless", kde=False, rug=True, bins=num_bins)
        # plt.show()
        max_men_length = max(len(x) for x in self.mention_indexes)

        mention_characters = F.pad(mention_characters, (0,0,0,max_men_length-1), "constant", 0)
        if not args.online:
            self.mention_representations, self.mention_mask, men_mask_rep, self.mention_reduced_mask = \
                self.elmo_representations(args, mention_characters, self.mention_indexes, elmo, mention=True)
        else:
            _, self.mention_mask, _,  self.mention_reduced_mask\
                = self.elmo_representations(args, mention_characters, self.mention_indexes, elmo, mention=True)
            self.mention_representations = mention_characters
            self.max_men_length = max(len(x) for x in self.mention_indexes)
            self.mention_indexes = np.array([xi + [-1] * (self.max_men_length - len(xi)) for xi in self.mention_indexes])
        # test stuff
        """



        if self.test_data:
            test_mention_sentences = []
            self.test_mention_indexes = []
            self.test_num_examples = sum([1 for file_key in test_doc_dict for mention in test_doc_dict[file_key]['concepts'] if mention['concept'] in self.cui_to_concept_info])

            self.test_mention_ids = np.zeros(self.test_num_examples)

            #self.test_mention_ids = np.zeros(self.test_num_examples)
            self.test_concept_ids = np.zeros(self.test_num_examples)
            test_men_uuids = []
            indx = 0
            inner_indx = 0

            metamap = MetaMap(
                metamap_path="")  # doesn't need actual path as we are just parsing cached output
            train, test, dev = load_share_clef_2013(partition='train'), load_share_clef_2013(
                partition='test'), load_share_clef_2013(partition='dev')
            for file in test:
                metamap_dict = metamap.load(file['note']._.metamapped_file)
                mapped_terms = metamap.extract_mapped_terms(metamap_dict)
                metamap_span_to_cui = {}
                for term in mapped_terms:
                    cui = term['CandidateCUI']
                    start, end = metamap.get_span_by_term(term)[0]
                    if (start, end) not in metamap_span_to_cui:
                        metamap_span_to_cui[(start, end)] = []
                    if cui not in metamap_span_to_cui[(start, end)]:
                        metamap_span_to_cui[(start, end)].append(cui)

                    mention_uuid = "{0}_{1}_{2}".format(file["id"],start, end)
                    if not mention_uuid in self.mention_to_info:
                        mention_map = {"comm_uuid": file["id"],
                                       "mention_uuid": mention_uuid,
                                       "index": indx
                                       }

                        self.mention_to_info[mention_uuid] = mention_map
                        self.test_id_to_mention_info[indx] = mention_map
                        try:
                            char_span = file['note'].char_span(start, end)
                            sent = char_span.sent


                        except:
                            span_list = []
                            found_start = False
                            for tok in file['note']:
                                tok_start = tok.idx
                                tok_end = tok_start + len(tok.text)
                                tok_range = range(tok_start, tok_end)


                                if start in range(tok_start, tok_end-1) or end in range(tok_start+1, tok_end):
                                    span_list.append(tok)
                                elif tok_start >= start and tok_end <= end:
                                    span_list.append(tok)

                            self.log.info("Error in {0}".format(mention_uuid))
                            char_span = file['note'].char_span(span_list[0].idx, span_list[-1].idx + len(span_list[-1].text))
                            sent = char_span.sent
                        sent_list, new_tok_index_list = self.remove_whitespace_mm(char_span)

                        self.test_mention_indexes.append(new_tok_index_list)

                        test_mention_sentences.append(sent_list)

                        self.test_mention_ids[inner_indx] = mention_map['index']

                        # self.test_concept_ids[inner_indx] = self.cui_to_concept_info[mention["concept"]][0]['index']
                        # self.concepts_used.add(mention["concept"])
                        # self.concepts_ids_used.add(self.cui_to_concept_info[mention["concept"]][0]['index'])
                        test_men_uuids.append(mention_uuid)

                        indx += 1
                        inner_indx += 1
            test_mention_characters = batch_to_ids(test_mention_sentences)
            test_max_men_length = max(len(x) for x in self.test_mention_indexes)

            test_mention_characters = F.pad(test_mention_characters, (0, 0, 0, test_max_men_length - 1), "constant", 0)
            if not args.online:
                self.test_mention_representations, self.test_mention_mask, test_mention_max, self.test_mention_reduced_mask = \
                    self.elmo_representations(args, test_mention_characters, self.test_mention_indexes, elmo, mention=True)

                padded_test_mention_representations = torch.zeros((self.test_mention_representations.shape[0],
                                                                   self.mention_representations.shape[1],
                                                                   self.mention_representations.shape[2]))
                padded_test_mention_representations[:, :self.test_mention_representations.shape[1], :] = self.test_mention_representations
                self.mention_representations = torch.cat((self.mention_representations, padded_test_mention_representations), 0)
            else:
                _, self.test_mention_mask, _, self.test_mention_reduced_mask = \
                    self.elmo_representations(args, test_mention_characters, self.test_mention_indexes, elmo, mention=True)
                self.test_mention_representations = test_mention_characters
                length = max(len(x) for x in self.test_mention_indexes)
                self.test_mention_indexes = np.array([xi + [-1] * (length - len(xi)) for xi in self.test_mention_indexes])
                self.max_men_length = max(self.max_men_length, length)

        #self.mention_indexes = np.array(self.mention_indexes)

        self.log.info("Loaded elmo characters")

    def load_elmo_ont(self, comm_dict, args, test_comm_dict):
        self.only_annotated_concepts = args.only_annotated_concepts
        self.concepts_used = set()
        self.concepts_ids_used = set()

        self.test_data = test_comm_dict is not None

        with open(os.path.join(args.mention_embeddings, 'mention_representations.npy'),
                  'rb') as mention_representations_npy, \
                open(os.path.join(args.mention_embeddings, 'mention_to_info.pkl'), 'rb') as mention_to_info_pkl, \
                open(os.path.join(args.mention_embeddings, 'id_to_mention_info.pkl'), 'rb') as id_to_mention_info_pkl:

            self.mention_representations = np.load(mention_representations_npy)
            self.id_to_mention_info = pickle.load(id_to_mention_info_pkl)
            self.mention_to_info = pickle.load(mention_to_info_pkl)

        w2v = KeyedVectors.load_word2vec_format(args.ont_w2v_filename)

        id_mapping = [x.split() for x in open(args.ont_id_mapping).readlines()]
        id_mapping_dict = {x[0]: x[1] for x in id_mapping}
        id_mapping_dict_rev = {x[1]: x[0] for x in id_mapping}

        name_map = [x.split() for x in open(args.ont_name_mapping, encoding="ISO-8859-1").readlines()]
        name_dict = {x[0]: " ".join(x[1:]) for x in name_map}

        lexicon_cui_set = set()
        with open(args.lexicon, encoding='utf-8') as lex_file:
            tsv_reader = csv.reader(lex_file, delimiter="\t")
            index = 0
            for row in tsv_reader:
                lexicon_cui_set.add(row[1])

        self.concept_representations = []
        self.cui_to_concept_info = {}
        self.id_to_concept_info = {}
        zero_excluded = set()
        for their_id in w2v.vocab.keys():
            cui = id_mapping_dict[their_id]
            if cui in lexicon_cui_set:
                our_id = len(self.concept_representations)
                vector = w2v[their_id]
                if vector.any():
                    self.concept_representations.append(vector)
                    name = name_dict[their_id]
                    concept_map = {"name": name,
                                   "concept_id": cui,
                                   "alternate": False,
                                   "index": our_id
                                   }
                    self.cui_to_concept_info[cui] = [concept_map]
                    self.id_to_concept_info[our_id] = concept_map
                else:
                    zero_excluded.add(cui)
        print("Excluding {0} all zero vectors:\n{1}".format(len(zero_excluded), zero_excluded))
        self.concept_representations = np.stack(self.concept_representations)
        rs = RobustScaler(quantile_range=(0.05, 0.95))
        self.concept_representations = rs.fit_transform(self.concept_representations)
        from scipy.stats import describe
        print(describe(self.concept_representations, axis=None))

        if self.test_data:
            combined_dict = {**comm_dict, **test_comm_dict}
        else:
            combined_dict = comm_dict

        if args.include_cuiless:
            self.cuiless_entries = set("CUI-less")
            cuiless_concept = np.random.random_sample(size=(1, self.concept_representations.shape[1]))
            self.concept_representations = np.vstack([self.concept_representations, cuiless_concept])
            c_map = {"name": "CUI-less",
                               "concept_id": "CUI-less",
                               "alternate": False,
                               "index": self.concept_representations.shape[0]-1
                               }
            self.id_to_concept_info[self.concept_representations.shape[0]-1] = c_map
            self.cui_to_concept_info["CUI-less"] = [c_map]

        else:
            self.cuiless_entries = set()
        self.num_examples = sum(1 for _, m in mention_iterator(comm_dict) if m.entityType in self.cui_to_concept_info)
        self.mention_ids = np.zeros(self.num_examples)
        self.concept_ids = np.zeros(self.num_examples)
        for i, (c_fn, m) in enumerate(mention_iterator(combined_dict)):
            if m.entityType in self.cui_to_concept_info:
                self.concepts_used.add(m.entityType)
                self.concepts_ids_used.update((x['index'], x['alternate'])
                                              for x in self.cui_to_concept_info[m.entityType])

        if self.only_annotated_concepts:
            self.log.info("Excluding concepts not annotated in dataset")
            new_concept_representations = np.zeros(shape=(len(self.concepts_ids_used),
                                                          self.concept_representations.shape[1]))
            new_id_to_concept_info = {}
            for new_id, (old_id, alt) in enumerate(self.concepts_ids_used):
                new_concept_representations[new_id] = self.concept_representations[old_id]
                new_id_to_concept_info[new_id] = self.id_to_concept_info[old_id]
                if not alt:
                    new_id_to_concept_info[new_id]["index"] = new_id
                cui = new_id_to_concept_info[new_id]["concept_id"]
                for i in range(len(self.cui_to_concept_info[cui])):
                    if self.cui_to_concept_info[cui][i]["index"] == old_id:
                        self.cui_to_concept_info[cui][i]["index"] = new_id

            self.concept_representations = new_concept_representations
            self.id_to_concept_info = new_id_to_concept_info




        indx = 0
        excluded_count = 0
        excluded_types = set()

        for i, (c_fn, m) in enumerate(mention_iterator(comm_dict)):
            if m.entityType in self.cui_to_concept_info:
                self.mention_ids[indx] = self.mention_to_info[m.uuid.uuidString]['index']
                self.concept_ids[indx] = self.cui_to_concept_info[m.entityType][0]['index']
                indx += 1
            else:
                excluded_count += 1
                excluded_types.add(m.entityType)

        self.log.info("Excluded {0} mentions with entity types {1}".format(excluded_count, ",".join(excluded_types)))

        # test stuff
        if self.test_data:
            self.test_num_examples = sum(1 for _, m in mention_iterator(test_comm_dict)
                                         if m.entityType in self.cui_to_concept_info)

            self.test_mention_ids = np.zeros(self.test_num_examples)
            self.test_concept_ids = np.zeros(self.test_num_examples)
            indx = 0
            for i, (c_fn, m) in enumerate(mention_iterator(test_comm_dict)):
                if m.entityType in self.cui_to_concept_info:
                    self.test_mention_ids[indx] = self.mention_to_info[m.uuid.uuidString]['index']
                    self.test_concept_ids[indx] = self.cui_to_concept_info[m.entityType][0]['index']
                    indx += 1
            self.test_num_mentions = self.test_mention_ids.shape[0]

        self.num_mentions = self.mention_ids.shape[0]
        self.num_concepts = self.concept_representations.shape[0]

    def load_elmo_old(self, comm_dict, args, test_comm_dict):
        #This still uses concrete
        self.only_annotated_concepts = args.only_annotated_concepts
        self.concepts_used = set()
        self.concepts_ids_used = set()

        self.test_data = test_comm_dict is not None

        with open(os.path.join(args.mention_embeddings, 'mention_representations.npy'),
                  'rb') as mention_representations_npy, \
                open(os.path.join(args.mention_embeddings, 'mention_to_info.pkl'), 'rb') as mention_to_info_pkl, \
                open(os.path.join(args.mention_embeddings, 'id_to_mention_info.pkl'), 'rb') as id_to_mention_info_pkl:

            self.mention_representations = np.load(mention_representations_npy)
            self.id_to_mention_info = pickle.load(id_to_mention_info_pkl)
            self.mention_to_info = pickle.load(mention_to_info_pkl)


        with open(os.path.join(args.concept_embeddings, 'concept_representations.npy'),
                  'rb') as concept_representations_npy, \
                open(os.path.join(args.concept_embeddings, 'id_to_concept_name_alt.pkl'),
                     'rb') as id_to_concept_name_alt_pkl, \
                open(os.path.join(args.concept_embeddings, 'concept_to_id_name_alt.pkl'),
                     'rb') as concept_to_id_name_alt_pkl:
            self.concept_representations = np.load(concept_representations_npy)
            self.id_to_concept_info = pickle.load(id_to_concept_name_alt_pkl)
            self.cui_to_concept_info = pickle.load(concept_to_id_name_alt_pkl)

        if self.test_data:
            combined_dict = {**comm_dict, **test_comm_dict}
        else:
            combined_dict = comm_dict

        if args.include_cuiless:
            self.cuiless_entries = set("CUI-less")
            cuiless_concept = np.random.random_sample(size=(1, self.concept_representations.shape[1]))
            self.concept_representations = np.vstack([self.concept_representations, cuiless_concept])
            c_map = {"name": "CUI-less",
                               "concept_id": "CUI-less",
                               "alternate": False,
                               "index": self.concept_representations.shape[0]-1
                               }
            self.id_to_concept_info[self.concept_representations.shape[0]-1] = c_map
            self.cui_to_concept_info["CUI-less"] = [c_map]

        else:
            self.cuiless_entries = set()


        self.num_examples = sum(1 for _, m in mention_iterator(comm_dict) if m.entityType in self.cui_to_concept_info)
        self.mention_ids = np.zeros(self.num_examples)
        self.concept_ids = np.zeros(self.num_examples)
        for i, (c_fn, m) in enumerate(mention_iterator(combined_dict)):
            if m.entityType in self.cui_to_concept_info:
                self.concepts_used.add(m.entityType)
                self.concepts_ids_used.update((x['index'], x['alternate'])
                                              for x in self.cui_to_concept_info[m.entityType])

        if self.only_annotated_concepts:
            self.log.info("Excluding concepts not annotated in dataset")
            new_concept_representations = np.zeros(shape=(len(self.concepts_ids_used),
                                                          self.concept_representations.shape[1]))
            new_id_to_concept_info = {}
            for new_id, (old_id, alt) in enumerate(self.concepts_ids_used):
                new_concept_representations[new_id] = self.concept_representations[old_id]
                new_id_to_concept_info[new_id] = self.id_to_concept_info[old_id]
                if not alt:
                    new_id_to_concept_info[new_id]["index"] = new_id
                cui = new_id_to_concept_info[new_id]["concept_id"]
                for i in range(len(self.cui_to_concept_info[cui])):
                    if self.cui_to_concept_info[cui][i]["index"] == old_id:
                        self.cui_to_concept_info[cui][i]["index"] = new_id

            self.concept_representations = new_concept_representations
            self.id_to_concept_info = new_id_to_concept_info




        indx = 0
        excluded_count = 0
        excluded_types = set()

        for i, (c_fn, m) in enumerate(mention_iterator(comm_dict)):
            if m.entityType in self.cui_to_concept_info:
                self.mention_ids[indx] = self.mention_to_info[m.uuid.uuidString]['index']
                self.concept_ids[indx] = self.cui_to_concept_info[m.entityType][0]['index']
                indx += 1
            else:
                excluded_count += 1
                excluded_types.add(m.entityType)

        self.log.info("Excluded {0} mentions with entity types {1}".format(excluded_count, ",".join(excluded_types)))

        # test stuff
        if self.test_data:
            if args.test_include_cuiless:
                c_map = {"name": "CUI-less",
                         "concept_id": "CUI-less",
                         "alternate": False,
                         "index": self.concept_representations.shape[0]
                         }
                self.id_to_concept_info[self.concept_representations.shape[0]] = c_map
                self.cui_to_concept_info["CUI-less"] = [c_map]
                self.log.info("Adding cuiless")

            self.test_num_examples = sum(1 for _, m in mention_iterator(test_comm_dict)
                                         if m.entityType in self.cui_to_concept_info)

            self.test_mention_ids = np.zeros(self.test_num_examples)
            self.test_concept_ids = np.zeros(self.test_num_examples)
            indx = 0
            for i, (c_fn, m) in enumerate(mention_iterator(test_comm_dict)):
                if m.entityType in self.cui_to_concept_info:
                    self.test_mention_ids[indx] = self.mention_to_info[m.uuid.uuidString]['index']
                    self.test_concept_ids[indx] = self.cui_to_concept_info[m.entityType][0]['index']
                    indx += 1
            self.test_num_mentions = self.test_mention_ids.shape[0]

        self.num_mentions = self.mention_ids.shape[0]
        self.num_concepts = self.concept_representations.shape[0]

        if args.dnorm_feats != "":
            self.load_dnorm_features(args)
            self.dnorm_features = True

    def load_dnorm_features(self, args):
        with open(os.path.join(args.dnorm_feats, 'dnorm_scores.npz'), 'rb') as dnorm_npy:
            self.dnorm_scores = load_npz(dnorm_npy)
        with open(os.path.join(args.dnorm_feats, 'dnorm_top5.npz'), 'rb') as dnorm_npy:
            self.dnorm_top5 = load_npz(dnorm_npy)
        with open(os.path.join(args.dnorm_feats, 'dnorm_top1.npz'), 'rb') as dnorm_npy:
            self.dnorm_top1 =load_npz(dnorm_npy)

        pass
    def test_dnorm_index(self, i_men, j_con, test=False):
        if test:
            return (i_men * self.test_num_examples) + j_con
        else:
            return (i_men * self.num_examples) + j_con

    def get_next_id(self, dict, num, key):
        dict[key] = num
        num += 1
        return dict, num


    def __repr__(self):

        return ('<LinkedMentions dataset ({num_mentions} mentions x {num_concepts} concepts '
                'x {linked_mentions} LinkedMentions)>'
                .format(
                    num_mentions=self.num_mentions,
                    num_concepts=self.num_concepts,
                    linked_mentions=len(self)
                ))

    def __len__(self):

        return len(self.mention_ids)

