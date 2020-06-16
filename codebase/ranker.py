"""
Elliot Schumacher, Johns Hopkins University
Created 2/12/19

"""
from time import time
import torch.nn as nn
import torch
from collections import OrderedDict
from allennlp.modules.elmo import Elmo
import numpy as np
from codebase import torch_utils
import logging
from pytorch_pretrained_bert import BertTokenizer, BertModel
from codebase.ElmoAllLayers import ElmoAllLayers
from codebase.self_attention import StructuredSelfAttention

class ElmoAttentionRanker(nn.Module):


    default_arguments = {
        "num_hidden_layers": 4,
        "hidden_layer_size": "512",
        "dropout_prob": 0.2,
        "freeze_emb_concept": True,
        "freeze_emb_mention": True,
        "elmo_options_file": "",
        "elmo_weight_file": "",
        "finetune_elmo": False,
        "elmo_mix" : False,
        "weighted_only" : False,
        "att_heads" : 1,
        "att_dim" : 256,
        "use_att_reg": False,
        "att_reg_val" : 0.0001
    }

    def __init__(self,
                 args,
                 mention_embedding_layer,
                 concept_embedding_layer,
                 mention_links,
                 transform = nn.ReLU,
                 final_transform=nn.Tanh):
        """

        :arg args: ConfigArgParse object containing program arguments
        :arg mention_embedding_layer: If using cached embeddings, embeddings for mentions
        :arg concept_embedding_layer: If using cached embeddings, embeddings for concepts
        :arg mention_links: A MentionLinks dataset
        :arg transform: The transformation function for n-1 layers
        :arg final_transform: The transformation function for the nth layer
        """

        super(ElmoAttentionRanker, self).__init__()
        self.output_size = 1
        self.args = args
        input_size = mention_embedding_layer.shape[1] + concept_embedding_layer.shape[1]

        self.log = logging.getLogger()
        self.dnorm_features = False
        self.attention = False

        if not args.online:

            reshaped_mentions = mention_embedding_layer.view((mention_embedding_layer.shape[0],
                                                              mention_embedding_layer.shape[1]*
                                                              mention_embedding_layer.shape[2]))
            reshaped_concepts = concept_embedding_layer.view((concept_embedding_layer.shape[0],
                                                              concept_embedding_layer.shape[1]*
                                                              concept_embedding_layer.shape[2]))
            self.emb_size = mention_embedding_layer.shape[2]
            self.max_concept_seq = concept_embedding_layer.shape[1]
            self.max_mention_seq = mention_embedding_layer.shape[1]

            self.mention_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(reshaped_mentions),
                                                                   freeze=args.freeze_emb_mention)

            self.concept_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(reshaped_concepts),
                                                                   freeze=args.freeze_emb_concept)
            self.mention_attention = torch_utils.gpu(StructuredSelfAttention(lstm_hid_dim=mention_embedding_layer.shape[2],
                                                     d_a = args.att_dim,
                                                     r=args.att_heads,
                                                     max_len = mention_embedding_layer.shape[1]
                                                                             , use_gpu=args.use_cuda), args.use_cuda)
            self.concept_attention = torch_utils.gpu(StructuredSelfAttention(lstm_hid_dim=concept_embedding_layer.shape[2],
                                                     d_a = args.att_dim,
                                                     r=args.att_heads,
                                                     max_len =concept_embedding_layer.shape[1], use_gpu=args.use_cuda), args.use_cuda)
            self.attention = True
            input_size = mention_embedding_layer.shape[2] + concept_embedding_layer.shape[2]

        else:


            self.elmo = torch_utils.gpu(ElmoAllLayers(args.elmo_options_file, args.elmo_weight_file, 1,
                                                 requires_grad=args.finetune_elmo, learn_mix_parameters=args.elmo_mix, dropout=0.0),
                                   gpu=args.use_cuda)
            input_size = self.elmo._elmo_lstm._elmo_lstm.input_size * 4
            self.emb_size = self.elmo._elmo_lstm._elmo_lstm.input_size

            self.mention_attention = torch_utils.gpu(StructuredSelfAttention(lstm_hid_dim=self.emb_size * 2,
                                                     d_a = args.att_dim,
                                                     r=args.att_heads,
                                                     max_len = mention_links.max_men_length,  use_gpu=args.use_cuda), args.use_cuda)
            self.concept_attention = torch_utils.gpu(StructuredSelfAttention(lstm_hid_dim=self.emb_size * 2,
                                                     d_a = args.att_dim,
                                                     r=args.att_heads,
                                                     max_len =concept_embedding_layer.shape[1], use_gpu=args.use_cuda), args.use_cuda)
            self.attention = True


        self.transform = transform
        self.final_transform = final_transform
        self.num_hidden_layers = args.num_hidden_layers
        self.layers = OrderedDict()


        if "," in args.hidden_layer_size:
            self.hidden_layer_size = [int(x) for x in args.hidden_layer_size.split(",")]
            if len(self.hidden_layer_size) != self.num_hidden_layers + 1:
                raise Exception("Wrong hidden layer size specification")
        else:
            self.hidden_layer_size = [int(args.hidden_layer_size) for _ in range(self.num_hidden_layers + 1)]

        self.layers['linear_input'] = nn.Linear(input_size,self.hidden_layer_size[0])
        self.layers['transform_input'] = self.transform()
        if args.dropout_prob > 0.0:
            self.layers['dropout_input'] = nn.Dropout(args.dropout_prob)

        for i in range(0, args.num_hidden_layers):
            self.layers['linear_h{i}'.format(i=i)] = nn.Linear(self.hidden_layer_size[i],
                                                               self.hidden_layer_size[i+1])
            self.layers['transform_h{i}'.format(i=i)] = self.transform()
            if args.dropout_prob > 0.0:
                self.layers['dropout_h{i}'.format(i=i)] = nn.Dropout(args.dropout_prob)

        self.layers['linear_output'] = nn.Linear(self.hidden_layer_size[-1], self.output_size)
        self.layers['transform_output'] = self.final_transform()
        if args.attention:
            self.sequential_net = torch_utils.gpu(nn.Sequential(self.layers), args.use_cuda)

        else:
            self.sequential_net = nn.Sequential(self.layers)
        self.log.info("Sequential net:{0}".format(self.sequential_net))

        self.log.info("Module:{0}".format(self))


    def forward(self, ids, concept_ids=None, mention_indexes=None, mention_mask=None, concept_mask=None,
                cached_emb=False, emb_only=False, mention_att=None, concept_att=None, use_att_reg = False, mention_mask_reduced=None):
        """
        Runs a forward pass over the model.  This has several different usages depending on the situation, documented below.


        :return loss of model
        """

        if emb_only:
            #This only returns the underlying embedding of the bert or elmo model.  This is used for caching in prediction.

            mention_embeddings_all = self.elmo(ids)["elmo_representations"][0]
            """
            mention_embedding_selected = torch_utils.gpu(
                torch.zeros((mention_embeddings_all.shape[0], self.mention_attention.max_len,
                             mention_embeddings_all.shape[2])), self.args.use_cuda)

            #TODO : select individual words!
            
            for k in range(0, len(mention_embeddings_all)):
                mention_embedding_selected[k, :len(mention_indexes[k]), :] = mention_embeddings_all[k,
                                                                             mention_indexes[k], :]
            """
            mention_embedding_masked = mention_embeddings_all[
                mention_mask.byte().unsqueeze(2).expand(mention_mask.shape[0], mention_mask.shape[1],
                                                        mention_embeddings_all.shape[2])].view(
                mention_embeddings_all.shape[0], mention_mask_reduced.shape[1],
                mention_embeddings_all.shape[2])
            mention_embedding = self.mention_attention(mention_embedding_masked, mask=mention_mask_reduced)
            return mention_embedding

        elif self.args.online:
            if cached_emb:
                mention_embedding = ids.squeeze()#ids will contain mention embedding
                concept_chars = concept_ids

                concept_embedding_all = self.elmo(concept_chars[0,:,:].view(1, concept_chars.shape[1], -1))["elmo_representations"][0]

                concept_embedding = self.concept_attention(concept_embedding_all, mask=concept_mask[0,:].unsqueeze(0))
                concept_embedding = concept_embedding.expand(len(concept_chars), concept_embedding.shape[1])


            else:
                if concept_ids is None:
                    mention_chars = ids[0]
                    concept_chars = ids[1]
                    mention_indexes = ids[2]
                else:
                    mention_chars = torch_utils.gpu(ids, gpu=self.args.use_cuda)
                    concept_chars = torch_utils.gpu(concept_ids, gpu=self.args.use_cuda)
                    mention_indexes = mention_indexes
                concept_embeddings_all = self.elmo(concept_chars)["elmo_representations"][0]

                mention_embeddings_all = self.elmo(mention_chars)["elmo_representations"][0]
                # TODO : select individual words!
                # mention_embedding_selected = torch_utils.gpu(
                #      torch.zeros(
                #          (mention_embeddings_all.shape[0], self.mention_attention.max_len,
                #           mention_embeddings_all.shape[2])) , self.args.use_cuda)

                # mention_embeddings_all = mention_embeddings_all * mention_mask.unsqueeze(2).expand(mention_mask.shape[0], mention_mask.shape[1],
                #                                                           mention_embeddings_all.shape[2])
                # start = time()
                # for k in range(0, len(mention_embeddings_all)):
                #      indexes = mention_indexes[k][mention_indexes[k] >=0 ]
                #      mention_embedding_selected[k, :len(indexes), :] =\
                #          mention_embeddings_all[k,indexes, :]
                # self.log.info("select elapsed :{0}".format(time()-start))
                # start = time()
                mention_embedding_masked = mention_embeddings_all[
                    mention_mask.byte().unsqueeze(2).expand(mention_mask.shape[0], mention_mask.shape[1],
                                                            mention_embeddings_all.shape[2])].view(
                    mention_embeddings_all.shape[0], mention_mask_reduced.shape[1],
                    mention_embeddings_all.shape[2])
                # self.log.info("mask elapsed :{0}".format(time()-start))
                #
                # if not torch.allclose(mention_embedding_masked, mention_embedding_selected):
                #      self.log.error(mention_embedding_masked)
                #      self.log.error(mention_embedding_selected)
                #      assert(torch.allclose(mention_embedding_masked, mention_embedding_selected))
                if use_att_reg:
                    mention_embedding, mention_att_penalty = self.mention_attention(mention_embedding_masked, use_reg=True, mask=mention_mask_reduced)
                    concept_embedding, concept_att_penalty = self.concept_attention(concept_embeddings_all, use_reg=True, mask=concept_mask)
                else:
                    mention_embedding = self.mention_attention(mention_embedding_masked, mask=mention_mask_reduced)
                    concept_embedding = self.concept_attention(concept_embeddings_all, mask=concept_mask)

        else:
            if concept_ids is None:

                mention_embedding = self.mention_embeddings(ids[0])
                concept_embedding = self.concept_embeddings(ids[1])
                concept_ids = ids[1]
                ids = ids[0]
            else:
                mention_embedding = self.mention_embeddings(ids)
                concept_embedding = self.concept_embeddings(concept_ids)

            mention_embedding = torch_utils.gpu(mention_embedding, self.args.use_cuda)\
                .view(-1, self.max_mention_seq, self.emb_size)
            concept_embedding = torch_utils.gpu(concept_embedding, self.args.use_cuda)\
                .view(-1, self.max_concept_seq, self.emb_size)

            mention_embedding = self.mention_attention(mention_embedding)
            concept_embedding = self.concept_attention(concept_embedding)


        input_rep = torch.cat([mention_embedding, concept_embedding], 1)

        out = self.sequential_net(input_rep)
        if use_att_reg:
            return out, mention_att_penalty, concept_att_penalty
        else:
            return out


class NeuralRanker(nn.Module):
    """

    :arg args: ConfigArgParse object containing program arguments
    :arg mention_embedding_layer: If using cached embeddings, embeddings for mentions
    :arg concept_embedding_layer: If using cached embeddings, embeddings for concepts
    :arg mention_links: A MentionLinks dataset
    :arg transform: The transformation function for n-1 layers
    :arg final_transform: The transformation function for the nth layer
    """

    default_arguments = {
        "num_hidden_layers": 4,
        "hidden_layer_size": "512",
        "dropout_prob": 0.2,
        "freeze_emb_concept": True,
        "freeze_emb_mention": True,
        "elmo_options_file": "",
        "elmo_weight_file": "",
        "finetune_elmo": False,
        "elmo_mix" : "",
        "weighted_only" : False,
        "att_heads" : 1,
        "att_dim" : 256
    }

    def __init__(self,
                 args,
                 mention_embedding_layer,
                 concept_embedding_layer,
                 mention_links,
                 transform = nn.ReLU,
                 final_transform=nn.Tanh):

        super(NeuralRanker, self).__init__()
        self.output_size = 1
        self.args = args
        input_size = mention_embedding_layer.shape[1] + concept_embedding_layer.shape[1]

        self.log = logging.getLogger()
        self.dnorm_features = False
        self.attention = False

        if not args.online and args.attention:

            reshaped_mentions = mention_embedding_layer.view((mention_embedding_layer.shape[0],
                                                              mention_embedding_layer.shape[1]*
                                                              mention_embedding_layer.shape[2]))
            reshaped_concepts = concept_embedding_layer.view((concept_embedding_layer.shape[0],
                                                              concept_embedding_layer.shape[1]*
                                                              concept_embedding_layer.shape[2]))
            self.emb_size = mention_embedding_layer.shape[2]
            self.max_concept_seq = concept_embedding_layer.shape[1]
            self.max_mention_seq = mention_embedding_layer.shape[1]

            self.mention_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(reshaped_mentions),
                                                                   freeze=args.freeze_emb_mention)

            self.concept_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(reshaped_concepts),
                                                                   freeze=args.freeze_emb_concept)
            self.mention_attention = torch_utils.gpu(StructuredSelfAttention(lstm_hid_dim=mention_embedding_layer.shape[2],
                                                     d_a = args.att_dim,
                                                     r=args.att_heads,
                                                     max_len = mention_embedding_layer.shape[1]), args.use_cuda)
            self.concept_attention = torch_utils.gpu(StructuredSelfAttention(lstm_hid_dim=concept_embedding_layer.shape[2],
                                                     d_a = args.att_dim,
                                                     r=args.att_heads,
                                                     max_len =concept_embedding_layer.shape[1]), args.use_cuda)
            self.attention = True
            input_size = mention_embedding_layer.shape[2] + concept_embedding_layer.shape[2]

        elif not args.online and not args.attention:
            self.mention_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(mention_embedding_layer),
                                                                   freeze=args.freeze_emb_mention)

            self.concept_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(concept_embedding_layer),
                                                                   freeze=args.freeze_emb_concept)
        elif args.online:
            """
            mix_weights = None
            if len(args.elmo_mix.split(",")) == 3:
                mix_weights = [float(x) for x in args.elmo_mix.split(",")]
                self.log.info("Using scalar mixed weights {0}".format(mix_weights))

            self.elmo = torch_utils.gpu(Elmo(args.elmo_options_file, args.elmo_weight_file, 1,
                                             requires_grad=self.args.finetune_elmo, scalar_mix_parameters=mix_weights),
                                            gpu=self.args.use_cuda)
            """
            self.elmo = torch_utils.gpu(ElmoAllLayers(args.elmo_options_file, args.elmo_weight_file, 1,
                                                 requires_grad=args.finetune_elmo, learn_mix_parameters=args.elmo_mix, dropout=0.0),
                                   gpu=args.use_cuda)
            input_size = self.elmo._elmo_lstm._elmo_lstm.input_size * 4
            self.emb_size = self.elmo._elmo_lstm._elmo_lstm.input_size


        if mention_links.dnorm_features:
            self.log.info("Using dnorm features")

            self.dnorm_score_emb = nn.Embedding.from_pretrained(torch.FloatTensor(mention_links.dnorm_scores.toarray().astype(np.float32)),
                                                                   freeze=args.freeze_emb_mention)

            self.dnorm_top1_emb = nn.Embedding.from_pretrained(torch.FloatTensor(mention_links.dnorm_top1.toarray().astype(np.float32)),
                                                                   freeze=args.freeze_emb_mention)

            self.dnorm_top5_emb = nn.Embedding.from_pretrained(torch.FloatTensor(mention_links.dnorm_top5.toarray().astype(np.float32)),
                                                                   freeze=args.freeze_emb_mention)
            self.dnorm_features = True

            self.dnorm_num_examples = mention_embedding_layer.shape[0]

            input_size += 3

        self.transform = transform
        self.final_transform = final_transform
        self.num_hidden_layers = args.num_hidden_layers
        self.layers = OrderedDict()

        if not self.args.weighted_only:

            if "," in args.hidden_layer_size:
                self.hidden_layer_size = [int(x) for x in args.hidden_layer_size.split(",")]
                if len(self.hidden_layer_size) != self.num_hidden_layers + 1:
                    raise Exception("Wrong hidden layer size specification")
            else:
                self.hidden_layer_size = [int(args.hidden_layer_size) for _ in range(self.num_hidden_layers + 1)]

            self.layers['linear_input'] = nn.Linear(input_size,self.hidden_layer_size[0])
            self.layers['transform_input'] = self.transform()
            if args.dropout_prob > 0.0:
                self.layers['dropout_input'] = nn.Dropout(args.dropout_prob)

            for i in range(0, args.num_hidden_layers):
                self.layers['linear_h{i}'.format(i=i)] = nn.Linear(self.hidden_layer_size[i],
                                                                   self.hidden_layer_size[i+1])
                self.layers['transform_h{i}'.format(i=i)] = self.transform()
                if args.dropout_prob > 0.0:
                    self.layers['dropout_h{i}'.format(i=i)] = nn.Dropout(args.dropout_prob)

            self.layers['linear_output'] = nn.Linear(self.hidden_layer_size[-1], self.output_size)
            self.layers['transform_output'] = self.final_transform()
            if args.attention:
                self.sequential_net = torch_utils.gpu(nn.Sequential(self.layers), args.use_cuda)

            else:
                self.sequential_net = nn.Sequential(self.layers)
            self.log.info("Sequential net:{0}".format(self.sequential_net))

        else:
            self.weight1 = nn.Parameter(torch.Tensor(self.mention_embeddings.weight.size()[1],
                                                     self.concept_embeddings.weight.size()[1]))
            nn.init.xavier_uniform_(self.weight1)
        self.log.info("Module:{0}".format(self))

    def forward_elmo(self, char_ids):
        results =self.elmo(char_ids)
        return results["elmo_representations"][0]

    def forward(self, ids, concept_ids=None, mention_indexes=None, mention_mask=None, concept_mask=None,
                cached_emb=False, emb_only=False, mention_att=None, concept_att=None):
        """
        Runs a forward pass over the model.  This has several different usages depending on the situation, documented below.


        :return loss of model
        """

        if emb_only:
            #This only returns the underlying embedding of the bert or elmo model.  This is used for caching in prediction.
            results = self.elmo(ids)
            return results["elmo_representations"][0]

        else:
            if self.args.online:
                # If using an online model, this will run the model end-to-end.  However, this should only be used
                # during training, as it's very slow for prediction.

                if cached_emb:
                    if concept_mask is not None: #ids will contain mention embedding
                        mention_embedding = ids.squeeze()
                        concept_chars = concept_ids

                        if self.args.embedding == "elmo":
                            concept_embedding_all = self.elmo(concept_chars)["elmo_representations"][0]


                        if self.args.comb_op == "max":
                            masked_concept = (concept_embedding_all + concept_mask[0,:]
                                              .view(concept_embedding_all.size()[0], concept_embedding_all.size()[1], 1)
                                              .expand(concept_embedding_all.size()[0], concept_embedding_all.size()[1],
                                                      concept_embedding_all.size()[2]))

                            concept_embedding = masked_concept.max(dim=1)[0]
                            concept_embedding = concept_embedding.view(1, concept_embedding.shape[1]) \
                                .expand(len(mention_embedding), concept_embedding.shape[1])

                    else:
                        mention_embedding = ids.squeeze()
                        concept_embedding = concept_ids.squeeze()
                else:
                    if concept_ids is None:
                        mention_chars = ids[0]
                        concept_chars = ids[1]
                        mention_indexes = ids[2]
                    else:
                        mention_chars = torch_utils.gpu(ids, gpu=self.args.use_cuda)
                        concept_chars = torch_utils.gpu(concept_ids, gpu=self.args.use_cuda)
                        mention_indexes = mention_indexes
                    concept_embedding_all = self.elmo(concept_chars)["elmo_representations"][0]

                    mention_embeddings_all = self.elmo(mention_chars)["elmo_representations"][0]

                    if self.args.comb_op == "max":
                        masked_concept = (concept_embedding_all + concept_mask
                                       .view(concept_embedding_all.size()[0], concept_embedding_all.size()[1], 1)
                                       .expand(concept_embedding_all.size()[0], concept_embedding_all.size()[1],
                                               concept_embedding_all.size()[2]))

                        concept_embedding = masked_concept.max(dim=1)[0]

                        masked_mens = (mention_embeddings_all + mention_mask
                                       .view(mention_embeddings_all.size()[0], mention_embeddings_all.size()[1], 1)
                                       .expand(mention_embeddings_all.size()[0], mention_embeddings_all.size()[1],
                                               mention_embeddings_all.size()[2]))

                        mention_embedding = masked_mens.max(dim=1)[0]

            elif self.attention:
                # This is used when there are cached embeddings provided in a lookup table.
                if concept_ids is None:

                    mention_embedding = self.mention_embeddings(ids[0])
                    concept_embedding = self.concept_embeddings(ids[1])
                    concept_ids = ids[1]
                    ids = ids[0]
                else:
                    mention_embedding = self.mention_embeddings(ids)
                    concept_embedding = self.concept_embeddings(concept_ids)

                mention_embedding = torch_utils.gpu(mention_embedding, self.args.use_cuda)\
                    .view(-1, self.max_mention_seq, self.emb_size)
                concept_embedding = torch_utils.gpu(concept_embedding, self.args.use_cuda)\
                    .view(-1, self.max_concept_seq, self.emb_size)

                mention_embedding = self.mention_attention(mention_embedding)
                concept_embedding = self.concept_attention(concept_embedding)

            else:
                # This is used when there are cached embeddings provided in a lookup table.
                if concept_ids is None:

                    mention_embedding = self.mention_embeddings(ids[0])
                    concept_embedding = self.concept_embeddings(ids[1])
                    concept_ids = ids[1]
                    ids = ids[0]
                else:
                    mention_embedding = self.mention_embeddings(ids)
                    concept_embedding = self.concept_embeddings(concept_ids)

            if not self.args.weighted_only:

                input_rep = torch.cat([mention_embedding, concept_embedding], 1)

                if self.dnorm_features:
                    dnorm_ids = ids * self.dnorm_num_examples + concept_ids
                    dnorm_scores = self.dnorm_score_emb(dnorm_ids)
                    dnorm_top1 = self.dnorm_top1_emb(dnorm_ids)
                    dnorm_top5 = self.dnorm_top5_emb(dnorm_ids)
                    input_rep = torch.cat([input_rep, dnorm_scores, dnorm_top1, dnorm_top5], 1)

                out = self.sequential_net(input_rep)
            else:
                mult = (torch.mm(mention_embedding, self.weight1) * concept_embedding)
                out = mult.sum(dim=1)
            return out

class BertAttentionNeuralRanker(nn.Module):
    """

    :arg args: ConfigArgParse object containing program arguments
    :arg mention_embedding_layer: If using cached embeddings, embeddings for mentions
    :arg concept_embedding_layer: If using cached embeddings, embeddings for concepts
    :arg mention_links: A MentionLinks dataset
    :arg transform: The transformation function for n-1 layers
    :arg final_transform: The transformation function for the nth layer
    """

    default_arguments = {
        "num_hidden_layers": 4,
        "hidden_layer_size": "512",
        "dropout_prob": 0.2,
        "freeze_emb_concept": True,
        "freeze_emb_mention": True,
        "elmo_options_file": "",
        "elmo_weight_file": "",
        "finetune_elmo": False,
        "elmo_mix" : "",
        "weighted_only" : False
    }

    def __init__(self,
                 args,
                 mention_embedding_layer,
                 concept_embedding_layer,
                 mention_links,
                 transform = nn.ReLU,
                 final_transform=nn.Tanh):

        super(BertAttentionNeuralRanker, self).__init__()
        self.output_size = 1
        self.args = args
        input_size = mention_embedding_layer.shape[1] + concept_embedding_layer.shape[1]

        self.log = logging.getLogger()


        bert_model = BertModel.from_pretrained(args.bert_path)
        self.bert = torch_utils.gpu(bert_model, args.use_cuda)

        #self.bert.eval()

        input_size = 768 * 2
        self.emb_size = 768


        self.transform = transform
        self.final_transform = final_transform
        self.num_hidden_layers = args.num_hidden_layers
        self.layers = OrderedDict()


        self.mention_attention = torch_utils.gpu(StructuredSelfAttention(lstm_hid_dim=self.emb_size,
                                                                         d_a=args.att_dim,
                                                                         r=args.att_heads,
                                                                         max_len=mention_links.max_men_length,
                                                                         use_gpu=args.use_cuda), args.use_cuda)
        self.concept_attention = torch_utils.gpu(StructuredSelfAttention(lstm_hid_dim=self.emb_size,
                                                                         d_a=args.att_dim,
                                                                         r=args.att_heads,
                                                                         max_len=concept_embedding_layer.shape[1],
                                                                         use_gpu=args.use_cuda), args.use_cuda)
        self.attention = True


        if "," in args.hidden_layer_size:
            self.hidden_layer_size = [int(x) for x in args.hidden_layer_size.split(",")]
            if len(self.hidden_layer_size) != self.num_hidden_layers + 1:
                raise Exception("Wrong hidden layer size specification")
        else:
            self.hidden_layer_size = [int(args.hidden_layer_size) for _ in range(self.num_hidden_layers + 1)]

        self.layers['linear_input'] = nn.Linear(input_size,self.hidden_layer_size[0])
        self.layers['transform_input'] = self.transform()
        if args.dropout_prob > 0.0:
            self.layers['dropout_input'] = nn.Dropout(args.dropout_prob)

        for i in range(0, args.num_hidden_layers):
            self.layers['linear_h{i}'.format(i=i)] = nn.Linear(self.hidden_layer_size[i],
                                                               self.hidden_layer_size[i+1])
            self.layers['transform_h{i}'.format(i=i)] = self.transform()
            if args.dropout_prob > 0.0:
                self.layers['dropout_h{i}'.format(i=i)] = nn.Dropout(args.dropout_prob)

        self.layers['linear_output'] = nn.Linear(self.hidden_layer_size[-1], self.output_size)
        self.layers['transform_output'] = self.final_transform()

        self.sequential_net = nn.Sequential(self.layers)
        self.log.info("Sequential net:{0}".format(self.sequential_net))


        self.log.info("Module:{0}".format(self))


    def forward(self, ids, concept_ids=None, mention_indexes=None, mention_mask=None, concept_mask=None,
                cached_emb=False, emb_only=False, mention_att=None, concept_att=None, use_att_reg=False, mention_mask_reduced=None):
        """
        Runs a forward pass over the model.  This has several different usages depending on the situation, documented below.


        :return loss of model
        """

        if emb_only:
            #This only returns the underlying embedding of the bert or elmo model.  This is used for caching in prediction.
            #embedding_layers, embedding_pool = self.bert(ids)
            mention_embeddings_layers, mention_embedding_pool = self.bert(ids)
            mention_embeddings_all = mention_embeddings_layers[self.args.emb_layer]

            mention_embedding_masked = mention_embeddings_all[
                mention_mask.byte().unsqueeze(2).expand(mention_mask.shape[0], mention_mask.shape[1],
                                                        mention_embeddings_all.shape[2])].view(
                mention_embeddings_all.shape[0], mention_mask_reduced.shape[1],
                mention_embeddings_all.shape[2])
            # masked_mens = (mention_embeddings_all + mention_mask
            #                .view(mention_embeddings_all.size()[0], mention_embeddings_all.size()[1], 1)
            #                .expand(mention_embeddings_all.size()[0], mention_embeddings_all.size()[1],
            #                        mention_embeddings_all.size()[2]))
            mention_embedding = self.mention_attention(mention_embedding_masked, mask=mention_mask_reduced)

            return mention_embedding
        elif self.args.online and cached_emb:
            if concept_mask is not None: #ids will contain mention embedding
                mention_embedding = ids.squeeze()
                concept_chars = concept_ids


                concept_embedding_layers, concept_embedding_pool = self.bert(concept_chars[0,:].view(1,concept_chars.shape[1]))

                concept_embedding_all = concept_embedding_layers[self.args.emb_layer]
                                    #\.expand(concept_chars.shape[0],concept_embedding_layers[self.args.emb_layer].shape[1])

                concept_embedding = self.concept_attention(concept_embedding_all, mask=concept_mask[0, :].unsqueeze(0))
                concept_embedding = concept_embedding.expand(len(concept_chars), concept_embedding.shape[1])

            else:
                mention_embedding = ids.squeeze()
                concept_embedding = concept_ids.squeeze()

        elif self.args.online and not cached_emb:
            if concept_ids is None:
                mention_chars = ids[0]
                concept_chars = ids[1]
                mention_indexes = ids[2]
            else:
                mention_chars = torch_utils.gpu(ids, gpu=self.args.use_cuda)
                concept_chars = torch_utils.gpu(concept_ids, gpu=self.args.use_cuda)
                mention_indexes = mention_indexes

            concept_embedding_layers, concept_embedding_pool = self.bert(concept_chars)
            concept_embedding_all = concept_embedding_layers[self.args.emb_layer]


            mention_embeddings_layers, mention_embedding_pool = self.bert(mention_chars)
            mention_embeddings_all = mention_embeddings_layers[self.args.emb_layer]


            mention_embedding_masked = mention_embeddings_all[
                mention_mask.byte().unsqueeze(2).expand(mention_mask.shape[0], mention_mask.shape[1],
                                                        mention_embeddings_all.shape[2])].view(
                mention_embeddings_all.shape[0], mention_mask_reduced.shape[1],
                mention_embeddings_all.shape[2])

            if use_att_reg:
                mention_embedding, mention_att_penalty = self.mention_attention(mention_embedding_masked, use_reg=True,
                                                                                mask=mention_mask_reduced)
                concept_embedding, concept_att_penalty = self.concept_attention(concept_embedding_all, use_reg=True,
                                                                                mask=concept_mask)
            else:
                mention_embedding = self.mention_attention(mention_embedding_masked, mask=mention_mask_reduced)
                concept_embedding = self.concept_attention(concept_embedding_all, mask=concept_mask)

        else:
            # This is used when there are cached embeddings provided in a lookup table.
            if concept_ids is None:

                mention_embedding = self.mention_embeddings(ids[0])
                concept_embedding = self.concept_embeddings(ids[1])
                concept_ids = ids[1]
                ids = ids[0]
            else:
                mention_embedding = self.mention_embeddings(ids)
                concept_embedding = self.concept_embeddings(concept_ids)

        input_rep = torch.cat([mention_embedding, concept_embedding], 1)


        out = self.sequential_net(input_rep)

        return out


class BertNeuralRanker(nn.Module):
    """

    :arg args: ConfigArgParse object containing program arguments
    :arg mention_embedding_layer: If using cached embeddings, embeddings for mentions
    :arg concept_embedding_layer: If using cached embeddings, embeddings for concepts
    :arg mention_links: A MentionLinks dataset
    :arg transform: The transformation function for n-1 layers
    :arg final_transform: The transformation function for the nth layer
    """

    default_arguments = {
        "num_hidden_layers": 4,
        "hidden_layer_size": "512",
        "dropout_prob": 0.2,
        "freeze_emb_concept": True,
        "freeze_emb_mention": True,
        "elmo_options_file": "",
        "elmo_weight_file": "",
        "finetune_elmo": False,
        "elmo_mix" : "",
        "weighted_only" : False
    }

    def __init__(self,
                 args,
                 mention_embedding_layer,
                 concept_embedding_layer,
                 mention_links,
                 transform = nn.ReLU,
                 final_transform=nn.Tanh):

        super(BertNeuralRanker, self).__init__()
        self.output_size = 1
        self.args = args
        input_size = mention_embedding_layer.shape[1] + concept_embedding_layer.shape[1]

        self.log = logging.getLogger()
        self.dnorm_features = False

        if args.embedding == "bert" and not args.online:
            self.mention_embeddings = nn.Embedding.from_pretrained(mention_embedding_layer,freeze=args.freeze_emb_mention)

            self.concept_embeddings = nn.Embedding.from_pretrained(concept_embedding_layer,freeze=args.freeze_emb_concept)

        elif args.embedding == "bert" and args.online:
            bert_model = BertModel.from_pretrained(args.bert_path)
            self.bert = torch_utils.gpu(bert_model, args.use_cuda)

            #self.bert.eval()

            input_size = 768 * 2
            self.emb_size = 768


        self.transform = transform
        self.final_transform = final_transform
        self.num_hidden_layers = args.num_hidden_layers
        self.layers = OrderedDict()

        if not self.args.weighted_only:

            if "," in args.hidden_layer_size:
                self.hidden_layer_size = [int(x) for x in args.hidden_layer_size.split(",")]
                if len(self.hidden_layer_size) != self.num_hidden_layers + 1:
                    raise Exception("Wrong hidden layer size specification")
            else:
                self.hidden_layer_size = [int(args.hidden_layer_size) for _ in range(self.num_hidden_layers + 1)]

            self.layers['linear_input'] = nn.Linear(input_size,self.hidden_layer_size[0])
            self.layers['transform_input'] = self.transform()
            if args.dropout_prob > 0.0:
                self.layers['dropout_input'] = nn.Dropout(args.dropout_prob)

            for i in range(0, args.num_hidden_layers):
                self.layers['linear_h{i}'.format(i=i)] = nn.Linear(self.hidden_layer_size[i],
                                                                   self.hidden_layer_size[i+1])
                self.layers['transform_h{i}'.format(i=i)] = self.transform()
                if args.dropout_prob > 0.0:
                    self.layers['dropout_h{i}'.format(i=i)] = nn.Dropout(args.dropout_prob)

            self.layers['linear_output'] = nn.Linear(self.hidden_layer_size[-1], self.output_size)
            self.layers['transform_output'] = self.final_transform()

            self.sequential_net = nn.Sequential(self.layers)
            self.log.info("Sequential net:{0}".format(self.sequential_net))

        else:
            self.weight1 = nn.Parameter(torch.Tensor(self.mention_embeddings.weight.size()[1],
                                                     self.concept_embeddings.weight.size()[1]))
            nn.init.xavier_uniform_(self.weight1)
        self.log.info("Module:{0}".format(self))


    def forward(self, ids, concept_ids=None, mention_indexes=None, mention_mask=None, concept_mask=None,
                cached_emb=False, emb_only=False, mention_att=None, concept_att=None):
        """
        Runs a forward pass over the model.  This has several different usages depending on the situation, documented below.


        :return loss of model
        """

        if emb_only:
            #This only returns the underlying embedding of the bert or elmo model.  This is used for caching in prediction.
            embedding_layers, embedding_pool = self.bert(ids)
            return embedding_pool
        elif self.args.online and cached_emb:
            if concept_mask is not None: #ids will contain mention embedding
                mention_embedding = ids.squeeze()
                concept_chars = concept_ids


                concept_embedding_layers, concept_embedding_pool = self.bert(concept_chars[0,:].view(1,concept_chars.shape[1])
                                                                             ,attention_mask=concept_att[0,:].view(1,concept_chars.shape[1]))


                if self.args.comb_op == "max":
                    concept_embedding_all = concept_embedding_layers[self.args.emb_layer]\
                        .expand(concept_chars.shape[0],concept_embedding_layers.shape[1])

                    masked_concept = (concept_embedding_all + concept_mask[0,:]
                                      .view(concept_embedding_all.size()[0], concept_embedding_all.size()[1], 1)
                                      .expand(concept_embedding_all.size()[0], concept_embedding_all.size()[1],
                                              concept_embedding_all.size()[2]))

                    concept_embedding = masked_concept.max(dim=1)[0]
                    concept_embedding = concept_embedding.view(1, concept_embedding.shape[1]) \
                        .expand(len(mention_embedding), concept_embedding.shape[1])
                elif self.args.comb_op == "cls" and self.args.embedding == "bert":
                    concept_embedding = concept_embedding_pool\
                        .expand(concept_chars.shape[0],concept_embedding_pool.shape[1])
            else:
                mention_embedding = ids.squeeze()
                concept_embedding = concept_ids.squeeze()

        elif self.args.online and not cached_emb:
            if concept_ids is None:
                mention_chars = ids[0]
                concept_chars = ids[1]
                mention_indexes = ids[2]
            else:
                mention_chars = torch_utils.gpu(ids, gpu=self.args.use_cuda)
                concept_chars = torch_utils.gpu(concept_ids, gpu=self.args.use_cuda)
                mention_indexes = mention_indexes

            concept_embedding_layers, concept_embedding_pool = self.bert(concept_chars, attention_mask=concept_att)
            concept_embedding_all = concept_embedding_layers[self.args.emb_layer]


            mention_embeddings_layers, mention_embedding_pool = self.bert(mention_chars, attention_mask=mention_att)
            mention_embeddings_all = mention_embeddings_layers[self.args.emb_layer]

            if self.args.comb_op == "max":
                masked_concept = (concept_embedding_all + concept_mask
                               .view(concept_embedding_all.size()[0], concept_embedding_all.size()[1], 1)
                               .expand(concept_embedding_all.size()[0], concept_embedding_all.size()[1],
                                       concept_embedding_all.size()[2]))

                concept_embedding = masked_concept.max(dim=1)[0]

                masked_mens = (mention_embeddings_all + mention_mask
                               .view(mention_embeddings_all.size()[0], mention_embeddings_all.size()[1], 1)
                               .expand(mention_embeddings_all.size()[0], mention_embeddings_all.size()[1],
                                       mention_embeddings_all.size()[2]))

                mention_embedding = masked_mens.max(dim=1)[0]
            elif self.args.comb_op == "cls" and self.args.embedding == "bert":
                concept_embedding = concept_embedding_pool
                mention_embedding = mention_embedding_pool
        else:
            # This is used when there are cached embeddings provided in a lookup table.
            if concept_ids is None:

                mention_embedding = self.mention_embeddings(ids[0])
                concept_embedding = self.concept_embeddings(ids[1])
                concept_ids = ids[1]
                ids = ids[0]
            else:
                mention_embedding = self.mention_embeddings(ids)
                concept_embedding = self.concept_embeddings(concept_ids)

        input_rep = torch.cat([mention_embedding, concept_embedding], 1)


        out = self.sequential_net(input_rep)

        return out
