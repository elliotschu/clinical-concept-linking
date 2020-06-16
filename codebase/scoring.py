"""
Elliot Schumacher, Johns Hopkins University
Created 2/1/19
"""
from codebase import losses
import numpy as np
import torch
from codebase import ranker
import torch.optim as optim
from codebase import torch_utils
from tensorboardX import SummaryWriter
import os
import gzip
from time import time
import shutil
import logging
from multiprocessing import Process
from codebase import evaluation
from codebase import sheets
import pandas
import pprint
from collections import OrderedDict


def zip_models(models_to_zip, new_zip):
    log = logging.getLogger()
    try:
        start_time = time()
        with gzip.open(new_zip, 'wb') as f_out:
            for path in models_to_zip:
                with open(path, 'rb') as f_in:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(path)
        log.info("Saved models {models} as {zip}, elapsed time {time}".format(models=",".join(models_to_zip),
                                                                              zip=new_zip,
                                                                              time=time() - start_time))
    except:
        log.info("Failed to zip {file}".format(file=models_to_zip))


class PairwiseRankingModel(object):
    """
    This class allows for training and evaluating the ranker using a MentionLinks dataset.

    :arg args: the ConfigArgParse object containing program arguments
    :arg mention_links: MentionLinks object containing the dataset.

    Some code borrowed from Spotlight codebase (see https://github.com/maciejkula/spotlight)
    """

    default_arguments = {
        "n_iter": 100,
        "batch_size": 256,
        "eval_batch_size": 4096,
        "learning_rate": 1e-4,
        "l2": 0.0,
        "num_negative_samples": 10,
        "use_cuda": False,
        "loss": "adaptive_hinge",
        "optimizer": "adam",
        "eval_every": 10,
        "save_every": 10,
        "comb_op" : "max",
        "weight_update_every" : 0,
        "eps_finetune" : 2,
        "save_if_better" : False,
        "save_eval_field": "accuracy",
    }

    def __init__(self,
                 args,
                 mention_links,
                 random_state=None):

        self._n_iter = args.n_iter
        self._learning_rate = args.learning_rate
        self._batch_size = args.batch_size
        self._l2 = args.l2
        self._use_cuda = args.use_cuda
        self._mention_representation = mention_links.mention_representations
        self._concept_representation = mention_links.concept_representations
        self.mention_links = mention_links
        self._optimizer_func = args.optimizer
        self._random_state = random_state or np.random.RandomState()
        self._num_negative_samples = args.num_negative_samples

        self.args = args

        self._num_concepts = None
        self._net = None
        self._optimizer = None

        if args.loss == "hinge":
            self._loss_func = losses.hinge_loss
        elif args.loss == "adaptive_hinge":
            self._loss_func = losses.adaptive_hinge_loss


        self.log = logging.getLogger()
        self.summary_writer = SummaryWriter(os.path.join(args.directory, 'tensorboard'))

        self.model_chkpt_dir = args.directory

        self.last_epoch = 0
        self.last_loss = 0
    def __repr__(self):

        if self._net is None:
            net_representation = '[uninitialised]'
        else:
            net_representation = repr(self._net)

        return ('<{}: {}>'
            .format(
            self.__class__.__name__,
            net_representation,
        ))

    @property
    def _initialized(self):
        return self._net is not None

    def _initialize(self, mention_links):
        """
        Initializes the pytorch model and optimizer

        :param mention_links: MentionLinks dataset
        """

        self._num_concepts = mention_links.num_concepts

        if self.args.embedding != "bert":
            if self.args.attention:
                if not self.args.online:
                    if torch.cuda.device_count() > 1:
                        self.log.info("Using {0} GPUs!".format(torch.cuda.device_count()))

                        self._net =torch.nn.DataParallel(
                                ranker.ElmoAttentionRanker(args=self.args,
                                                    concept_embedding_layer=self._concept_representation,
                                                    mention_embedding_layer=self._mention_representation,
                                                    mention_links=self.mention_links))
                    else:

                        self._net = ranker.ElmoAttentionRanker(args=self.args,
                                                concept_embedding_layer=self._concept_representation,
                                                mention_embedding_layer=self._mention_representation,
                                                mention_links=self.mention_links)
                else:
                    if torch.cuda.device_count() > 1:
                        self.log.info("Using {0} GPUs!".format(torch.cuda.device_count()))

                        self._net = torch_utils.gpu(
                            torch.nn.DataParallel(
                                ranker.ElmoAttentionRanker(args=self.args,
                                                    concept_embedding_layer=self._concept_representation,
                                                    mention_embedding_layer=self._mention_representation,
                                                    mention_links=self.mention_links)),
                            self._use_cuda
                        )
                    else:

                        self._net = torch_utils.gpu(
                            ranker.ElmoAttentionRanker(args=self.args,
                                                concept_embedding_layer=self._concept_representation,
                                                mention_embedding_layer=self._mention_representation,
                                                mention_links=self.mention_links),
                            self._use_cuda
                        )
            else:
                if torch.cuda.device_count() > 1:
                    self.log.info("Using {0} GPUs!".format(torch.cuda.device_count()))

                    self._net = torch_utils.gpu(
                        torch.nn.DataParallel(
                        ranker.NeuralRanker(args=self.args,
                                          concept_embedding_layer=self._concept_representation,
                                          mention_embedding_layer=self._mention_representation,
                                          mention_links=self.mention_links)),
                        self._use_cuda
                    )
                else:

                    self._net = torch_utils.gpu(
                        ranker.NeuralRanker(args=self.args,
                                          concept_embedding_layer=self._concept_representation,
                                          mention_embedding_layer=self._mention_representation,
                                          mention_links=self.mention_links),
                        self._use_cuda
                    )
        else:
            if torch.cuda.device_count() > 1 and not self.args.attention:
                self.log.info("Using {0} GPUs!".format(torch.cuda.device_count()))

                self._net = torch_utils.gpu(
                    torch.nn.DataParallel(
                        ranker.BertNeuralRanker(args=self.args,
                                            concept_embedding_layer=self._concept_representation,
                                            mention_embedding_layer=self._mention_representation,
                                            mention_links=self.mention_links)),
                    self._use_cuda
                )
            elif torch.cuda.device_count() > 1 :
                self.log.info("Using {0} GPUs!".format(torch.cuda.device_count()))

                self._net = torch_utils.gpu(
                    torch.nn.DataParallel(
                        ranker.BertAttentionNeuralRanker(args=self.args,
                                                concept_embedding_layer=self._concept_representation,
                                                mention_embedding_layer=self._mention_representation,
                                                mention_links=self.mention_links)),
                    self._use_cuda
                )
            elif self.args.attention:

                self._net = torch_utils.gpu(
                    ranker.BertAttentionNeuralRanker(args=self.args,
                                        concept_embedding_layer=self._concept_representation,
                                        mention_embedding_layer=self._mention_representation,
                                        mention_links=self.mention_links),
                    self._use_cuda
                )
            else:

                self._net = torch_utils.gpu(
                    ranker.BertNeuralRanker(args=self.args,
                                        concept_embedding_layer=self._concept_representation,
                                        mention_embedding_layer=self._mention_representation,
                                        mention_links=self.mention_links),
                    self._use_cuda
                )
        if self._optimizer_func == "adam":
            self._optimizer = optim.Adam(
                self._net.parameters(),
                weight_decay=self._l2,
                lr=self._learning_rate
            )
        #if self.args.model_path is not None:
        #    self.load_model(self.args.model_path)
        """
        dummy_mentions = torch_utils.gpu(
            torch.randint(size=(self.args.batch_size,), high=int(interactions.mention_ids.max().astype(np.int64))),
            self._use_cuda
        )
        dummy_concepts = torch_utils.gpu(
            torch.randint(size=(self.args.batch_size,), high=int(interactions.concept_ids.max().astype(np.int64))),
            self._use_cuda
        )
        self.summary_writer.add_graph(self._net, (dummy_mentions, dummy_concepts))"""

    def fit_online(self, mention_links):
        epoch_loss = 0.0

        if self._random_state is None:
            random_state = np.random.RandomState()
        shuffle_indices = np.arange(self.mention_links.mention_representations.shape[0])
        self._random_state.shuffle(shuffle_indices)
        if self.args.embedding == "elmo":
            mentions = self.mention_links.mention_representations[shuffle_indices, :, :]
        else:
            mentions = self.mention_links.mention_representations[shuffle_indices, :]

        concept_ids = self.mention_links.concept_ids[shuffle_indices]
        concepts = self.mention_links.concept_representations[concept_ids, :]
        concept_mask = torch_utils.gpu(mention_links.concept_mask[concept_ids, :], gpu=self._use_cuda)
        mention_indexes = self.mention_links.mention_indexes[shuffle_indices, :]
        mention_mask = torch_utils.gpu(self.mention_links.mention_mask[shuffle_indices, :], gpu=self._use_cuda)

        if self.args.attention:
            concept_att = torch_utils.gpu(mention_links.concept_att[concept_ids, :], gpu=self._use_cuda)
            mention_att = torch_utils.gpu(self.mention_links.mention_att[shuffle_indices, :], gpu=self._use_cuda)
        # mention_ids_tensor = torch_utils.gpu(mentions, self._use_cuda)
        # concept_ids_tensor = torch_utils.gpu(concepts, self._use_cuda)
        minibatch_num = 0
        for i in range(0, len(mentions), self._batch_size):
            if self.args.embedding == "elmo":

                batch_mention = mentions[i:i + self._batch_size, :, :]
                batch_concept = concepts[i:i + self._batch_size, :, :]
            else:

                batch_mention = mentions[i:i + self._batch_size, :]
                batch_concept = concepts[i:i + self._batch_size, :]
            batch_mention_indexes = mention_indexes[i:i + self._batch_size]
            """
            for con, men in zip(concepts, mentions):
                con_tok = [x for x in self.mention_links.tokenizer.convert_ids_to_tokens(con.numpy()) if x != '[PAD]']
                men_tok = [x for x in self.mention_links.tokenizer.convert_ids_to_tokens(men.numpy()) if x != '[PAD]']
                print("{0}\t{1}".format(con_tok, men_tok))
            """
            if self.args.attention:
                positive_prediction = self._net(ids=batch_mention,
                                            concept_ids=batch_concept,
                                            mention_indexes=batch_mention_indexes,
                                            mention_mask=mention_mask[i:i + self._batch_size, :],
                                            concept_mask=concept_mask[i:i + self._batch_size, :],
                                            mention_att=mention_att[i:i + self._batch_size, :],
                                            concept_att=concept_att[i:i + self._batch_size, :])
            else:
                positive_prediction = self._net(ids=batch_mention,
                                            concept_ids=batch_concept,
                                            mention_indexes=batch_mention_indexes,
                                            mention_mask=mention_mask[i:i + self._batch_size, :],
                                            concept_mask=concept_mask[i:i + self._batch_size, :])
            if self.args.embedding == "elmo":

                negative_prediction = self._get_multiple_negative_predictions_elmo_online(
                    batch_mention, mention_mask[i:i + self._batch_size, :], n=self._num_negative_samples)
            else:
                negative_prediction = self._get_multiple_negative_predictions_bert_online(
                    batch_mention,
                    mention_mask[i:i + self._batch_size, :],
                    mention_att[i:i + self._batch_size, :],
                    n=self._num_negative_samples)

            loss = self._loss_func(positive_prediction, negative_prediction)
            epoch_loss += float(loss.item())

            self._optimizer.zero_grad()

            loss.backward()
            self._optimizer.step()
            minibatch_num += 1


        return epoch_loss, minibatch_num

    def fit_cached(self, mention_links):
        epoch_loss = 0.0
        concept_ids = mention_links.concept_ids.astype(np.int64)

        mentions, concepts = torch_utils.shuffle(mention_links.mention_ids.astype(np.int64),
                                                 concept_ids,
                                                 random_state=self._random_state)
        if self.args.attention:
            mention_ids_tensor = torch.from_numpy(mentions)
            concept_ids_tensor = torch.from_numpy(concepts)
        else:
            mention_ids_tensor = torch_utils.gpu(torch.from_numpy(mentions),
                                                 self._use_cuda)
            concept_ids_tensor = torch_utils.gpu(torch.from_numpy(concepts),
                                                 self._use_cuda)

        for (minibatch_num,
             (batch_mention,
              batch_concept)) in enumerate(torch_utils.minibatch(mention_ids_tensor,
                                                                 concept_ids_tensor,
                                                                 batch_size=self._batch_size)):

            positive_prediction = self._net(ids=batch_mention, concept_ids=batch_concept)

            if self._loss_func == losses.adaptive_hinge_loss:
                negative_prediction = self._get_multiple_negative_predictions(
                    batch_mention, n=self._num_negative_samples)
            else:
                negative_prediction = self._get_negative_prediction(batch_mention)

            self._optimizer.zero_grad()

            loss = self._loss_func(positive_prediction, negative_prediction)
            epoch_loss += float(loss.item())

            loss.backward()
            self._optimizer.step()
        return epoch_loss, minibatch_num

    def fit_bert_attention(self, mention_links):
        epoch_loss = 0.0

        if self._random_state is None:
            random_state = np.random.RandomState()
        shuffle_indices = np.arange(self.mention_links.mention_representations.shape[0])
        self._random_state.shuffle(shuffle_indices)
        mentions = self.mention_links.mention_representations[shuffle_indices, :]


        concept_ids = self.mention_links.concept_ids[shuffle_indices]
        concepts = self.mention_links.concept_representations[concept_ids, :]
        concept_mask = torch_utils.gpu(mention_links.concept_mask[concept_ids, :], gpu=self._use_cuda)
        mention_indexes = torch_utils.gpu(torch.tensor(self.mention_links.mention_indexes[shuffle_indices, :]), gpu=self._use_cuda)
        mention_mask = torch_utils.gpu(self.mention_links.mention_mask[shuffle_indices, :], gpu=self._use_cuda)

        #concept_att = torch_utils.gpu(mention_links.concept_att[concept_ids, :], gpu=self._use_cuda)
        #mention_att = torch_utils.gpu(self.mention_links.mention_att[shuffle_indices, :], gpu=self._use_cuda)

        mention_mask_reduced = torch_utils.gpu(self.mention_links.mention_reduced_mask[shuffle_indices, :], gpu=self._use_cuda)


        minibatch_num = 0
        for i in range(0, len(mentions), self._batch_size):

            batch_mention = mentions[i:i + self._batch_size, :]
            batch_concept = concepts[i:i + self._batch_size, :]

            batch_mention_indexes = mention_indexes[i:i + self._batch_size]

            if self.args.use_att_reg:
                positive_prediction,mention_att_penalty, concept_att_penalty = self._net(ids=batch_mention,
                                                concept_ids=batch_concept,
                                                mention_indexes=batch_mention_indexes,
                                                mention_mask=mention_mask[i:i + self._batch_size, :],
                                                concept_mask=concept_mask[i:i + self._batch_size, :],
                                                mention_mask_reduced = mention_mask_reduced[i:i + self._batch_size, :],
                                                use_att_reg=True
                                                )

            else:
                positive_prediction = self._net(ids=batch_mention,
                                                concept_ids=batch_concept,
                                                mention_indexes=batch_mention_indexes,
                                                mention_mask=mention_mask[i:i + self._batch_size, :],
                                                concept_mask=concept_mask[i:i + self._batch_size, :],
                                                mention_mask_reduced = mention_mask_reduced[i:i + self._batch_size, :],
                                                )


            negative_prediction = self._get_multiple_negative_predictions_elmo_att(
                batch_mention,
                batch_mention_indexes,
                mention_mask[i:i + self._batch_size, :] ,
                mention_mask_reduced[i:i + self._batch_size, :],
                n=self._num_negative_samples)

            loss = self._loss_func(positive_prediction, negative_prediction)
            if self.args.use_att_reg:


                loss += torch.sum(self.args.att_reg_val * mention_att_penalty/batch_mention.shape[0])
                loss += torch.sum(self.args.att_reg_val * concept_att_penalty/batch_concept.shape[0])

            epoch_loss += float(loss.item())

            self._optimizer.zero_grad()

            loss.backward()
            self._optimizer.step()

            minibatch_num += 1

            #self.log.info("BREAK\n"*5)
            #break


        return epoch_loss, minibatch_num

    def fit_elmo_attention(self, mention_links):
        epoch_loss = 0.0

        if self._random_state is None:
            random_state = np.random.RandomState()
        shuffle_indices = np.arange(self.mention_links.mention_representations.shape[0])
        self._random_state.shuffle(shuffle_indices)
        mentions = self.mention_links.mention_representations[shuffle_indices, :, :]


        concept_ids = self.mention_links.concept_ids[shuffle_indices]
        concepts = self.mention_links.concept_representations[concept_ids, :]
        concept_mask = torch_utils.gpu(mention_links.concept_mask[concept_ids, :], gpu=self._use_cuda)
        mention_indexes = torch_utils.gpu(torch.tensor(self.mention_links.mention_indexes[shuffle_indices, :]), gpu=self._use_cuda)
        mention_mask = torch_utils.gpu(self.mention_links.mention_mask[shuffle_indices, :], gpu=self._use_cuda)

        #concept_att = torch_utils.gpu(mention_links.concept_att[concept_ids, :], gpu=self._use_cuda)
        #mention_att = torch_utils.gpu(self.mention_links.mention_att[shuffle_indices, :], gpu=self._use_cuda)

        mention_mask_reduced = torch_utils.gpu(self.mention_links.mention_reduced_mask[shuffle_indices, :], gpu=self._use_cuda)


        minibatch_num = 0
        for i in range(0, len(mentions), self._batch_size):

            batch_mention = mentions[i:i + self._batch_size, :, :]
            batch_concept = concepts[i:i + self._batch_size, :, :]

            batch_mention_indexes = mention_indexes[i:i + self._batch_size]

            if self.args.use_att_reg:
                positive_prediction,mention_att_penalty, concept_att_penalty = self._net(ids=batch_mention,
                                                concept_ids=batch_concept,
                                                mention_indexes=batch_mention_indexes,
                                                mention_mask=mention_mask[i:i + self._batch_size, :],
                                                concept_mask=concept_mask[i:i + self._batch_size, :],
                                                mention_mask_reduced = mention_mask_reduced[i:i + self._batch_size, :],
                                                use_att_reg=True
                                                )

            else:
                positive_prediction = self._net(ids=batch_mention,
                                                concept_ids=batch_concept,
                                                mention_indexes=batch_mention_indexes,
                                                mention_mask=mention_mask[i:i + self._batch_size, :],
                                                concept_mask=concept_mask[i:i + self._batch_size, :],
                                                mention_mask_reduced = mention_mask_reduced[i:i + self._batch_size, :],
                                                )


            negative_prediction = self._get_multiple_negative_predictions_elmo_att(
                batch_mention,
                batch_mention_indexes,
                mention_mask[i:i + self._batch_size, :] ,
                mention_mask_reduced[i:i + self._batch_size, :],
                n=self._num_negative_samples)

            loss = self._loss_func(positive_prediction, negative_prediction)
            if self.args.use_att_reg:


                loss += torch.sum(self.args.att_reg_val * mention_att_penalty/batch_mention.shape[0])
                loss += torch.sum(self.args.att_reg_val * concept_att_penalty/batch_concept.shape[0])

            epoch_loss += float(loss.item())

            self._optimizer.zero_grad()

            loss.backward()
            self._optimizer.step()

            minibatch_num += 1


        return epoch_loss, minibatch_num


    def fit(self, mention_links, test_dict=None):
        """
        Fit the model.

        :arg mention_links: dataset (a mention_links instance)
        :arg test_dict: test dataset will be evaluated every 'eval_every'
        """


        if not self._initialized:
            self._initialize(mention_links)

        models_to_zip = []
        zip_num = 0
        max_eval_metric = 0.
        eval_file = os.path.join(self.model_chkpt_dir, "results.csv")
        eval_first = True
        first_epoch = 0
        if self.last_epoch > 0:
            first_epoch = self.last_epoch + 1
            self.log.info("Starting at epoch {0}".format(first_epoch))

        for epoch_num in range(first_epoch, self._n_iter + first_epoch):

            self._net.train()
            #self.log.info("Elmo training:{0}".format(self._net.module.elmo.training))
            if self.args.embedding == "elmo" and self.args.attention:
                epoch_loss, minibatch_num = self.fit_elmo_attention(mention_links)
            elif self.args.embedding == "bert" and self.args.attention:
                epoch_loss, minibatch_num = self.fit_bert_attention(mention_links)

            elif (self.args.embedding == "elmo" or self.args.embedding == "bert") and self.args.online:
                epoch_loss, minibatch_num = self.fit_online(mention_links)

            else:
                epoch_loss, minibatch_num = self.fit_cached(mention_links)

            epoch_loss /= minibatch_num + 1

            self.log.info('Epoch {}: loss {}'.format(epoch_num, epoch_loss))

            self.last_epoch = epoch_num
            self.last_loss = epoch_loss


            # tensorboard stuff
            self.summary_writer.add_scalar('loss', epoch_loss, epoch_num)

            if self.args.embedding == "bert" and self.args.online and self.args.eps_finetune == (epoch_num+1):
                try:
                    for param in self._net.bert.parameters():
                        param.requires_grad = False
                except:
                    for param in self._net.module.bert.parameters():
                        param.requires_grad = False
                self.log.warning("Frozen bert weights at epoch {0}".format(epoch_num))


            if (self.args.embedding == "elmo") and self.args.online and self.args.eps_finetune == (epoch_num+1):
                try:
                    self.dfs_freeze(self._net.elmo._elmo_lstm)
                except:
                    self.dfs_freeze(self._net.module.elmo._elmo_lstm)

                self.log.warning("Frozen elmo weights at epoch {0}".format(epoch_num))

            for name, param in self._net.named_parameters():
                self.summary_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch_num)




            #evaluation
            current_metric = 0.
            if ((epoch_num + 1) % int(self.args.eval_every)) == 0 and test_dict is not None:
                with torch.no_grad():

                    self.log.info("Evaluating at {0}".format(epoch_num))
                    if self.args.attention and self.args.embedding == "elmo":
                        predictions = self.predict_elmo_att(mention_links)
                    elif self.args.online and self.args.embedding == "elmo":
                        predictions = self.predict_elmo_online(mention_links)
                    elif self.args.attention and self.args.embedding == "bert":
                        predictions = self.predict_bert_att(mention_links)
                    elif self.args.online and self.args.embedding == "bert":
                        predictions = self.predict_bert_online(mention_links)
                    else:
                        predictions = self.predict(mention_links)
                    output_path = os.path.join(self.model_chkpt_dir, "eval_{0}.csv".format(epoch_num))

                    scores = evaluation.score(mention_links, predictions, test_dict, outpath=output_path)
                    current_metric = scores[self.args.save_eval_field]
                    for key, val in scores.items():
                        self.summary_writer.add_scalar(key, val, epoch_num)
                    self.log.info("Epoch:{0}, scores:{1}".format(epoch_num, scores))


                    scores['epoch'] = epoch_num
                    scores['epoch_loss'] = epoch_loss

                    with open(eval_file, 'a') as eval_csv:
                        dataframe = pandas.DataFrame.from_dict([scores])
                        dataframe.to_csv(eval_csv, header=eval_first, index=False)
                        eval_first = False

                    run = {**vars(self.args), **scores}

                    sheets.update_run(run)

            # save checkpoint
            if ((epoch_num + 1) % int(self.args.save_every)) == 0:
                save_model = False
                if self.args.save_if_better:
                    if current_metric >= max_eval_metric:
                        max_eval_metric = current_metric
                        save_model = True
                    else:
                        self.log.info("Not saving model. Previous {0} :{1}, current:{2}"
                                      .format(self.args.save_eval_field, max_eval_metric, current_metric))
                else:
                    save_model = True

                if save_model:
                    filename = "checkpoint_{0}.tar".format(epoch_num)
                    if self.args.save_if_better:
                        filename = "checkpoint.tar"
                    model_path = os.path.join(self.model_chkpt_dir, filename)
                    torch.save({
                        'epoch': epoch_num,
                        'model_state_dict': self._net.state_dict(),
                        'optimizer_state_dict': self._optimizer.state_dict(),
                        'loss': epoch_loss,
                    },
                        model_path)
                    models_to_zip.append(model_path)
                    if len(models_to_zip) == 1:
                        archive_name = os.path.join(self.model_chkpt_dir, filename+".gz")
                        p = Process(target=zip_models, args=(models_to_zip, archive_name))
                        p.start()
                        zip_num += 1
                        models_to_zip = []

    def dfs_freeze(self, model):
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = False
            self.dfs_freeze(child)

    def eval_saved(self, mention_links, model_path, test_dict):
        if not self._initialized:
            self._initialize(mention_links)

        self.model_chkpt_dir = os.path.dirname(model_path)
        self.load_model(model_path)
        self.log.info("Evaluating model saved at {0}".format(model_path))
        if self.args.embedding == "elmo" and self.args.online:
            predictions = self.predict_elmo_online(mention_links)

        else:
            predictions = self.predict(mention_links)
        output_path = os.path.join(os.path.dirname(model_path), "eval.csv")

        scores = evaluation.score(mention_links, predictions, test_dict, outpath=output_path)
        pprinter = pprint.PrettyPrinter()
        self.log.info("Scores:\n{0}".format(pprinter.pformat(scores)))

    def compare_saved(self, mention_links, model_path, test_dict):
        if not self._initialized:
            self._initialize(mention_links)

        #self.load_model(model_path)
        #self.log.info("Evaluating model saved at {0}".format(model_path))
        self.compare(mention_links, test_dict)


    def _sample_concepts(self, num_concepts, shape, random_state=None):
        """
        Randomly sample a number of concepts.

        Parameters
        ----------

        num_concepts: int
            Total number of concepts from which we should sample:
            the maximum value of a sampled concept id will be smaller
            than this.
        shape: int or tuple of ints
            Shape of the sampled array.
        random_state: np.random.RandomState instance, optional
            Random state to use for sampling.

        Returns
        -------

        concepts: np.array of shape [shape]
            Sampled concept ids.
        """

        """if random_state is None:
            random_state = np.random.RandomState()

        concepts = random_state.randint(0, num_concepts, shape, dtype=np.int64)
        """
        concepts = torch.LongTensor(shape).random_(0, num_concepts)

        return concepts

    def _get_negative_prediction(self, mention_ids, concept_ids = None):

        negative_concepts = self._sample_concepts(
            self._num_concepts,
            len(mention_ids),
            random_state=self._random_state)
        if self.args.attention and not self.args.online:
            negative_var = negative_concepts#torch.from_numpy(negative_concepts)

        else:
            negative_var = torch_utils.gpu(negative_concepts, self._use_cuda)

        negative_prediction = self._net(ids=mention_ids, concept_ids=negative_var)

        return negative_prediction

    def _get_multiple_negative_predictions(self, mention_ids, n=5):

        batch_size = mention_ids.size(0)

        negative_prediction = self._get_negative_prediction(mention_ids
                                                            .view(batch_size, 1)
                                                            .expand(batch_size, n)
                                                            .reshape(batch_size * n))

        return negative_prediction.view(n, len(mention_ids))

    def _get_multiple_negative_predictions_elmo_online(self, mention_ids, batch_mention_mask, n=5):

        batch_size = mention_ids.size(0)
        if len(mention_ids.size()) == 3:
            reshaped_mentions = mention_ids\
                .view(1, mention_ids.size()[0], mention_ids.size()[1], mention_ids.size()[2])\
                .expand(n,mention_ids.size()[0], mention_ids.size()[1], mention_ids.size()[2])\
                .reshape(n*mention_ids.size()[0], mention_ids.size()[1], mention_ids.size()[2])
        else:
            reshaped_mentions = mention_ids\
                .view(1, mention_ids.size()[0], mention_ids.size()[1])\
                .expand(n,mention_ids.size()[0], mention_ids.size()[1])\
                .reshape(n*mention_ids.size()[0], mention_ids.size()[1])
        reshaped_mask = batch_mention_mask\
            .view(1, batch_mention_mask.shape[0], batch_mention_mask.shape[1])\
            .expand(n,batch_mention_mask.shape[0], batch_mention_mask.shape[1])\
            .reshape(n*batch_mention_mask.shape[0], batch_mention_mask.shape[1])

        negative_prediction = self._get_negative_prediction_elmo_online(reshaped_mentions, reshaped_mask)

        return negative_prediction.view(n, mention_ids.size()[0])

    def _get_multiple_negative_predictions_bert_online(self, mention_ids, batch_mention_mask, batch_mention_att, n=5):

        batch_size = mention_ids.size(0)
        if len(mention_ids.size()) == 3:
            reshaped_mentions = mention_ids\
                .view(1, mention_ids.size()[0], mention_ids.size()[1], mention_ids.size()[2])\
                .expand(n,mention_ids.size()[0], mention_ids.size()[1], mention_ids.size()[2])\
                .reshape(n*mention_ids.size()[0], mention_ids.size()[1], mention_ids.size()[2])
        else:
            reshaped_mentions = mention_ids\
                .view(1, mention_ids.size()[0], mention_ids.size()[1])\
                .expand(n,mention_ids.size()[0], mention_ids.size()[1])\
                .reshape(n*mention_ids.size()[0], mention_ids.size()[1])
        reshaped_mask = batch_mention_mask\
            .view(1, batch_mention_mask.shape[0], batch_mention_mask.shape[1])\
            .expand(n,batch_mention_mask.shape[0], batch_mention_mask.shape[1])\
            .reshape(n*batch_mention_mask.shape[0], batch_mention_mask.shape[1])

        reshaped_att = batch_mention_att\
            .view(1, batch_mention_att.shape[0], batch_mention_att.shape[1])\
            .expand(n,batch_mention_att.shape[0], batch_mention_att.shape[1])\
            .reshape(n*batch_mention_att.shape[0], batch_mention_att.shape[1])

        negative_prediction = self._get_negative_prediction_bert_online(reshaped_mentions, reshaped_mask, reshaped_att)

        return negative_prediction.view(n, mention_ids.size()[0])

    def _get_negative_prediction_elmo_online(self, mention_ids, batch_mention_mask):

        concepts = self._random_state.randint(0, self._num_concepts, len(mention_ids), dtype=np.int64)

        this_concept_mask = torch_utils.gpu(self.mention_links.concept_mask[concepts], gpu=self._use_cuda)

        negative_var = torch_utils.gpu(self.mention_links.concept_representations[concepts], gpu=self._use_cuda)

        negative_prediction = self._net(ids=mention_ids, concept_ids=negative_var,
                                        mention_mask=batch_mention_mask, concept_mask = this_concept_mask)

        return negative_prediction

    def _get_multiple_negative_predictions_elmo_att(self, mention_ids, batch_mention_index, batch_mention_mask, batch_mention_reduced_mask, n=5):

        batch_size = mention_ids.size(0)
        if len(mention_ids.size()) == 3:
            reshaped_mentions = mention_ids\
                .view(1, mention_ids.size()[0], mention_ids.size()[1], mention_ids.size()[2])\
                .expand(n,mention_ids.size()[0], mention_ids.size()[1], mention_ids.size()[2])\
                .reshape(n*mention_ids.size()[0], mention_ids.size()[1], mention_ids.size()[2])
        else:
            reshaped_mentions = mention_ids\
                .view(1, mention_ids.size()[0], mention_ids.size()[1])\
                .expand(n,mention_ids.size()[0], mention_ids.size()[1])\
                .reshape(n*mention_ids.size()[0], mention_ids.size()[1])
        reshaped_indexes = batch_mention_index\
            .view(1, batch_mention_index.shape[0], batch_mention_index.shape[1])\
            .expand(n,batch_mention_index.shape[0], batch_mention_index.shape[1])\
            .reshape(n*batch_mention_index.shape[0], batch_mention_index.shape[1])
        reshaped_masks = batch_mention_mask\
            .view(1, batch_mention_mask.shape[0], batch_mention_mask.shape[1])\
            .expand(n,batch_mention_mask.shape[0], batch_mention_mask.shape[1])\
            .reshape(n*batch_mention_mask.shape[0], batch_mention_mask.shape[1])
        reshaped_reduced_masks = batch_mention_reduced_mask\
            .view(1, batch_mention_reduced_mask.shape[0], batch_mention_reduced_mask.shape[1])\
            .expand(n,batch_mention_reduced_mask.shape[0], batch_mention_reduced_mask.shape[1])\
            .reshape(n*batch_mention_reduced_mask.shape[0], batch_mention_reduced_mask.shape[1])

        negative_prediction = self._get_negative_prediction_elmo_att(reshaped_mentions, reshaped_indexes, reshaped_masks, reshaped_reduced_masks)

        return negative_prediction.view(n, mention_ids.size()[0])

    def _get_negative_prediction_elmo_att(self, mention_ids, batch_mention_index, reshaped_masks, reshaped_reduced_masks):

        #concepts = self._random_state.randint(0, self._num_concepts, len(mention_ids), dtype=np.int64)
        concepts = torch.LongTensor(len(mention_ids)).random_(0, self._num_concepts)


        this_concept_mask = torch_utils.gpu(self.mention_links.concept_mask[concepts], gpu=self._use_cuda)

        negative_var = torch_utils.gpu(self.mention_links.concept_representations[concepts], gpu=self._use_cuda)

        negative_prediction = self._net(ids=mention_ids, concept_ids=negative_var,
                                        mention_indexes=batch_mention_index,
                                        concept_mask = this_concept_mask,
                                        mention_mask = reshaped_masks,
                                        mention_mask_reduced=reshaped_reduced_masks)

        return negative_prediction

    def _get_negative_prediction_bert_online(self, mention_ids, batch_mention_mask, batch_mention_att):

        concepts = self._random_state.randint(0, self._num_concepts, len(mention_ids), dtype=np.int64)

        this_concept_mask = torch_utils.gpu(self.mention_links.concept_mask[concepts], gpu=self._use_cuda)

        this_concept_att = torch_utils.gpu(self.mention_links.concept_att[concepts], gpu=self._use_cuda)

        negative_var = torch_utils.gpu(self.mention_links.concept_representations[concepts], gpu=self._use_cuda)



        negative_prediction = self._net(ids=mention_ids,
                                        concept_ids=negative_var,
                                        mention_mask=batch_mention_mask,
                                        concept_mask = this_concept_mask,
                                        concept_att = this_concept_att,
                                        mention_att = batch_mention_att)

        return negative_prediction

    def load_model(self, path: str):
        """
        Load a previously trained pytorch model
        :param path: A string containing the path name for the model
        """


        if "tar.gz" in path:
            new_path = path.replace("tar.gz", "tar")
            if not os.path.exists(new_path):
                with gzip.open(path, 'rb') as f_in:
                    with open(new_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            path = new_path

        device = 'cpu'
        if self.args.use_cuda:
            device = 'cuda'

        checkpoint = torch.load(path, map_location=device)
        ignore_states = ["mention_embeddings"]
        checkpoint_dict = checkpoint['model_state_dict']
        # Ignore mention_embeddings from previous runs
        if len(ignore_states) > 0:
            new_checkpoint_dict = self._net.state_dict()
            for k, v in checkpoint_dict.items():
                excluded = False
                for ig in ignore_states:
                    if ig in k:
                        excluded = True
                        print("Excluding {0}".format(k))
                if not excluded:
                    new_checkpoint_dict[k] = v
            checkpoint_dict = new_checkpoint_dict
        try:
            self._net.load_state_dict(checkpoint_dict)
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            # Handle loading a model to a CPU which was trained on GPU
            new_state_dict = OrderedDict()
            for k, v in checkpoint_dict.items():
                name = k.replace("module.", "")  # remove `module.`
                new_state_dict[name] = v
            self._net.load_state_dict(new_state_dict)
            op_new_state_dict = OrderedDict()
            for k, v in checkpoint['optimizer_state_dict'].items():
                name = k.replace("module.", "")  # remove `module.`
                op_new_state_dict[name] = v
            self._optimizer.load_state_dict(op_new_state_dict)
        self.last_epoch = checkpoint['epoch']
        self.last_loss = checkpoint['loss']

        self.log.info("Loaded model {path}.\nLoss:{loss}, epoch:{epoch}".format(path=path,
                                                                                epoch=self.last_epoch,
                                                                                loss=self.last_loss))

    def predict(self, mention_links):
        """
        The batch prediction function for cached embedding.  Alternatives are provided for BERT/ELMo in-model embeddings
        :param mention_links: A mention links objects (needs to contain test data)
        :return: a numpy matrix of size [num test mentions] x [num concepts], with each row representing the scores for
        a given mention
        """
        self._net.eval()
        if not self._initialized:
            self._initialize(mention_links)

        # self._check_input(mention_links, None, allow_concepts_none=True)

        mention_ids = mention_links.test_mention_ids.astype(np.int64)
        gold_concept_ids = mention_links.test_concept_ids.astype(np.int64)

        concept_ids = np.arange(self._num_concepts, dtype=np.int64)
        concept_ids = torch.from_numpy(concept_ids.reshape(-1, 1).astype(np.int64))
        if not self.args.attention:

            concept_ids = torch_utils.gpu(concept_ids, self.args.use_cuda).squeeze()

        mention_ids = torch.from_numpy(mention_ids.reshape(-1, 1).astype(np.int64))

        results = np.zeros((mention_ids.shape[0], self._num_concepts))
        for i in range(mention_ids.shape[0]):
            mention_var = mention_ids[i].expand(concept_ids.size())
            if not self.args.attention:
                mention_var = torch_utils.gpu(mention_var, self.args.use_cuda).squeeze()
            if self.args.embedding != "tfidf" and not self.args.attention:
                results[i, :] = torch_utils.cpu(self._net(mention_var, concept_ids).flatten()).detach().numpy()
            else:
                for j in range(0, len(mention_var), self.args.eval_batch_size):
                    upper = min(j + self.args.eval_batch_size, len(mention_var))
                    results[i, j:upper] = torch_utils.cpu(self._net(mention_var[j:upper], concept_ids[j:upper])
                                                          .flatten()).detach().numpy()

        return results

    def compare(self, mention_links, test_dict):

        """
        Compares online elmo representations to those cached from tensorflow

        :param mention_links: A mention_links dataset.  This function only uses test data
        :param test_dict:  Dictionary of test comms
        """
        import pickle
        from scipy.spatial.distance import cdist
        from scipy.stats import describe
        from sklearn.metrics.pairwise import cosine_similarity

        self._net.eval()
        if not self._initialized:
            self._initialize(mention_links)

        #self._check_input(mention_links, None, allow_concepts_none=True)

        with open(os.path.join(self.args.mention_embeddings, 'mention_representations.npy'),
                  'rb') as mention_representations_npy, \
                open(os.path.join(self.args.mention_embeddings, 'mention_to_info.pkl'), 'rb') as mention_to_info_pkl, \
                open(os.path.join(self.args.mention_embeddings, 'id_to_mention_info.pkl'), 'rb') as id_to_mention_info_pkl:

            c_mention_representations = np.load(mention_representations_npy)
            c_id_to_mention_info = pickle.load(id_to_mention_info_pkl)
            c_mention_to_info = pickle.load(mention_to_info_pkl)

        mention_ids = torch_utils.gpu(mention_links.test_mention_representations, self.args.use_cuda)
        gold_concept_ids = mention_links.test_concept_ids.astype(np.int64)


        #concept_ids = np.arange(self._num_concepts, dtype=np.int64)
        concept_ids = mention_links.concept_representations
        concept_ids = torch_utils.gpu(concept_ids, self.args.use_cuda).squeeze()
        men_indx = torch.tensor(self.mention_links.test_mention_indexes)

        #mention_ids = torch.from_numpy(mention_ids.reshape(-1, 1).astype(np.int64))
        test_mention_mask = torch_utils.gpu(self.mention_links.test_mention_mask, gpu=self._use_cuda)
        distances = []
        all_close = []
        for i in range (0, len(mention_ids)):
            men_uuid = mention_links.test_id_to_mention_info[i]["mention_uuid"]
            c_id = c_mention_to_info[men_uuid]["index"]
            comm = test_dict[c_mention_to_info[men_uuid]["comm_uuid"]]
            mention = [x for x in comm.entityMentionSetList[0].mentionList
                       if x.uuid.uuidString == men_uuid][0]
            mention_elmo = self._net(mention_ids[i,:, :].view(1, mention_ids.size()[1], mention_ids.size()[2]), emb_only=True)

            masked_mens = (mention_elmo + test_mention_mask[i,:]
                           .view(mention_elmo.size()[0], mention_elmo.size()[1], 1)
                           .expand(mention_elmo.size()[0], mention_elmo.size()[1],
                                   mention_elmo.size()[2]))

            self.log.info("men elmo:{0}".format(mention_elmo.size()))
            self.log.info("self.mention_links.test_mention_mask[i:upper,:] :{0}".format(self.mention_links.test_mention_mask[i,:].size()))
            self.log.info("masked_men :{0}".format(masked_mens.size()))
            if self.args.comb_op == "sum":
                men_divisor = torch_utils.gpu(
                    self.mention_links.test_mention_mask[i, :].sum(dim=1).type(torch.FloatTensor),
                    self.args.use_cuda)
                menemb =  masked_mens.sum(dim=1).t().div(men_divisor).t().detach().numpy()
            else:
                menemb =  masked_mens.max(dim=1)[0].detach().numpy()

            selected_men = mention_elmo[0, men_indx[i, :][men_indx[i, :] >= 0]]
            max_emb = torch.max(selected_men, dim=0, keepdim=True)[0]

            c_menemb = c_mention_representations[c_id, :]
            c_menemb = np.max(c_mention_representations[c_id, :][0][:len(np.nonzero(men_indx[i, :] >= 0))], axis=0)
            d = cosine_similarity(c_menemb.reshape(1, -1), menemb)
            all_close.append(np.allclose(menemb, max_emb.detach().numpy()))
            distances.append(d[0][0])
        self.log.info(describe(distances))
        self.log.info("All close?:{0}".format(describe(all_close)))

    def predict_elmo_online(self, mention_links):
        """
        The prediction module for online elmo.  To manage GPU memory, all mention representations are cached (with a
        batch size of eval_batch_size, and then as a batch scored against each concept (one by one).

        :param mention_links: A mention links objects (needs to contain test data)
        :return: a numpy matrix of size [num test mentions] x [num concepts], with each row representing the scores for
        a given mention

        """
        self._net.eval()
        if not self._initialized:
            self._initialize(mention_links)

        # self._check_input(mention_links, None, allow_concepts_none=True)
        with torch.no_grad():


            mention_ids = torch_utils.gpu(mention_links.test_mention_representations, self.args.use_cuda)
            gold_concept_ids = mention_links.test_concept_ids.astype(np.int64)

            # concept_ids = np.arange(self._num_concepts, dtype=np.int64)
            concept_ids = mention_links.concept_representations
            concept_ids = torch_utils.gpu(concept_ids, self.args.use_cuda).squeeze()
            men_indx = torch.tensor(self.mention_links.test_mention_indexes)

            # mention_ids = torch.from_numpy(mention_ids.reshape(-1, 1).astype(np.int64))

            results = np.zeros((mention_ids.shape[0], self._num_concepts))
            test_mention_mask = torch_utils.gpu(self.mention_links.test_mention_mask, gpu=self._use_cuda)

            mention_embedding_list = []
            mention_embedding = torch_utils.gpu(torch.zeros(mention_ids.shape[0], self._net.emb_size*2), gpu=self._use_cuda)
            for i in range(0, len(mention_ids), self.args.eval_batch_size):
                upper = min(i + self.args.eval_batch_size, len(mention_ids))
                if self.args.embedding == "elmo":
                    mention_elmo = self._net(mention_ids[i:upper, :, :], emb_only=True)
                else:
                    mention_elmo = self._net(mention_ids[i:upper, :], emb_only=True)

                masked_mens = (mention_elmo + test_mention_mask[i:upper, :]
                               .view(mention_elmo.size()[0], mention_elmo.size()[1], 1)
                               .expand(mention_elmo.size()[0], mention_elmo.size()[1],
                                       mention_elmo.size()[2]))
                # self.log.info("i={0}".format(i))
                # self.log.info("men elmo:{0}".format(mention_elmo.size()))
                # self.log.info("self.mention_links.test_mention_mask[i:upper,:] :{0}".format(self.mention_links.test_mention_mask[i:upper,:].size()))
                # self.log.info("masked_men :{0}".format(masked_mens.size()))
                intermed = masked_mens.max(dim=1)[0]
                mention_embedding[i:upper, :] = intermed

            # np.save(file='emlo_mentions', arr=mention_embedding.detach().numpy())
            start = time()

            for j in range(0, self._num_concepts):
                if self.args.embedding == "elmo":
                    concept_rep = torch_utils.gpu(
                        self._net(ids=concept_ids[j, :, :].view(1, concept_ids.shape[1], concept_ids.shape[2]),
                                  emb_only=True)
                        , self._use_cuda)
                else:
                    concept_rep = torch_utils.gpu(
                        self._net(ids=concept_ids[j, :].view(1, concept_ids.shape[1]), emb_only=True)
                        , self._use_cuda)
                # self.log.info("cr :{0}".format(concept_rep.size()))
                # self.log.info("ci:{0}".format(concept_ids[j,:]))
                # self.log.info("ci:{0}".format(concept_ids[j,:].size()))

                # concept_embedding = concept_rep.max(dim=1)[0]

                masked_concept = (concept_rep + torch_utils.gpu(mention_links.concept_mask[j, :], gpu=self._use_cuda)
                                  .view(concept_rep.size()[0], concept_rep.size()[1], 1)
                                  .expand(concept_rep.size()[0], concept_rep.size()[1],
                                          concept_rep.size()[2]))

                # self.log.info("cr 2:{0}".format(concept_rep.size()))

                concept_embedding = masked_concept.max(dim=1)[0]
                # self.log.info("ce:{0}".format(concept_embedding))
                # self.log.info("ce:{0}".format(concept_embedding.size()))

                concept_rep_view = concept_embedding.view(1, concept_embedding.shape[1]) \
                    .expand(len(mention_embedding), concept_embedding.shape[1])

                intermed = self._net(ids=mention_embedding,
                                     concept_ids=concept_rep_view,
                                     cached_emb=True).flatten()
                results[:, j] = torch_utils.cpu(intermed).detach().numpy().transpose()
            self.log.info("Eval time={0}".format(time() - start))

            return results

    def predict_bert_att(self, mention_links):
        """
        The prediction module for online bert.  To manage GPU memory, n mention representations (set by eval_batch_size)
        are cached, and then as a batch scored against each concept (one by one).

        :param mention_links: A mention links objects (needs to contain test data)
        :return: a numpy matrix of size [num test mentions] x [num concepts], with each row representing the scores for
        a given mention
        """
        self._net.eval()
        if not self._initialized:
            self._initialize(mention_links)

        #self._check_input(mention_links, None, allow_concepts_none=True)
        with torch.no_grad():#, torch.autograd.profiler.profile(use_cuda=self.args.use_cuda) as prof:

            mention_ids = torch_utils.gpu(mention_links.test_mention_representations, self.args.use_cuda)
            mention_indxs = torch_utils.gpu(torch.tensor(mention_links.test_mention_indexes), self.args.use_cuda)

            #gold_concept_ids = mention_links.test_concept_ids.astype(np.int64)

            results = np.zeros((mention_ids.shape[0], self._num_concepts))
            test_mention_mask = torch_utils.gpu(self.mention_links.test_mention_mask, gpu=self._use_cuda)
            test_mention_reduced_mask = torch_utils.gpu(self.mention_links.test_mention_reduced_mask, gpu=self._use_cuda)

            start = time()


            try:
                emb_size = self._net.emb_size
            except:
                emb_size = self._net.module.emb_size
            num_gpus = 1
            if torch.cuda.device_count() > 1:
                num_gpus = torch.cuda.device_count()

            mention_embedding = torch.zeros(len(mention_ids), emb_size) # this should remain on the cpu!
            for i in range(0, len(mention_ids), self.args.eval_batch_size):
                k = min(i + self.args.eval_batch_size, len(mention_ids))
                mention_elmo = self._net(ids=mention_ids[i:k,:],
                                         emb_only=True,
                                         mention_indexes=mention_indxs[i:k,:],
                                         mention_mask = test_mention_mask[i:k,:],
                                         mention_mask_reduced = test_mention_reduced_mask[i:k,:])
                mention_embedding[i:k,:] = torch_utils.cpu(mention_elmo).detach()

            self.log.info("Processed embeddings :{0}".format(time()-start))
            del test_mention_mask
            del mention_indxs
            del mention_ids

            gpu_mention_embeddings = torch_utils.gpu(mention_embedding, gpu=self._use_cuda)
            concept_ids = mention_links.concept_representations
            concept_ids = torch_utils.gpu(concept_ids, self.args.use_cuda).squeeze()
            #concept_att = torch_utils.gpu(self.mention_links.concept_att,  gpu=self._use_cuda)
            gpu_concept_mask = torch_utils.gpu(mention_links.concept_mask,  gpu=self._use_cuda)
            emb_start = time()
            for j in range(0, self._num_concepts):

                these_concept_ids = concept_ids[j, :].view(1, concept_ids.shape[1])\
                    .expand(gpu_mention_embeddings.shape[0], concept_ids.shape[1])
                these_concept_mask = gpu_concept_mask[j, :].view(1, mention_links.concept_mask.shape[1])\
                                                                  .expand(gpu_mention_embeddings.shape[0], mention_links.concept_mask.shape[1])
                """

                these_concept_att = concept_att[j, :].view(1, concept_att.shape[1])\
                                                                  .expand(gpu_mention_embeddings.shape[0], concept_att.shape[1])"""
                intermed = self._net(ids=gpu_mention_embeddings,
                                     concept_ids=these_concept_ids,
                                     concept_mask=these_concept_mask,
                                     cached_emb=True).flatten()
                results[:, j] = torch_utils.cpu(intermed).detach().numpy().transpose()

                if (j+1) % 10000 == 0:

                    #torch.cuda.synchronize()
                    self.log.info("Processed {0} concepts : {1}".format(j+1, time()-emb_start))

        #torch.cuda.synchronize()
        #self.log.info(prof)
        self.log.info("Total eval time={0}".format(time() - start))

        return results

    def predict_elmo_att(self, mention_links):
        """
        The prediction module for online bert.  To manage GPU memory, n mention representations (set by eval_batch_size)
        are cached, and then as a batch scored against each concept (one by one).

        :param mention_links: A mention links objects (needs to contain test data)
        :return: a numpy matrix of size [num test mentions] x [num concepts], with each row representing the scores for
        a given mention
        """
        self._net.eval()
        if not self._initialized:
            self._initialize(mention_links)

        #self._check_input(mention_links, None, allow_concepts_none=True)
        with torch.no_grad():#, torch.autograd.profiler.profile(use_cuda=self.args.use_cuda) as prof:

            mention_ids = torch_utils.gpu(mention_links.test_mention_representations, self.args.use_cuda)
            mention_indxs = torch_utils.gpu(torch.tensor(mention_links.test_mention_indexes), self.args.use_cuda)

            #gold_concept_ids = mention_links.test_concept_ids.astype(np.int64)

            results = np.zeros((mention_ids.shape[0], self._num_concepts))
            test_mention_mask = torch_utils.gpu(self.mention_links.test_mention_mask, gpu=self._use_cuda)
            test_mention_reduced_mask = torch_utils.gpu(self.mention_links.test_mention_reduced_mask, gpu=self._use_cuda)

            start = time()


            try:
                emb_size = self._net.emb_size
            except:
                emb_size = self._net.module.emb_size
            num_gpus = 1
            if torch.cuda.device_count() > 1:
                num_gpus = torch.cuda.device_count()

            mention_embedding = torch.zeros(len(mention_ids), emb_size*2) # this should remain on the cpu!
            for i in range(0, len(mention_ids), self.args.eval_batch_size):
                k = min(i + self.args.eval_batch_size, len(mention_ids))
                mention_elmo = self._net(ids=mention_ids[i:k,:],
                                         emb_only=True,
                                         mention_indexes=mention_indxs[i:k,:],
                                         mention_mask = test_mention_mask[i:k,:],
                                         mention_mask_reduced = test_mention_reduced_mask[i:k,:])
                mention_embedding[i:k,:] = torch_utils.cpu(mention_elmo).detach()
            self.log.info("Processed embeddings :{0}".format(time()-start))
            del test_mention_mask
            del mention_indxs
            del mention_ids

            gpu_mention_embeddings = torch_utils.gpu(mention_embedding, gpu=self._use_cuda)
            concept_ids = mention_links.concept_representations
            concept_ids = torch_utils.gpu(concept_ids, self.args.use_cuda).squeeze()
            #concept_att = torch_utils.gpu(self.mention_links.concept_att,  gpu=self._use_cuda)
            gpu_concept_mask = torch_utils.gpu(mention_links.concept_mask,  gpu=self._use_cuda)
            emb_start = time()
            for j in range(0, self._num_concepts):

                these_concept_ids = concept_ids[j, :].view(1, concept_ids.shape[1], concept_ids.shape[2])\
                    .expand(gpu_mention_embeddings.shape[0], concept_ids.shape[1], concept_ids.shape[2])
                these_concept_mask = gpu_concept_mask[j, :].view(1, mention_links.concept_mask.shape[1])\
                                                                  .expand(gpu_mention_embeddings.shape[0], mention_links.concept_mask.shape[1])
                """

                these_concept_att = concept_att[j, :].view(1, concept_att.shape[1])\
                                                                  .expand(gpu_mention_embeddings.shape[0], concept_att.shape[1])"""
                intermed = self._net(ids=gpu_mention_embeddings,
                                     concept_ids=these_concept_ids,
                                     concept_mask=these_concept_mask,
                                     cached_emb=True).flatten()
                results[:, j] = torch_utils.cpu(intermed).detach().numpy().transpose()

                if (j+1) % 10000 == 0:

                    #torch.cuda.synchronize()
                    self.log.info("Processed {0} concepts : {1}".format(j+1, time()-emb_start))

        #torch.cuda.synchronize()
        #self.log.info(prof)
        self.log.info("Total eval time={0}".format(time() - start))

        return results

    def predict_bert_online(self, mention_links):
        """
        The prediction module for online bert.  To manage GPU memory, n mention representations (set by eval_batch_size)
        are cached, and then as a batch scored against each concept (one by one).

        :param mention_links: A mention links objects (needs to contain test data)
        :return: a numpy matrix of size [num test mentions] x [num concepts], with each row representing the scores for
        a given mention
        """
        self._net.eval()
        if not self._initialized:
            self._initialize(mention_links)

        #self._check_input(mention_links, None, allow_concepts_none=True)
        with torch.no_grad():

            mention_ids = torch_utils.gpu(mention_links.test_mention_representations, self.args.use_cuda)
            mention_att = torch_utils.gpu(mention_links.test_mention_att, self.args.use_cuda)
            #gold_concept_ids = mention_links.test_concept_ids.astype(np.int64)


            men_indx = torch.tensor(self.mention_links.test_mention_indexes)


            results = np.zeros((mention_ids.shape[0], self._num_concepts))
            test_mention_mask = torch_utils.gpu(self.mention_links.test_mention_mask, gpu=self._use_cuda)
            start = time()


            try:
                emb_size = self._net.emb_size
            except:
                emb_size = self._net.module.emb_size
            num_gpus = 1
            if torch.cuda.device_count() > 1:
                num_gpus = torch.cuda.device_count()

            mention_embedding = torch.zeros(len(mention_ids), emb_size) # this should remain on the cpu!
            for i in range(0, len(mention_ids), self.args.eval_batch_size):
                k = min(i + self.args.eval_batch_size, len(mention_ids))
                mention_elmo = self._net(ids=mention_ids[i:k,:], emb_only=True, mention_att=mention_att[i:k,:])
                mention_embedding[i:k,:] = torch_utils.cpu(mention_elmo).detach()
            self.log.info("Processed embeddings :{0}".format(time()-start))
            del test_mention_mask
            del mention_att
            del mention_ids

            gpu_mention_embeddings = torch_utils.gpu(mention_embedding, gpu=self._use_cuda)
            concept_ids = mention_links.concept_representations
            concept_ids = torch_utils.gpu(concept_ids, self.args.use_cuda).squeeze()
            concept_att = torch_utils.gpu(self.mention_links.concept_att,  gpu=self._use_cuda)
            gpu_concept_mask = torch_utils.gpu(mention_links.concept_mask,  gpu=self._use_cuda)
            emb_start = time()
            for j in range(0, self._num_concepts):

                these_concept_ids = concept_ids[j, :].view(1, concept_ids.shape[1])\
                    .expand(gpu_mention_embeddings.shape[0], concept_ids.shape[1])

                these_concept_mask = gpu_concept_mask[j, :].view(1, mention_links.concept_mask.shape[1])\
                                                                  .expand(gpu_mention_embeddings.shape[0], mention_links.concept_mask.shape[1])
                these_concept_att = concept_att[j, :].view(1, concept_att.shape[1])\
                                                                  .expand(gpu_mention_embeddings.shape[0], concept_att.shape[1])
                intermed = self._net(ids=gpu_mention_embeddings,
                                     concept_ids=these_concept_ids,
                                     concept_mask = these_concept_mask,
                                     concept_att = these_concept_att,
                                     cached_emb=True).flatten()
                results[:, j] = torch_utils.cpu(intermed).detach().numpy().transpose()

                if (j+1) % 1000 == 0:
                    self.log.info("Processed {0} concepts : {1}".format(j, time()-emb_start))

            self.log.info("Total eval time={0}".format(time() - start))

            return results

    def predict_bert_online2(self, mention_links):
        """
        The prediction module for online bert.  To manage GPU memory, n mention representations (set by eval_batch_size)
        are cached, and then as a batch scored against each concept (one by one).

        :param mention_links: A mention links objects (needs to contain test data)
        :return: a numpy matrix of size [num test mentions] x [num concepts], with each row representing the scores for
        a given mention
        """
        self._net.eval()
        if not self._initialized:
            self._initialize(mention_links)

        # self._check_input(mention_links, None, allow_concepts_none=True)

        men_indx = torch.tensor(self.mention_links.test_mention_indexes)

        test_mention_mask = self.mention_links.test_mention_mask
        test_concept_mask = torch_utils.gpu(self.mention_links.concept_mask, gpu=self._use_cuda)

        mention_ids = mention_links.test_mention_ids.astype(np.int64)
        gold_concept_ids = mention_links.test_concept_ids.astype(np.int64)

        concept_ids = torch_utils.gpu(self.mention_links.concept_representations, gpu=self._use_cuda)

        mention_ids = mention_links.test_mention_representations

        results = np.zeros((mention_ids.shape[0], self._num_concepts))
        start = time()
        for i in range(mention_ids.shape[0]):
            mention_id_repeat = torch_utils.gpu(mention_ids[i, :], gpu=self._use_cuda)\
                .view(1, mention_ids.shape[1])\
                .expand(len(concept_ids), mention_ids.shape[1])

            mention_indx_repeat = men_indx[i, :].view(1, men_indx.shape[1])\
                .expand(len(concept_ids), men_indx.shape[1])

            mention_mask_repeat = torch_utils.gpu(test_mention_mask[i, :], gpu=self._use_cuda)\
                .view(1, test_mention_mask.shape[1])\
                .expand(len(concept_ids), test_mention_mask.shape[1])


            for j in range(0, len(mention_id_repeat), self.args.eval_batch_size):
                upper = min(j + self.args.eval_batch_size, len(mention_id_repeat))

                res = self._net(ids=mention_id_repeat[j:upper, :],
                                concept_ids=concept_ids[j:upper, :],
                                mention_indexes=mention_indx_repeat[j:upper, :],
                                mention_mask=mention_mask_repeat[j:upper, :],
                                concept_mask=test_concept_mask[j:upper, :])
                results[i, j:upper] = torch_utils.cpu(res.flatten()).detach().numpy()

            if (i+1) % 100 == 0:
                current = time()
                self.log.info("At i={0}, elapsed {1}".format(i, time() - start))

        self.log.info("Total eval time:{0}".format(time() - start))
        return results
