"""
Elliot Schumacher, Johns Hopkins University
Created 2/1/19
"""
import sys
import torch
import time
import os
import subprocess
import logging
import configargparse
import socket
import glob
from codebase import ranker
from codebase import scoring
from codebase import mention_links
from codebase import torch_utils
from codebase import evaluation
from codebase import sheets
from n2c2_2019.clinical_concept_linker import load_share_clef_2013
from n2c2_2019.clinical_concept_linker import load_train_test, load_data as n2c2_load_data

import shutil

log = logging.getLogger()

def load_data(args):
    train_dict = {}
    test_dict = {}
    if args.dataset == "n2c2":
        train, dev = load_train_test()
        for file in train:
            train_dict[file['id']] = file
        for file in dev:
            test_dict[file['id']] = file

        log.info("Loaded n2c2 dataset")
    else:
        for file in load_share_clef_2013(partition='train'):
            train_dict[file['id']] = file
        for file in load_share_clef_2013(partition='dev'):
            test_dict[file['id']] = file
        log.info("Loaded SHaRE dataset")

    return train_dict, test_dict

def load_test_data(args):
    train_dict = {}
    test_dict = {}
    if args.dataset == "n2c2":
        train = n2c2_load_data('train')
        for file in train:
            train_dict[file['id']] = file
        test = n2c2_load_data('test')
        for file in test:
            test_dict[file['id']] = file

        log.info("Loaded n2c2 dataset")
    else:
        for file in load_share_clef_2013(partition='train'):
            train_dict[file['id']] = file
        for file in load_share_clef_2013(partition='test'):
            test_dict[file['id']] = file
        log.info("Loaded SHaRE dataset")

    return train_dict, test_dict
def model(args):
    if not args.test:
        train_dict, test_dict = load_data(args)

        links = mention_links.MentionLinks(train_dict,args, test_dict)

        model = scoring.PairwiseRankingModel(args, links)


        if args.model_path or args.keep_training:
            if not model._initialized:
                model._initialize(links)

            if args.keep_training:
                try:
                    all_files = glob.glob(os.path.join(args.directory,'checkpoint*.tar.gz'))
                    latest_model = max(all_files, key=os.path.getctime)
                    log.info("Loading model {0}".format(latest_model))

                    model.load_model(latest_model)
                except:
                    log.error("Cannot find any checkpoints in directory :{0}".format(args.directory))
            else:
                model.load_model(args.model_path)

        model.fit(links, test_dict)
    else:
        if args.test_partition == 'test':
            train_dict, test_dict = load_test_data(args)
        elif args.test_partition == 'dev':
            train_dict, test_dict = load_data(args)

        links = mention_links.MentionLinks(train_dict, args, test_dict)

        model = scoring.PairwiseRankingModel(args, links)
        if not model._initialized:
            model._initialize(links)
        model.load_model(args.model_path)
        if args.attention and args.embedding == "elmo":
            predictions = model.predict_elmo_att(links)
        elif args.online and args.embedding == "elmo":
            predictions = model.predict_elmo_online(links)
        elif args.attention and args.embedding == "bert":
            predictions = model.predict_bert_att(links)
        elif args.online and args.embedding == "bert":
            predictions = model.predict_bert_online(links)
        else:
            predictions = model.predict(links)
        output_path = os.path.join(model.model_chkpt_dir, "eval_final.csv")
        scores = evaluation.score(links, predictions, test_dict, outpath=output_path)
        log.info(scores)

def save_code(args):
    """
    This function zips the current state of the codebase as a replication backup.
    :param args: ConfigArgParse program arguments
    """
    try:
        current_file = sys.argv[0]
        pathname = os.path.abspath(os.path.dirname(current_file))
        outzip = os.path.join(args.directory, "codebase")
        shutil.make_archive(base_name=outzip,
                            format='zip',
                            root_dir=pathname)
        log.info("Saving codebase at {0}".format(outzip))

        result = subprocess.run(['git',  'show', '--oneline',  '-s'], stdout=subprocess.PIPE)
        log.info("Git version:\t{0}".format(result.stdout.decode('utf-8')))
    except Exception as e:
        log.error("Error saving code to zip")
        log.error(e)

if __name__ == "__main__":

    config_file = "../config.ini"
    p = configargparse.ArgParser()#default_config_files=["../config.ini", "./config.ini","../config.local.ini", "./config.local.ini"])
    p.add('-c', '--my-config', is_config_file=True, help='config file path')
    p.add('--dataset', default='share')
    p.add('--root', help='root directory for storing results',default='./results')
    p.add_argument('--only_annotated_concepts', dest='only_annotated_concepts', action='store_true')
    p.set_defaults(only_annotated_concepts=False)
    p.add('--mention_embeddings', help='directory for mention embeddings')
    p.add('--concept_embeddings', help='directory for concept embeddings')
    p.add('--model_path', help='pre-trained model path', default=None)
    p.add('--lexicon', help='path to lexicon', default=None)
    #p.add('--gpus', help='gpus to use', default="0,1")
    p.add('--test', default=False)
    p.add('--test_partition', default='test')
    p.add('--timestamp', default=None)
    p.add('--directory')
    p.add('--keep_training', default=False)


    # Adds arguments got ranker, mention_links, and scoring classes.
    for arg, val in scoring.PairwiseRankingModel.default_arguments.items():
        if type(val) is not bool:
            p.add("--{0}".format(arg), default=val, type=type(val))
        else:
            p.add_argument('--{0}'.format(arg), dest=arg, action='store_true')
            p.add_argument('--{0}_false'.format(arg), dest=arg, action='store_false')
            p.set_defaults(**{arg:val})

    for arg, val in ranker.ElmoAttentionRanker.default_arguments.items():
        if type(val) is not bool:
            p.add("--{0}".format(arg), default=val, type=type(val))
        else:
            p.add_argument('--{0}'.format(arg), dest=arg, action='store_true')
            p.add_argument('--{0}_false'.format(arg), dest=arg, action='store_false')
            p.set_defaults(**{arg:val})
    for arg, val in mention_links.MentionLinks.default_arguments.items():
        if type(val) is not bool:
            p.add("--{0}".format(arg), default=val, type=type(val))
        else:
            p.add_argument('--{0}'.format(arg), dest=arg, action='store_true')
            p.add_argument('--{0}_false'.format(arg), dest=arg, action='store_false')
            p.set_defaults(**{arg:val})

    args = p.parse_args()

    # Setting up timestamped directory for log, model, and other object storage
    if args.timestamp is None:
        args.timestamp = time.strftime("run_%Y_%m_%d_%H_%M_%S") + "_{machine}".format(machine = socket.gethostname())
    else:
        args.keep_training=True
        log.info("Using timestamp: {0}".format(args.timestamp))

    args.directory = os.path.join(args.root, args.timestamp)
    os.makedirs(args.directory, exist_ok=True)

    log.handlers.clear()
    formatter = logging.Formatter('%(message)s')
    fh = logging.FileHandler(os.path.join(args.directory, "log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.setLevel(logging.INFO)
    log.addHandler(ch)
    log.info(p.format_values())

    sheets.add_run(vars(args))

    torch_utils.set_seed(cuda=args.use_cuda)

    p.write_config_file(args, [os.path.join(args.directory, 'config.ini')])

    try:
        for i in range (torch.cuda.device_count()):
            torch_utils.gpu(torch.zeros((1)), gpu=args.use_cuda).to('cuda:{0}'.format(i))
        if torch.cuda.is_available():
            log.info("Using CUDA device:{0}".format(torch.cuda.current_device()))
        else:
            log.info("Not using CUDA :(")
        save_code(args)
        model(args)
        sheets.update_run(vars(args), end=True)
    except Exception as e:
        sheets.error_run(vars(args), e)
        raise e
    except KeyboardInterrupt:
        sheets.error_run(vars(args), "KeyboardInterrupt", end_type="Aborted")
    except InterruptedError:
        sheets.error_run(vars(args), "OS Interrupt", end_type="Aborted")

