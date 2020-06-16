"""
Elliot Schumacher, Johns Hopkins University
Created 3/19/19
"""

from allennlp.modules.elmo import Elmo, batch_to_ids
import torch
options_file = "/Users/elliotschumacher/Dropbox/git/synonym_detection/resources/bilm/out_max/options.json"
weight_file = "/Users/elliotschumacher/Dropbox/git/synonym_detection/resources/bilm/out_max/std-weights.hdf5"

# Compute two different representation for each token.
# Each representation is a linear weighted combination for the
# 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
elmo = Elmo(options_file, weight_file, 1, dropout=0)

# use batch_to_ids to convert sentences to character ids
sentences = [['First', 'sentence', '.'], ['Another', '.'], ["The", "patient", "displayed", "signs", "of", "diabetes"]]
character_ids = batch_to_ids(sentences)

embeddings = elmo(character_ids)
print(embeddings)
elmo.train()
# embeddings['elmo_representations'] is length two list of tensors.
# Each element contains one layer of ELMo representations with shape
# (2, 3, 1024).
#   2    - the batch size
#   3    - the sequence length of the batch
#   1024 - the length of each ELMo vector