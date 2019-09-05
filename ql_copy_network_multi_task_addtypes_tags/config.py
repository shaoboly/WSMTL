import sys
import time
import os
import tensorflow as tf
import numpy as np
from collections import namedtuple
#from tensorflow.python import debug as tf_debug
import argparse
import logging

tf.set_random_seed(111)  # a seed value for randomness

FLAGS = tf.app.flags.FLAGS

# Where to find data
tf.app.flags.DEFINE_string('data_path', '', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', '', 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('train_name', 'train.txt.types.new', 'train file.')
tf.app.flags.DEFINE_string('dev_name', 'simple_test.txt', 'dev file.')
tf.app.flags.DEFINE_string('test_name', 'test_tmp.txt', 'test file.')
tf.app.flags.DEFINE_string('qq_name', 'test_tmp.txt', 'test file.')

# Important settings
tf.app.flags.DEFINE_string('mode', 'decode', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')
tf.app.flags.DEFINE_string('cell_name', 'gru', 'must be one of lstm/gru')

tf.app.flags.DEFINE_boolean('shared_vocab', False, "if True, vocab.out = vocab.in")
tf.app.flags.DEFINE_boolean('use_glove', False,"use pretrain word2vec")
tf.app.flags.DEFINE_string('glove_dir', r"D:\data\glove.6B\glove.6B.300d.txt", 'glove dir')
tf.app.flags.DEFINE_boolean('use_grammer_dict', False,"use_grammer_dict")
tf.app.flags.DEFINE_boolean('dict_loss', False, 'dict_loss of use_grammer_dict')


tf.app.flags.DEFINE_boolean('match_attention', False,"match_attention")
tf.app.flags.DEFINE_boolean('cor_embedding', False,"match_attention")
tf.app.flags.DEFINE_string('cor_embedding_dir', r"D:\data\seq2seq\MSPaD.Merge\MSPaD\data_dir_lower\all_predict\new_fresh_fix_s\vector.tsv", 'glove dir')

tf.app.flags.DEFINE_boolean('position_embedding', False,"match_attention")
tf.app.flags.DEFINE_boolean('types', True, 'query type loss')
tf.app.flags.DEFINE_boolean('qq_loss', True, 'qq_loss')
tf.app.flags.DEFINE_boolean('use_pos_tag', True,"use_pos_tag")

# Where to save output
tf.app.flags.DEFINE_string('log_root', 'train_model', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', '', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 300, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 300, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 32, 'minibatch size')
tf.app.flags.DEFINE_integer('max_enc_steps', 30, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 30, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('beam_size', 1, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 1, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5, 'learning rate')
tf.app.flags.DEFINE_integer('pos_tag_dim', 50, 'dimension of word embeddings')



tf.app.flags.DEFINE_integer('badvalid', 10, 'badvalid.')

tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('max_grad_norm', 1.0, 'for gradient clipping')
tf.app.flags.DEFINE_float('dropout', 0.5, 'for gradient clipping')

# Pointer-generator or baseline model
tf.app.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')

# Coverage hyperparameters
tf.app.flags.DEFINE_boolean('coverage', False, 'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
tf.app.flags.DEFINE_float('cov_loss_wt', 1.0, 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')

# Utility flags, for restoring and changing checkpoints
tf.app.flags.DEFINE_boolean('convert_to_coverage_model', False, 'Convert a non-coverage model to a coverage model. Turn this on and run in train mode. Your current training model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')
tf.app.flags.DEFINE_boolean('restore_best_model', False, 'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')

# Debugging. See https://www.tensorflow.org/programmers_guide/debugger
tf.app.flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")
tf.app.flags.DEFINE_integer('max_run_steps', 10000000,
                            'Maximum number of run steps.')
# Save frequency
tf.app.flags.DEFINE_integer('save_model_step', 100, 'How often to save the model')
tf.app.flags.DEFINE_integer('valid_step', 1000, 'How often to save the model')
tf.app.flags.DEFINE_integer('best_k_hyp', 1, 'Best k hypotheses')
tf.app.flags.DEFINE_float('beta', 0, 'Weight for timestep in beamsearch score')


def retype_FLAGS():
    # Make a namedtuple hps, containing the values of the hyperparameters that the model needs


    hps_dict = {}
    for key, val in FLAGS.__flags.items():  # for each flag
        hps_dict[key] = val  # add it to the dict
    hps = namedtuple("HParams", hps_dict.keys())._make(hps_dict.values())
    return hps

def generate_nametuple(hps_dict):
    return namedtuple("HParams", hps_dict.keys())._make(hps_dict.values())