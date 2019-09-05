import tensorflow as tf
import config
import logging
from data import Vocab
from batcher import Batcher
from model_gru import SummarizationModel
from decode import BeamSearchDecoder

from tensorflow.python.ops import variable_scope
import data
import numpy as np

import os
import copy

import matrix


def train_with_eval(FLAGS):
    FLAGS = config.retype_FLAGS()

    logging.info("hidden_dim:" + str(FLAGS.hidden_dim))
    logging.info("emb_dim:" + str(FLAGS.emb_dim))
    logging.info("batch_size:" + str(FLAGS.batch_size))
    logging.info("max_enc_steps:" + str(FLAGS.max_enc_steps))
    logging.info("max_dec_steps:" + str(FLAGS.max_dec_steps))
    logging.info("learning rate:" + str(FLAGS.lr))

    # load dictionary
    vocab_in, vocab_out = data.load_dict_data(FLAGS)

    checkpoint_basename = os.path.join(FLAGS.log_root, "PointerGenerator_model")
    logging.info(checkpoint_basename)

    logging.info("creating model...")
    train_model, dev_model = create_training_model(FLAGS, vocab_in,vocab_out)
    train_model.save_model(checkpoint_basename)

    best_bleu, best_acc, dev_loss = validation_acc(dev_model, FLAGS)
    logging.info("bleu_now {}".format(best_bleu))

    tmpDevModel = checkpoint_basename+"tmp"
    #train_model.save_model(tmpDevModel, False)
    bad_valid = 0
    bestDevModel = tf.train.get_checkpoint_state(FLAGS.log_root).model_checkpoint_path
    while True:
        step = train_model.get_specific_variable(train_model.global_step)
        if step > FLAGS.max_run_steps:
            break

        loss = train_one_epoch(train_model, dev_model, FLAGS)
        if np.isnan(loss) or loss<0:
            logging.info("loss is nan, restore")
            train_model.load_specific_model(bestDevModel)
            bleu = -1
            acc = -1
        else:
            train_model.save_model(tmpDevModel,False)
            bleu, acc, dev_loss = validation_acc(dev_model, FLAGS)


        if acc>=best_acc:
            lr = train_model.get_specific_variable(train_model.learning_rate)
            logging.info("save new best model, learning rate {} step {}".format(lr,step))
            train_model.save_model(checkpoint_basename)
            bad_valid = 0

            best_bleu = bleu
            best_acc = acc
            bestDevModel = tf.train.get_checkpoint_state(FLAGS.log_root).model_checkpoint_path
        else:
            if FLAGS.badvalid==0:
                continue
            bad_valid += 1
            logging.info("bad valid {} compared with bestDevModel {} bleu {} acc {}".format(bad_valid,bestDevModel,best_bleu,best_acc))
            lr = train_model.get_specific_variable(train_model.learning_rate)
            logging.info("current learning rate {}".format(lr))
            if bad_valid>=FLAGS.badvalid:
                logging.info("restore model {} for {}".format(step,bestDevModel))
                train_model.load_specific_model(bestDevModel)
                train_model.run_decay_lr()
                train_model.save_model(checkpoint_basename)
                bestDevModel = tf.train.get_checkpoint_state(FLAGS.log_root).model_checkpoint_path
                bad_valid = 0
                if lr<0.001:
                    logging.info("lr = {}, stop".format(lr))
                    break
    train_model.load_specific_model(bestDevModel)
    train_model.save_model(bestDevModel,False)


def validation_acc(dev_model,FLAGS):
    dev_model.create_or_load_recent_model()
    dev_loss = 0
    valid_batcher = dev_model.batcher
    numBatches = 0
    totalLoss = 0

    output_result = []
    list_of_reference = []

    step = dev_model.get_specific_variable(dev_model.global_step)
    out_f = open(r"train_model/{}.test".format(step),"w",encoding='utf-8')

    #gate_f = open(os.path.join(FLAGS.log_root, "gate.txt"), 'w', encoding="utf-8")
    def write_pgens( pgen_label, gate_prob, gate_f):
        tmp_pgen_label = []
        for i in range(len(pgen_label)):
            tmp_pgen_label.append(str(pgen_label[i]))
            #pgen_label[i] = str(pgen_label[i])

        gate_f.write(' '.join(tmp_pgen_label) + '\t')
        gate_f.flush()
        tmp_gate = []
        for i in range(len(gate_prob)):
            tmp_gate.append(str(gate_prob[i]))
            gate_prob[i] = str(gate_prob[i])
        gate_f.write(' '.join(tmp_gate) + '\n')
        gate_f.flush()

    with dev_model.graph.as_default():
        while True:
            valid_batch = valid_batcher.next_batch()
            if valid_batch is None:
                break

            if len(valid_batch.art_oovs) < len(valid_batch.enc_batch):
                continue
            results = dev_model.run_eval_step(valid_batch)
            loss = results['loss']
            ids = np.array(results['final_ids']).T
            if np.isnan(loss):
                logging.debug("Nan")

            '''gate_p = np.array(results['gate_prob'])
            gate_p = np.argmax(gate_p,axis=-1)
            gate_p = gate_p.T'''
            for i,instance in enumerate(ids):
                if i>=valid_batch.real_length:
                    print("eval done with {} isntances".format(len(output_result)))
                    break
                if i==len(valid_batch.art_oovs):
                    break
                out_words = data.outputids2words(instance, dev_model._vocab_out, valid_batch.art_oovs[i])
                #refer = data.outputids2words(valid_batch.target_batch[i],dev_model._vocab,valid_batch.art_oovs[i])

                if data.STOP_DECODING in out_words:
                    out_words = out_words[:out_words.index(data.STOP_DECODING)]
                #if data.STOP_DECODING in refer:
                #    refer = refer[:refer.index(data.STOP_DECODING)]

                output_now = " ".join(out_words)
                output_result.append(output_now)
                #refer = " ".join(refer)

                refer = valid_batch.original_abstracts[i].strip()
                list_of_reference.append([refer])

                out_f.write(valid_batch.original_articles[i]+ '\t' + valid_batch.original_abstracts[i]+'\t'+output_now+'\n')

                #gate_f.write(str(valid_batch.enc_lens[i])+'\t')
                #write_pgens(valid_batch.pgen_label[i], gate_p[i],gate_f)

            totalLoss += loss
            numBatches += 1

    bleu = matrix.bleu_score(list_of_reference,output_result)
    acc = matrix.compute_acc(list_of_reference,output_result)

    logging.info("dev_bleu {}".format(bleu))

    logging.info("right acc {}".format(acc))

    import random
    for i in range(2):
        idx_sample = random.randint(0,len(output_result)-1)
        logging.info("real {}".format(list_of_reference[idx_sample][0]))
        logging.info("fake {}\n\n".format(output_result[idx_sample]))


    # print("totalLoss{}".format(float(totalLoss) / float(numBatches)))
    return bleu,acc,dev_loss


def train_one_epoch(train_model, dev_model, FLAGS):
    train_model.graph.as_default()
    loss =0
    lr = train_model.get_specific_variable(train_model.learning_rate)
    while True:
        batch = train_model.batcher.next_batch()
        if batch==None:
            logging.info("Finish {} epoch, lr {}".format(train_model.batcher.c_epoch,lr))
            return loss

        if batch.real_length!=FLAGS.batch_size or len(batch.art_oovs)<len(batch.enc_batch):
            continue
        results = train_model.run_train_step(batch)

        loss = results['loss']
        step = train_model.get_specific_variable(train_model.global_step)
        if np.isnan(loss):
            train_model.batcher.reset()
            return loss
        if step % 100 == 0:
            logging.info("[{}/{}]".format(train_model.batcher.c_index, train_model.batcher._length))
            logging.info("step %d \t loss %f  " % (step, loss))
            prog = float(step) / float(FLAGS.max_run_steps) * 100.0
            logging.info('PROGRESS: %.2f%%\n' % prog)






def create_training_model(FLAGS,vocab_in, vocab_out = None):
    batcher_train = Batcher(FLAGS.data_path, vocab_in,vocab_out, FLAGS, data_file=FLAGS.train_name)

    train_model = SummarizationModel(FLAGS, vocab_in,vocab_out,batcher_train)

    logging.info("Building graph...")
    train_model.build_graph()

    # Create dev model
    # I can't deepCopy tf.flags, so I change flags into nametuple.
    # Find another way in the future
    FLAGS_eval = FLAGS._asdict()
    FLAGS_eval["mode"] = "eval"
    FLAGS_eval = config.generate_nametuple(FLAGS_eval)


    #variable_scope.get_variable_scope().reuse_variables()

    batcher_dev = Batcher(FLAGS.data_path, vocab_in,vocab_out, FLAGS, data_file=FLAGS.dev_name)
    dev_model = SummarizationModel(FLAGS_eval, vocab_in,vocab_out,batcher_dev)
    dev_model.build_graph()

    train_model.create_or_load_recent_model()

    return train_model,dev_model



def decode_Beam(FLAGS):
    # If in decode mode, set batch_size = beam_size
    # Reason: in decode mode, we decode one example at a time.
    # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.
    #if FLAGS.mode == 'decode':
    #    FLAGS.batch_size = FLAGS.beam_size

    # If single_pass=True, check we're in decode mode
    #if FLAGS.single_pass and FLAGS.mode != 'decode':
    #    raise Exception("The single_pass flag should only be True in decode mode")


    vocab_in, vocab_out = data.load_dict_data(FLAGS)

    FLAGS_batcher = config.retype_FLAGS()

    FLAGS_decode = FLAGS_batcher._asdict()
    FLAGS_decode["max_dec_steps"] = 1
    FLAGS_decode["mode"] = "decode"
    FLAGS_decode = config.generate_nametuple(FLAGS_decode)
    # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries
    batcher = Batcher(FLAGS.data_path, vocab_in,vocab_out, FLAGS_batcher,  data_file=FLAGS.test_name)

    model = SummarizationModel(FLAGS_decode, vocab_in,vocab_out,batcher)
    decoder = BeamSearchDecoder(model, batcher, vocab_out)
    decoder.decode()

import time
def decode_my(FLAGS):
    vocab_in, vocab_out = data.load_dict_data(FLAGS)
    batcher = Batcher(FLAGS.data_path, vocab_in, vocab_out, FLAGS, data_file=FLAGS.test_name,shuffle=False)
    import eval
    FLAGS_decode = config.retype_FLAGS()._asdict()
    FLAGS_decode["max_dec_steps"] = 1
    FLAGS_decode = config.generate_nametuple(FLAGS_decode)
    model = SummarizationModel(FLAGS_decode, vocab_in, vocab_out, batcher)
    decoder = eval.EvalDecoder(model, batcher, vocab_out)

    time_start = time.time()
    decoder.decode()
    time_end = time.time()
    print(time_end-time_start)



def main(unused_argv):
    FLAGS = config.FLAGS
    head = '[%(asctime)-15s %(levelname)s] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('Starting seq2seq_attention in %s mode...', (FLAGS.mode))


    if FLAGS.mode == 'train':
        train_with_eval(FLAGS)

    elif FLAGS.mode == 'decode' and FLAGS.beam_size>1:
        decode_Beam(FLAGS)
    elif FLAGS.mode == 'decode':
        decode_my(FLAGS)
    else:
        raise ValueError("The 'mode' flag must be one of train/eval/decode")


if __name__ == '__main__':
    tf.app.run()