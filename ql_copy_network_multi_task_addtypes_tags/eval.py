import os
import time
import tensorflow as tf
import beam_search
import data
import json
# import pyrouge
import util
import logging
import numpy as np
import codecs
import matrix

# from beam_search import Hypothesis

FLAGS = tf.app.flags.FLAGS

SECS_UNTIL_NEW_CKPT = 60  # max number of seconds before loading new checkpoint

def sort_hyps(hyps):
  """Return a list of Hypothesis objects, sorted by descending average log probability"""
  return sorted(hyps, key=lambda h: h.avg_log_prob + FLAGS.beta * h.len, reverse=True)
class Hypothesis(object):
    """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

    def __init__(self, tokens, log_probs, state, attn_dists, p_gens, coverage, len, start_decode_Flag = None):
        """Hypothesis constructor.

        Args:
          tokens: List of integers. The ids of the tokens that form the summary so far.
          log_probs: List, same length as tokens, of floats, giving the log probabilities of the tokens so far.
          state: Current state of the decoder, a LSTMStateTuple.
          attn_dists: List, same length as tokens, of numpy arrays with shape (attn_length). These are the attention distributions so far.
          p_gens: List, same length as tokens, of floats, or None if not using pointer-generator model. The values of the generation probability so far.
          coverage: Numpy array of shape (attn_length), or None if not using coverage. The current coverage vector.
        """
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.attn_dists = attn_dists
        self.p_gens = p_gens
        self.coverage = coverage
        self.len = len

    def extend(self, token, log_prob, state, attn_dist, p_gen, coverage, len):
        """Return a NEW hypothesis, extended with the information from the latest step of beam search.

        Args:
          token: Integer. Latest token produced by beam search.
          log_prob: Float. Log prob of the latest token.
          state: Current decoder state, a LSTMStateTuple.
          attn_dist: Attention distribution from latest step. Numpy array shape (attn_length).
          p_gen: Generation probability on latest step. Float.
          coverage: Latest coverage vector. Numpy array shape (attn_length), or None if not using coverage.
        Returns:
          New Hypothesis for next step.
        """
        return Hypothesis(tokens=self.tokens + [token],
                          log_probs=self.log_probs + [log_prob],
                          state=state,
                          attn_dists=self.attn_dists + [attn_dist],
                          p_gens=self.p_gens + [p_gen],
                          coverage=coverage,
                          len=len)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def log_prob(self):
        # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
        return self.log_prob / len(self.tokens)

class Candidate_batch():
    def __init__(self,tokens, state , scores ,len):
        self.tokens = tokens
        self.last_states = state
        self.logScores = scores
        self.len = len


class EvalDecoder(object):
    """Beam search decoder."""

    def __init__(self, model, batcher, vocab):
        """Initialize decoder.

        Args:
          model: a Seq2SeqAttentionModel object.
          batcher: a Batcher object.
          vocab: Vocabulary object
        """
        self._model = model
        self._model.build_graph()
        self._batcher = batcher
        self._vocab = vocab
        self._saver = self._model.saver  # we use this to load checkpoints for decoding
        self._sess = self._model.gSess_train

        # Load an initial checkpoint to use for decoding
        # ckpt_path = util.load_ckpt(self._saver, self._sess)

        # if FLAGS.single_pass:
        #  # Make a descriptive decode directory name
        #  ckpt_name = "ckpt-" + ckpt_path.split('-')[-1] # this is something of the form "ckpt-123456"
        #  self._decode_dir = os.path.join(FLAGS.log_root, get_decode_dir_name(ckpt_name))
        #  if os.path.exists(self._decode_dir):
        #    raise Exception("single_pass decode directory %s should not already exist" % self._decode_dir)

        # else: # Generic decode dir name
        #  self._decode_dir = os.path.join(FLAGS.log_root, "decode")

        ## Make the decode dir if necessary
        # if not os.path.exists(self._decode_dir): os.mkdir(self._decode_dir)

        ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to decode yet at %s', FLAGS.log_root)
            return

        tf.logging.info('checkpoint path %s', ckpt_state.model_checkpoint_path)
        ckpt_path = os.path.join(
            FLAGS.log_root, os.path.basename(ckpt_state.model_checkpoint_path))
        tf.logging.info('renamed checkpoint path %s', ckpt_path)
        self._saver.restore(self._sess, ckpt_path)
        # if FLAGS.single_pass:
        #  # Make the dirs to contain output written in the correct format for pyrouge
        #  self._rouge_ref_dir = os.path.join(self._decode_dir, "reference")
        #  if not os.path.exists(self._rouge_ref_dir): os.mkdir(self._rouge_ref_dir)
        #  self._rouge_dec_dir = os.path.join(self._decode_dir, "decoded")
        #  if not os.path.exists(self._rouge_dec_dir): os.mkdir(self._rouge_dec_dir)

    def decode_one_question(self,batch):
        """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
        t0 = time.time()
        counter = 0

        f = os.path.join(FLAGS.log_root, "output.txt")
        # print("----------------"+f)
        outputfile = codecs.open(f, "w", "utf8")
        output_result = []
        list_of_reference = []
        print(self._batcher.c_index)

        # Run beam search to get best Hypothesis
        result = self.eval_one_batch(self._sess, self._model, self._vocab, batch)

        i=0
        out_words = data.outputids2words(result[i], self._model._vocab_out, batch.art_oovs[i])
        if data.STOP_DECODING in out_words:
            out_words = out_words[:out_words.index(data.STOP_DECODING)]
        output_now = " ".join(out_words)
        output_result.append(output_now)
        # refer = " ".join(refer)

        refer = batch.original_abstracts[i].strip()
        list_of_reference.append([refer])

        return batch.original_articles[i], batch.original_abstracts[i], output_now

    def get_condidate_predicate(self,case,candidates,art_oovs):
        import re
        predicate_regrex = "(r-mso|mso|dev|r-dev):.*?\.(.)+"
        result = []
        for i in range(len(case)):
            if re.match(predicate_regrex, case[i]):
                candidate_predicates_words = data.outputids2words(candidates[i], self._model._vocab_out, art_oovs)
                result.append("|||".join(candidate_predicates_words))
        return result

    #pair_wise decode input batch output batch


    def pair_wise_decode_batch_return_batch(self,batch):
        # Run beam search to get best Hypothesis
        result = self.eval_one_batch(self._sess, self._model, self._vocab, batch)
        # result = self.eval_one_batch(self._sess, self._model, self._vocab, batch)

        result_batch = []

        for i, instance in enumerate(result):
            if i == len(batch.art_oovs):
                break
            if i >= batch.real_length:
                print("not enough")
                break
            out_words = data.outputids2words(instance, self._model._vocab_out, batch.art_oovs[i])
            if data.STOP_DECODING in out_words:
                out_words = out_words[:out_words.index(data.STOP_DECODING)]

            output_now = " ".join(out_words)
            result_batch.append([batch.original_articles[i],output_now])

            #outputfile.write(batch.original_articles[i] + '\t' + output_now + '\n')

        i = 0
        out_batch = []
        while i<len(result_batch):
            in1 = result_batch[i]
            in2 = result_batch[2]
            tmp = in1[1]
            in1[1] = in2[1]
            in2[1] = tmp
            out_batch.append(in1)
            out_batch.append(in2)
            i+=2

        return out_batch
        #return result_batch


    def pair_wise_decode(self):
        f = os.path.join(FLAGS.data_path, "output.txt")
        outputfile = codecs.open(f, "w", "utf8")
        output_result = []
        list_of_reference = []
        while True:
            batch = self._batcher.next_pairwised_decode_batch()  # 1 example repeated across batch
            if batch is None:  # finished decoding dataset in single_pass mode
                logging.info("eval_finished")
                outputfile.close()
                break
            print(self._batcher.c_index)
            original_article = batch.original_articles[0]  # string
            original_abstract = batch.original_abstracts[0]  # string
            original_abstract_sents = batch.original_abstracts_sents[0]  # list of strings

            article_withunks = data.show_art_oovs(original_article, self._vocab)  # string
            abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab,
                                                   (batch.art_oovs[0] if FLAGS.pointer_gen else None))  # string

            # Run beam search to get best Hypothesis
            result = self.eval_one_batch(self._sess, self._model, self._vocab, batch)
            # result = self.eval_one_batch(self._sess, self._model, self._vocab, batch)


            for i, instance in enumerate(result):
                if i == len(batch.art_oovs):
                    break
                if i >= batch.real_length:
                    print("eval done with {} isntances".format(len(output_result)))
                    break
                out_words = data.outputids2words(instance, self._model._vocab_out, batch.art_oovs[i])
                if data.STOP_DECODING in out_words:
                    out_words = out_words[:out_words.index(data.STOP_DECODING)]

                output_now = " ".join(out_words)
                output_result.append(output_now)
                # refer = " ".join(refer)

                refer = batch.original_abstracts[i].strip()
                list_of_reference.append([refer])

                outputfile.write(batch.original_articles[i]  + '\t' + output_now + '\n')

        bleu = matrix.bleu_score(list_of_reference, output_result)
        acc = matrix.compute_acc(list_of_reference, output_result)

        print("bleu : {}   acc : {}".format(bleu, acc))
        return

    def decode(self):
        """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
        t0 = time.time()
        counter = 0


        f = os.path.join(FLAGS.log_root, "output.txt")
        # print("----------------"+f)
        outputfile = codecs.open(f, "w", "utf8")
        output_result = []
        list_of_reference = []
        while True:
            batch = self._batcher.next_batch()  # 1 example repeated across batch
            if batch is None:  # finished decoding dataset in single_pass mode
                logging.info("eval_finished")
                outputfile.close()
                break
            print(self._batcher.c_index)
            original_article = batch.original_articles[0]  # string
            original_abstract = batch.original_abstracts[0]  # string
            original_abstract_sents = batch.original_abstracts_sents[0]  # list of strings

            article_withunks = data.show_art_oovs(original_article, self._vocab)  # string
            abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab,
                                                   (batch.art_oovs[0] if FLAGS.pointer_gen else None))  # string

            # Run beam search to get best Hypothesis
            result,all_candidate = self.eval_one_batch_with_candidate(self._sess, self._model, self._vocab, batch)
            #result = self.eval_one_batch(self._sess, self._model, self._vocab, batch)


            for i,instance in enumerate(result):
                if i == len(batch.art_oovs):
                    break
                if i>=batch.real_length:
                    print("eval done with {} isntances".format(len(output_result)))
                    break
                out_words = data.outputids2words(instance, self._model._vocab_out, batch.art_oovs[i])
                if data.STOP_DECODING in out_words:
                    out_words = out_words[:out_words.index(data.STOP_DECODING)]

                candidates_value = self.get_condidate_predicate(out_words,all_candidate[i],batch.art_oovs[i])
                candidates_value = "_||_".join(candidates_value)

                output_now = " ".join(out_words)
                output_result.append(output_now)
                    # refer = " ".join(refer)

                refer = batch.original_abstracts[i].strip()
                list_of_reference.append([refer])

                outputfile.write(batch.original_articles[i] + '\t' + batch.original_abstracts[i] + '\t' + output_now +'\t'+candidates_value+ '\n')


        bleu = matrix.bleu_score(list_of_reference, output_result)
        acc = matrix.compute_acc(list_of_reference, output_result)

        print("bleu : {}   acc : {}".format(bleu,acc))
        return

    def eval_one_batch(self,sess, model, vocab_out, batch):
        enc_states, dec_in_state = model.run_encoder_eval(sess, batch)

        # Initialize beam_size-many hyptheses

        results = []
        steps = 0
        latest_tokens = [vocab_out.word2id(data.START_DECODING) for i in
                         range(FLAGS.batch_size)]  # latest token produced by each hypothesis

        pre_state = dec_in_state
        prev_coverage = None

        while steps < FLAGS.max_dec_steps:
            latest_tokens = [t if t in range(vocab_out.size()) else vocab_out.word2id(data.UNKNOWN_TOKEN) for t in
                             latest_tokens]
            # Run one step of the decoder to get the new info
            (topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage) = model.decode_onestep(sess=sess,
                                                                                                            batch=batch,
                                                                                                            latest_tokens=latest_tokens,
                                                                                                            enc_states=enc_states,
                                                                                                            dec_init_states=pre_state,
                                                                                                            prev_coverage=prev_coverage,
                                                                                                            first=(
                                                                                                            steps == 0))

            topk_ids = np.reshape(topk_ids,[-1])
            results.append(topk_ids)

            pre_state = new_states
            latest_tokens = topk_ids

            steps += 1

        results = np.array(results).T
        return results
    def eval_one_batch_with_candidate(self,sess, model, vocab_out, batch):
        enc_states, dec_in_state = model.run_encoder_eval(sess, batch)

        # Initialize beam_size-many hyptheses

        results = []
        steps = 0
        latest_tokens = [vocab_out.word2id(data.START_DECODING) for i in range(FLAGS.batch_size)]  # latest token produced by each hypothesis

        pre_state = dec_in_state
        prev_coverage = None

        all_candidate = [[] for i in range(FLAGS.batch_size)]

        while steps < FLAGS.max_dec_steps:
            latest_tokens = [t if t in range(vocab_out.size()) else vocab_out.word2id(data.UNKNOWN_TOKEN) for t in latest_tokens]
            # Run one step of the decoder to get the new info
            (topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage) = model.decode_onestep(sess=sess,
                                                                                                            batch=batch,
                                                                                                            latest_tokens=latest_tokens,
                                                                                                            enc_states=enc_states,
                                                                                                            dec_init_states=pre_state,
                                                                                                            prev_coverage=prev_coverage,
                                                                                                            first=(steps == 0))

            all_ids = topk_ids
            topk_ids = [topk_ids[i][0] for i in range(len(topk_ids))]

            for i in range(len(all_ids)):
                all_candidate[i].append(all_ids[i])


            #topk_ids = np.reshape(topk_ids,[-1])
            results.append(topk_ids)

            pre_state = new_states
            latest_tokens = topk_ids

            steps+=1

        results = np.array(results).T
        return results,all_candidate

    def basic_beam_search(self,sess, model, vocab_out, batch,top_k):
        enc_states, dec_in_state = model.run_encoder(sess, batch)
        hyps = [Hypothesis(tokens=[vocab_out.word2id(data.START_DECODING)],
                           log_probs=[0.0],
                           state=dec_in_state,
                           attn_dists=[],
                           p_gens=[],
                           coverage=np.zeros([batch.enc_batch.shape[1]]),  # zero vector of length attention_length
                           len=0) for _ in range(FLAGS.beam_size)]
        results = []  # this will contain finished hypotheses (those that have emitted the [STOP] token)
        steps = 0
        while steps < FLAGS.max_dec_steps and len(results) < FLAGS.beam_size:
            latest_tokens = [h.latest_token for h in hyps]  # latest token produced by each hypothesis
            latest_tokens = [t if t in range(vocab_out.size()) else vocab_out.word2id(data.UNKNOWN_TOKEN) for t in
                             latest_tokens]  # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings
            states = [h.state for h in hyps]  # list of current decoder states of the hypotheses
            prev_coverage = [h.coverage for h in hyps]  # list of coverage vectors (or None)
            (topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage) = model.decode_onestep(sess=sess,
                                                                                                            batch=batch,
                                                                                                            latest_tokens=latest_tokens,
                                                                                                            enc_states=enc_states,
                                                                                                            dec_init_states=states,
                                                                                                            prev_coverage=prev_coverage,
                                                                                                            first=(
                                                                                                            steps == 0))

            # Extend each hypothesis and collect them all in all_hyps
            all_hyps = []
            num_orig_hyps = 1 if steps == 0 else len(
                hyps)  # On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
            for i in range(num_orig_hyps):
                h, new_state, attn_dist, p_gen, new_coverage_i = hyps[i], new_states[i], attn_dists[i], p_gens[i], \
                                                                 new_coverage[
                                                                     i]  # take the ith hypothesis and new decoder state info
                for j in range(FLAGS.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    # Extend the ith hypothesis with the jth option
                    new_hyp = h.extend(token=topk_ids[i, j],
                                       log_prob=topk_log_probs[i, j],
                                       state=new_state,
                                       attn_dist=attn_dist,
                                       p_gen=p_gen,
                                       coverage=new_coverage_i,
                                       len=steps + 1)
                    all_hyps.append(new_hyp)

            # Filter and collect any hypotheses that have produced the end token.
            hyps = []  # will contain hypotheses for the next step
            for h in sort_hyps(all_hyps):  # in order of most likely h
                if h.latest_token == vocab_out.word2id(data.STOP_DECODING):  # if stop token is reached...
                    # If this hypothesis is sufficiently long, put in results. Otherwise discard.
                    if steps >= FLAGS.min_dec_steps:
                        results.append(h)
                else:  # hasn't reached stop token, so continue to extend this hypothesis
                    hyps.append(h)
                if len(hyps) == FLAGS.beam_size or len(results) == FLAGS.beam_size:
                    # Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop.
                    break

            steps += 1

            # At this point, either we've got beam_size results, or we've reached maximum decoder steps

        if len(results) == 0:  # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
            results = hyps

        # Sort hypotheses by average log probability
        hyps_sorted = sort_hyps(results)

        # Return the hypothesis with highest average log prob
        return hyps_sorted[:top_k]

    def beam_search_eval(self,sess, model, vocab_out, batch):
        enc_states, dec_in_state = model.run_encoder_eval(sess, batch)

        # Initialize beam_size-many hyptheses

        results = []
        steps = 0


        latest_tokens = [vocab_out.word2id(data.START_DECODING) for i in range(FLAGS.batch_size)]  # latest token produced by each hypothesis

        pre_state = dec_in_state
        prev_coverage = None

        candidate_batch_tmp = Candidate_batch(latest_tokens,pre_state,np.zeros([FLAGS.batch_size,1],dtype=np.float32),None)
        candidate_list = [candidate_batch_tmp]

        final_result = []

        while steps < FLAGS.max_dec_steps:

            def run_hypo(candidate_list):
                topk_result = None
                top_log_probs_result = None
                for candidate_batch_now in candidate_list:
                    latest_tokens = candidate_batch_now.tokens
                    latest_tokens = [t if t in range(vocab_out.size()) else vocab_out.word2id(data.UNKNOWN_TOKEN) for t in latest_tokens]
                    pre_state = candidate_batch_now.last_states

                    # Run one step of the decoder to get the new info
                    (topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage) = model.decode_onestep(
                        sess=sess,
                        batch=batch,
                        latest_tokens=latest_tokens,
                        enc_states=enc_states,
                        dec_init_states=pre_state,
                        prev_coverage=prev_coverage,
                        first=(steps == 0))

                    topk_log_probs = np.reshape(topk_log_probs,[FLAGS.batch_size,FLAGS.beam_size])
                    topk_log_probs+=candidate_batch_now.logScores

                    topk_ids = np.reshape(topk_ids, [FLAGS.batch_size, FLAGS.beam_size])

                    if topk_result==None:
                        topk_result = topk_ids
                        top_log_probs_result = topk_log_probs
                    else:
                        topk_result = np.hstack([topk_result,topk_ids])
                        top_log_probs_result = np.hstack([top_log_probs_result,topk_log_probs])
                return topk_result,top_log_probs_result

            topk_result, top_log_probs_result = run_hypo(candidate_list)

            def generate_top_result(topk_result,top_log_probs_result):
                all_top_beam_result = []
                all_top_beam_logScore = []
                for i,logs in enumerate(top_log_probs_result):
                    topk_tokens = topk_result[i]
                    tmp_result = np.stack([logs,topk_tokens],axis=1)
                    tmp_result = sorted(tmp_result,key=lambda x:x[0],reverse=True)
                    tmp_result = np.array(tmp_result[:FLAGS.beam_size])
                    tmp_result = tmp_result.T
                    all_top_beam_logScore.append(tmp_result[0])
                    all_top_beam_result.append(tmp_result[1])
                return all_top_beam_result,all_top_beam_logScore

            all_top_beam_result, all_top_beam_logScore = generate_top_result(topk_result, top_log_probs_result)


            # Run one step of the decoder to get the new info
            (topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage) = model.decode_onestep(sess=sess,
                                                                                                            batch=batch,
                                                                                                            latest_tokens=latest_tokens,
                                                                                                            enc_states=enc_states,
                                                                                                            dec_init_states=pre_state,
                                                                                                            prev_coverage=prev_coverage,
                                                                                                            first=(steps == 0))

            topk_ids = np.reshape(topk_ids, [-1])
            results.append(topk_ids)

            pre_state = new_states
            latest_tokens = topk_ids
            latest_tokens = [t if t in range(vocab_out.size()) else vocab_out.word2id(data.UNKNOWN_TOKEN) for t in
                             latest_tokens]

            steps += 1

        results = np.array(results).T
        return results