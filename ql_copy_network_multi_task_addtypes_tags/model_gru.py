# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import time
import numpy as np
import tensorflow as tf
from attention_decoder import attention_decoder,attention_decoder_fixed_context
#from tensorflow.contrib.tensorboard.plugins import projector
import logging
from tensorflow.contrib.rnn import LSTMCell,GRUCell
import copy

FLAGS = tf.app.flags.FLAGS

def linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_start: starting value to initialize the bias; 0 by default.
        scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]
            # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(axis=1, values=args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
    return res + bias_term

class SummarizationModel(object):
    """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""

    def __init__(self, hps, vocab_in,vocab_out,batcher,append_data = None):
        self._hps = hps
        self._vocab_in = vocab_in
        self._vocab_out = vocab_out
        self.batcher = batcher
        if self._hps.use_grammer_dict:
            self.grammer_index = self._vocab_out.load_special_vocab_indexes(os.path.join(self._hps.data_path, "grammer"))
        self.feed_dict = None
        self.append_data = append_data


    def _add_placeholders(self):
        """Add placeholders to the graph. These are entry points for any input data."""
        hps = self._hps
        self.learning_rate = tf.Variable(float(self._hps.lr), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * self._hps.learning_rate_decay_factor)

        # encoder part
        self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch')
        self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens')
        self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='enc_padding_mask')
        if self._hps.pointer_gen:
            self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch_extend_vocab')
            self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')

        if self._hps.position_embedding:
            self._en_position = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_position')

        if self._hps.types:
            self.types_label = tf.placeholder(tf.int32, [hps.batch_size], name='types_label')

        if self._hps.use_grammer_dict and self._hps.dict_loss:
            self._type_target = tf.placeholder(tf.int32, [hps.batch_size, None], name='type_target')

        # decoder part
        self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
        self._target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
        self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps], name='dec_padding_mask')

        if self._hps.use_pos_tag:
            self._dec_batch_tags = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
            self._target_batch_tags = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps],name='target_batch_tags')

        if self._hps.use_grammer_dict:
            self.grammer_indices_mask = tf.constant(self.grammer_index,dtype=tf.float32)
            self.reverse_grammer = 1-self.grammer_indices_mask

        if hps.mode == "decode":
            self.mask_context = tf.placeholder(tf.float32, [hps.batch_size, hps.hidden_dim * 2], name='mask_context')
        if hps.mode=="decode" and hps.coverage:
            self.prev_coverage = tf.placeholder(tf.float32, [hps.batch_size, None], name='prev_coverage')


    def _make_feed_dict(self, batch, just_enc=False):
        """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

        Args:
            batch: Batch object
            just_enc: Boolean. If True, only feed the parts needed for the encoder.
        """
        feed_dict = {}
        feed_dict[self._enc_batch] = batch.enc_batch
        feed_dict[self._enc_lens] = batch.enc_lens
        feed_dict[self._enc_padding_mask] = batch.enc_padding_mask


        if self._hps.pointer_gen:
            feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
            feed_dict[self._max_art_oovs] = batch.max_art_oovs


        if not just_enc:
            feed_dict[self._dec_batch] = batch.dec_batch
            feed_dict[self._target_batch] = batch.target_batch
            feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
            if self._hps.use_pos_tag:
                # feed_dict[self._enc_pos] = batch.enc_pos
                feed_dict[self._dec_batch_tags] = batch.dec_pos
                feed_dict[self._target_batch_tags] = batch.target_pos
            if self._hps.types:
                feed_dict[self.types_label] = batch.types

        if self._hps.position_embedding:
            feed_dict[self._en_position] = batch.en_position

        if self._hps.use_grammer_dict and self._hps.dict_loss:
            feed_dict[self._type_target] = batch.pgen_label
        return feed_dict

    def _add_encoder(self, encoder_inputs, seq_len):
        """Add a single-layer bidirectional LSTM encoder to the graph.

        Args:
            encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
            seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

        Returns:
            encoder_outputs:
                A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
            fw_state, bw_state:
                Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        """
        if self._hps.cell_name == "lstm":
            self.cell_build = LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=False)

        if self._hps.cell_name == "gru":
            self.cell_build = GRUCell(self._hps.hidden_dim)

        if self._hps.mode =="train" and self._hps.dropout<1:
            self.cell_build = tf.contrib.rnn.DropoutWrapper(cell=self.cell_build, input_keep_prob=(self._hps.dropout))


        with tf.variable_scope('encoder'):
            cell_fw = copy.deepcopy(self.cell_build)
            cell_bw = copy.deepcopy(self.cell_build)
            (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
            encoder_outputs = tf.concat(axis=2, values=encoder_outputs) # concatenate the forwards and backwards states
        return encoder_outputs, fw_st, bw_st

    def _add_tag_decoder(self,inputs):
        """Add attention decoder to the graph. In train or eval mode, you call this once to get output on ALL steps. In decode (beam search) mode, you call this once for EACH decoder step.

                Args:
                    inputs: inputs to the decoder (word embeddings). A list of tensors shape (batch_size, emb_dim)

                Returns:
                    outputs: List of tensors; the outputs of the decoder
                    out_state: The final state of the decoder
                    attn_dists: A list of tensors; the attention distributions
                    p_gens: A list of scalar tensors; the generation probabilities
                    coverage: A tensor, the current coverage vector
                """
        hps = self._hps
        cell = copy.deepcopy(self.cell_build)

        if self._hps.match_attention:
            # new_match_embedding = self.find_predicate_attention()
            new_match_embedding = self.embedding_out
            #new_match_embedding = self.find_predicate_attention_gru()
        else:
            new_match_embedding = None
        match_probs = None
        prev_coverage = self.prev_coverage if hps.mode == "decode" and hps.coverage else None  # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time
        if hps.mode == "decode":
            outputs, out_state, attn_dists, p_gens, coverage, p_grammers, match_probs = attention_decoder_fixed_context(
                                                                                            inputs, self._dec_in_state,
                                                                                            self._enc_states,
                                                                                            self._enc_padding_mask, cell,
                                                                                            initial_state_attention=(
                                                                                                hps.mode == "decode"),
                                                                                            pointer_gen=hps.pointer_gen,
                                                                                            use_coverage=hps.coverage,
                                                                                            prev_coverage=prev_coverage,
                                                                                            mask_context=self.mask_context,
                                                                                            embedding=new_match_embedding)
        else:
            outputs, out_state, attn_dists, p_gens, coverage, p_grammers, match_probs = attention_decoder(inputs,
                                                                                                          self._dec_in_state,
                                                                                                          self._enc_states,
                                                                                                          self._enc_padding_mask,
                                                                                                          cell,
                                                                                                          initial_state_attention=(
                                                                                                              hps.mode == "decode"),
                                                                                                          pointer_gen=hps.pointer_gen,
                                                                                                          use_coverage=hps.coverage,
                                                                                                          prev_coverage=prev_coverage,
                                                                                                          embedding=new_match_embedding)

        self.p_grammers_no_use = p_gens
        self.match_probs_no_use = match_probs
        return outputs, out_state, attn_dists, p_gens, coverage

    def _reduce_states(self, fw_st, bw_st):
        """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

        Args:
            fw_st: LSTMStateTuple with hidden_dim units.
            bw_st: LSTMStateTuple with hidden_dim units.

        Returns:
            state: LSTMStateTuple with hidden_dim units.
        """
        hidden_dim = self._hps.hidden_dim
        with tf.variable_scope('reduce_final_st'):

            # Define weights and biases to reduce the cell and reduce the state
            w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
            w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
            bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
            bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

            # Apply linear layer
            old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c]) # Concatenation of fw and bw cell
            old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h]) # Concatenation of fw and bw state
            new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c) # Get new cell from old cell
            new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h) # Get new state from old state
            return tf.contrib.rnn.LSTMStateTuple(new_c, new_h) # Return new cell and state

    def _reduce_states_GRU(self,fw_st, bw_st):
        encode_s = tf.concat(axis=1, values=[fw_st, bw_st])  # Concatenation of fw and bw cell
        init_s = linear(encode_s,fw_st.get_shape()[1].value,True,scope="reduce_final_st")
        return init_s

    def _add_decoder(self, inputs):
        """Add attention decoder to the graph. In train or eval mode, you call this once to get output on ALL steps. In decode (beam search) mode, you call this once for EACH decoder step.

        Args:
            inputs: inputs to the decoder (word embeddings). A list of tensors shape (batch_size, emb_dim)

        Returns:
            outputs: List of tensors; the outputs of the decoder
            out_state: The final state of the decoder
            attn_dists: A list of tensors; the attention distributions
            p_gens: A list of scalar tensors; the generation probabilities
            coverage: A tensor, the current coverage vector
        """
        hps = self._hps
        cell = copy.deepcopy(self.cell_build)

        if self._hps.match_attention:
            #new_match_embedding = self.find_predicate_attention()
            #new_match_embedding = self.embedding_out
            new_match_embedding = self.find_predicate_attention_gru()

        else:
            new_match_embedding = None
        match_probs = None
        prev_coverage = self.prev_coverage if hps.mode=="decode" and hps.coverage else None # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time
        if hps.mode == "decode":
            outputs, out_state, attn_dists, p_gens, coverage,p_grammers, match_probs = attention_decoder_fixed_context(inputs,self._dec_in_state,
                                                                                               self._enc_states,
                                                                                               self._enc_padding_mask,cell,
                                                                                               initial_state_attention=(
                                                                                               hps.mode == "decode"),
                                                                                               pointer_gen=hps.pointer_gen,
                                                                                               use_coverage=hps.coverage,
                                                                                               prev_coverage=prev_coverage,
                                                                                               mask_context=self.mask_context,
                                                                                               embedding=new_match_embedding)
        else:
            outputs, out_state, attn_dists, p_gens, coverage,p_grammers,match_probs = attention_decoder(inputs, self._dec_in_state,
                                                                                 self._enc_states,
                                                                                 self._enc_padding_mask, cell,
                                                                                 initial_state_attention=(
                                                                                 hps.mode == "decode"),
                                                                                 pointer_gen=hps.pointer_gen,
                                                                                 use_coverage=hps.coverage,
                                                                                 prev_coverage=prev_coverage,
                                                                                 embedding=new_match_embedding)

        self.p_grammers = p_gens
        self.match_probs = match_probs
        return outputs, out_state, attn_dists, p_gens, coverage

    def _calc_final_dist(self, vocab_dists, attn_dists):
        """Calculate the final distribution, for the pointer-generator model

        Args:
            vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
            attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays

        Returns:
            final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
        """
        with tf.variable_scope('final_distribution'):
            if self._hps.use_grammer_dict:
                new_vocab_with_grammer = []
                new_attention_dist = []
                for i,dist in enumerate(vocab_dists):

                    p_gram = self.p_gens[i]
                    p_first = tf.slice(p_gram,[0,0],[self._hps.batch_size,1])
                    p_second = tf.slice(p_gram, [0, 1], [self._hps.batch_size, 1])
                    p_third = tf.slice(p_gram, [0, 2], [self._hps.batch_size, 1])

                    weights = self.grammer_indices_mask*p_first+self.reverse_grammer*p_second
                    tmp = dist * weights

                    new_vocab_with_grammer.append(tmp)
                    tmp2 = attn_dists[i]*p_third
                    new_attention_dist.append(tmp2)
                vocab_dists = new_vocab_with_grammer
                attn_dists = new_attention_dist

            # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
            else:
                vocab_dists = [p_gen * dist for (p_gen, dist) in zip(self.p_gens, vocab_dists)]
                attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(self.p_gens, attn_dists)]

            # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
            extended_vsize = self._vocab_out.size() + self._max_art_oovs # the maximum (over the batch) size of the extended vocabulary
            extra_zeros = tf.zeros((self._hps.batch_size, self._max_art_oovs))
            vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists] # list length max_dec_steps of shape (batch_size, extended_vsize)

            # Project the values in the attention distributions onto the appropriate entries in the final distributions
            # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
            # This is done for each decoder timestep.
            # This is fiddly; we use tf.scatter_nd to do the projection
            batch_nums = tf.range(0, limit=self._hps.batch_size) # shape (batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
            attn_len = tf.shape(self._enc_batch_extend_vocab)[1] # number of states we attend over
            batch_nums = tf.tile(batch_nums, [1, attn_len]) # shape (batch_size, attn_len)
            indices = tf.stack( (batch_nums, self._enc_batch_extend_vocab), axis=2) # shape (batch_size, enc_t, 2)
            shape = [self._hps.batch_size, extended_vsize]
            attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists] # list length max_dec_steps (batch_size, extended_vsize)

            # Add the vocab distributions and the copy distributions together to get the final distributions
            # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
            # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
            final_dists = [vocab_dist + copy_dist for (vocab_dist,copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]

            return final_dists

    def find_predicate_attention_gru_spo(self):
        #subwords_embedding, mask_np = self._vocab_out.compute_predicate_indices_split_mask(self._vocab_in)
        subwords_embedding,mask_np = self._vocab_out.compute_predicate_indices_mask(self._vocab_in)
        attention_subwords = tf.constant(subwords_embedding, dtype=tf.int32)
        attention_embedding = tf.nn.embedding_lookup(self.embedding_in, attention_subwords)

        words_mask = tf.constant(mask_np, dtype=tf.int32)

        attention_embedding_size = attention_embedding.get_shape()[-1].value
        tile_embedding_out = tf.expand_dims(self.embedding_out, axis=1)
        attention_embedding = tf.concat([tile_embedding_out, attention_embedding], axis=1)
        attention_embedding_size_in = attention_embedding.get_shape()[-1].value

        cell_fw = copy.deepcopy(self.cell_build)
        cell_bw = copy.deepcopy(self.cell_build)
        (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, attention_embedding,
                                                                            dtype=tf.float32, sequence_length=words_mask,
                                                                            swap_memory=True)
        #encoder_outputs = tf.concat(axis=2, values=encoder_outputs)  # concatenate the forwards and backwards states
        encode_s = tf.concat([fw_st,bw_st],axis=1)
        init_s = linear(encode_s, attention_embedding_size, True, scope="mode_predicate")

        '''gramar_indices_ex = tf.expand_dims(self.grammer_indices_mask,axis=-1)
        reverse_grammer = tf.expand_dims(self.reverse_grammer,axis=-1)
        init_s = self.embedding_out*gramar_indices_ex + init_s*reverse_grammer'''

        return init_s

    def find_predicate_attention_gru(self):
        subwords_embedding,words_mask = self._vocab_out.compute_predicate_indices_split_mask(self._vocab_in)
        #subwords_embedding, words_mask = self._vocab_out.compute_lcquad_indices_mask(self._vocab_in)
        attention_subwords = tf.constant(subwords_embedding, dtype=tf.int32)
        attention_embedding = tf.nn.embedding_lookup(self.embedding_in, attention_subwords)

        attention_embedding_size = attention_embedding.get_shape()[-1].value
        tile_embedding_out = tf.expand_dims(self.embedding_out, axis=1)
        attention_embedding = tf.concat([tile_embedding_out, attention_embedding], axis=1)
        attention_embedding_size_in = attention_embedding.get_shape()[-1].value

        cell_fw = copy.deepcopy(self.cell_build)
        cell_bw = copy.deepcopy(self.cell_build)
        (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, attention_embedding,
                                                                            dtype=tf.float32, sequence_length=words_mask,
                                                                            swap_memory=True)
        #encoder_outputs = tf.concat(axis=2, values=encoder_outputs)  # concatenate the forwards and backwards states
        encode_s = tf.concat([fw_st,bw_st],axis=1)
        init_s = linear(encode_s, attention_embedding_size, True, scope="mode_predicate")

        if self._hps.use_grammer_dict:
            gramar_indices_ex = tf.expand_dims(self.grammer_indices_mask,axis=-1)
            reverse_grammer = tf.expand_dims(self.reverse_grammer,axis=-1)
            init_s = self.embedding_out*gramar_indices_ex + init_s*reverse_grammer

        return init_s


    def find_predicate_attention(self):
        #subwords_embedding = self._vocab_out.compute_predicate_indices(self._vocab_in)
        subwords_embedding = self._vocab_out.compute_predicate_indices_split(self._vocab_in)
        attention_subwords = tf.constant(subwords_embedding, dtype=tf.int32)
        attention_embedding = tf.nn.embedding_lookup(self.embedding_in, attention_subwords)

        attention_embedding_size = attention_embedding.get_shape()[-1].value
        tile_embedding_out = tf.expand_dims(self.embedding_out,axis=1)
        attention_embedding = tf.concat([tile_embedding_out,attention_embedding],axis=1)

        attention_embedding_size_in =  attention_embedding.get_shape()[-1].value

        with tf.variable_scope("attention_cnn"):
            w = tf.get_variable("W_h", [1, 1, attention_embedding_size_in, attention_embedding_size])
            b = tf.get_variable("b", [attention_embedding_size])
            attention_embedding = tf.expand_dims(attention_embedding,axis=1)
            attention_embedding_out = tf.nn.conv2d(attention_embedding, w, strides=[1, 1, 1, 1], padding='VALID')+b
            attention_embedding_out = tf.nn.max_pool(attention_embedding_out, ksize=[1, 1, 4, 1], strides=[1, 1, 1, 1], padding='VALID')

            attention_embedding_out = tf.squeeze(attention_embedding_out)

        gramar_indices_ex = tf.expand_dims(self.grammer_indices_mask,axis=-1)
        reverse_grammer = tf.expand_dims(self.reverse_grammer,axis=-1)
        attention_embedding_out = self.embedding_out*gramar_indices_ex + attention_embedding_out*reverse_grammer
        #attention_embedding = tf.nn.dropout(attention_embedding_out,0.5)
        if False:
            reduce_attention_embedding = tf.reduce_mean(attention_embedding,axis=1)
            return reduce_attention_embedding
        return attention_embedding_out

    def mask_softmax(self,vocab_score):
        normal_vocab_dists = []
        for dist in vocab_score:
            dist = tf.exp(dist)
            predicates = dist * self.reverse_grammer
            sum_w = tf.reduce_sum(predicates, axis=-1, keep_dims=True)
            #sum_w = tf.clip_by_value(sum_w, 1e-20, 10)
            predicates = predicates / sum_w

            grammars = dist * self.grammer_indices_mask
            sum_w = tf.reduce_sum(grammars, axis=-1, keep_dims=True)
            #sum_w = tf.clip_by_value(sum_w, 1e-20, 10)
            grammars = grammars / sum_w

            final_out = predicates + grammars
            normal_vocab_dists.append(final_out)
        return normal_vocab_dists

    def add_type_loss(self,encode_hidden_status):
        t_pre = linear(encode_hidden_status, 9, True, scope="type_loss")
        t_pre = tf.nn.softmax(t_pre)

        t_pre = tf.clip_by_value(t_pre, 1e-20, 10.0)
        self.type_pre = t_pre
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.types_label,logits=t_pre)
        self.types_loss = tf.reduce_mean(loss)

    def add_tag_multi_task(self):
        with tf.variable_scope('add_tag_multi_task'):
            hps = self._hps
            vocab_tag = self._vocab_out.vocab_tag
            vsize_out = vocab_tag.size()

            embedding_out_tags = tf.get_variable('embedding_out_tag', [vsize_out, self._hps.emb_dim], dtype=tf.float32,
                                                 initializer=self.trunc_norm_init)
            emb_dec_inputs_tags = [tf.nn.embedding_lookup(embedding_out_tags, x) for x in
                                   tf.unstack(self._dec_batch_tags, axis=1)]
            decoder_outputs_tags, self._dec_out_state_tags, attn_dists_tags, p_gens_tags, coverage_tags = self._add_tag_decoder(
                emb_dec_inputs_tags)

            with tf.variable_scope('output_projection'):
                w = tf.get_variable('w', [hps.hidden_dim, vsize_out], dtype=tf.float32,
                                    initializer=self.trunc_norm_init)
                v = tf.get_variable('v', [vsize_out], dtype=tf.float32, initializer=self.trunc_norm_init)
                vocab_scores = []  # vocab_scores is the vocabulary distribution before applying softmax. Each entry on the list corresponds to one decoder step
                for i, output in enumerate(decoder_outputs_tags):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()

                    tmp = tf.nn.xw_plus_b(output, w, v)  # apply the linear layer
                    vocab_scores.append(tmp)
                # vocab_dists = self.mask_softmax(vocab_scores)
                vocab_dists = [tf.nn.softmax(s) for s in vocab_scores]
                final_dists = vocab_dists

            self.final_ids_tags = []
            for dist_now in final_dists:
                self.final_ids_tags.append(tf.argmax(dist_now,axis=-1))

            if hps.mode in ['train', 'eval']:
                # Calculate the loss
                with tf.variable_scope('loss'):
                    loss_per_step = []  # will be list length max_dec_steps containing shape (batch_size)
                    batch_nums = tf.range(0, limit=hps.batch_size)  # shape (batch_size)
                    for dec_step, dist in enumerate(final_dists):
                        targets = self._target_batch_tags[:,
                                  dec_step]  # The indices of the target words. shape (batch_size)
                        indices = tf.stack((batch_nums, targets), axis=1)  # shape (batch_size, 2)
                        gold_probs = tf.gather_nd(dist,indices)  # shape (batch_size). prob of correct words on this step
                        gold_probs = tf.clip_by_value(gold_probs, 1e-20, 10.0)
                        # losses = -(1 - gold_probs) * (1 - gold_probs) * tf.log(gold_probs)
                        losses = -tf.log(gold_probs)
                        loss_per_step.append(losses)
                        # Apply dec_padding_mask and get loss
                        self._loss_tags = _mask_and_avg(loss_per_step, self._dec_padding_mask)


    def _add_seq2seq(self):
        """Add the whole sequence-to-sequence model to the graph."""
        hps = self._hps
        vsize_en = self._vocab_in.size()  # size of the vocabulary
        vsize_out = self._vocab_out.size()

        with tf.variable_scope('seq2seq'):
            # Some initializers
            self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)

            # Add embedding matrix (shared by the encoder and decoder inputs)
            with tf.variable_scope('embedding'):
                embedding_in = tf.get_variable('embedding_in', [vsize_en, hps.emb_dim], dtype=tf.float32,initializer=self.trunc_norm_init)
                self.original_embedding_in = embedding_in
                if hps.cor_embedding:
                    self._extra_embedding_in = tf.get_variable('extra_embedding_in', [vsize_en, 100], dtype=tf.float32,initializer=self.trunc_norm_init)
                    embedding_in = tf.concat([embedding_in,self._extra_embedding_in],axis=-1)

                if self._hps.shared_vocab:
                    embedding_out = embedding_in
                else:
                    embedding_out = tf.get_variable('embedding_out', [vsize_out, hps.emb_dim], dtype=tf.float32,initializer=self.trunc_norm_init)
                    self.original_embedding_out = embedding_out
                    if hps.cor_embedding:
                        self._extra_embedding_out = tf.get_variable('extra_embedding_out', [vsize_out, 100], dtype=tf.float32,initializer=self.trunc_norm_init)
                        embedding_out = tf.concat([embedding_out,self._extra_embedding_out],axis=-1)

                #if hps.mode=="train": self._add_emb_vis(embedding) # add to tensorboard
                emb_enc_inputs = tf.nn.embedding_lookup(embedding_in, self._enc_batch) # tensor with shape (batch_size, max_enc_steps, emb_size)
                emb_dec_inputs = [tf.nn.embedding_lookup(embedding_out, x) for x in tf.unstack(self._dec_batch, axis=1)] # list length max_dec_steps containing shape (batch_size, emb_size)
                '''if hps.use_pos_tag:
                    self._enc_pos = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_pos')
                    pos_embedding = tf.get_variable('pos_tag', [self._vocab_in.pos_len, hps.pos_tag_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
                    enc_pos_embedding = tf.nn.embedding_lookup(pos_embedding, self._enc_pos)
                    emb_enc_inputs = tf.concat(axis=-1, values=[emb_enc_inputs, enc_pos_embedding])'''
                if hps.position_embedding:
                    position_embedding = tf.get_variable('position', [hps.max_enc_steps, 50], dtype=tf.float32, initializer=self.trunc_norm_init)
                    position_feature = tf.nn.embedding_lookup(position_embedding,self._en_position)
                    emb_enc_inputs = tf.concat(axis=-1, values=[emb_enc_inputs, position_feature])

            self.embedding_in = embedding_in
            self.embedding_out = embedding_out

            # Add the encoder.
            enc_outputs, fw_st, bw_st = self._add_encoder(emb_enc_inputs, self._enc_lens)
            self._enc_states = enc_outputs

            # Our encoder is bidirectional and our decoder is unidirectional so we need to reduce the final encoder hidden state to the right size to be the initial decoder hidden state
            self._dec_in_state = self._reduce_states_GRU(fw_st, bw_st)

            if hps.types:
                self.add_type_loss(self._dec_in_state)

            # Add the decoder.
            with tf.variable_scope('decoder'):
                decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage = self._add_decoder(emb_dec_inputs)

            if self._hps.mode=='train':
                for i in range(len(decoder_outputs)):
                    decoder_outputs[i] = tf.nn.dropout(decoder_outputs[i],self._hps.dropout)
            # Add the output projection to obtain the vocabulary distribution
            with tf.variable_scope('output_projection'):
                w = tf.get_variable('w', [hps.hidden_dim, vsize_out], dtype=tf.float32,initializer=self.trunc_norm_init)
                v = tf.get_variable('v', [vsize_out], dtype=tf.float32, initializer=self.trunc_norm_init)
                vocab_scores = []  # vocab_scores is the vocabulary distribution before applying softmax. Each entry on the list corresponds to one decoder step
                for i, output in enumerate(decoder_outputs):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()

                    tmp = tf.nn.xw_plus_b(output, w, v)# apply the linear layer
                    if self._hps.match_attention:
                        predicate_weights = self.match_probs[i]
                        #predicate_weights = self.match_probs[i] * self.reverse_grammer + self.grammer_indices_mask

                        tmp = tmp * predicate_weights
                    vocab_scores.append(tmp)
                #vocab_dists = self.mask_softmax(vocab_scores)
                vocab_dists = [tf.nn.softmax(s) for s in vocab_scores]  # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.

                if self._hps.use_grammer_dict==True:
                    normal_vocab_dists = []
                    for i, output in enumerate(vocab_dists):
                        predicates = output* self.reverse_grammer
                        sum_w = tf.reduce_sum(predicates,axis=-1,keep_dims=True)
                        sum_w = tf.clip_by_value(sum_w,1e-20,10)
                        predicates = predicates/sum_w

                        grammars = output*self.grammer_indices_mask
                        sum_w = tf.reduce_sum(grammars,axis=-1,keep_dims=True)
                        sum_w = tf.clip_by_value(sum_w, 1e-20, 10)
                        grammars = grammars/sum_w

                        final_out= predicates+grammars
                        normal_vocab_dists.append(final_out)
                    vocab_dists = normal_vocab_dists


            # For pointer-generator model, calc final distribution from copy distribution and vocabulary distribution
            if FLAGS.pointer_gen:
                final_dists = self._calc_final_dist(vocab_dists, self.attn_dists)
            else: # final distribution is just vocabulary distribution
                final_dists = vocab_dists

            self.final_dists = final_dists

            self.final_ids = []
            for dist_now in final_dists:
                self.final_ids.append(tf.argmax(dist_now,axis=-1))

            if self._hps.use_pos_tag:
                self.add_tag_multi_task()

            if hps.mode in ['train', 'eval']:
                # Calculate the loss
                with tf.variable_scope('loss'):
                    if FLAGS.pointer_gen:
                        # Calculate the loss per step
                        # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
                        loss_per_step = [] # will be list length max_dec_steps containing shape (batch_size)
                        batch_nums = tf.range(0, limit=hps.batch_size) # shape (batch_size)
                        type_batch_number = tf.range(0, limit=hps.batch_size)
                        for dec_step, dist in enumerate(final_dists):
                            targets = self._target_batch[:,dec_step] # The indices of the target words. shape (batch_size)
                            indices = tf.stack( (batch_nums, targets), axis=1) # shape (batch_size, 2)
                            gold_probs = tf.gather_nd(dist, indices) # shape (batch_size). prob of correct words on this step
                            gold_probs = tf.clip_by_value(gold_probs,1e-20,10.0)
                            #losses = -(1 - gold_probs) * (1 - gold_probs) * tf.log(gold_probs)
                            losses = -tf.log(gold_probs)
                            if self._hps.use_grammer_dict and self._hps.dict_loss:
                                p_gram = self.p_gens[dec_step]
                                types = self._type_target[:, dec_step]
                                indices_types = tf.stack((type_batch_number, types), axis=1)
                                type_probs = tf.gather_nd(p_gram, indices_types)
                                type_probs = tf.clip_by_value(type_probs, 1e-20, 10.0)
                                type_loss = -tf.log(type_probs)
                                losses +=0.01 * type_loss

                            loss_per_step.append(losses)


                        # Apply dec_padding_mask and get loss
                        self._original_loss = _mask_and_avg(loss_per_step, self._dec_padding_mask)
                        self._loss = self._original_loss
                        if hps.types:
                            self._loss = self._loss + 0.1*self.types_loss

                        if self._hps.use_pos_tag:
                            self._loss += 0.1 * self._loss_tags


                    else: # baseline model
                        self._loss = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1), self._target_batch, self._dec_padding_mask) # this applies softmax internally

                    tf.summary.scalar('loss', self._loss)

                    # Calculate coverage loss from the attention distributions
                    if hps.coverage:
                        with tf.variable_scope('coverage_loss'):
                            self._coverage_loss = _coverage_loss(self.attn_dists, self._dec_padding_mask)
                            tf.summary.scalar('coverage_loss', self._coverage_loss)
                        self._total_loss = self._loss + hps.cov_loss_wt * self._coverage_loss
                        tf.summary.scalar('total_loss', self._total_loss)

        if hps.mode == "decode":
            # We run decode beam search mode one decoder step at a time
            assert len(final_dists)==1 # final_dists is a singleton list containing shape (batch_size, extended_vsize)
            final_dists = final_dists[0]
            #final_dists = final_dists
            if self._hps.beam_size>1:
                topk_probs, self._topk_ids = tf.nn.top_k(final_dists, self._hps.beam_size*2) # take the k largest probs. note batch_size=beam_size in decode mode
            else:
                topk_probs, self._topk_ids = tf.nn.top_k(final_dists,1)  # if need candidate set 1 to n
            self._topk_log_probs = tf.log(topk_probs)


    def _add_train_op(self):
        """Sets self._train_op, the op to run for training."""
        # Take gradients of the trainable variables w.r.t. the loss function to minimize
        loss_to_minimize = self._total_loss if self._hps.coverage else self._loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

        # Clip the gradients
        with tf.device("/gpu:0"):
            grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

        # Add a summary
        tf.summary.scalar('global_norm', global_norm)

        # Apply adagrad optimizer
        optimizer = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)
        with tf.device("/gpu:0"):
            self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')

    def _add_ql_only_train_op(self):
        """Sets self._train_op, the op to run for training."""
        # Take gradients of the trainable variables w.r.t. the loss function to minimize
        loss_to_minimize = self._original_loss if self._hps.coverage else self._loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

        # Clip the gradients
        with tf.device("/gpu:0"):
            grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

        # Add a summary
        tf.summary.scalar('global_norm', global_norm)

        # Apply adagrad optimizer
        optimizer = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)
        with tf.device("/gpu:0"):
            self._original_train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='original_train_step')


    def build_graph(self):
        """Add the placeholders, model, global step, train_op and summaries to the graph"""
        t0 = time.time()

        _config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
        self.graph = tf.Graph()

        self.gSess_train = tf.Session(config=_config,graph= self.graph)

        logging.debug("Graph id: {}{}".format(id(self.graph),self.graph))

        with self.graph.as_default():
            self._add_placeholders()
            with tf.device("/gpu:0"):
                self._add_seq2seq()
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            if self._hps.mode == 'train':
                self._add_train_op()
            self._summaries = tf.summary.merge_all()
            t1 = time.time()
            tf.logging.info('Time to build graph: %i seconds', t1 - t0)
            self.saver = tf.train.Saver(tf.global_variables(),max_to_keep=10)

            logging.debug(self._hps.mode+" model")
            for tmp in tf.global_variables():
                logging.debug(tmp)



    def assign_word_embedding(self):
        logging.info("load glove embedding from {}".format(self._hps.glove_dir))
        with self.graph.as_default():
            embedding = self._vocab_in.load_word_embedding(self._hps.glove_dir,self._hps.emb_dim)
            self.gSess_train.run(self.original_embedding_in.assign(embedding))

            if self._hps.cor_embedding ==True:
                logging.info("load cor_embedding from {}".format(self._hps.cor_embedding_dir))
                cor_embedding = self._vocab_in.load_word_embedding(self._hps.cor_embedding_dir,100)
                self.gSess_train.run(self._extra_embedding_in.assign(cor_embedding))

            if self._hps.shared_vocab==False:
                embedding = self._vocab_out.load_word_embedding(self._hps.glove_dir, self._hps.emb_dim)
                self.gSess_train.run(self.original_embedding_out.assign(embedding))
                if self._hps.cor_embedding == True:
                    cor_embedding = self._vocab_out.load_word_embedding(self._hps.cor_embedding_dir, 100)
                    self.gSess_train.run(self._extra_embedding_out.assign(cor_embedding))

    #think again
    def run_single_train_step(self, batch):
        """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
        feed_dict = self._make_feed_dict(batch)
        to_return = {
                'train_op': self._original_train_op,
                'summaries': self._summaries,
                'loss': self._original_loss,
                'global_step': self.global_step,
        }

        return self.gSess_train.run(to_return, feed_dict)


    def run_train_step(self, batch):
        """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
        feed_dict = self._make_feed_dict(batch)
        to_return = {
                'train_op': self._train_op,
                'summaries': self._summaries,
                'loss': self._loss,
                'global_step': self.global_step,
        }
        if self._hps.coverage:
            to_return['coverage_loss'] = self._coverage_loss
        return self.gSess_train.run(to_return, feed_dict)

    def run_eval_step(self, batch):
        """Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss."""
        sess = self.gSess_train
        feed_dict = self._make_feed_dict(batch)
        to_return = {
                'summaries': self._summaries,
                'loss': self._loss,
                'global_step': self.global_step,
                'final_ids': self.final_ids,
                'gate_prob': self.p_gens,
                'types':self.type_pre,
                'final_ids_tags':self.final_ids_tags

        }
        if self._hps.coverage:
            to_return['coverage_loss'] = self._coverage_loss
        return sess.run(to_return, feed_dict)

    def run_encoder(self, sess, batch):
        """For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.

        Args:
            sess: Tensorflow session.
            batch: Batch object that is the same example repeated across the batch (for beam search)

        Returns:
            enc_states: The encoder states. A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
            dec_in_state: A LSTMStateTuple of shape ([1,hidden_dim],[1,hidden_dim])
        """
        feed_dict = self._make_feed_dict(batch, just_enc=True) # feed the batch into the placeholders
        (enc_states, dec_in_state, global_step) = sess.run([self._enc_states, self._dec_in_state, self.global_step], feed_dict) # run the encoder

        # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
        #dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
        dec_in_state =dec_in_state[0]
        return enc_states, dec_in_state

    def run_encoder_eval(self, sess, batch):
        """For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.

        Args:
            sess: Tensorflow session.
            batch: Batch object that is the same example repeated across the batch (for beam search)

        Returns:
            enc_states: The encoder states. A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
            dec_in_state: A LSTMStateTuple of shape ([1,hidden_dim],[1,hidden_dim])
        """
        feed_dict = self._make_feed_dict(batch, just_enc=True) # feed the batch into the placeholders
        (enc_states, dec_in_state, global_step) = sess.run([self._enc_states, self._dec_in_state, self.global_step], feed_dict) # run the encoder

        # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
        #dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
        dec_in_state =dec_in_state
        return enc_states, dec_in_state

    def decode_onestep(self, sess, batch, latest_tokens, enc_states, dec_init_states, prev_coverage,first):
        """For beam search decoding. Run the decoder for one step.

        Args:
            sess: Tensorflow session.
            batch: Batch object containing single example repeated across the batch
            latest_tokens: Tokens to be fed as input into the decoder for this timestep
            enc_states: The encoder states.
            dec_init_states: List of beam_size LSTMStateTuples; the decoder states from the previous timestep
            prev_coverage: List of np arrays. The coverage vectors from the previous timestep. List of None if not using coverage.

        Returns:
            ids: top 2k ids. shape [beam_size, 2*beam_size]
            probs: top 2k log probabilities. shape [beam_size, 2*beam_size]
            new_states: new states of the decoder. a list length beam_size containing
                LSTMStateTuples each of shape ([hidden_dim,],[hidden_dim,])
            attn_dists: List length beam_size containing lists length attn_length.
            p_gens: Generation probabilities for this step. A list length beam_size. List of None if in baseline mode.
            new_coverage: Coverage vectors for this step. A list of arrays. List of None if coverage is not turned on.
        """

        beam_size = len(dec_init_states)

        # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
        '''cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
        hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
        new_c = np.concatenate(cells, axis=0)	# shape [batch_size,hidden_dim]
        new_h = np.concatenate(hiddens, axis=0)	# shape [batch_size,hidden_dim]
        new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)'''
        new_dec_in_state = dec_init_states

        if first:
            mask_context = np.zeros([self._hps.batch_size, self._hps.hidden_dim * 2])
        else:
            mask_context = np.ones([self._hps.batch_size, self._hps.hidden_dim * 2])

        feed = {
            self._enc_states: enc_states,
            self._enc_padding_mask: batch.enc_padding_mask,
            self._dec_in_state: new_dec_in_state,
            self._dec_batch: np.transpose(np.array([latest_tokens])),
            self.mask_context: mask_context
        }

        to_return = {
            "ids": self._topk_ids,
            "probs": self._topk_log_probs,
            "states": self._dec_out_state,
            "attn_dists": self.attn_dists
        }

        if FLAGS.pointer_gen:
            feed[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
            feed[self._max_art_oovs] = batch.max_art_oovs
            to_return['p_gens'] = self.p_gens

        if self._hps.coverage:
            feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
            to_return['coverage'] = self.coverage

        results = sess.run(to_return, feed_dict=feed) # run the decoder step

        # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
        new_states = [results['states'][i, :] for i in range(beam_size)]

        # Convert singleton list containing a tensor to a list of k arrays
        assert len(results['attn_dists'])==1
        attn_dists = results['attn_dists'][0].tolist()

        if FLAGS.pointer_gen:
            # Convert singleton list containing a tensor to a list of k arrays
            assert len(results['p_gens'])==1
            p_gens = results['p_gens'][0].tolist()
        else:
            p_gens = [None for _ in range(beam_size)]

        # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
        if FLAGS.coverage:
            new_coverage = results['coverage'].tolist()
            assert len(new_coverage) == beam_size
        else:
            new_coverage = [None for _ in range(beam_size)]

        return results['ids'], results['probs'], new_states, attn_dists, p_gens, new_coverage

    def create_or_load_recent_model(self):
        with self.graph.as_default():
            if not os.path.isdir(FLAGS.log_root):
                os.mkdir(FLAGS.log_root)
            ckpt = tf.train.get_checkpoint_state(FLAGS.log_root)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                self.saver.restore(self.gSess_train, ckpt.model_checkpoint_path)
            else:
                print("Created model with fresh parameters.")
                self.gSess_train.run(tf.global_variables_initializer())
                if FLAGS.use_glove and self._hps.mode=="train":
                    self.assign_word_embedding()

    def load_specific_model(self,best_path):
        with self.graph.as_default():
            self.saver.restore(self.gSess_train, best_path)

    def save_model(self,checkpoint_basename,with_step = True):
        with self.graph.as_default():
            if with_step:
                self.saver.save(self.gSess_train, checkpoint_basename, global_step=self.global_step)
            else:
                self.saver.save(self.gSess_train, checkpoint_basename)

            logging.debug("model save {}".format(checkpoint_basename))

    def get_specific_variable(self,cur_varaible):
        with self.graph.as_default():
            return cur_varaible.eval(self.gSess_train)
    def run_decay_lr(self):
        with self.graph.as_default():
            self.gSess_train.run(self.learning_rate_decay_op)

    def assign_specific_value(self,tensor,val):
        self.gSess_train.run(tensor.assign(val))

def _mask_and_avg(values, padding_mask):
    """Applies mask to values then returns overall average (a scalar)

    Args:
        values: a list length max_dec_steps containing arrays shape (batch_size).
        padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

    Returns:
        a scalar
    """

    dec_lens = tf.reduce_sum(padding_mask, axis=1) # shape batch_size. float32
    values_per_step = [v * padding_mask[:,dec_step] for dec_step,v in enumerate(values)]
    values_per_ex = sum(values_per_step)/dec_lens # shape (batch_size); normalized value for each batch member
    return tf.reduce_mean(values_per_ex) # overall average


def _coverage_loss(attn_dists, padding_mask):
    """Calculates the coverage loss from the attention distributions.

    Args:
        attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
        padding_mask: shape (batch_size, max_dec_steps).

    Returns:
        coverage_loss: scalar
    """
    coverage = tf.zeros_like(attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
    covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
    for a in attn_dists:
        covloss = tf.reduce_sum(tf.minimum(a, coverage), [1]) # calculate the coverage loss for this step
        covlosses.append(covloss)
        coverage += a # update the coverage vector
    coverage_loss = _mask_and_avg(covlosses, padding_mask)
    return coverage_loss
