import random
import data
import numpy as np
import os
#one record in dataset
class Example:
    def __init__(self, article, abstract_sentences,pos,  vocab_in,vocab_out, config,types = None):
        self._config = config

        # Get ids of special tokens
        start_decoding = vocab_in.word2id(data.START_DECODING)
        stop_decoding = vocab_in.word2id(data.STOP_DECODING)

        # Process the article
        article_words = article.split()
        if len(article_words) > config.max_enc_steps:
            article_words = article_words[:config.max_enc_steps]
        self.enc_len = len(article_words)  # store the length after truncation but before padding
        self.enc_input = [vocab_in.word2id(w) for w in
                          article_words]  # list of word ids; OOVs are represented by the id for UNK token


        if config.use_pos_tag:
            if pos==None:
                self.decode_pos, self.target_pos = [],[]
                self.enc_pos = []
            else:
                pos_words = pos.split()
                if len(pos_words) > config.max_enc_steps:
                    pos_words = pos_words[:config.max_enc_steps]
                assert len(pos_words)==len(article_words)
                #self.enc_pos = [vocab_in.tag2id[w] for w in pos_words]
                self.enc_pos = [vocab_out.vocab_tag.word2id(w) for w in pos_words]
                self.decode_pos, self.target_pos = self.get_dec_inp_targ_seqs(self.enc_pos, config.max_dec_steps,start_decoding, stop_decoding)


        if config.types:
            self.types = types

        # Process the abstract
        abstract = ' '.join(abstract_sentences)  # string
        abstract_words = abstract.split()  # list of strings
        abs_ids = [vocab_out.word2id(w) for w in abstract_words]  # list of word ids; OOVs are represented by the id for UNK token



        # Get the decoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, config.max_dec_steps, start_decoding,stop_decoding)
        self.dec_len = len(self.dec_input)

        # If using pointer-generator mode, we need to store some extra info
        if config.pointer_gen:
            # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
            self.enc_input_extend_vocab, self.article_oovs = data.article2ids(article_words, vocab_out)

            # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
            abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab_out, self.article_oovs)

            # Overwrite decoder target sequence so it uses the temp article OOV ids
            _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, config.max_dec_steps, start_decoding,stop_decoding)

        # Store the original strings
        self.original_article = article
        self.original_abstract = abstract
        self.original_abstract_sents = abstract_sentences


    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        """Given the reference summary as a sequence of tokens, return the input sequence for the decoder, and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len. The input sequence must start with the start_id and the target sequence must end with the stop_id (but not if it's been truncated).

        Args:
          sequence: List of ids (integers)
          max_len: integer
          start_id: integer
          stop_id: integer

        Returns:
          inp: sequence length <=max_len starting with start_id
          target: sequence same length as input, ending with stop_id only if there was no truncation
        """
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # truncate
            inp = inp[:max_len]
            target = target[:max_len]  # no end_token
        else:  # no truncation
            target.append(stop_id)  # end token
        assert len(inp) == len(target)
        return inp, target

    def pad_decoder_inp_targ(self, max_len, pad_id):
        """Pad decoder input and target sequences with pad_id up to max_len."""
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_encoder_input(self, max_len, pad_id):
        """Pad the encoder input sequence with pad_id up to max_len."""
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if self._config.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)

    def pad_pos_input(self,max_len,pad_id):
        while len(self.enc_pos) < max_len:
            self.enc_pos.append(pad_id)

    def pad_tags_decode(self,max_len,pad_id):
        while len(self.decode_pos) < max_len:
            self.decode_pos.append(pad_id)
        while len(self.target_pos) < max_len:
            self.target_pos.append(pad_id)

class Batch(object):
    """Class representing a minibatch of train/val/test examples for text summarization."""

    def __init__(self, example_list, hps, vocab,vocab_out,real_length =None):
        """Turns the example_list into a Batch object.

        Args:
             example_list: List of Example objects
             hps: hyperparameters
             vocab: Vocabulary object
        """
        self.hps = hps
        self.pad_id = vocab.word2id(data.PAD_TOKEN) # id of the PAD token used to pad sequences
        self.vocab = vocab
        self.vocab_out = vocab_out
        self.init_encoder_seq(example_list, hps) # initialize the input to the encoder
        self.init_decoder_seq(example_list, hps) # initialize the input and targets for the decoder
        self.store_orig_strings(example_list) # store the original strings
        self.real_length =real_length

    def init_encoder_seq(self, example_list, hps):
        """Initializes the following:
                self.enc_batch:
                    numpy array of shape (batch_size, <=max_enc_steps) containing integer ids (all OOVs represented by UNK id), padded to length of longest sequence in the batch
                self.enc_lens:
                    numpy array of shape (batch_size) containing integers. The (truncated) length of each encoder input sequence (pre-padding).
                self.enc_padding_mask:
                    numpy array of shape (batch_size, <=max_enc_steps), containing 1s and 0s. 1s correspond to real tokens in enc_batch and target_batch; 0s correspond to padding.

            If hps.pointer_gen, additionally initializes the following:
                self.max_art_oovs:
                    maximum number of in-article OOVs in the batch
                self.art_oovs:
                    list of list of in-article OOVs (strings), for each example in the batch
                self.enc_batch_extend_vocab:
                    Same as self.enc_batch, but in-article OOVs are represented by their temporary article OOV number.
        """
        # Determine the maximum length of the encoder input sequence in this batch
        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)
            if hps.use_pos_tag:
                ex.pad_pos_input(max_enc_seq_len,self.vocab.pos_pad_id)
                ex.pad_tags_decode(hps.max_dec_steps, self.pad_id)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        self.enc_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
        if hps.use_pos_tag:
            self.enc_pos = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
            self.dec_pos = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
            self.target_pos = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)

        self.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.float32)

        if hps.types:
            self.types = np.zeros(shape=[hps.batch_size], dtype=np.int32)
            for i, ex in enumerate(example_list):
                if ex.types!=None:
                    self.types[i] = ex.types

        # Fill in the numpy arrays

        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1

            if hps.use_pos_tag:
                self.enc_pos[i, :] = ex.enc_pos[:]
                self.dec_pos[i, :] = ex.decode_pos[:]
                self.target_pos[i, :] = ex.target_pos[:]

        if hps.position_embedding:
            self.en_position = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                position_ex = np.arange(0,max_enc_seq_len,1)
                self.en_position[i,:] = position_ex[:]

        # For pointer-generator mode, need to store some extra info
        if hps.pointer_gen:
            # Determine the max number of in-article OOVs in this batch
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
            # Store the in-article OOVs themselves
            self.art_oovs = [ex.article_oovs for ex in example_list]
            # Store the version of the enc_batch that uses the article OOV ids
            self.enc_batch_extend_vocab = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]


    def init_deocde_point_label(self,vocab_out):
        self.pgen_label = np.zeros_like(self.target_batch,dtype=np.int32)
        grammar_indices = vocab_out.get_special_vocab_indexes(os.path.join(self.hps.data_path, "grammer"))
        for i, target_out in enumerate(self.target_batch):
            for j,item in enumerate(target_out):
                #if item in grammar_indices or item <4:
                if item in grammar_indices:
                    self.pgen_label[i][j] = 0
                else:
                    '''if vocab_out.is_predicate(item):
                        self.pgen_label[i][j]= 1
                    else:
                        self.pgen_label[i][j] = 2'''
                    if item>=vocab_out.size():
                    #if vocab_out.is_predicate(item):
                        self.pgen_label[i][j]= 2
                    else:
                        self.pgen_label[i][j] = 1

    def init_decoder_seq(self, example_list, hps):
        """Initializes the following:
                self.dec_batch:
                    numpy array of shape (batch_size, max_dec_steps), containing integer ids as input for the decoder, padded to max_dec_steps length.
                self.target_batch:
                    numpy array of shape (batch_size, max_dec_steps), containing integer ids for the target sequence, padded to max_dec_steps length.
                self.dec_padding_mask:
                    numpy array of shape (batch_size, max_dec_steps), containing 1s and 0s. 1s correspond to real tokens in dec_batch and target_batch; 0s correspond to padding.
                """
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp_targ(hps.max_dec_steps, self.pad_id)

        # Initialize the numpy arrays.
        # Note: our decoder inputs and targets must be the same length for each batch (second dimension = max_dec_steps) because we do not use a dynamic_rnn for decoding. However I believe this is possible, or will soon be possible, with Tensorflow 1.0, in which case it may be best to upgrade to that.
        self.dec_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            for j in range(ex.dec_len):
                self.dec_padding_mask[i][j] = 1

        if hps.use_grammer_dict and hps.dict_loss:
            self.init_deocde_point_label(vocab_out=self.vocab_out)


    def store_orig_strings(self, example_list):
        """Store the original article and abstract strings in the Batch object"""
        self.original_articles = [ex.original_article for ex in example_list] # list of lists
        self.original_abstracts = [ex.original_abstract for ex in example_list] # list of lists
        self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list] # list of list of lists


class Batcher:
    def __init__(self,data_path, vocab_in,vocab_out, config, data_file,shuffle=True):
        self._data_path = data_path
        self._data_file = os.path.join(data_path,data_file)
        self._vocab_in = vocab_in
        self._vocab_out = vocab_out
        self._config = config
        self._shuffle = shuffle

        self._raw_text = open(self._data_file,encoding="utf-8").readlines()

        if shuffle:
            random.shuffle(self._raw_text)

        self.c_epoch = 0
        self.c_index = 0
        self._length = len(self._raw_text)
        if self._config.types:
            self.read_types(os.path.join(data_path,"types.txt"))

    def read_types(self,data_dir):
        self.types = {}
        in_f = open(data_dir,encoding="utf-8").readlines()
        for i,line in enumerate(in_f):
            line= line.strip().split('\t')[0]
            self.types[line] = i

    def next_batch(self):
        if self.c_index>= self._length:
            self.c_epoch+=1
            self.c_index=0
            random.shuffle(self._raw_text)
            return None

        last_index = self.c_index+self._config.batch_size
        batch_now =  self._raw_text[self.c_index:last_index]
        self.c_index = last_index

        example_list = []
        for i,instance in enumerate(batch_now):
            pos=None
            article,abstract = instance.strip().split("\t")[:2]
            if self._config.use_pos_tag:
                pos = instance.strip().split("\t")[2]

            types=None
            if self._config.types:
                types = instance.strip().split("\t")[3]
                types = self.types[types]

            abstract=abstract.replace(data.SENTENCE_START,"").replace(data.SENTENCE_END,"")
            example = Example(article, [abstract],pos, self._vocab_in,self._vocab_out, self._config,types=types)
            example_list.append(example)

        real_length = len(example_list)
        while len(example_list)<self._config.batch_size:
            example_list+=random.sample(example_list,1)

        batch = Batch(example_list,self._config,self._vocab_in,self._vocab_out,real_length=real_length)
        return batch

    def batch_from_decode_result(self,result):
        example_list = []
        pos = None
        for i, instance in enumerate(result):
            article, abstract = instance
            '''
            if self._config.use_pos_tag:
                pos = instance.strip().split("\t")[2]'''

            abstract = abstract.replace(data.SENTENCE_START, "").replace(data.SENTENCE_END, "")
            example = Example(article, [abstract], pos, self._vocab_in, self._vocab_out, self._config)
            example_list.append(example)

        real_length = len(example_list)
        while len(example_list) < self._config.batch_size:
            example_list += random.sample(example_list, 1)

        batch = Batch(example_list, self._config, self._vocab_in, self._vocab_out, real_length=real_length)
        return batch

    def next_pairwised_decode_batch(self):
        if self.c_index>= self._length:
            self.c_epoch+=1
            self.c_index=0
            random.shuffle(self._raw_text)

        real_batch = int(self._config.batch_size/2)

        last_index = self.c_index+real_batch
        batch_now =  self._raw_text[self.c_index:last_index]
        self.c_index = last_index

        example_list = []
        for i,instance in enumerate(batch_now):
            pos=None
            q1,q2 = instance.strip().split("\t")[:2]
            abstract = "kong"
            #article,abstract = instance.strip().split("\t")[:2]

            abstract=abstract.replace(data.SENTENCE_START,"").replace(data.SENTENCE_END,"")
            example = Example(q1, [abstract],pos, self._vocab_in,self._vocab_out, self._config)
            example_list.append(example)
            example = Example(q2, [abstract], pos, self._vocab_in, self._vocab_out, self._config)
            example_list.append(example)

        real_length = len(example_list)
        while len(example_list)<self._config.batch_size:
            example_list+=random.sample(example_list,1)

        batch = Batch(example_list,self._config,self._vocab_in,self._vocab_out,real_length=real_length)
        return batch

    def batch_one_data(self,raw_txt,pos=None):
        article, abstract = raw_txt.strip(),"kong"
        example = Example(article, [abstract], pos, self._vocab_in, self._vocab_out, self._config)
        tiles_example = []
        for i in range(self._config.batch_size):
            tiles_example.append(example)
        batch = Batch(tiles_example, self._config, self._vocab_in, self._vocab_out)
        return batch

    def next_single_decode_batch(self):
        if self.c_index>= self._length:
            self.c_epoch+=1
            self.c_index=0
            random.shuffle(self._raw_text)
            return None

        last_index = self.c_index+1
        batch_now =  self._raw_text[self.c_index:last_index]
        self.c_index = last_index

        example_list = []
        for i,instance in enumerate(batch_now):
            pos=None
            article,abstract = instance.strip().split("\t")[:2]
            if self._config.use_pos_tag:
                pos = instance.strip().split("\t")[2]

            abstract=abstract.replace(data.SENTENCE_START,"").replace(data.SENTENCE_END,"")
            example = Example(article, [abstract],pos, self._vocab_in,self._vocab_out, self._config)
            example_list.append(example)

        tiles_example = []
        for i in range(self._config.batch_size):
            tiles_example.append(example_list[0])

        batch = Batch(tiles_example, self._config, self._vocab_in, self._vocab_out)
        return batch

    def reset(self):
        self.c_index = 0
        random.shuffle(self._raw_text)

    def reload(self):
        self._raw_text = open(self._data_file, encoding="utf-8").readlines()
        random.shuffle(self._raw_text)
        self.c_epoch = 0
        self.c_index = 0
        self._length = len(self._raw_text)