import numpy as np
import logging,os
import re
# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences

# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.
predicate_regrex = "(r-mso|mso|dev|r-dev):.*?\.(.)+"

class Vocab(object):
    """Vocabulary class for mapping between words and ids (integers)"""

    def __init__(self, vocab_file, max_size):
        """Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.

        Args:
            vocab_file: path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with most frequent word first. This code doesn't actually use the frequencies, though.
            max_size: integer. The maximum size of the resulting Vocabulary."""
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0 # keeps track of total number of words in the Vocab
        self.grammar_id =None
        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        self.vocab_tag = None
        self.pos_pad_id = 0
        for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r',encoding="utf-8") as vocab_f:
            for line in vocab_f:
                if '\t' in line:
                    pieces = line.split('\t')
                else:
                    pieces = line.split()

                if len(pieces) != 2:
                    print('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue
                w = pieces[0]
                if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
                    break

        print("Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1]))

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

    def load_special_vocab_indexes(self,sp_dir):
        lines = open(sp_dir,encoding="utf-8")
        result = np.zeros((self.size()),dtype = np.float32)
        for line in lines:
            word = line.strip()
            idx = self.word2id(word)
            result[idx] = 1
        return result
    def get_special_vocab_indexes(self,sp_dir):
        if self.grammar_id!=None:
            return self.grammar_id
        lines = open(sp_dir,encoding="utf-8")
        grammar_id = []
        for line in lines:
            word = line.split('\t')[0].strip()
            idx = self.word2id(word)
            grammar_id.append(idx)
        self.grammar_id = grammar_id
        return grammar_id


    def load_word_embedding(self,embedding_dir,embedding_dim =300):
        in_f = open(embedding_dir,encoding="utf-8")
        final_embedding = np.random.normal(scale = 0.001,size=(self._count,embedding_dim))
        for line in in_f:
            line = line.strip().split()
            word,embedding = line[0],line[1:]
            if word not in self._word_to_id:
                continue
            idx = self._word_to_id[word]
            for i in range(len(embedding)):
                final_embedding[idx][i] = float(embedding[i])
            #final_embedding[idx] = norm_vector(final_embedding[idx])
        return final_embedding


    def load_Pos_dict(self,pos_dir):
        in_f = open(pos_dir, encoding="utf-8")
        self.tag2id ={}
        self.id2tag ={}
        for line in in_f:
            #line = line.strip()
            line = line.strip().split('\t')[0]
            idx = len(self.id2tag)
            self.tag2id[line] = idx
            self.id2tag[idx] = line
        if 'O' not in self.tag2id:
            idx = len(self.id2tag)
            self.tag2id['O'] = idx
            self.id2tag[idx] = 'O'
        self.pos_len = len(self.tag2id)
        self.pos_pad_id = self.tag2id["O"]


    def is_predicate(self,index):
        if index in self._id_to_word:
            w = self.id2word(index)
        else:
            return False
        if re.match(predicate_regrex, w) or re.match("<http://dbpedia.org/.*?>", w):
            return True
        else:
            return False

    def compute_predicate_indices_mask(self,embedding_dict):
        all_predicate_indexes = []
        mask = []
        for i in range(self._count):
            w = self.id2word(i)
            if re.match(predicate_regrex,w):
                w = w.replace("r-mso:","")
                w = w.replace("mso:", "")
                subwords = w.split('.')
                tmp = []
                for sub in subwords:
                    tmp.append(embedding_dict.word2id(sub))
                all_predicate_indexes.append(np.array(tmp))
                mask.append(len(tmp)+1)
            else:
                c = embedding_dict.word2id(w)
                mask.append(1)
                tmp = [c,c,c]
                all_predicate_indexes.append(np.array(tmp))

        return np.array(all_predicate_indexes), np.array(mask,dtype=np.int32)

    def compute_predicate_indices(self,embedding_dict):
        all_predicate_indexes = []
        for i in range(self._count):
            w = self.id2word(i)
            if re.match("(r-mso|mso):.*?\..*?\.(.)+",w):
                w = w.replace("r-mso:","")
                w = w.replace("mso:", "")
                subwords = w.split('.')
                tmp = []
                for sub in subwords:
                    tmp.append(embedding_dict.word2id(sub))
                all_predicate_indexes.append(np.array(tmp))
            else:
                c = embedding_dict.word2id(w)
                tmp = [c,c,c]
                all_predicate_indexes.append(np.array(tmp))

        return np.array(all_predicate_indexes)

    def compute_predicate_indices_split_mask(self, embedding_dict, max=3):
        all_predicate_indexes = []
        mask = []
        for i in range(self._count):
            w = self.id2word(i)
            if re.match(predicate_regrex, w):
                w = w.replace("r-mso:", "")
                w = w.replace("mso:", "")

                second_word = w.split('.')[1]
                subwords = [second_word]+w.split('.')[-1].split('_')

                tmp = []
                for i, sub in enumerate(subwords):
                    if i >= max:
                        break
                    tmp.append(np.array(embedding_dict.word2id(sub)))
                mask.append(len(tmp)+1)
                while len(tmp) < max:
                    tmp.append(np.zeros_like(tmp[-1]))
                all_predicate_indexes.append(np.array(tmp))
            else:
                c = embedding_dict.word2id(w)
                tmp = [c for i in range(max)]
                all_predicate_indexes.append(np.array(tmp))
                mask.append(1)

        return np.array(all_predicate_indexes),np.array(mask,dtype=np.int32)

    def compute_predicate_indices_split(self,embedding_dict,max=3):
        all_predicate_indexes = []
        for j in range(self._count):
            w = self.id2word(j)
            if re.match("(r-mso|mso):.*?\..*?\.(.)+",w):
                w = w.replace("r-mso:","")
                w = w.replace("mso:", "")
                subwords = w.split('.')[-1].split('_')
                tmp = []
                for i,sub in enumerate(subwords):
                    if i>=max:
                        break
                    tmp.append(np.array(embedding_dict.word2id(sub)))
                while len(tmp)<max:
                    tmp.append(np.zeros_like(tmp[-1]))
                all_predicate_indexes.append(np.array(tmp))
            else:
                c = embedding_dict.word2id(w)
                tmp = [c for i in range(max)]
                all_predicate_indexes.append(np.array(tmp))

        return np.array(all_predicate_indexes)

    def compute_lcquad_indices_mask(self,embedding_dict, max=3):
        all_predicate_indexes = []
        mask = []
        for i in range(self._count):
            w = self.id2word(i)
            if re.match("<http://.*?>", w):
                w = w.replace('<http://dbpedia.org/', '').replace('>', '')
                subwords = w.split('/')
                tmp = []
                for i,sub in enumerate(subwords):
                    if i>=max:
                        break
                    tmp.append(embedding_dict.word2id(sub))
                if len(tmp)==0:
                    print("?")
                mask.append(len(tmp) + 1)
                while len(tmp) < max:
                    tmp.append(1)
                all_predicate_indexes.append(np.array(tmp))

            else:
                c = embedding_dict.word2id(w)
                mask.append(1)
                tmp = [c for i in range(max)]
                all_predicate_indexes.append(np.array(tmp))

        return np.array(all_predicate_indexes), np.array(mask,dtype=np.int32)

def load_dict_data(FLAGS):

    logging.info(FLAGS.vocab_path)
    vocab_in = Vocab(os.path.join(FLAGS.data_path, "vocab.in"), FLAGS.vocab_size)  # create a vocabulary
    if FLAGS.use_pos_tag:
        #vocab_in.load_Pos_dict(os.path.join(FLAGS.data_path, "pos_tag"))
        vocab_in.vocab_tag = load_tag_vocab(FLAGS)
    if FLAGS.shared_vocab:
        vocab_out = vocab_in
    else:
        vocab_out = Vocab(os.path.join(FLAGS.data_path, "vocab.out"), FLAGS.vocab_size)
    if FLAGS.use_pos_tag:
        vocab_out.vocab_tag = load_tag_vocab(FLAGS)

    return vocab_in,vocab_out

def load_tag_vocab(FLAGS):
    vocab_tag = Vocab(os.path.join(FLAGS.data_path, "vocab.tags"), FLAGS.vocab_size)  # create a vocabulary
    return vocab_tag


def norm_vector(x,std = 1e-4):
    x = (x - np.average(x)) / np.std(x) *std
    return x



def article2ids(article_words, vocab):
    """Map the article words to their ids. Also return a list of OOVs in the article.

    Args:
        article_words: list of words (strings)
        vocab: Vocabulary object

    Returns:
        ids:
            A list of word ids (integers); OOVs are represented by their temporary article OOV number. If the vocabulary size is 50k and the article has 3 OOVs, then these temporary OOV numbers will be 50000, 50001, 50002.
        oovs:
            A list of the OOV words in the article (strings), in the order corresponding to their temporary article OOV numbers."""
    ids = []
    oovs = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id: # If w is OOV
            if w not in oovs: # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_num) # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs):
    """Map the abstract words to their ids. In-article OOVs are mapped to their temporary OOV numbers.

    Args:
        abstract_words: list of words (strings)
        vocab: Vocabulary object
        article_oovs: list of in-article OOV words (strings), in the order corresponding to their temporary article OOV numbers

    Returns:
        ids: List of ids (integers). In-article OOV words are mapped to their temporary OOV numbers. Out-of-article OOV words are mapped to the UNK token id."""
    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id: # If w is an OOV word
            if w in article_oovs: # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w) # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else: # If w is an out-of-article OOV
                ids.append(unk_id) # Map to the UNK token id
        else:
            ids.append(i)
    return ids

def outputids2words(id_list, vocab, article_oovs):
    """Maps output ids to words, including mapping in-article OOVs from their temporary ids to the original OOV string (applicable in pointer-generator mode).

    Args:
        id_list: list of ids (integers)
        vocab: Vocabulary object
        article_oovs: list of OOV words (strings) in the order corresponding to their temporary article OOV ids (that have been assigned in pointer-generator mode), or None (in baseline mode)

    Returns:
        words: list of words (strings)
    """
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i) # might be [UNK]
        except ValueError as e: # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                if article_oov_idx>=len(article_oovs):
                    words.append(UNKNOWN_TOKEN+str(article_oov_idx))
                    continue
                w = article_oovs[article_oov_idx]
            except ValueError as e: # i doesn't correspond to an article oov
                raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words

def show_art_oovs(article, vocab):
    """Returns the article string, highlighting the OOVs by placing __underscores__ around them"""
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.word2id(w)==unk_token else w for w in words]
    out_str = ' '.join(words)
    return out_str


def show_abs_oovs(abstract, vocab, article_oovs):
    """Returns the abstract string, highlighting the article OOVs with __underscores__.

    If a list of article_oovs is provided, non-article OOVs are differentiated like !!__this__!!.

    Args:
        abstract: string
        vocab: Vocabulary object
        article_oovs: list of words (strings), or None (in baseline mode)
    """
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = abstract.split(' ')
    new_words = []
    for w in words:
        if vocab.word2id(w) == unk_token: # w is oov
            if article_oovs is None: # baseline mode
                new_words.append("__%s__" % w)
            else: # pointer-generator mode
                if w in article_oovs:
                    new_words.append("__%s__" % w)
                else:
                    new_words.append("!!__%s__!!" % w)
        else: # w is in-vocab word
            new_words.append(w)
    out_str = ' '.join(new_words)
    return out_str
