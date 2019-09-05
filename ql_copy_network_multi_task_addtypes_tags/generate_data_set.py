import collections
import random

datadir= r"D:\data\seq2seq\MSPAD5W\trainning_data\patterns_0528\head_position"

train_f = open(datadir+r"\train.txt.tags",encoding="utf-8").readlines()
valid_f = open(datadir+r"\dev.txt.tags",encoding="utf-8").readlines()
test_f = open(datadir+r"\test.txt.tags",encoding="utf-8").readlines()
#train_f +=valid_f+test_f
train_out_f = open("train.txt","w",encoding="utf-8")
valid_out_f = open("validation.txt","w",encoding="utf-8")
test_out_f = open("input.txt","w",encoding="utf-8")

source_min = 0
target_min = 0
vocat_fenge = "\t"
import re

predicate_regrex = "(r-mso|mso|dev|r-dev):.*?\.(.)+"

def generate_vocab(train_f,word_size):
    counter = collections.Counter()

    predict = {}
    for line in train_f:
        #line = str(line).strip().lower().split('\t')
        line = str(line).strip().split('\t')
        line = line[0].split()+line[1].split()

        for w in line:
            w = w.strip()
            counter[w] += 1

            #if re.match("ns:.*?\..*?\.(.)+", w):
            if re.match(predicate_regrex,w):
                predict[w] =1

    counter = counter.most_common(word_size)

    vocab = open("vocab","w",encoding="utf-8")

    for i, w in enumerate(counter):
        if w[0] in predict:
            vocab.write("{} {}\n".format(w[0],w[1]))
        else:
            if w[1]>=source_min:
                vocab.write("{} {}\n".format(w[0], w[1]))
    vocab.close()

def generate_grammar_words(train_f,word_size):
    counter = collections.Counter()
    grammar = {}
    for line in train_f:
        #line = str(line).strip().lower().split('\t')
        line = str(line).strip().split('\t')
        logic = line[1].split()

        for w in logic:
            if not re.match("<http://.*?>", w):
                counter[w] = 10

    counter = counter.most_common(word_size)

    vocab = open("grammar", "w", encoding="utf-8")
    for i, w in enumerate(counter):
        vocab.write("{}{}{}\n".format(w[0],vocat_fenge, w[1]))

def generate_pos_words(train_f,word_size):
    counter = collections.Counter()
    grammar = {}
    for line in train_f:
        #line = str(line).strip().lower().split('\t')
        line = str(line).strip().split('\t')
        logic = line[2].split()

        for w in logic:
            counter[w] +=1

    counter = counter.most_common(word_size)

    vocab = open("pos_tag", "w", encoding="utf-8")
    for i, w in enumerate(counter):
        vocab.write("{}{}{}\n".format(w[0],vocat_fenge, w[1]))


def generate_predicate_vocab(train_f,word_size):
    counter = collections.Counter()
    predict = {}
    for line in train_f:
        #line = str(line).strip().lower().split('\t')
        line = str(line).strip().split('\t')
        predicate = line[1].split()

        line = line[0].split()

        for w in predicate:
            #if not re.match("ns:.*?\..*?\.(.)+", w):
            if re.match(predicate_regrex, w):
                w =w.split(':')[1]
                w_all = w.split('.')
                for tmp in w_all:
                    counter[tmp] += target_min

    counter = counter.most_common(word_size)

    vocab = open("grammer", "w", encoding="utf-8")

    for i, w in enumerate(counter):
        if w[1] >= target_min:
            vocab.write("{}{}{}\n".format(w[0],vocat_fenge, w[1]))
    vocab.close()


import re
def generate_target_vocab(train_f,word_size):
    counter = collections.Counter()
    predict = {}
    for line in train_f:
        #line = str(line).strip().lower().split('\t')
        line = str(line).strip().split('\t')
        line = line[1].split()

        for w in line:
            w = w.strip()
            if re.match(predicate_regrex,w):
            #if "http://dbpedia.org" in w:
                predict[w] = 1
                counter[w] = 1
            else:
                w = w.strip()
                counter[w] += 1



    counter = counter.most_common(word_size)

    vocab = open("vocab.out", "w", encoding="utf-8")

    print(len(predict))
    for i, w in enumerate(counter):
        if w[0] in predict:
            vocab.write("{}{}{}\n".format(w[0], vocat_fenge, w[1]))
        else:
            if w[1] >= target_min:
                vocab.write("{}{}{}\n".format(w[0], vocat_fenge, w[1]))
    vocab.close()

def generate_target_vocab_pre_only(train_f,word_size):
    counter = collections.Counter()
    predict = {}
    for line in train_f:
        #line = str(line).strip().lower().split('\t')
        line = str(line).strip().split('\t')
        line = line[1].split()

        for w in line:
            w = w.strip()
            if not re.match("(r-mso|mso):.*?\..*?\.(.)+", w):
                counter[w] += 1

            if re.match("(r-mso|mso):.*?\..*?\.(.)+",w):
                predict[w] = 1

    counter = counter.most_common(word_size)

    vocab = open("vocab.out", "w", encoding="utf-8")

    for i, w in enumerate(counter):
        if w[1] >= target_min:
            vocab.write("{}{}{}\n".format(w[0], vocat_fenge, w[1]))

    for w in predict:
        vocab.write("{}{}{}\n".format(w[0], vocat_fenge, w[1]))
    vocab.close()

def generate_source_vocab(train_f,word_size):
    counter = collections.Counter()
    predict = {}
    for line in train_f:
        #line = str(line).strip().lower().split('\t')
        line = str(line).strip().split('\t')
        line = line[0].split()

        for w in line:
            w = w.strip()
            counter[w] += 1


    counter = counter.most_common(word_size)

    vocab = open("vocab.in", "w", encoding="utf-8")

    for i, w in enumerate(counter):
        if w[1] >= source_min:
            vocab.write("{}{}{}\n".format(w[0], vocat_fenge, w[1]))
    vocab.close()


def append_target_subwords():
    vocab = open("vocab.in", encoding="utf-8")
    word2count = {}
    for line in vocab:
        word,cnt = line.strip().split()
        word2count[word] = cnt
    vocab = open("vocab.in_new", "w", encoding="utf-8")
    vocab_out = open("vocab.out", encoding="utf-8")

    for line in vocab_out:
        w,cnt = line.strip().split()

        if re.match(predicate_regrex, w):
            #subwords = w.split('.')[-1].split('_')
            subwords = [w.split('.')[1]] + w.split('.')[-1].split('_')
            for sub in subwords:
                word2count[sub]=100

        '''if re.match("<http://.*?>", w):
            w = w.replace('<http://dbpedia.org/','').replace('>','')
            subwords = w.split('/')
            for sub in subwords:
                word2count[sub] = 100'''
    for key in word2count.keys():
        vocab.write(key+'\t'+str(word2count[key])+'\n')



def generate_training_file(train_f,train_out_f):
    random.shuffle(train_f)

    for line in train_f:
        #line =  str(line).strip().lower().split("\t")
        line = str(line).strip().split("\t")
        line[1] = ' '.join(line[1].strip().split())
        train_out_f.write("{}\t{}\n".format(line[0],line[1]))




#generate_training_file(train_f,train_out_f)
#generate_training_file(valid_f,valid_out_f)
#generate_training_file(test_f,test_out_f)
import re
_SPLIT = re.compile("([,.])")
def generate_fresh_data(train_f,train_out_f):
    predicate_pattern = "(r-mso|mso):.*?\..*?\.(.)+"
    for j,line in enumerate(train_f):
        q,l = str(line).strip().lower().split("\t")[:2]
        q = ' '.join(_SPLIT.split(q))

        q = q.replace("’s", " 's")
        q = q.replace("'s", " 's")

        q = ' '.join(q.strip().split())

        l = l.strip().split()
        l_words = []
        for i,w in enumerate(l):
            if re.match(predicate_pattern,w) or w =="’s" or w =="'s":
                l_words.append(w)
            else:
                w = " ".join(_SPLIT.split(w))
                w = " _||_ ".join(w.split())
                w = w.replace("’s"," _||_ 's")
                w = w.replace("'s", " _||_ 's")
                l_words.append(w)

        l = ' '.join(l_words)
        train_out_f.write("{}\t{}\n".format(q, l))


def reverse_seq2seq_data(name):

    data = open(name,encoding="utf-8")
    reverse_data = open(name+".rev","w", encoding="utf-8")

    for line in data:
        line = line.strip().split('\t')
        reverse_data.write(line[1]+'\t'+line[0]+'\n')

#reverse_seq2seq_data(r"D:\data\seq2seq\MSPaD.Merge\MSPaD\data_dir_lower\all_predict\new_fresh_fix_s\input.txt")
#reverse_seq2seq_data(r"D:\data\seq2seq\MSPaD.Merge\MSPaD\data_dir_lower\all_predict\new_fresh_fix_s\validation.txt")
#reverse_seq2seq_data(r"D:\data\seq2seq\MSPaD.Merge\MSPaD\data_dir_lower\all_predict\new_fresh_fix_s\train.txt")

#generate_grammar_words(train_f,50000)
#append_target_subwords()

#generate_training_file(train_f,train_out_f)
#generate_training_file(valid_f,valid_out_f)
#generate_training_file(test_f,test_out_f)

'''
generate_vocab(train_f,50000)
generate_target_vocab(train_f,50000)
generate_source_vocab(train_f,50000)
append_target_subwords()
generate_pos_words(train_f,50000)'''


#generate_predicate_vocab(train_f,10000)
#generate_predicate_vocab(train_f,50000)'''

def sample_data(inname):
    in_f = open(inname,encoding="utf-8").readlines()

    cnt = len(in_f)

    import random

    for i in range(2,10,2):
        part = int(cnt/10*i)
        sample_data = random.sample(in_f,part)

        out_f = open(inname+str(i),"w",encoding="utf-8")

        for line in sample_data:
            out_f.write(line)


sample_data(r"D:\data\seq2seq\complexwebquestions\new_fresh_0809\complex.train.fresh_new.tags.real_tag")