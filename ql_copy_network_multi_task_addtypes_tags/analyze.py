def check_candidate_matching(gold_p,candidate,topk):
    find = True
    for i in range(len(gold_p)):
        if len(gold_p)!=len(candidate):
            return False
        if gold_p[i] not in candidate[i][:topk]:
            return False
    return find

def candidate_check():
    import re
    in_f1 = open(r"D:\code\ABCNN-master\data\eval\test_candidate_CE.txt", encoding="utf-8").readlines()
    out_f = open("tmp","w",encoding="utf-8")
    cnt = [0.0 for i in range(10)]

    for i,line in enumerate(in_f1):
        if len(line.strip().split('\t'))<4:
            print(line)
            continue
        q,r,f,c =  line.strip().split('\t')[:4]
        candidates = [item.split("|||") for item in c.split('_||_')]
        gold_p = []
        predicate_regrex = "(r-mso|mso|dev|r-dev):.*?\.(.)+"
        for word in r.split():
            if re.match(predicate_regrex, word):
                gold_p.append(word)

        for topk in range(0,10):
            if check_candidate_matching(gold_p,candidates,topk+1):
                cnt[topk]+=1
        if not check_candidate_matching(gold_p, candidates, 1):
            out_f.write(line)

    for topk in range(0, 10):
        print(cnt[topk]/len(in_f1))


def change_mso(predicate):
    predicate = str(predicate)
    if predicate[0]=='r':
        predicate = predicate.replace("r-","")
        predicate=predicate.split('.')
        tmp = predicate[1]
        predicate[1] = predicate[2]
        predicate[2] = tmp
        predicate = '.'.join(predicate)
        return predicate
    else:
        return predicate

def coverage():
    in_f1 = open(r"Satori.TypeTriples.tsv", encoding="utf-8").readlines()
    predicates = []
    for line in in_f1:
        predicates.append(line.strip().split('\t')[1])
    predicates = list(set(predicates))

    my_file = open(r"D:\data\seq2seq\MSPAD5W\predicate.list", encoding="utf-8").readlines()
    all_predicates = []
    overall = len(my_file)
    cnt = 0.0
    for line in my_file:
        line = line.strip().split()[0]

        if line in predicates:
            cnt+=1
        else:
            print(line)
    print(cnt/overall)


def compute_beam_precision(inname,topk = 20):
    in_f1 = open(inname).readlines()

    cnt = 0
    total = 0.0

    result = {}

    for line in in_f1:
        q,r,f = line.strip().split('\t')[:3]

        if q not in result:
            result[q] = ["",[]]
            result[q][0] = r

        #result[q][1].append(f)
        '''if "[unk]" in f or "[UNK]" in f:
            continue'''
        result[q][1].append(f)

    for key in result.keys():
        r, candidates = result[key]
        total += 1
        candidates = candidates[:topk]
        if r in candidates:
            cnt+=1

    print(cnt/total)


compute_beam_precision(r"D:\code\ABCNN-master\data\eval\match_beam.txt",32)

for topk in [1,2,3,5,10,20,30]:
    print(topk)
    compute_beam_precision(r"D:\code\ABCNN-master\data\eval\match_beam.txt", topk)

#candidate_check()
#coverage()

