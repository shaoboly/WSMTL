from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction


def bleu_score(list_of_references,predictions):
    return corpus_bleu(list_of_references, predictions,smoothing_function=SmoothingFunction(epsilon=0.01).method1) * 100

def compute_acc(list_of_references,predictions):
    cnt=0.0
    for i,output in enumerate(predictions):
        refer = list_of_references[i][0]
        if refer==output:
            cnt+=1
    return cnt/len(predictions) *100