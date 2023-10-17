from sacrebleu.metrics import BLEU, CHRF, TER
from rouge import Rouge
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from easse.sari import corpus_sari
from mosestokenizer import *

bleu = BLEU(tokenize='zh')
def cal_bleu(bleu_model,output_texts, ref_texts):
    bleu_score = bleu_model.corpus_score(output_texts, ref_texts).score
    return bleu_score

def cal_cls_score(pred_list, label_list,metric='acc'):
    pred_list = [p.lower() for p in pred_list]
    label_list = [l.lower() for l in label_list]
    if metric == 'f1':
        score = f1_score(label_list, pred_list, average='macro')
    elif metric == 'acc':
        score = accuracy_score(label_list, pred_list)
    return score

def cal_rouge(output_texts, ref_texts):
    print("calculating rouge score...")
    print(output_texts)
    print(ref_texts)
    rouge = Rouge()
    output_texts = [" ".join(MosesTokenizer('en')(sent)) for sent in output_texts]
    ref_texts = [" ".join(MosesTokenizer('en')(sent)) for sent in ref_texts]
    scores = rouge.get_scores(output_texts, ref_texts, avg=True)
    print(scores)
    return scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f'] 

def cal_sari(orig_sents, sys_sents, refs_sents):
    sari = corpus_sari(orig_sents=orig_sents,  
                sys_sents=sys_sents, 
                refs_sents=refs_sents)
    return sari

if __name__ == '__main__':
    from utils import load_sim_data, read_lines
    _,_, src, tgt = load_sim_data('asset', 5)
    sys = read_lines('../../data/sim/asset/test/refine.5.post')
    print(cal_sari(src, sys, tgt))