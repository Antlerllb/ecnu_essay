import os
import argparse
from collections import defaultdict
from torchmetrics import SacreBLEUScore
from bert_score import score
import torch
import distance
import subprocess
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertForMaskedLM
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from scorer.ChERRANT.compare_m2_for_evaluation import *
from utils import *

DATA_PATH = './datas'
BERT_PATH = '/home/xinshu/pt/bert-base-chinese'
LABEL_PATH = os.path.join(DATA_PATH,'label2id.json')

def bert_ppl(sentences):
    """bert ppl"""
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    model = BertForMaskedLM.from_pretrained(BERT_PATH)
    model = model.cuda()
    scores = []
    for s in sentences:
        input_ids = torch.tensor(tokenizer.encode(s)).unsqueeze(0).to(model.device) # [ba, seq_len]
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs[0]
        scores.append(torch.exp(loss).cpu().item())
    return sum(scores)/len(scores)

def bleu(preds, golds):
    """bleu"""
    # bleu1 = SacreBLEUScore(tokenize='zh', n_gram=1)
    # bleu2 = SacreBLEUScore(tokenize='zh', n_gram=2)
    # bleu3 = SacreBLEUScore(tokenize='zh', n_gram=3)
    bleu4 = SacreBLEUScore(tokenize='zh', n_gram=4)

    golds = [[g] for g in golds]

    # print(f"bleu1: {bleu1(preds, golds).cpu().item()*100:.2f}")
    # print(f"bleu2: {bleu2(preds, golds).cpu().item()*100:.2f}")
    # print(f"bleu3: {bleu3(preds, golds).cpu().item()*100:.2f}")
    print(f"bleu4: {bleu4(preds, golds).cpu().item()*100:.2f}")
    
    return bleu4(preds, golds)

def bert_score(preds, golds):
    """bert score"""
    P, R, F1 = score(preds, golds, verbose=True, model_type=BERT_PATH)
    return torch.mean(F1).cpu().item()

def levenshtein(preds, inputs):
    """编辑距离"""
    dist = [distance.levenshtein(p, g) for p, g in zip(preds, inputs)]
    return sum(dist)/len(dist)

def calc_metrics(tp, p, t, percent=True):
    """
    compute overall precision, recall and FB1 (default values are 0.0)
    if percent is True, return 100 * original decimal value
    """
    precision = tp / p if p else 0
    recall = tp / t if t else 0
    fb1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    if percent:
        return 100 * precision, 100 * recall, 100 * fb1
    else:
        return precision, recall, fb1

def get_result(correct_counts, true_counts, pred_counts):

    # sum counts
    sum_correct_counts = sum(correct_counts.values())
    sum_true_counts = sum(true_counts.values())
    sum_pred_counts = sum(pred_counts.values())

    chunk_types = sorted(list(set(list(true_counts) + list(pred_counts))))

    # compute overall precision, recall and FB1 (default values are 0.0)
    prec, rec, f1 = calc_metrics(sum_correct_counts, sum_pred_counts, sum_true_counts)
    res = (prec, rec, f1)

    # print overall performance, and performance per type
    print("processed %i labels; " % (sum_true_counts), end='')
    print("found: %i labels; correct: %i.\n" % (sum_pred_counts, sum_correct_counts), end='')
    print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" % (prec, rec, f1),end="")
    print("  (%d & %d) = %d" % (sum_pred_counts,sum_true_counts,sum_correct_counts))

    # for each type, compute precision, recall and FB1 (default values are 0.0)
    for t in chunk_types:
        prec, rec, f1 = calc_metrics(correct_counts[t], pred_counts[t], true_counts[t])
        print("%17s: " %t , end='')
        print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" %
                    (prec, rec, f1), end='')
        print("  (%d & %d) = %d" % (pred_counts[t],true_counts[t],correct_counts[t]))

    return res

def New_multilabel_evaluate(preds, golds, verbose=False, id2labels=None, error_only=True):

    # if error_only:
    #     gold_results = [g.cpu().tolist()[:-1] for g in golds]
    #     pred_results = [p.cpu().tolist()[:-1] for p in preds]
    # else:
    if type(golds) != list:
        gold_results = [g.cpu().tolist() for g in golds]
        pred_results = [p.cpu().tolist() for p in preds]
    else:
        gold_results = golds
        pred_results = preds

    pre, recall, macro_f1, _ = precision_recall_fscore_support(gold_results, pred_results, average='macro', zero_division=0)
    mipre, mirecall, micro_f1, _ = precision_recall_fscore_support(gold_results, pred_results, average='micro', zero_division=0)

    print(f"micro precision: {mipre*100:.2f}, micro recall: {mirecall*100:.2f}, micro f1: {micro_f1*100:.2f}")
    print(f"macro precision: {pre*100:.2f}, macro recall: {recall*100:.2f}, macro f1: {macro_f1*100:.2f}")

    if verbose:
        pre, recall, f1, _ = precision_recall_fscore_support(gold_results, pred_results, average=None, zero_division=0)
        if id2labels is None:
            return macro_f1, micro_f1
        else:
            for idx, (p, r, f) in enumerate(zip(pre, recall, f1)):
                print(f"{id2labels[idx]}: precision:{p*100:.2f}, recall:{r*100:.2f}, f1:{f*100:.2f}")
    return macro_f1, micro_f1


def MultiLabel_evaluate(pred_seqs, true_seqs):

    # label2ids = read_json(LABEL_PATH)

    correct_counts = defaultdict(int)
    true_counts = defaultdict(int)
    pred_counts = defaultdict(int)

    # macro_f1 = f1_score(true_seqs, pred_seqs, average='macro')

    # print(f"macro f1: {macro_f1}")

    for true_tag, pred_tag in zip(true_seqs, pred_seqs):
        if true_tag == []:
            true_tag = ['正确']
        if pred_tag == []:
            pred_tag = ['正确']
        for c in set(true_tag) & set(pred_tag):
            correct_counts[c] += 1
        for t in true_tag:
            true_counts[t] += 1
        for p in pred_tag:
            pred_counts[p] += 1
    
    results = get_result(correct_counts, true_counts, pred_counts)
    return results

def preprocess(para_file, m2_file, granularity='char'):
    """
        ref_para: gold的para文件
        hyp_para: pred的para文件
    """
    result = subprocess.check_output([
            "python",
            "scorer/ChERRANT/parallel_to_m2.py",
            "-f",
            para_file,
            "-o",
            m2_file,
            "-g",
            granularity,
            # "-d",
            # "7"
        ]).decode().split()
    print(f"process success:{result}")

def cherrant(hyp, ref, args):
    """
        hyp: hyp path
        ref: ref path
    """
    hyp_m2 = open(hyp).read().strip().split("\n\n")
    ref_m2 = open(ref).read().strip().split("\n\n")
    print(f"len ref_m2:{len(ref_m2)}, len hyp_m2: {len(hyp_m2)}")
    # Make sure they have the same number of sentences
    ref_m2 = ref_m2[:len(hyp_m2)]
    assert len(hyp_m2) == len(ref_m2), print(len(hyp_m2), len(ref_m2))

    # Store global corpus level best counts here
    best_dict = Counter({"tp":0, "fp":0, "fn":0})
    best_cats = {}
    # Process each sentence
    sents = zip(hyp_m2, ref_m2)
    for sent_id, sent in enumerate(sents):
        # Simplify the edits into lists of lists
        # if "A1" in sent[0] or "A1" in sent[1] or sent_id in sent_id_cons:
        #     sent_id_cons.append(sent_id)
        src = sent[0].split("\n")[0]
        hyp_edits = simplify_edits(sent[0], None)
        ref_edits = simplify_edits(sent[1], None)
        # Process the edits for detection/correction based on args
        hyp_dict = process_edits(hyp_edits, args)
        ref_dict = process_edits(ref_edits, args)
        
        # if  args.reference_num is None or len(ref_dict.keys()) == args.reference_num:
        # Evaluate edits and get best TP, FP, FN hyp+ref combo.
        count_dict, cat_dict = evaluate_edits(src,
            hyp_dict, ref_dict, best_dict, sent_id, args)
        # Merge these dicts with best_dict and best_cats
        best_dict += Counter(count_dict)
        best_cats = merge_dict(best_cats, cat_dict)
    # Print results
    return print_results(best_dict, best_cats, args)

def em_eval(true_seqs, pred_seqs):
    correct_num = 0
    for g, p in zip(true_seqs, pred_seqs):
        if g == p:
            correct_num += 1
    return correct_num / len(true_seqs)

def rewrite_eval(args, true_seqs, pred_seqs, sent_ids, save_path, data_path=None, raw_inputs=None):
    """
    args:
        ChERRANT的evaluation
        true_seqs: list，gold句子的list
        pred_seqs: list，pred句子的list
        sent_ids: list， 句子的id
        save_path: pred的成对句子（正确句子和错误句子）要保存的路径
        data_path: gold file的路径
        raw_inputs: 原始输入：true_seqs、pred_seqs、raw_inputs长度相同
    """
    filename = data_path + '.ref.para'
    
    if data_path == None:
        error_ids = []
    else:
        datas = read_json(data_path)
        error_ids = {d['sent_id']:d for d in datas} # if d['errorType'] != ['正确']}
        if not os.path.exists(filename):
            # 如果文件不存在，则要创建.ref.para文件
            convert_mydatas2para(filename, datas)
            # process(filename, filename.replace('.ref.para', '.ref.m2'))

    R = []
    for idx, (pred, true, sent_id) in enumerate(zip(pred_seqs, true_seqs, sent_ids)):
        if not sent_id in error_ids.keys():
            continue
        if raw_inputs == None:
            R.append(f'{sent_id}\t{error_ids[sent_id]["sent"]}\t{pred}')
        else:
            if pred == '':
                print("pred null")
            R.append(f'{sent_id}\t{raw_inputs[idx]}\t{pred}')
    save_txt(save_path, R)

    char_ref = filename.replace('.ref.para', '.ref.m2.char')
    char_hyp = save_path.replace('.hyp.para', '.hyp.m2.char')

    word_ref = filename.replace('.ref.para', '.ref.m2.word')
    word_hyp = save_path.replace('.hyp.para', '.hyp.m2.word')

    if not os.path.exists(char_ref) and not os.path.exists(word_ref):
        preprocess(filename, char_ref, granularity='char')
        preprocess(filename, word_ref, granularity='word')
    
    preprocess(save_path, char_hyp, granularity='char')
    preprocess(save_path, word_hyp, granularity='word')

    char_pre, char_recall, char_f = cherrant(char_hyp, char_ref, args)

    # word_pre, word_recall, word_f = cherrant(word_hyp, word_ref, args)
    
    return char_f*100, 0


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--preds", type=str, default=os.path.join(DATA_PATH, 'dev_r.json'), help="pred data path")
    parser.add_argument("--golds", type=str, default=os.path.join(DATA_PATH, 'dev_r.json'), help="gold data path")

    args = parser.parse_args()

    preds = read_json(args.preds)
    golds = read_json(args.golds)

    pred_tags = [p['revisedSent'] for p in preds]

    gold_tags = [g['revisedSent'] for g in golds if g['errorType']!=['正确']]
    # for g in golds:
    #     if g['errorType']==['正确']:
    #         continue
    #     gold_tags.append(g['revisedSent'])

    input_tags = [g['rawText'] for g in golds if g['errorType']!=['正确']]

    l_s = levenshtein(gold_tags, input_tags)
    print(f"数据集中revisedSent与rawText的编辑距离： {l_s}")