# -*- coding: utf-8 -*-
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from meteor_score import meteor
from pycocoevalcap.eval import COCOEvalCap

sentiment_word_list = ['positive', 'negative', 'neutral']
opinion2word = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}
opinion2word_under_o2m = {'good': 'positive', 'great': 'positive', 'best': 'positive',
                          'bad': 'negative', 'okay': 'neutral', 'ok': 'neutral', 'average': 'neutral'}

numopinion2word = {'SP1': 'positive', 'SP2': 'negative', 'SP3': 'neutral'}
import parse

def extract_spans_para(task, seq, seq_type):
    quads = []
    sents = [s.strip() for s in seq.split('[SSEP]')]

    if task == 'asqp' or task == 'gen_scl_nat_wo_intra':
        for s in sents:
            # food quality is bad because pizza is over cooked.
            try:
                ac_sp, at_ot = s.split(' because ')
                ac, sp = ac_sp.split(' is ')
                at, ot = at_ot.split(' is ')

                # if the aspect term is implicit
                if at.lower() == 'it':
                    at = 'NULL'
            except ValueError:
                try:
                    # print(f'In {seq_type} seq, cannot decode: {s}')
                    pass
                except UnicodeEncodeError:
                    # print(f'In {seq_type} seq, a string cannot be decoded')
                    pass
                try:
                    result = list(parse.parse('{0} is {1} because {2} is{3}', s, case_sensitive=True))
                    ac, sp, at, ot = result[0], result[1], result[2], result[3].lstrip(' ')
                except:
                    print(s)
                    ac, at, sp, ot = '', '', '', ''

            quads.append((ac, at, sp, ot))


    elif task.startswith('gen_scl_nat'):
        if seq:
            for s in sents:
                # food quality is bad because pizza is over cooked.
                try:
                    result = list(parse.parse('{0}|{1}|{2}|{3}', s, case_sensitive=True))
                    ac, at, ot, sp = [elt.strip(' ') for elt in result]
                    if at.lower == 'it':
                        at = 'NULL'
                except:
                    try:
                        result = list(parse.parse('{0}| THE{1}IS{2}|{3}', s, case_sensitive=True))
                        ac, at, ot, sp = [elt.strip(' ') for elt in result]
                        if at.lower == 'it':
                            at = 'NULL'
                    except:
                        print(s)
                        ac = ''
                        sp = ''
                        at = 'NULL'
                        ot = 'NULL'

                quads.append((ac, at, sp, ot))

    else:
        raise NotImplementedError
    return quads
    
def compute_f1_scores(pred_pt, gold_pt, silent=True):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(set(gold_pt[i]))
        n_pred += len(set(pred_pt[i]))

        for t in set(pred_pt[i]):
            if t in gold_pt[i]:
                n_tp += 1

    if not silent:
        print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    if recall > 1.0:
        import pdb
        pdb.set_trace()
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores


def compute_scores(pred_seqs, gold_seqs, task, silent=True):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):
        gold_list = extract_spans_para(task, gold_seqs[i], 'gold')
        pred_list = extract_spans_para(task, pred_seqs[i], 'pred')

        all_labels.append(gold_list)
        all_preds.append(pred_list)

    if not silent:
        print("\nResults:")
        scores = compute_f1_scores(all_preds, all_labels, silent)
        print(scores)
    else:
        scores = compute_f1_scores(all_preds, all_labels, silent)

    return scores, all_labels, all_preds


def compute_gen_metrics(pred_sentences, true_sentences, silent=True):
    """Computes overall BLEU, ROUGE, METEOR, and CIDEr scores for a list of sentences.

    Args:
        pred_sentences (list[str]): A list of predicted sentences.
        true_sentences (list[str]): A list of true sentences.
        silent (bool=True): Whether not to print the score
    Returns:
        dict: A dictionary containing the overall scores.
    """

    # BLEU
    bleu_score = corpus_bleu([[true_sentence] for true_sentence in true_sentences], pred_sentences)

    # ROUGE
    rouge = Rouge()
    scores = rouge.get_scores(pred_sentences, true_sentences)
    rouge_1 = sum(score["rouge-1"]["f"] for score in scores) / len(scores)
    rouge_2 = sum(score["rouge-2"]["f"] for score in scores) / len(scores)
    rouge_l = sum(score["rouge-l"]["f"] for score in scores) / len(scores)

    # METEOR
    meteor_scores = [meteor(pred_sentence, true_sentence) for pred_sentence, true_sentence in zip(pred_sentences, true_sentences)]
    meteor_score = sum(meteor_scores) / len(meteor_scores)

    # CIDEr
    coco_caps = [{"caption": pred_sentence} for pred_sentence in pred_sentences]
    coco_refs = [{"caption": true_sentence} for true_sentence in true_sentences]
    coco_eval = COCOEvalCap(coco_caps, coco_refs)
    coco_eval.evaluate()
    cider_score = coco_eval.eval['CIDEr']

    scores = {
        "bleu": bleu_score,
        "rouge_1": rouge_1,
        "rouge_2": rouge_2,
        "rouge_l": rouge_l,
        "meteor": meteor_score,
        "cider": cider_score
    }

    if not silent:
        print("\nGenerative Evaluation Results:")
        print(scores)

    return scores

