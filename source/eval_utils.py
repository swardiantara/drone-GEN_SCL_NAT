# -*- coding: utf-8 -*-

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


def f1_score(n_tp, n_gold, n_pred):
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    if recall > 1.0:
        import pdb
        pdb.set_trace()
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    
def compute_f1_scores(pred_pt, gold_pt, silent=True):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    quad_tp, n_gold, n_pred = 0, 0, 0
    tp_ac, tp_at, tp_ot, tp_sp = 0, 0, 0, 0

    # loop over all samples
    for i in range(len(gold_pt)):
        n_gold += len(gold_pt[i])   # num of quads in gold sample
        n_pred += len(pred_pt[i])   # num of quads in pred sample
        
        # loop over quads in gold labels
        for j in range(len(gold_pt[i])):
            gold_quad = gold_pt[i][j]       # check if the quad exists in pred_quad
            if gold_quad in pred_pt[i]:     # count misordered quad as TP
                quad_tp += 1                # strict quad-level true positive
            
            # prevent out of range index error
            # count per-element TP
            # the quad order must be the correct
            if j < len(pred_pt[i]):
                gold_ac, gold_at, gold_sp, gold_ot = gold_quad
                pred_ac, pred_at, pred_sp, pred_ot = pred_pt[i][j]
                tp_ac += 1 if pred_ac == gold_ac else 0
                tp_at += 1 if pred_at == gold_at else 0
                tp_ot += 1 if pred_ot == gold_ot else 0
                tp_sp += 1 if pred_sp == gold_sp else 0

    if not silent:
        print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {quad_tp}")
        
    # compute F1-score
    quad_scores = f1_score(quad_tp, n_gold, n_pred)
    ac_scores = f1_score(tp_ac, n_gold, n_pred)
    at_scores = f1_score(tp_at, n_gold, n_pred)
    ot_scores = f1_score(tp_ot, n_gold, n_pred)
    sp_scores = f1_score(tp_sp, n_gold, n_pred)
    total_tp = tp_ac + tp_at + tp_ot + tp_sp
    micro_average = f1_score(total_tp, (n_gold * 4),  (n_pred * 4))
    if not silent:
        print(f"number of gold quads: {n_gold}, predicted quads: {n_pred}, hit: {quad_tp}")
        print(f'Quad-level eval: {quad_scores}')
        print(f'Element-level eval: {micro_average}')

    return {
        'quad_score': quad_scores,
        'ac_score': ac_scores,
        'at_score': at_scores,
        'ot_score': ot_scores,
        'sp_score': sp_scores,
        'micro_average': micro_average
    }


def compute_scores(pred_seqs, gold_seqs, task, silent=False):
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


def compute_gen_metrics(pred_sents, gold_sents, silent=False):
    bleu = evaluate.load('bleu')
    bleu_score = bleu.compute(predictions=pred_sents, references=gold_sents)

    wer = evaluate.load('wer')
    wer_score = wer.compute(predictions=pred_sents, references=gold_sents)

    rouge = evaluate.load('rouge')
    rouge_score = rouge.compute(predictions=pred_sents, references=gold_sents)

    return {
        'bleu_score': bleu_score,
        'wer_score': wer_score,
        'wer_score': rouge_score
    }