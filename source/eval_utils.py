# -*- coding: utf-8 -*-

from collections import Counter

sentiment_word_list = ['positive', 'negative', 'neutral']
opinion2word = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}
opinion2word_under_o2m = {'good': 'positive', 'great': 'positive', 'best': 'positive',
                          'bad': 'negative', 'okay': 'neutral', 'ok': 'neutral', 'average': 'neutral'}

numopinion2word = {'SP1': 'positive', 'SP2': 'negative', 'SP3': 'neutral'}
import parse
import evaluate

def extract_spans_para(task, absa_task, seq, seq_type):
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
                    if absa_task == 'quad':
                        result = list(parse.parse('{0} is {1} because {2} is {3}', s, case_sensitive=True))
                        ac, sp, at, ot = [elt.strip(' ') for elt in result]
                    elif absa_task == 'tasd':
                        result = list(parse.parse('{0} is {1} because {2} is {3}', s, case_sensitive=True))
                        ac, sp, at, ot = [elt.strip(' ') for elt in result]
                    elif absa_task == 'aste':
                        result = list(parse.parse('it is {0} because {1} is {2}', s, case_sensitive=True))
                        sp, at, ot =[elt.strip(' ') for elt in result]
                        ac = 'NULL'
                    else:
                        raise NotImplementedError
                    # ac, sp, at, ot = result[0], result[1], result[2], result[3].lstrip(' ')
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
    tp_ac, tp_at, tp_ot, tp_sp, tp_acsp = 0, 0, 0, 0, 0

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
                tp_acsp += 1 if ((pred_ac == gold_ac) and (pred_sp == gold_sp)) else 0

    if not silent:
        print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {quad_tp}")
        
    # compute F1-score
    quad_scores = f1_score(quad_tp, n_gold, n_pred)
    acsp_scores = f1_score(tp_acsp, n_gold, n_pred)
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
        'acsp_score': acsp_scores,
        'at_score': at_scores,
        'ot_score': ot_scores,
        'sp_score': sp_scores,
        'micro_average': micro_average
    }


def _prf(tp, n_gold, n_pred):
    precision = float(tp) / float(n_pred) if n_pred else 0.0
    recall = float(tp) / float(n_gold) if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _match_counts(pred_quads, gold_quads, multiset):
    """
    Computes true-positive/gold/pred counts for one example, either under
    set semantics (duplicates collapsed) or multiset/bag semantics
    (duplicates counted with multiplicity, via Counter intersection).
    """
    if multiset:
        pred_counter = Counter(pred_quads)
        gold_counter = Counter(gold_quads)
        tp = sum((pred_counter & gold_counter).values())
        n_pred = sum(pred_counter.values())
        n_gold = sum(gold_counter.values())
    else:
        pred_set = set(pred_quads)
        gold_set = set(gold_quads)
        tp = len(pred_set & gold_set)
        n_pred = len(pred_set)
        n_gold = len(gold_set)
    return tp, n_gold, n_pred


def compute_prf_averaged(pred_pt, gold_pt, multiset=False, silent=True):
    """
    Computes micro/macro/weighted precision, recall and F1 over a list of
    per-example predicted/gold quadruple lists.

    - set-based (multiset=False): each distinct quadruple within an example
      counts once, regardless of how many times it occurs. This under-counts
      examples with repeated gold quadruples (see source/analyze_duplicates.py).
    - multiset-based (multiset=True): quadruples are matched with
      multiplicity (bag/multiset intersection via collections.Counter), so
      repeated gold quadruples must be predicted the matching number of
      times to all count as true positives.

    micro:    global TP/gold/pred counts aggregated across all examples,
              then P/R/F1 computed once.
    macro:    per-example P/R/F1 computed independently, then unweighted
              mean across examples.
    weighted: per-example P/R/F1 averaged, weighted by each example's
              number of gold quadruples (support).
    """
    n = len(pred_pt)
    total_tp = total_gold = total_pred = 0
    per_example = []

    for i in range(n):
        tp, n_gold, n_pred = _match_counts(pred_pt[i], gold_pt[i], multiset)
        total_tp += tp
        total_gold += n_gold
        total_pred += n_pred
        p, r, f = _prf(tp, n_gold, n_pred)
        per_example.append((p, r, f, n_gold))

    micro_p, micro_r, micro_f = _prf(total_tp, total_gold, total_pred)

    if per_example:
        macro_p = sum(p for p, r, f, s in per_example) / len(per_example)
        macro_r = sum(r for p, r, f, s in per_example) / len(per_example)
        macro_f = sum(f for p, r, f, s in per_example) / len(per_example)
    else:
        macro_p = macro_r = macro_f = 0.0

    total_support = sum(s for p, r, f, s in per_example)
    if total_support > 0:
        weighted_p = sum(p * s for p, r, f, s in per_example) / total_support
        weighted_r = sum(r * s for p, r, f, s in per_example) / total_support
        weighted_f = sum(f * s for p, r, f, s in per_example) / total_support
    else:
        weighted_p = weighted_r = weighted_f = 0.0

    scores = {
        'micro': {'precision': micro_p, 'recall': micro_r, 'f1': micro_f},
        'macro': {'precision': macro_p, 'recall': macro_r, 'f1': macro_f},
        'weighted': {'precision': weighted_p, 'recall': weighted_r, 'f1': weighted_f},
        'support': {'n_gold': total_gold, 'n_pred': total_pred, 'n_tp': total_tp, 'n_examples': n},
    }

    if not silent:
        kind = 'multiset' if multiset else 'set'
        print(f"[{kind}] gold={total_gold} pred={total_pred} tp={total_tp}")
        print(scores)

    return scores


def compute_scores_from_quads(all_preds, all_labels, silent=True):
    """
    Computes both set-based and multiset-based (micro/macro/weighted) scores
    given already-extracted per-example predicted/gold quadruple lists. This
    is purely additive: it does not alter compute_f1_scores's own (positional,
    element-level) metrics.
    """
    return {
        'set': compute_prf_averaged(all_preds, all_labels, multiset=False, silent=silent),
        'multiset': compute_prf_averaged(all_preds, all_labels, multiset=True, silent=silent),
    }


# quad tuple layout used throughout this file and extract_spans_para: (ac, at, sp, ot)
ELEMENT_INDEX = {'ac': 0, 'at': 1, 'sp': 2, 'ot': 3}


def compute_element_scores(all_preds, all_labels, silent=True):
    """
    Content-matched (multiset/bag) per-element scores for ac/at/sp/ot,
    computed independently of full-quad matching: for each example, the
    predicted and gold values of one element (e.g. just the sentiment
    labels) are compared as a multiset via compute_prf_averaged, the same
    Counter-intersection machinery compute_scores_from_quads uses for full
    quads. A predicted element value counts as a true positive as soon as
    that value also appears (with matching multiplicity) among the gold
    values for that element in the same example -- it does not need to come
    from a quad that also matches on the other three elements.

    This replaces the earlier ac_score/at_score/ot_score/sp_score in
    compute_f1_scores, which aligned gold_quads[j] against pred_quads[j] by
    list position: whenever the model predicts a different number of quads
    than gold, or predicts the same quads in a different order (both common
    for a generative model with no guaranteed output order), that positional
    alignment can score an element as wrong even though the correct value
    was predicted -- just in a different slot. This function is not
    position-sensitive, so it isn't subject to that failure mode. Only
    multiset (bag) matching is provided, not set-based, since a repeated
    element value should require matching multiplicity to be fully credited
    (see the multiset justification in compute_scores_from_quads/
    source/analyze_duplicates.py) -- e.g. two independent "negative"
    quadruples in one message are two problems, not one.
    """
    scores = {}
    for name, idx in ELEMENT_INDEX.items():
        pred_values = [[quad[idx] for quad in quads] for quads in all_preds]
        gold_values = [[quad[idx] for quad in quads] for quads in all_labels]
        scores[name] = compute_prf_averaged(pred_values, gold_values, multiset=True, silent=silent)
    return scores


def compute_scores(pred_seqs, gold_seqs, task, absa_task, silent=False):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):
        gold_list = extract_spans_para(task, absa_task, gold_seqs[i], 'gold')
        pred_list = extract_spans_para(task, absa_task, pred_seqs[i], 'pred')

        all_labels.append(gold_list)
        all_preds.append(pred_list)

    if not silent:
        print("\nResults:")
        scores = compute_f1_scores(all_preds, all_labels, silent)
        print(scores)
    else:
        scores = compute_f1_scores(all_preds, all_labels, silent)

    # additive: set-based and multiset-based (bag) precision/recall/F1, with
    # micro/macro/weighted averaging, justified by source/analyze_duplicates.py
    scores['set_multiset_scores'] = compute_scores_from_quads(all_preds, all_labels, silent)

    # additive: content-matched (multiset) per-element scores -- see
    # compute_element_scores's docstring for why this replaces the
    # position-aligned ac_score/at_score/ot_score/sp_score above
    scores['element_scores'] = compute_element_scores(all_preds, all_labels, silent)

    return scores, all_labels, all_preds


def compute_gen_metrics(pred_sents, gold_sents, silent=False):
    bleu = evaluate.load('bleu')
    bleu_score = bleu.compute(predictions=pred_sents, references=gold_sents)

    wer = evaluate.load('wer')
    wer_score = wer.compute(predictions=pred_sents, references=gold_sents)

    rouge = evaluate.load('rouge')
    rouge_score = rouge.compute(predictions=pred_sents, references=gold_sents)
    
    meteor = evaluate.load('meteor')
    meteor_score = meteor.compute(predictions=pred_sents, references=gold_sents)

    return {
        'bleu_score': bleu_score,
        'wer_score': wer_score,
        'rouge_score': rouge_score,
        'meteor_score': meteor_score
    }
