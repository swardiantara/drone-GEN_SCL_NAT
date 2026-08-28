# -*- coding: utf-8 -*-

from collections import Counter

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
      counts once, regardless of how many times it occurs (this is the
      standard ABSA-QUAD/GEN-SCL-NAT evaluation, and under-counts examples
      with repeated gold quadruples).
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
    given already-extracted per-example predicted/gold quadruple lists.
    """
    scores = {
        'set': compute_prf_averaged(all_preds, all_labels, multiset=False, silent=silent),
        'multiset': compute_prf_averaged(all_preds, all_labels, multiset=True, silent=silent),
    }
    # backward-compatible flat keys: original set-based micro P/R/F1
    scores['precision'] = scores['set']['micro']['precision']
    scores['recall'] = scores['set']['micro']['recall']
    scores['f1'] = scores['set']['micro']['f1']
    return scores


def compute_scores(pred_seqs, gold_seqs, task, silent=True):
    """
    Compute model performance from raw predicted/gold sequences: parses each
    sequence into quadruples, then computes both set-based and multiset-based
    (bag) precision/recall/F1, each with micro/macro/weighted averaging.
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
    scores = compute_scores_from_quads(all_preds, all_labels, silent)

    return scores, all_labels, all_preds
