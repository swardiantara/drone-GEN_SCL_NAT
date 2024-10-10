# -*- coding: utf-8 -*-

# This script contains all data transformation and reading

import random
from torch.utils.data import Dataset
import torch
from generate_data import sentword2opinion, dronesent2opinion, get_domain, domain_map, senttag2opinion, get_gen_scl_nat_data


def read_line_examples_from_file(data_path, silence=False):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                sents.append(words.split())
                labels.append(eval(tuples))
    if silence:
        print(f"Total examples = {len(sents)}")
    return sents, labels


def get_para_asqp_targets(sents, labels, drone_sp=None, truncated=False):
    """
    Obtain the target sentence under the paraphrase paradigm
    This replicates the ABSA-QUAD approach
    """
    targets = []
    for label in labels:
        all_quad_sentences = []
        for quad in label:
            at, raw_ac, sp, ot = quad
            # TRUNCATED
            if truncated == True:
                raw_ac = raw_ac.split("#")[0]

            domain = get_domain(raw_ac)
            ac = domain_map[domain].get(raw_ac, raw_ac)
            
            if drone_sp is None:
                man_ot = sentword2opinion[sp]  # 'positive' -> 'good'    
            else:
                man_ot = dronesent2opinion[sp] if drone_sp == 'binary' else sp

            if at == 'NULL':  # for implicit aspect term
                at = 'it'

            one_quad_sentence = f"{ac} is {man_ot} because {at} is {ot}"
            all_quad_sentences.append(one_quad_sentence)

        target = ' [SSEP] '.join(all_quad_sentences)
        targets.append(target)
    return sents.copy(), targets


def get_para_tasd_targets(sents, labels, drone_sp=None, truncated=False):

    targets = []
    for label in labels:
        all_tri_sentences = []
        for quad in label:
            at, raw_ac, sp, _ = quad

            domain = get_domain(raw_ac)
            ac = domain_map[domain].get(raw_ac, raw_ac)

            if drone_sp is None:
                man_ot = sentword2opinion[sp]  # 'positive' -> 'good'    
            else:
                man_ot = dronesent2opinion[sp] if drone_sp == 'binary' else sp

            if at == 'NULL':
                at = 'it'
            one_tri = f"{ac} is {man_ot} because {at} is {man_ot}"
            all_tri_sentences.append(one_tri)

        target = ' [SSEP] '.join(all_tri_sentences)
        targets.append(target)
    return sents.copy(), targets


def get_para_aste_targets(sents, labels, drone_sp=None, truncated=False):
    targets = []
    for label in labels:
        all_tri_sentences = []
        for quad in label:
            at, _, sp, ot = quad

            if drone_sp is None:
                man_ot = sentword2opinion[sp]  # 'positive' -> 'good'    
            else:
                man_ot = dronesent2opinion[sp] if drone_sp == 'binary' else sp

            if at == 'NULL':
                at = 'it'
            one_tri = f"It is {man_ot} because {at} is {ot}"
            all_tri_sentences.append(one_tri)

        target = ' [SSEP] '.join(all_tri_sentences)
        targets.append(target)
    return sents.copy(), targets


def get_transformed_io(data_path, data_dir, task, absa_task, data_type, truncate=False):
    """
    The main function to transform input & target according to the task
    """
    sents, labels = read_line_examples_from_file(data_path)
    drone_sp = None if data_dir.split('_')[-1] == 'data' else data_dir.split('_')[-1]

    # the input is just the raw sentence
    if task == 'asqp':
        if absa_task == 'quad':
            inputs, targets = get_para_asqp_targets(sents, labels, drone_sp, truncate)
        elif absa_task == 'tasd':
            inputs, targets = get_para_tasd_targets(sents, labels, drone_sp, truncate)
        elif absa_task == 'aste':
            inputs, targets = get_para_aste_targets(sents, labels, drone_sp, truncate)
        else:
            raise NotImplementedError
        return inputs, targets, labels
        
    elif task.startswith('gen_scl_nat'):
        inputs, targets = get_gen_scl_nat_data(sents, labels, task, drone_sp, truncate)
        return inputs, targets, labels
    else:
        raise NotImplementedError


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, task, absa_task, max_len=256, data_path=None, truncate=False):
        # './data/rest16/train.txt'
        if data_path:
            self.data_path = data_path
        else:
            self.data_path = f'data/{data_dir}/{data_type}.txt'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.task = task
        self.absa_task = absa_task
        self.data_type = data_type
        self.inputs = []
        self.targets = []
        self.contrastive_labels = {}
        self.sentence_strings = []
        self.truncate = truncate
        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, 
                "target_ids": target_ids, "target_mask": target_mask}

    def get_raw_labels(self):
        results = get_transformed_io(self.data_path, self.data_dir, self.task, self.absa_task, self.data_type, self.truncate)
        return results
        
    def _build_examples(self):

        inputs, targets = get_transformed_io(self.data_path, self.data_dir, self.task, self.absa_task, self.data_type, self.truncate)
        self.sentence_strings = inputs
        for i in range(len(inputs)):
            # change input and target to two strings
            input = ' '.join(inputs[i])
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
              [input], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
              [target], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )
            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)
        print(f'self.inputs: {self.inputs}')
        print(f'self.inputs: {self.targets}')

class GenSCLNatDataset(ABSADataset):

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        sentiment_label = torch.tensor(self.contrastive_labels['sentiment'][index])
        aspect_label = torch.tensor(self.contrastive_labels['aspect'][index])
        opinion_label = torch.tensor(self.contrastive_labels['opinion'][index])
        
        return {"source_ids": source_ids,
                "source_mask": src_mask, 
                "target_ids": target_ids,
                "target_mask": target_mask,
                'sentiment_labels': sentiment_label,
                'opinion_labels': opinion_label,
                'aspect_labels': aspect_label,
                }

    def _build_examples(self):

        inputs, targets, labels = get_transformed_io(self.data_path, self.data_dir, self.task, self.absa_task, self.data_type, self.truncate)
        
        self.sentence_strings = inputs
        for i in range(len(inputs)):
            # change input and target to two strings

            input = ' '.join(inputs[i])
            target = targets[i]
            if isinstance(targets[i], list):
                target = " ".join(targets[i])

            tokenized_input = self.tokenizer.batch_encode_plus(
              [input], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
              [target], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)

        def get_sentiment_labels(labels_in):
            drone_sp = self.data_dir.split('_')[-1]

            if drone_sp == 'binary':
                sentiment_dict = {
                    'negative': 0,
                    'positive': 1,
                    'mixed': 2,
                }
            elif drone_sp == 'multi':
                sentiment_dict = {
                    'normal': 0,
                    'minor': 1,
                    'moderate': 2,
                    'severe': 3,
                    'mixed': 4,
                }
            else:
                sentiment_dict = {
                    'negative': 0,
                    'neutral': 1,
                    'positive': 2,
                    'mixed': 3
                }
            sentiment_labels = []
            for ex in labels_in:
                label = list(set([quad[2] for quad in ex]))
                if len(label) == 1:
                    label = sentiment_dict[label[0]]
                else:
                    label = sentiment_dict['mixed']
                sentiment_labels.append(label)
            from collections import Counter
            print("Sentiment distribution")
            print(Counter(sentiment_labels))
            return sentiment_labels

        def get_opinion_labels(labels_in):
            opinion_dict = {
                'NULL': 0,
                'EXPLICIT': 1,
                'BOTH': 2,
            }
            opinion_labels = []
            for ex in labels_in:
                opinions = set([quad[3] for quad in ex])

                if 'NULL' not in opinions:
                    label = opinion_dict['EXPLICIT']
                else:
                    if len(opinions) == 1:
                        label = opinion_dict['NULL']
                    else:
                        label = opinion_dict['BOTH']

                opinion_labels.append(label)
            return opinion_labels

        def get_aspect_labels(labels_in):
            aspect_dict = {
                'NULL': 0,
                'EXPLICIT': 1,
                'BOTH': 2,
            }
            aspect_labels = []
            for ex in labels_in:
                aspects = set([quad[0] for quad in ex])

                if 'NULL' not in aspects:
                    label = aspect_dict['EXPLICIT']
                else:
                    if len(aspects) == 1:
                        label = aspect_dict['NULL']
                    else:
                        label = aspect_dict['BOTH']

                aspect_labels.append(label)
            return aspect_labels
        
        self.contrastive_labels['sentiment'] = get_sentiment_labels(labels)
        self.contrastive_labels['opinion'] = get_opinion_labels(labels)
        self.contrastive_labels['aspect'] = get_aspect_labels(labels)