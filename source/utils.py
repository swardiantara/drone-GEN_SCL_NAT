"""
Module containing helper functions for GEN_SCL_NAT project
"""
from transformers import LogitsProcessor
import torch


def load_mappings():
    """
    Load category mappings used to map existing labelset to human-readable variant
    """
    import os
    import json
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, 'category_mappings.json')) as ofile:
        data_json = json.load(ofile)
    return data_json


class BatchConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, aspect_categories, special_tokens):
        self.tokenizer = tokenizer
        self.aspect_category_tokens = set(tokenizer.encode(' '.join(aspect_categories), add_special_tokens=False)) if aspect_categories is not None else None
        self.special_tokens = set(tokenizer.encode(' '.join(special_tokens), add_special_tokens=False)) if special_tokens is not None else None
        
        # Pre-compute a mask for aspect categories and special tokens
        self.static_mask = torch.zeros(len(tokenizer), dtype=torch.bool)
        self.static_mask[list(self.aspect_category_tokens.union(self.special_tokens))] = True

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, vocab_size = scores.shape
        device = scores.device

        # Create a mask for input tokens (varies for each item in the batch)
        input_mask = torch.zeros((batch_size, vocab_size), dtype=torch.bool, device=device)
        for i, seq in enumerate(input_ids):
            input_mask[i, seq.unique()] = True

        # Combine with the static mask (same for all items in the batch)
        combined_mask = input_mask | self.static_mask.to(device)

        # Apply the mask
        scores = scores.masked_fill(~combined_mask, -float('inf'))
        return scores
    

class ToggleableConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, aspect_categories=None, special_tokens=None, use_constraints=False):
        self.tokenizer = tokenizer
        self.aspect_category_tokens = set(tokenizer.encode(' '.join(aspect_categories), add_special_tokens=False)) if aspect_categories is not None else None
        self.special_tokens = set(tokenizer.encode(' '.join(special_tokens), add_special_tokens=False)) if special_tokens is not None else None
        self.use_constraints = use_constraints
        
        # Pre-compute a mask for aspect categories and special tokens
        self.static_mask = torch.zeros(len(tokenizer), dtype=torch.bool)
        # self.static_mask[list(self.aspect_category_tokens.union(self.special_tokens))] = True
        allowed_tokens = self.aspect_category_tokens.union(self.special_tokens)
        allowed_tokens = [token for token in allowed_tokens if token < len(tokenizer)]
        self.static_mask[allowed_tokens] = True

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.use_constraints:
            return scores  # Return original scores if constraints are disabled

        batch_size, vocab_size = scores.shape
        device = scores.device

        # Create a mask for input tokens (varies for each item in the batch)
        input_mask = torch.zeros((batch_size, vocab_size), dtype=torch.bool, device=device)
        for i, seq in enumerate(input_ids):
            # input_mask[i, seq.unique()] = True
            valid_tokens = seq[seq < vocab_size].unique()
            input_mask[i, valid_tokens] = True

        # Combine with the static mask (same for all items in the batch)
        combined_mask = input_mask | self.static_mask.to(device)

        # Apply the mask
        scores = scores.masked_fill(~combined_mask, -float('inf'))
        return scores

    def toggle_constraints(self, use_constraints):
        self.use_constraints = use_constraints