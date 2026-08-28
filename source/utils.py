"""
Module containing helper functions for GEN_SCL_NAT project
"""


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


# NOTE: constrained decoding used to be implemented here via
# BatchConstrainedLogitsProcessor / ToggleableConstrainedLogitsProcessor,
# which restricted generation to category/sentiment/template tokens only
# (no source-copy allowance). That blocks aspect/opinion span tokens from
# ever being generated, since they aren't part of that closed vocabulary and
# haven't been generated yet on the first decoding step -- this is why that
# wiring was disabled (see gen_scl_nat_main.py history). Constrained decoding
# is now implemented in constrained_decoding.py, which additionally allows
# each example's own source tokens so spans can still be copied.