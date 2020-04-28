from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re

def filter_inputs(caption_features_pairs):
    return caption_features_pairs

def normalize_strings(strings):
    return [normalize_string(s) for s in strings]

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s