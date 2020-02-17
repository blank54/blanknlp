#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import pickle as pk

from config import Config
with open('./custom.cfg', 'r') as f:
    cfg = Config(f)

import sys
sys.path.append(cfg.root)
from blanknlp.function import Preprocess


# Data Import
fname_docs = os.path.join(cfg.root, cfg.fname_docs_sample_eng)
with open(fname_docs, 'rb') as f:
    docs = pk.load(f)

# Run Preprocess
pr_config = {
    'do_synonym': True,
    'do_lower': True,
    'do_stop': True,
    'stoplist': 'custom'
}
pr = Preprocess(**pr_config)

docs_prep = []
for doc in docs:
    sents = pr.cleaning(doc).split('  ')
    docs_prep.append([pr.stemmize(sent) for sent in sents])

    

# Run Preprocess (cleaning)
for doc in docs:
    doc_prep = pr.cleaning(doc)
    print('BEFORE: {}'.format(doc))
    print('AFTER: {}'.format(doc_prep))

# Run Preprocess (synonym)
for doc in docs:
    sents = pr.cleaning(doc).split('  ')
    for sent in sents:
        sent_prep = pr.synonym(sent)
        print('BEFORE: {}'.format(sent))
        print('AFTER: {}'.format(sent_prep))

# Run Preprocess (marking)
for doc in docs:
    sents = pr.cleaning(doc).split('  ')
    for sent in sents:
        sent_prep = pr.marking(sent)
        print('BEFORE: {}'.format(sent))
        print('AFTER: {}'.format(sent_prep))

# Run Preprocess (tokenize)
for doc in docs:
    sents = pr.cleaning(doc).split('  ')
    for sent in sents:
        sent_prep = pr.tokenize(sent)
        print('BEFORE: {}'.format(sent))
        print('AFTER: {}'.format(sent_prep))

# Run Preprocess (stopword removal)
for doc in docs:
    sents = pr.cleaning(doc).split('  ')
    for sent in sents:
        sent_prep = pr.stopword_removal(sent)
        print('BEFORE: {}'.format(sent))
        print('AFTER: {}'.format(sent_prep))