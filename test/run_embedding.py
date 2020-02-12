#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import pickle as pk

from config import Config
with open('/data/blank54/workspace/blanknlp/custom.cfg', 'r') as f:
    cfg = Config(f)

import sys
sys.path.append(cfg.root)
from function import makedir
from embedding import embedding_tfidf, embedding_doc2vec

# Data Import
fname_docs_sample = os.path.join(cfg.root, cfg.fname_docs_sample)
with open(fname_docs_sample, 'rb') as f:
    docs = pk.load(f)

# Data Preparation
tagged_docs = [(str(idx), ' '.join(doc)) for idx, doc in enumerate(docs)]

# Embedding TF-IDF
tfidf_model = embedding_tfidf(tagged_docs=tagged_docs)
id2idx, tfidf_matrix, tfidf_vocab = tfidf_model
print('Shape of TF-IDF (# of docs, # of terms): {}'.format(tfidf_matrix.shape))

fname_tfidf_model = os.path.join(cfg.root, cfg.fname_tfidf_model)
makedir(fname_tfidf_model)
with open(fname_tfidf_model, 'wb') as f:
    pk.dump(tfidf_model, f)

# TF-IDF Model Usage
fname_tfidf_model = os.path.join(cfg.root, cfg.fname_tfidf_model)
with open(fname_tfidf_model, 'rb') as f:
    tfidf_model = pk.load(f)
    id2idx, tfidf_matrix, tfidf_vocab = tfidf_model

idx = 7
doc_id, doc_text = tagged_docs[idx]
tfidf_vector = tfidf_matrix[id2idx[doc_id],:] # TF-IDF vector for tagged_docs[idx]
print('TF-IDF ID of Doc[{}]: {}'.format(idx, id2idx[str(idx)]))
print('TF-IDF Vector of Doc[{}]:'.format(id2idx[str(idx)]))
print(tfidf_vector)
print(tfidf_vector.shape)