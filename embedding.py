#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from sklearn.feature_extraction.text import TfidfVectorizer

from config import Config
with open('./custom.cfg', 'r') as f:
    cfg = Config(f)

import sys
sys.path.append(cfg.root)
from function import Preprocess

pr_config = {
    'do_marking': False,
    'do_synonym': False,
    'do_lower': False,
    'do_stop': False
}
pr = Preprocess(**pr_config)

def embedding_tfidf(tagged_docs):
    '''
    tagged_docs = [(id, doc), (id, doc), ...]
    type(id) == str
    type(doc) == str
    '''

    vectorizer = TfidfVectorizer()
    sorted_docs = sorted(tagged_docs, key=lambda x:x[0])
    id2idx = {id: idx for idx, (id, doc) in enumerate(sorted_docs)}
    docs_for_tfidf = [doc for id, doc in sorted_docs]

    tfidf_matrix = vectorizer.fit_transform(docs_for_tfidf)
    tfidf_vocab = vectorizer.vocabulary_
    tfidf_model = (id2idx, tfidf_matrix, tfidf_vocab)
    return tfidf_model

def embedding_doc2vec(tagged_docs, parameters, verbose=True):
    '''
    tagged_docs = [(id, doc), (id, doc), ...]
    type(id) == str
    type(doc) == str
    '''

    docs_for_d2v = [TaggedDocument(words=pr.tokenize(doc), tags=[id]) for id, doc in tagged_docs]
    model = Doc2Vec(
        vector_size=parameters.get('vector_size', cfg.d2v_vector_size),
        alpha=parameters.get('alpha', cfg.d2v_alpha),
        min_alpha=parameters.get('min_alpha', cfg.d2v_min_alpha),
        min_count=parameters.get('min_count', cfg.d2v_min_count),
        window=parameters.get('window', cfg.d2v_window),
        workers=parameters.get('workers', cfg.d2v_workers),
        dm=parameters.get('dm', cfg.d2v_dm)
    )
    model.build_vocab(docs_for_d2v)

    max_epoch = parameters.get('max_epoch', cfg.d2v_max_epoch)
    alpha_step = parameters.get('alpha_step', cfg.d2v_alpha_step)
    with tqdm(total=max_epoch) as pbar:
        for epoch in range(max_epoch):
            model.train(
                documents=docs_for_d2v,
                total_examples=model.corpus_count,
                epochs=epoch
            )
            model.alpha -= alpha_step
            if verbose:
                pbar.update(1)
    return model