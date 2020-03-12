#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
abspath = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.sep.join(abspath.split(os.path.sep)[:-1])

from time import time
from tqdm import tqdm
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer

from config import Config
with open(os.path.join(config_path, 'custom.cfg'), 'r') as f:
    cfg = Config(f)


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


def embedding_word2vec(docs, parameters, verbose=True):
    '''
    docs = [[w1, w2, ...], [w3, ...], ...]
    '''

    _start = time()
    model = Word2Vec(
        size=parameters.get('size', cfg.w2v_size),
        window=parameters.get('window', cfg.w2v_window),
        min_count=parameters.get('min_count', cfg.w2v_min_count),
        workers=parameters.get('workers', cfg.w2v_workers),
        sg=parameters.get('sg', cfg.w2v_sg),
        hs=parameters.get('hs', cfg.w2v_hs),
        negative=parameters.get('negative', cfg.w2v_negative),
        ns_exponent=parameters.get('ns_exponent', cfg.w2v_ns_exponent),
        iter=parameters.get('iter', cfg.w2v_iter)
    )
    model.build_vocab(sentences=docs)
    model.train(sentences=docs, total_examples=model.corpus_count, epochs=model.iter)
    _end = time()
    if verbose:
        print('Training Word2Vec: {:,.02f} minutes'.format((_end-_start)/60))
    return model


def update_word2vec(current_model, new_docs, verbose=True):
    _start = time()
    model = current_model
    model.min_count = 0
    model.build_vocab(sentences=new_docs, update=True)
    model.train(sentences=new_docs, total_examples=model.corpus_count, epochs=model.iter)
    _end = time()
    if verbose:
        print('Training Word2Vec: {:,.02f} minutes'.format((_end-_start)/60))
    return model


def embedding_doc2vec(tagged_docs, parameters, verbose=True):
    '''
    https://code.google.com/archive/p/word2vec/

    tagged_docs = [(id, doc), (id, doc), ...]
    type(id) == str
    type(doc) == list of words
    '''

    _start = time()
    docs_for_d2v = [TaggedDocument(words=doc, tags=[id]) for id, doc in tagged_docs]
    model = Doc2Vec(
        vector_size=parameters.get('vector_size', cfg.d2v_vector_size),
        window=parameters.get('window', cfg.d2v_window),
        min_count=parameters.get('min_count', cfg.d2v_min_count),
        workers=parameters.get('workers', cfg.d2v_workers),
        dm=parameters.get('dm', cfg.d2v_dm),
        negative=parameters.get('negative', cfg.d2v_negative),
        epochs=parameters.get('epochs', cfg.d2v_epochs),
        dbow_words=parameters.get('dbow_words', cfg.d2v_dbow_words)
    )
    model.build_vocab(docs_for_d2v)
    model.train(documents=docs_for_d2v, total_examples=model.corpus_count, epochs=model.epochs)
    _end = time()
    if verbose:
        print('Training Doc2Vec: {:,.02f} minutes'.format((_end-_start)/60))
    return model