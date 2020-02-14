#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
abspath = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.sep.join(abspath.split(os.path.sep)[:-1])

import re
import pickle as pk
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer

from config import Config
with open(os.path.join(config_path, 'custom.cfg'), 'r') as f:
    cfg = Config(f)

def makedir(path):
    if path.endswith('/'):
        os.makedirs(path, exist_ok=True)
    else:
        os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)

def save_df2excel(data, fname, verbose=False):
    makedir(fname)

    writer = pd.ExcelWriter(fname)
    data.to_excel(writer, "Sheet1", index=False)
    writer.save()
    if verbose:
        print("Saved data as: {}".format(fname))

def flist_archive(fdir):
    flist = []
    for (path, _, files) in os.walk(fdir):
        flist.extend([os.path.join(path, file) for file in files if file.endswith('.pk')])
    return flist

class Preprocess:
    def __init__(self, **kwargs):
        nltk.download('wordnet', quiet=True)
        self.stemmer = LancasterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        self.do_marking = kwargs.get('do_marking', False)
        self.do_synonym = kwargs.get('do_synonym', False)
        self.do_lower = kwargs.get('do_lower', False)
        self.do_stop = kwargs.get('do_stop', False)

        self.fname_synonym_list = kwargs.get('fname_synonym_list', os.path.join(cfg.root, cfg.fname_synonym_list))
        self.fname_stopphrase_list = kwargs.get('fname_stopphrase_list', os.path.join(cfg.root, cfg.fname_stopphrase_list))
        self.fname_stopword_list = kwargs.get('fname_stopword_list', os.path.join(cfg.root, cfg.fname_stopword_list))

        self.synonym_rule = self.__read_synonym_rule()
        self.stopphrase_list = self.__read_stopphrase_list()
        self.stopword_list = self.__read_stopword_list()

        self.note = kwargs.get('note', '')

    def __read_synonym_rule(self):
        synonym_rule = {}
        with open(self.fname_synonym_list, 'r', encoding='utf-8') as f:
            synonym_pairs = [pair.split('\t') for pair in re.sub('  ', '\t', f.read().strip()).split('\n')]
            try:
                synonym_rule = {left: right for left, right in synonym_pairs}
            except:
                pass
        return synonym_rule

    def __read_stopphrase_list(self):
        with open(self.fname_stopphrase_list, 'r', encoding='utf-8') as f:
            stopword_list = list(set(f.read().strip().split('\n')))
        with open(self.fname_stopphrase_list, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(stopword_list)).strip())
        return stopword_list

    def __read_stopword_list(self):
        with open(self.fname_stopword_list, 'r', encoding='utf-8') as f:
            stopword_list = list(set(f.read().strip().split('\n')))
        with open(self.fname_stopword_list, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(stopword_list)).strip())
        return stopword_list

    def cleaning(self, text):
        for stopphrase in self.stopphrase_list:
            text = re.sub(stopphrase, '', text)
        text = re.sub('[^ \'-\./0-9a-zA-Zㄱ-힣\n]', '', text)

        text = text.replace('\x0c', '')
        text = re.sub('\.+\.', ' ', text)
        text = text.replace('\\\\', '\\').replace('\\r\\n', '')

        text = text.replace('\n', ' ')
        text = re.sub('\.(?=[A-Zㄱ-힣])', '. ', text)
        text = re.sub('\. ', '  ', text)
        text = re.sub('\s+\s', '  ', text).strip()
        return text

    def synonym(self, text):
        try:
            for left in self.synonym_rule:
                text_syno = re.sub(left, self.synonym_rule[left], text)
            return text_syno
        except:
            return text

    def marking(self, words):
        result = words
        for i, w in enumerate(words):
            if re.match('www.', str(w)):
                result[i] = 'LINK'
            elif re.search('\d+\d\.\d+', str(w)):
                result[i] = 'REF'
            elif re.search('\d', str(w)):
                result[i] = 'NUM'
        return result

    def tokenize(self, text):
        text = self.cleaning(text)
        if self.do_synonym:
            text = self.synonym(text)
        if self.do_lower:
            text = text.lower()

        words = [w for w in re.split(' |  |\n', text) if len(w)>0]
        if self.do_marking:
            words = self.marking(words)
        return words

    def stopword_removal(self, text):
        words = self.tokenize(text)
        return [w for w in words if w not in self.stopword_list]

    def stemmize(self, text):
        if self.do_stop:
            words = self.stopword_removal(text)
        else:
            words = self.tokenize(text)
        return [self.stemmer.stem(w) for w in words]

    def lemmatize(self, text):
        if self.do_stop:
            words = self.stopword_removal(text)
        else:
            words = self.tokenize(text)
        return [self.lemmatizer.lemmatize(w) for w in text]