#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer

from config import Config
with open('/data/blank54/workspace/blanknlp/custom.cfg', 'r') as f:
    cfg = Config(f)

# Configuration
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

class Preprocess:
    def __init__(self, **kwargs):
        nltk.download('wordnet', quiet=True)
        self.stemmer = LancasterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        self.do_marking = kwargs.get('do_marking', True)
        self.do_synonym = kwargs.get('do_synonym', True)
        self.do_lower = kwargs.get('do_lower', True)
        self.do_stop = kwargs.get('do_stop', True)

        self.fname_synonym_list = kwargs.get('fname_synonym_list', os.path.join(cfg.root, cfg.fname_synonym_list))
        self.fname_stopword_list = kwargs.get('fname_stopword_list', os.path.join(cfg.root, cfg.fname_stopword_list))

        self.synonym_rule = self.__read_synonym_rule()
        self.stop_list = self.__read_stop_list()

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

    def __read_stop_list(self):
        with open(self.fname_stopword_list, 'r', encoding='utf-8') as f:
            stop_list = list(set(f.read().strip().split('\n')))
        with open(self.fname_stopword_list, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(stop_list)).strip())
        return stop_list

    def cleaning(self, text):
        text = re.sub('[^ \'-\./0-9a-zA-Z\n]', '', text)

        text = text.replace('\x0c', '')
        text = re.sub('\.+\.', ' ', text)
        text = text.replace('\\\\', '\\').replace('\\r\\n', '')

        text = text.replace('\n', ' ')
        text = re.sub('\.(?=[A-Z])', '. ', text)
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
        return [w for w in words if w not in self.stop_list]

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