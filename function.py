#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
abspath = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.sep.join(abspath.split(os.path.sep)[:-1])

import re
import math
import pickle as pk
import pandas as pd
from konlpy.tag import Komoran
from collections import defaultdict

import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')

from soynlp.normalizer import only_text
from soynlp.word import WordExtractor
from soynlp.tokenizer import RegexTokenizer, MaxScoreTokenizer, LTokenizer

from config import Config
with open(os.path.join(config_path, 'custom.cfg'), 'r') as f:
    cfg = Config(f)


class DataHandler:
    def __init__(self, **kwargs):
        self.note = kwargs.get('note', '')

    def makedir(self, path):
        if path.endswith('/'):
            os.makedirs(path, exist_ok=True)
        else:
            os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)

    def export_excel(self, data, fname, index=False, orient=None, verbose=False):
        self.makedir(fname)

        if type(data) != 'pandas.core.frame.DataFrame':
            if orient == 'index':
                data = pd.DataFrame.from_dict(data=data, orient=orient)
            else:
                data = pd.DataFrame(data)

        writer = pd.ExcelWriter(fname)
        data.to_excel(writer, "Sheet1", index=index)
        writer.save()
        if verbose:
            print("Saved data as: {}".format(fname))

    def flist_archive(self, fdir):
        flist = []
        for (path, _, files) in os.walk(fdir):
            flist.extend([os.path.join(path, file) for file in files if file.endswith('.pk')])
        return flist

    def f1_score(self, p, r):
        if p != 0 or r != 0:
            return (2*p*r)/(p+r)
        else:
            return 0

    def get_latest_fpath(self, fpath):
        if os.path.isfile(fpath):
            pass
        else:
            fdir = os.path.sep.join(fpath.split(os.path.sep)[:-1])
            fpath = os.path.join(fdir, os.listdir(fdir)[-1])
        return fpath


class KoreanWordScore:
    def __init__(self, **kwargs):
        self.scores = {}
        self.fpath = kwargs.get('fpath', '')

    def __call__(self):
        return self.scores

    def train(self, sentences, **kwargs):
        word_extractor = WordExtractor(min_frequency=0, min_cohesion_forward=0.05, min_right_branching_entropy=0.0)
        word_extractor.train(sentences)
        words = word_extractor.extract()

        scores = {}
        for word, score in words.items():
            scores[word] = score.cohesion_forward * math.exp(score.right_branching_entropy)

        self.scores = scores
        return self.scores

    def save(self, **kwargs):
        fpath = kwargs.get('fpath', self.fpath)
        if self.scores:
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(['  '.join((str(word), str(score))) for word, score in sorted(self.scores.items())]))
            sys.stdout.write('Success: save word scores ({})'.format(fpath))
        else:
            sys.stdout.write('Error: No scores')

    def load(self, **kwargs):
        fpath = kwargs.get('fpath', self.fpath)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = f.read()
                for pair in data.strip().split('\n'):
                    word, score = pair.split('  ')
                    self.scores[word] = float(score)
        except FileNotFoundError:
            sys.stdout.write('Error: No File ({})'.format(fpath))


class KoreanPreprocessor:
    def normalize(self, text):
        text = re.sub('_', ' ', text)
        text = only_text(text)
        return text


class KoreanTokenizer:
    def __init__(self, **kwargs):
        self.word_score = ''
        self.regex_tokenizer = RegexTokenizer()
        self.max_score_tokenizer = MaxScoreTokenizer(scores=self.word_score)
        self.ltokenizer = LTokenizer(scores=self.word_score)

    def train(self, docs, fpath):
        word_score = KoreanWordScore()
        word_score.train(sentences=docs)
        word_score.save(fpath=fpath)
        self.words_score = word_score

    def load(self, fpath):
        word_score = WordScore()
        word_score.load(fpath=fpath)
        self.word_score = word_score

    def tokenize(self, text):
        text = ' '.join(self.regex_tokenizer.tokenize(text))
        text = ' '.join(self.ltokenizer.tokenize(text))
        text = ' '.join(self.max_score_tokenizer.tokenize(text))
        return self.__post_process(text.split(' '))

    def __post_process(self, tokens):
        tokens_post = []
        i = 0
        while i < len(tokens):
            left = tokens[i]

            # concatenating numbers
            if left.isdecimal():
                for j in range(i+1, len(tokens)):
                    right = tokens[j]
                    if right.isdecimal():
                        left = left+right
                    else:
                        i = j
                        break

            else:
                i += 1
                pass
            tokens_post.append(left)
        return tokens_post


class StopwordRemover:
    def __init__(self, **kwargs):
        self.fpath = ''
        self.stoplist = self.__read_stoplist()

    def read_stoplist(self, fpath):
        self.fpath = fpath
        try:
            with open(self.fpath, 'r', encoding='utf-8') as f:
                data = f.read()
                stoplist = [word.strip() for word in data.strip().split('\n') if word]
                return stoplist
        except FileNotFoundError:
            sys.stdout.write('Error: No File ({})'.format(self.fpath))

    def __update_stoplist(self):
        self.stoplist = self.__read_stoplist()

    def remove(self, sent):
        if not self.fpath:
            sys.stdout.write('Error: Read stoplist first')
            return None
        else:
            self.__update_stoplist()
            return [word for word in sent if word not in self.stoplist]


# class TextHandler:
#     def __init__(self, **kwargs):
#         self.custom_cfg = kwargs.get('custom_cfg', cfg)

#         self.fname_stopphrase_list = kwargs.get('fname_stopphrase_list', os.path.join(cfg.root, cfg.fname_stopphrase_list))
#         self.fname_synonym_list = kwargs.get('fname_synonym_list', os.path.join(cfg.root, cfg.fname_synonym_list))
#         self.fname_unit_list = kwargs.get('fname_unit_list', os.path.join(cfg.root, cfg.fname_unit_list))

#         self.stopphrase_list = self.__read_stopphrase_list()
#         self.delete_word_list = ['\x0c', ',', '\(', '\)', '*']
#         self.synonym_rule = self.__read_synonym_rule()
#         self.unit_list = self.__read_unit_list()

#         self.do_synonym = kwargs.get('do_synonym', True)
#         self.do_marking = kwargs.get('do_marking', True)
#         self.do_unit = kwargs.get('do_unit', True)
#         self.do_lower = kwargs.get('do_lower', False)

#         self.delimiter = kwargs.get('delimiter', '  ')
#         self.note = kwargs.get('note', '')

#         self.language = kwargs.get('language', 'eng')
#         if self.language == 'eng':
#             self.fname_userword = kwargs.get('fname_userword', os.path.join(self.custom_cfg.root, self.custom_cfg.fname_userword_eng))
#             self.fname_userdic = kwargs.get('fname_userdic', os.path.join(self.custom_cfg.root, self.custom_cfg.fname_userdic_eng))
#         elif self.language == 'kor':
#             self.fname_userword = kwargs.get('fname_userword', os.path.join(self.custom_cfg.root, self.custom_cfg.fname_userword_kor))
#             self.fname_userdic = kwargs.get('fname_userdic', os.path.join(self.custom_cfg.root, self.custom_cfg.fname_userdic_kor))

#         self.userdic = self.__build_userdic()

#         self.note = kwargs.get('note', '')

#     def __call__(self, text):
#         return self.cleaning(text)

#     def __build_userdic(self):
#         if not os.path.isfile(self.fname_userword):
#             return []

#         with open(self.fname_userword, 'r', encoding='utf-8') as f:
#             wordlist = f.read().strip().split('\n')
#             userdic = ['{}\tNNP'.format(str(w)) for w in sorted(set(wordlist)) if len(w)]

#         with open(self.fname_userdic, 'w', encoding='utf-8') as f:
#             f.write('\n'.join(userdic).replace('\ufeff', ''))
#         return [pair.split('\t') for pair in userdic]

#     def __read_stopphrase_list(self):
#         with open(self.fname_stopphrase_list, 'r', encoding='utf-8') as f:
#             stopphrase_list = list(set(f.read().strip().split('\n')))
#         with open(self.fname_stopphrase_list, 'w', encoding='utf-8') as f:
#             f.write('\n'.join(sorted(stopphrase_list)).strip())
#         return stopphrase_list

#     def __read_unit_list(self):
#         with open(self.fname_unit_list, 'r', encoding='utf-8') as f:
#             unit_list = [pair.split('  ') for pair in f.read().strip().split('\n')]
#         return unit_list

#     def __read_synonym_rule(self):
#         synonym_rule = {}
#         with open(self.fname_synonym_list, 'r', encoding='utf-8') as f:
#             synonym_pairs = [pair.split('\t') for pair in re.sub('  ', '\t', f.read().strip()).split('\n')]
#             try:
#                 synonym_rule = {left: right for left, right in synonym_pairs}
#             except:
#                 pass
#         return synonym_rule

#     def __remove_char(self, text):
#         for stopphrase in self.stopphrase_list:
#             text = text.replace(stopphrase, '')

#         text = re.sub('[^ \'\?\./0-9a-zA-Zㄱ-힣\n]', '', text)
#         for word in self.delete_word_list:
#             text = text.replace(word, '')

#         text = text.replace(' / ', '/')
#         text = re.sub('\.+\.', ' ', text)
#         text = text.replace('\\\\', '\\').replace('\\r\\n', '')

#         text = text.replace('\n', '  ')
#         text = re.sub('\. ', '  ', text)
#         text = re.sub('\s+\s', ' ', text).strip()
#         return text

#     def __separate_sentence(self, text):
#         sents = []
#         for sent in text.split('  '):
#             sent_for_split = re.sub('\.(?=[A-Zㄱ-힣])', '  ', sent).split('  ')
#             if all((len(s.split(' '))>=5 for s in sent_for_split)):
#                 sents.extend(sent_for_split)
#             else:
#                 sents.append(sent)
#         return sents

#     def __sentence_hierarchy(self, sents):
#         if not any((s.startswith('?') for s in sents)):
#             return sents

#         sents_with_sub = []
#         origin = ''
#         origin2 = ''
#         for i in range(len(sents)):
#             if not sents[i].startswith('?'):
#                 sents_with_sub.append(sents[i])
#                 origin = sents[i]
#             else:
#                 if not sents[i].startswith('??'):
#                     sents_with_sub.append(origin+sents[i])
#                     origin2 = sents[i]
#                 else:
#                     sents_with_sub.append(origin+origin2+sents[i])
#         del origin, origin2
#         return [s.replace('?', '') for s in sents_with_sub]

#     def __replace_unit(self, sent):
#         sent = ' '+sent.strip()+' '
#         for abbr, full in self.unit_list:
#             sent = sent.replace(' {} '.format(str(abbr)), ' {} '.format(str(full))).strip()
#         return sent

#     def synonym(self, text):
#         if not text:
#             return ''

#         if not self.synonym_rule:
#             return text

#         for left in self.synonym_rule:
#             text_syno = re.sub(left, self.synonym_rule[left], text)
#         return text_syno

#     def marking(self, words):
#         if type(words) != list:
#             words = words.split(' ')

#         for i, w in enumerate(words):
#             if re.match('www.', str(w)):
#                 words[i] = 'LINK'
#             elif re.search('\d+\d\.\d+', str(w)):
#                 words[i] = 'REF'
#             elif re.match('\d', str(w)):
#                 words[i] = 'NUM'
#         return words

#     def cleaning(self, text):
#         if not text:
#             return ''

#         if self.do_lower:
#             text = text.lower()

#         # EOS
#         sents = self.__separate_sentence(text)

#         # Remove character
#         text = '  '.join([self.__remove_char(sent) for sent in sents])
        
#         # Synonym
#         if self.do_synonym:
#             text = self.synonym(text)

        
#         # Marking
#         if self.do_marking:
#             sents = [' '.join(self.marking(sent)) for sent in sents]

#         # Sub sentences
#         sents = self.__sentence_hierarchy(sents)

#         # Convert unit
#         if self.do_unit:
#             sents = [self.__replace_unit(sent) for sent in sents]

#         final_text = self.delimiter.join(sents)
#         return self.__remove_char(final_text)

#     def text2sent(self, text):
#         if not text:
#             return []
#         text = self.cleaning(text)
        
#         return text.split(self.delimiter)


# class NgramParser:
#     def __init__(self, **kwargs):
#         self.min_count = kwargs.get('min_count', 10)
#         self.n_range = kwargs.get('n_range', (2,5))

#         self.note = kwargs.get('note', '')

#     def __doc2ngrams(self, doc, n):
#         ngrams = []
#         for idx in range(0, len(doc)-n+1):
#             ngrams.append(tuple(doc[idx:idx+n]))
#         return ngrams

#     ## TODO: PPMI
#     def get_ngram_counter(self, docs, **kwargs):
#         min_count = kwargs.get('min_count', self.min_count)
#         n_range = kwargs.get('n_range', self.n_range)

#         n_begin, n_end = n_range
#         ngram_counter = defaultdict(int)
#         for doc in docs:
#             for n in range(n_begin, n_end+1):
#                 for ngram in self.__doc2ngrams(doc, n):
#                     ngram_counter[ngram] += 1

#         ngram_counter = {ngram: count for ngram, count in ngram_counter.items() if count >= min_count}
#         return Ngrams(ngram_counter=ngram_counter, min_count=min_count, n_range=n_range)


# class Ngrams:
#     def __init__(self, ngram_counter, min_count, n_range, **kwargs):
#         self.custom_cfg = kwargs.get('custom_cfg', cfg)

#         self.counter = ngram_counter
#         self.list = self.__counter2list()
#         self.min_count = min_count
#         self.n_range = n_range
#         self._from = self.n_range[0]
#         self._to = self.n_range[1]

#         self.fname = kwargs.get('fname', '')
#         self.note = kwargs.get('note', '')

#     def __call__(self):
#         return self.counter

#     def __len__(self):
#         return len(self.counter)

#     def __counter2list(self):
#         ngram_list = [ngram for ngram in self.counter]
#         return ngram_list

#     ## TODO: consider n-grams
#     def words2ngram(self, words, n):
#         ngrams = []
#         already = False
#         for b in range(0, len(words)-n+1):
#             ngram = tuple(words[b:b+n])
#             if ngram in self.list:
#                 ngrams.append('-'.join(ngram))
#                 already = True
#             else:
#                 if already == True:
#                     already = False
#                 else:
#                     ngrams.append(words[b])
#                     already = False

#         return ngrams


# class Tokenizer:
#     def __init__(self, **kwargs):
#         self.iter_unit = kwargs.get('iter_unit', 'word')
#         self.do_lower = kwargs.get('do_lower', False)

#         self.do_synonym = kwargs.get('do_synonym', True)
#         self.do_marking = kwargs.get('do_marking', False)
#         self.do_unit = kwargs.get('do_unit', True)
#         self.text_handler = TextHandler(do_synonym=self.do_synonym, do_marking=self.do_marking, do_unit=self.do_unit)

#         self.note = kwargs.get('note', '')

#     def __call__(self, text):
#         return self.tokenize(text)

#     def tokenize(self, text):
#         if not text:
#             return []

#         text = self.text_handler.cleaning(text)
#         if self.do_lower:
#             text = text.lower()

#         unigrams = [w for w in re.split(' |  |\n', text) if len(w)>0]
#         return unigrams

#     def tokenize_ngram(self, text, n_grams): # n_grams: Class "Ngrams"
#         ngrams = self.tokenize(text)
#         if not n_grams:
#             return ngrams

#         for n in range(n_grams._to, n_grams._from-1, -1):
#             _ngrams = []
#             for ngram in n_grams.words2ngram(ngrams, n):
#                 _ngrams.append(ngram)
#             ngrams = _ngrams
#         return ngrams


# class Preprocess:
#     def __init__(self, **kwargs):
#         self.custom_cfg = kwargs.get('custom_cfg', cfg)
#         nltk.download('wordnet', quiet=True)
#         self.stemmer = LancasterStemmer()
#         self.lemmatizer = WordNetLemmatizer()

#         self.language = kwargs.get('language', 'eng')
#         if self.language == 'eng':
#             self.fname_stopword_list = kwargs.get('fname_stopword_list', os.path.join(cfg.root, cfg.fname_stopword_list_eng))
#             self.stopword_list_nltk = stopwords.words('english')
#         elif self.language == 'kor':
#             self.fname_stopword_list = kwargs.get('fname_stopword_list', os.path.join(cfg.root, cfg.fname_stopword_list_kor))
#             self.stopword_list_nltk = []

#         self.do_stop = kwargs.get('do_stop', False)
#         self.stoplist = kwargs.get('stoplist', 'nltk')
#         self.stopword_list = self.__read_stopword_list()

#         self.note = kwargs.get('note', '')

#     def __read_stopword_list(self):
#         with open(self.fname_stopword_list, 'r', encoding='utf-8') as f:
#             stopword_list_custom_raw = list(set(f.read().strip().split('\n')))
#             stopword_list_custom_stem = [self.stemmer.stem(w) for w in stopword_list_custom_raw]
#             stopword_list_custom_lemma = [self.lemmatizer.lemmatize(w) for w in stopword_list_custom_raw]
#             stopword_list_custom = list(set(stopword_list_custom_raw+stopword_list_custom_stem+stopword_list_custom_lemma))
#         with open(self.fname_stopword_list, 'w', encoding='utf-8') as f:
#             f.write('\n'.join(sorted(stopword_list_custom)).strip())

#         if self.stoplist == 'nltk':
#             return self.stopword_list_nltk
#         elif self.stoplist == 'custom':
#             return stopword_list_custom
#         elif self.stoplist == 'merge':
#             return list(set(self.stopword_list_nltk+stopword_list_custom))

#     def stopword_removal(self, words):
#         return [w for w in words if w not in self.stopword_list]

#     def stemmize(self, words):
#         if self.do_stop:
#             words = self.stopword_removal(words)
#         return [self.stemmer.stem(w) for w in words]

#     def lemmatize(self, words):
#         if self.do_stop:
#             words = self.stopword_removal(words)
#         return [self.lemmatizer.lemmatize(w) for w in words]