#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
abspath = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.sep.join(abspath.split(os.path.sep)[:-1])

import csv
import numpy as np
import pickle as pk
import pandas as pd
from time import time
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import Dense, Bidirectional, LSTM, TimeDistributed
from keras_contrib.layers import CRF

from config import Config
with open(os.path.join(config_path, 'custom.cfg'), 'r') as f:
    cfg = Config(f)
    
from blanknlp.function import DataHandler, TextHandler
from blanknlp.embedding import update_word2vec
data_handler = DataHandler()
text_handler = TextHandler()

class LabeledSentence:
    def __init__(self, tag, sent, labels, **kwargs):
        self.tag = tag
        self.sent = sent
        self.labels = labels

        self.note = kwargs.get('note', 'Class: LabeledSentence')

    def __len__(self):
        return len(self.sent)

    def __str__(self):
        return ' '.join(self.sent)
    
    def __call__(self):
        return (self.tag, self.sent, self.labels)


class NER_LabeledDocs:
    def __init__(self, fdir_ner_labeled_docs):
        self.fdir_ner_labeled_docs = fdir_ner_labeled_docs
        self.docs = self.__read_labeled_data()
        
        self.words = self.__words()
        self.word2id = {w: i for i, w in enumerate(self.words)}
        self.id2word = {i: w for i, w in enumerate(self.words)}

        self._error_file = []
        self._error_sent = []

    def __read_labeled_data(self):
        docs = {}
        for fname in os.listdir(self.fdir_ner_labeled_docs):
            with open(os.path.join(self.fdir_ner_labeled_docs, fname), 'r', encoding='utf-8') as f:
                lines = [line for line in csv.reader(f)]

            if len(lines) % 2 == 1:
                self._error_file.append(fname)
                continue

            for idx in range(len(lines)):
                if idx % 2 == 0:
                    tag = '{}_{}'.format(fname.replace('.csv', ''), (int(idx/2)))
                    words = [w.lower() for w in lines[idx] if w]
                else:
                    labels = [l for l in lines[idx] if l]
                    if len(words) == len(labels):
                        docs[tag] = LabeledSentence(tag=tag, sent=words, labels=labels)
                    else:
                        self._error_sent.append((fname, tag))
        return docs

    def __words(self):
        words = []
        for tag in self.docs:
            words.extend(self.docs[tag].sent)
        words.append('__PAD__')
        words.append('__UNK__')
        return list(set(words))
    
    def __call__(self):
        return self.docs

    def __iter__(self):
        for tag in sorted(self.docs.keys()):
            yield self.docs[tag]

    def __len__(self):
        return len(self.docs)


class NER_Labels:
    def __init__(self, fpath_ner_labels, **kwargs):
        self.fpath_ner_labels = fpath_ner_labels
        self.labels = self.__read_ner_labels()
        self.n_labels = len(self.labels)
        
        self.label2id = {l: i for i, l in enumerate(self.labels)}
        self.id2label = {i: l for i, l in enumerate(self.labels)}

        self.note = kwargs.get('note', 'Class: NER_Labels')

    def __read_ner_labels(self):
        with open(self.fpath_ner_labels, 'r', encoding='utf-8') as f:
            labels = [l.strip() for l in f.read().strip().split('\n')]
            labels.append('__PAD__')
            labels.append('__UNK__')
        return labels

    def __str__(self):
        return ', '.join((self.labels))

    def __len__(self):
        return self.n_labels

    def __iter__(self):
        for label in self.labels:
            yield label
            
    def __call__(self):
        return self.labels


class NER_WeightLabels:
    def __init__(self, fname_ner_weight_labels, **kwargs):
        self.fname_ner_weight_labels = fname_ner_weight_labels
        self.labels = self.__read_weight_labels()

    def __call__(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        for label in self.labels:
            yield label

    def __read_weight_labels(self):
        try:
            with open(self.fname_ner_weight_labels, 'r', encoding='utf-8') as f:
                weight_labels = [l.strip() for l in f.read().strip().split('\n') if l]
            return weight_labels
        except:
            return []


class NER_Result:
    def __init__(self, input_sent, pred_labels, **kwargs):
        self.sent = input_sent
        self.pred = pred_labels
        self.result = self.__assign_labels()

        self.note = kwargs.get('note', 'Class: NER_Result')
        

    def __assign_labels(self):
        result = defaultdict(list)
        for (word, label) in zip(self.sent, self.pred):
            result[label].append(word)
        return result

    def __call__(self):
        return self.result
    
    def __len__(self):
        return len(self.sent)
    
    def __str__(self):
        return ' '.join(self.pred)
    
    def __iter__(self):
        for label in self.result:
            yield self.result[label]


class NER_Corpus:
    def __init__(self, ner_labeled_docs, ner_labels, w2v_model, max_sent_len, feature_size, **kwargs):
        self.labeled_docs = ner_labeled_docs
        self.labels = ner_labels

        self.do_duplicate = kwargs.get('do_duplicate', False)
        self.fname_ner_weight_labels = kwargs.get('fname_ner_weight_labels', '')
        self.weight_labels = NER_WeightLabels(fname_ner_weight_labels=self.fname_ner_weight_labels)
        
        self.max_sent_len = max_sent_len
        self.feature_size = w2v_model.vector_size

        self.update_w2v = kwargs.get('update_w2v', False)
        self.word_vector = self.__get_word_vector(current_model=w2v_model)
        
        self.x_words, self.y_labels = self.__padding()
        self.x, self.y = self.__embedding()

        self.test_size = kwargs.get('test_size', 0.3)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=self.test_size)
        
    def __call__(self):
        return self.__embedding()
    
    def __len__(self):
        return len(self.labeled_docs)
    
    def __iter__(self):
        for i in range(self.__len__()):
            yield (self.x_words[i], self.y_labels[i])
        
    def __get_word_vector(self, **kwargs):
        print('>>Update Word2Vec model')
        fpath_w2v_model_for_ner = os.path.join('/data/blank54/workspace/spec/model/w2v/', 'w2v_for_ner.pk')
        
        if self.update_w2v:
            w2v_model = kwargs.get('current_model', '')
            new_docs = [d.sent for d in self.labeled_docs]
            try:
                new_w2v_model = update_word2vec(current_model=w2v_model, new_docs=new_docs)
            except:
                new_w2v_model = update_word2vec(current_model=w2v_model.model, new_docs=new_docs)
            data_handler.makedir(fpath_w2v_model_for_ner)
            with open(fpath_w2v_model_for_ner, 'wb') as f:
                pk.dump(new_w2v_model, f)
        else:
            with open(fpath_w2v_model_for_ner, 'rb') as f:
                new_w2v_model = pk.load(f)
            
        word_vector = new_w2v_model.wv
        word_vector['__PAD__'] = np.zeros(self.feature_size)
        word_vector['__UNK__'] = np.zeros(self.feature_size)
        del new_w2v_model
        return word_vector

    def __padding(self):
        print('>>Padding LabeledDocs')
        x_words = []
        y_labels = []
        for doc in self.labeled_docs:
            x_words.append([self.labeled_docs.word2id[w] for w in doc.sent])
            y_labels.append(doc.labels)

            if self.do_duplicate and any(w in [self.labels.id2label[int(l)] for l in doc.labels] for w in self.weight_labels.labels):
                x_words.append([self.labeled_docs.word2id[w] for w in doc.sent])
                y_labels.append(doc.labels)
        
        x_words_pad = pad_sequences(
            maxlen=self.max_sent_len,
            sequences=x_words,
            padding='post',
            value=self.labeled_docs.word2id['__PAD__'])
        y_labels_pad = pad_sequences(
            maxlen=self.max_sent_len,
            sequences=y_labels,
            padding='post',
            value=self.labels.label2id['__PAD__'])
        return x_words_pad, y_labels_pad
    
    def __embedding(self):
        print('>>Embedding PaddedDocs')
        id2word = self.labeled_docs.id2word
        x = np.zeros((len(self.labeled_docs), self.max_sent_len, self.feature_size), dtype=list)
        y = np.zeros((len(self.labeled_docs), self.max_sent_len, self.labels.n_labels), dtype=list)
        with tqdm(total=len(self.labeled_docs)) as pbar:
            for i in range(len(self.labeled_docs)):
                for j, id in enumerate(self.x_words[i]):
                    for k in range(self.feature_size):
                        word = id2word[id]
                        x[i, j, k] = self.word_vector[word][k]

                y[i] = to_categorical(self.y_labels[i], num_classes=(self.labels.n_labels))
                pbar.update(1)
        return x, y


class NER_Model:    
    def __init__(self, fpath_ner_model, **kwargs):
        self.do_train = kwargs.get('do_train', False)

        self.fpath_ner_model = fpath_ner_model
        self.fpath_ner_corpus = kwargs.get('fpath_ner_corpus', self.fpath_ner_model.replace('model', 'corpus').replace('.h5', '.pk'))
        self.corpus = self.__read_corpus(self.fpath_ner_corpus)
        self.word_vector = self.corpus.word_vector

        self.x_train = self.corpus.x_train
        self.x_test = self.corpus.x_test
        self.y_train = self.corpus.y_train
        self.y_test = self.corpus.y_test
        
        self.data_len = len(self.corpus)
        self.max_sent_len = self.corpus.max_sent_len
        self.feature_size = self.corpus.feature_size
        self.input_shape = (self.max_sent_len, self.feature_size)

        self.labels = self.corpus.labels
        self.n_labels = self.labels.n_labels
        self.word2id = self.corpus.labeled_docs.word2id
        self.id2word = self.corpus.labeled_docs.id2word
                
        self.parameters = kwargs.get('parameters', {})
        self.lstm_units = self.parameters.get('lstm_units', 512)
        self.lstm_return_sequences = self.parameters.get('lstm_return_sequences', True)
        self.lstm_recurrent_dropout = self.parameters.get('lstm_recurrent_dropout', 0.2)
        self.dense_units = self.parameters.get('dense_units', 50)
        self.dense_activation = self.parameters.get('dense_activation', 'relu')
        self.ner_batch_size = self.parameters.get('ner_batch_size', 32)
        self.ner_epochs = self.parameters.get('ner_epochs', 200)
        self.ner_validation_split = self.parameters.get('ner_validation_split', 0.1)
        self.ner_verbose = self.parameters.get('ner_verbose', True)
        self.time_verbose = self.parameters.get('time_verbose', False)
        self.show_summary = self.parameters.get('show_summary', True)
        
        self.model = self.__initialization()

        self.confusion_matrix = ''
        self.confusion_matrix_size = self.n_labels-2
        self.f1 = ''

        self.note = kwargs.get('note', 'Class: NER_Model')
        
    def __read_corpus(self, fpath_ner_corpus):
        try:
            with open(fpath_ner_corpus, 'rb') as f:
                corpus = pk.load(f)
            return corpus
        except:
            return None

    def __initialization(self):
        input = Input(shape=(self.input_shape))
        model = Bidirectional(LSTM(units=self.lstm_units,
                                   return_sequences=self.lstm_return_sequences,
                                   recurrent_dropout=self.lstm_recurrent_dropout))(input)
        model = TimeDistributed(Dense(units=self.dense_units,
                                      activation=self.dense_activation))(model)
        crf = CRF(self.n_labels)
        out = crf(model)

        model = Model(inputs=input, outputs=out)
        model.compile(optimizer='rmsprop',
                      loss=crf.loss_function,
                      metrics=[crf.accuracy])
        return model
        
    def fit(self, **kwargs):
        show_summary = kwargs.get('show_summary', self.show_summary)
        time_verbose = kwargs.get('time_verbose', self.time_verbose)

        _start = time()
        if self.do_train:
            history = self.model.fit(
                x=self.x_train,
                y=self.y_train,
                batch_size=self.ner_batch_size,
                epochs=self.ner_epochs,
                validation_split=self.ner_validation_split,
                verbose=self.ner_verbose
            )
            
            data_handler.makedir(self.fpath_ner_model)
            self.model.save(self.fpath_ner_model)
        else:
            self.model.load_weights(self.fpath_ner_model)
        _end = time()    

        self.confusion_matrix = self.__confusion_matrix()
        self.f1 = self.__f1_score()

        if show_summary:
            self.model.summary()

        if time_verbose:
            print('Fitting Time of NER Model: {:,.02f} minutes'.format((_end-_start)/60))
            
    # TODO:
    def __pred2labels(self, sents, prediction):
        pred_labels = []
        for sent, pred in zip(sents, prediction):
            try:
                sent_len = np.where(sent==self.word2id['__PAD__'])[0][0]
            except:
                sent_len = self.max_sent_len
                
            labels = []
            for i in range(sent_len):
                labels.append(self.labels.id2label[np.argmax(pred[i])])
            pred_labels.append(labels)
        return pred_labels

    def __confusion_matrix(self):
        matrix = np.zeros((self.confusion_matrix_size+1, self.confusion_matrix_size+1))
        prediction = self.model.predict(self.x_test, verbose=self.ner_verbose)
        pred_labels = self.__pred2labels(self.x_test, prediction)
        test_labels = self.__pred2labels(self.y_test, self.y_test)
        
        for i in range(len(pred_labels)):
            for j, pred in enumerate(pred_labels[i]):
                matrix[self.labels.label2id[test_labels[i][j]], self.labels.label2id[pred]] += 1
                
        for i in range(self.confusion_matrix_size):
            matrix[i, self.confusion_matrix_size] = sum(matrix[i, 0:self.confusion_matrix_size])
            matrix[self.confusion_matrix_size, i] = sum(matrix[0:self.confusion_matrix_size, i])
            
        matrix[self.confusion_matrix_size, self.confusion_matrix_size] = sum(matrix[self.confusion_matrix_size, 0:self.confusion_matrix_size])
        return matrix

    def __f1_score(self):
        f1_list = []
        for i in range(self.confusion_matrix_size):
            corr = self.confusion_matrix[i, i]
            pred = self.confusion_matrix[self.confusion_matrix_size, i]
            real = self.confusion_matrix[i, self.confusion_matrix_size]

            precision = corr/max(pred, 1)
            recall = corr/max(real, 1)
            f1_list.append(data_handler.f1_score(precision, recall))
        return np.mean(f1_list).round(3)

    def evaluation(self):
        print('\nCONFUSION MATRIX:\n')
        print(self.confusion_matrix.astype(int))
        print()

        print('CLASS  %12s  %12s  %12s' %('PRECISION', 'RECALL', 'F1 SCORE'))
        for i in range(self.confusion_matrix_size):
            corr = self.confusion_matrix[i,i]
            pred = self.confusion_matrix[self.confusion_matrix_size, i]
            real = self.confusion_matrix[i, self.confusion_matrix_size]

            precision = corr/max(pred, 1)
            recall = corr/max(real, 1)
            f1 = data_handler.f1_score(precision, recall)

            print('%5s  %12.03f  %12.03f  %12.03f' %(self.labels.labels[i], precision, recall, f1))
            print('       (%5d/%5d) (%5d/%5d)\n' %(corr, pred, corr, real))
        print('____________________________________________________________')
        print('Average F1 Score: %.03f' %self.f1)

    def predict(self, sent):
        '''
        sent: list of words (e.g., [w1, w2, ...])
        '''
        sent_by_id = []
        for w in sent:
            try:
                sent_by_id.append(self.word2id[w])
            except:
                sent_by_id.append(self.word2id['__UNK__'])

        sent_by_id_pad = pad_sequences(maxlen=self.max_sent_len, sequences=[sent_by_id], padding='post', value=self.word2id['__PAD__'])
        x_input = np.zeros((1, self.max_sent_len, self.feature_size), dtype=list)
        for j, id in enumerate(sent_by_id_pad[0]):
            for k in range(self.feature_size):
                word = self.id2word[id]
                x_input[0, j, k] = self.word_vector[word][k]
        
        predictions = self.model.predict(x_input, verbose=self.ner_verbose)
        pred_labels = self.__pred2labels(sent_by_id_pad, predictions)[0]
        ner_result = NER_Result(input_sent=sent, pred_labels=pred_labels)
        return ner_result


class NER_Compare:
    def __init__(self, ner_result_left, ner_result_right, **kwargs):
        self.result_left = ner_result_left
        self.result_right = ner_result_right

        self.fpath_ner_labels = kwargs.get('fpath_ner_labels', '')
        self.labels = self.__read_ner_labels()

        self.left2right = self.__compare(self.result_left, self.result_right)
        self.right2left = self.__compare(self.result_right, self.result_left)

        self.note = kwargs.get('note', 'Class: NER_Compare')

    def __read_ner_labels(self):
        labels = []
        if self.fpath_ner_labels:
            labels = NER_Labels(fpath_ner_labels=self.fpath_ner_labels).labels
        else:
            labels = list(set(list(self.result_left.result.keys())+list(self.result_right.result.keys())))
        
        try:
            labels.pop(labels.index('__PAD__'))
        except ValueError:
            pass
            
        return labels

    def __compare(self, a, b):
        diff = defaultdict(list)
        for label in self.labels:
            for e in a.result[label]:
                if e not in b.result[label]:
                    diff[label].append(e)
        return diff

    def table(self):
        result = defaultdict(list)
        for label in self.labels:
            result['labels'].append(label)
            result['left'].append(', '.join(self.left2right[label]))
            result['right'].append(', '.join(self.right2left[label]))

        data = pd.DataFrame(result)
        data = data.set_index('labels')
        return data