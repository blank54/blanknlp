#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
installpath = os.path.sep.join(os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
matplotlib.rc('font', family='NanumBarunGothic')

from config import Config
with open('{}/custom.cfg'.format(installpath), 'r') as f:
    cfg = Config(f)

from function import *

class WordNetwork:
    def __init__(self, **kwargs):
        self.docs = kwargs.get('docs', []) # list of list of words: [[w1, w2, ...], [w3, ...], ...]
        self.docs_cnt = len(self.docs)

        self.count_option = kwargs.get('count_option', 'dist')
        self.combinations = self.__combinations()

        self.top_n = kwargs.get('top_n', 50)
        self.fig_width = kwargs.get('fig_width', 10)
        self.fig_height = kwargs.get('fig_height', 8)
        self.fig_dpi = kwargs.get('fig_dpi', 300)

        self.nx_edge_color = kwargs.get('nx_edge_color', 'grey')
        self.nx_node_color = kwargs.get('nx_node_color', 'purple')
        self.nx_box_color = kwargs.get('nx_box_color', 'white')
        self.nx_box_transparency = kwargs.get('nx_box_transparency', 0)
        self.nx_box_edge_color = kwargs.get('nx_box_edge_color', 'white')
        self.nx_font_size = kwargs.get('nx_font_size', 6) # Font Size

        self.save_plt = kwargs.get('save_plt', True)
        self.fname_plt = kwargs.get('fname_plt', os.path.join(cfg.root, cfg.fdir_word_network, 'tmp_word_network.png'))
        self.show_plt = kwargs.get('show_plt', True)

    def __combinations(self):
        combs = defaultdict(float)
        print('Calculating word combinations ...')
        with tqdm(total=self.docs_cnt) as pbar:
            for idx, doc in enumerate(self.docs):
                for i in range(self.docs_cnt):
                    w1 = doc[i]
                    for j in range(i+1, self.docs_cnt):
                        w2 = doc[j]
                        key = '{}__{}'.format(w1, w2)

                        if self.count_option == 'dist':
                            dist = np.abs(j-i)/(self.docs_cnt-1)
                            combs[key] += dist
                        elif self.count_option == 'occur':
                            combs[key] += 1
                        else:
                            print('Error: wrong count option')
                            break
                pbar.update(1)
        return combs

    def __top_n(self, combs):
        sorted_items = sorted(combs.items(), key=lambda x:x[1], reverse=True)
        sorted_combs = {}

        words = []
        for key, value in sorted_items:
            words.extend(key.split('__'))
            words = list(set(words))
            if len(words) < self.top_n:
                sorted_combs[key] = np.round(value, 3)
                continue
            else:
                break
        return sorted_combs

    def network(self):
        combs_df = pd.DataFrame(self.__top_n(self.combinations).items(), columns=['comb', 'count'])
        combs_dict = combs_df.set_index('comb').T.to_dict('records')
        graph = nx.Graph()
        for key, value in combs_dict[0].items():
            w1, w2 = key.split('__')
            graph.add_edge(w1, w2, weight=(value*10))

        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height), dpi=self.fig_dpi)
        pos = nx.spring_layout(graph, k=1)

        nx.draw_networkx(
            graph, pos, 
            node_size=10,
            font_size=0,
            width=0.8,
            edge_color=self.nx_edge_color,
            node_color=self.nx_node_color,
            with_labels=False,
            ax=ax)

        for key, value in pos.items():
            x, y = value[0], value[1]+0.025
            ax.text(
                x, y, s=key,
                bbox=dict(
                    facecolor=self.nx_box_color,
                    alpha=self.nx_box_transparency,
                    edgecolor=self.nx_box_edge_color),
                horizontalalignment='center',
                fontsize=self.nx_font_size)

        if self.save_plt:
            makedir(self.fname_plt)
            plt.savefig(self.fname_plt, dpi=self.fig_dpi)

        if self.show_plt:
            plt.show()