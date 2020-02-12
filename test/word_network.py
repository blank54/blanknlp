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
from visualize import *

# Data Import
fname_docs_sample = os.path.join(cfg.root, cfg.fname_docs_sample)
with open(fname_docs_sample, 'rb') as f:
    docs = pk.load(f)

for doc in docs[:10]:
    print(doc[:10])

# Word Network
word_network = WordNetwork(docs=docs, save_plt=False)
word_network.network()