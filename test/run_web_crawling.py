#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
abspath = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.sep.join(abspath.split(os.path.sep)[:-1])

from config import Config
with open(os.path.join(config_path, 'custom.cfg'), 'r') as f:
    cfg = Config(f)

import sys
sys.path.append(cfg.root)
from web_crawling import NewsCrawler

# Input Query
query = '교량+사고+유지관리'
date_from = '20190701'
date_to = '20190710'

# Build News Crawler
crawling_config = {
    'query': query,
    'date_from': date_from,
    'date_to': date_to
}
news_crawler = NewsCrawler(**crawling_config)

# Run Crawling
news_crawler.get_url_list()
news_crawler.get_articles()