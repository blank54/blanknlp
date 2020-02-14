#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
from config import Config
with open('./custom.cfg', 'r') as f:
    cfg = Config(f)

from blanknlp.web_crawling import NewsCrawler

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