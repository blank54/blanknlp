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
date_to = '20190930'

# Run Crawling
def run_crawling():
    crawling_config = {
        'query': query,
        'date_from': date_from,
        'date_to': date_to
    }
    news_crawler = NewsCrawler(**crawling_config)

    news_crawler.get_url_list()
    news_crawler.get_articles()
# run_crawling()

# Usage of Crawled Data
from blanknlp.web_crawling import read_articles

articles = read_articles(query, date_from, date_to)

for article in articles:
    print('Title: {}'.format(article.title))
    print('Date: {}'.format(article.date))
    print('Contents: \n{}...'.format(article.content[:200]))
    print()