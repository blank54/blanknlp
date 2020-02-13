#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import re
import time
import random
import numpy as np
import pickle as pk
import pandas as pd
import requests
import itertools
import urllib.request
from urllib.parse import quote
from bs4 import BeautifulSoup
from collections import defaultdict
from tqdm import tqdm, tqdm_notebook
from datetime import datetime, timedelta

from config import Config
with open('./custom.cfg', 'r') as f:
    cfg = Config(f)

import sys
sys.path.append(cfg.root)
from function import makedir, save_df2excel, flist_archive, Preprocess
pr = Preprocess()

def get_url_uniq(url):
    return url.split('/')[-1]

class Article:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', '')
        self.url = kwargs.get('url', '')
        self.url_uniq = get_url_uniq(self.url)

        self.title = kwargs.get('title', '')
        self.date = kwargs.get('date', '')
        self.category = kwargs.get('category', '')
        self.content = kwargs.get('content', '')
        
        self.token = kwargs.get('token', pr.tokenize(self.content))
        self.stop = kwargs.get('stop', '')
        self.stem = kwargs.get('stem', '')
        self.lemma = kwargs.get('token', '')

        self.likeit_good = kwargs.get('likeit_good', '')
        self.likeit_warm = kwargs.get('likeit_warm', '')
        self.likeit_sad = kwargs.get('likeit_sad', '')
        self.likeit_angry = kwargs.get('likeit_angry', '')
        self.likeit_want = kwargs.get('likeit_want', '')

        self.comment_list = kwargs.get('comment_list', 'none')
        self.comment_count = kwargs.get('comment_count', 0)

    def __call__(self):
        return (self.id, self.url)

    def __str__(self):
        return '{}: {}'.format(self.id, self.url)

    def __len__(self):
        return len(self.content)

class NewsCrawler:
    def __init__(self, **kwargs):
        self.time_lag = np.random.normal(loc=kwargs.get('time_lag', 3.0), scale=1.0)
        self.headers = {'User-Agent': '''
            [Windows64,Win64][Chrome,58.0.3029.110][KOS] 
            Mozilla/5.0 Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) 
            Chrome/58.0.3029.110 Safari/537.36
            '''}

        self.query = NewsQuery(kwargs.get('query', ''))
        self.date_from = NewsDate(kwargs.get('date_from', ''))
        self.date_to = NewsDate(kwargs.get('date_to', ''))
        self.query_info = '{}_{}_{}'.format(self.query.query, self.date_from.date, self.date_to.date)

        self.do_sampling = kwargs.get('do_samples', False)
        self.news_num_samples = kwargs.get('num_samples', 100)

        self.url_base = 'https://search.naver.com/search.naver?&where=news&query={}&sm=tab_pge&sort=1&photo=0&field=0&reporter_article=&pd=3&ds={}&de={}&docid=&nso=so:dd,p:from{}to{},a:all&mynews=0&start={}&refresh_start=0'
        self.url_start = 'https://news.naver.com/'
        self.url_list = kwargs.get('url_list', [])
        self.articles = kwargs.get('articles', [])

        self.fname_news_url_list = kwargs.get('fname_news_url_list', os.path.join(cfg.root, cfg.fdir_news_url_list, '{}.pk'.format(self.query_info)))
        self.fdir_news_data_articles = kwargs.get('fdir_news_data_articles', os.path.join(cfg.root, cfg.fdir_news_data_articles, '{}').format(self.query.query))
        self.fdir_news_corpus_articles = kwargs.get('fdir_news_corpus_articles', os.path.join(cfg.root, cfg.fdir_news_corpus_articles, '{}').format(self.query.query))
        self.news_archive = kwargs.get('news_archive', flist_archive(self.fdir_news_corpus_articles))
        self.fname_articles_xlsx = kwargs.get('fname_articles', os.path.join(self.fdir_news_data_articles, '{}_{}_{}.xlsx'.format(self.query.query, self.date_from.date, self.date_to.date)))

        self._errors = []
        self.fname_errors = kwargs.get('fname_errors', os.path.join(cfg.root, cfg.fdir_news_errors))

    def __get_last_page(self):
        start_idx = 1
        url_list_page = self.url_base.format(self.query(),
                                             self.date_from.formatted,
                                             self.date_to.formatted,
                                             self.date_from.date,
                                             self.date_to.date,
                                             start_idx)

        req = urllib.request.Request(url=url_list_page, headers=self.headers)
        html = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(html, 'lxml')
        time.sleep(self.time_lag)

        _text = soup.select('div.title_desc.all_my')[0].text
        last_page = int(re.sub(',', '', _text.split('/')[1])[:-1].strip())
        return last_page

    def __parse_list_page(self, url_list_page):
        req = urllib.request.Request(url=url_list_page, headers=self.headers)
        html = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(html, 'lxml')
        time.sleep(self.time_lag)
        href = soup.select('dl dd a')
        return [h.attrs['href'] for h in href if h.attrs['href'].startswith(self.url_start)]

    def get_url_list(self):
        print('=========================================================')
        print('  >>Parsing List Page\n\tQuery: {} ({} to {}) ...'.format(self.query.query, self.date_from.date, self.date_to.date))
        
        try:
            with open(self.fname_news_url_list, 'rb') as f:
                url_list = pk.load(f)
        except FileNotFoundError:
            url_list = []
            last_page = self.__get_last_page()
            max_start_idx = int(round(last_page, -1)) + 1
            index_list = list(range(1, max_start_idx, 10)) # 네이버는 최대 4000개까지만 제공함
            with tqdm(total=len(index_list)) as pbar:
                for start_idx in index_list:
                    url_list_page = self.url_base.format(self.query(),
                                                         self.date_from.formatted,
                                                         self.date_to.formatted,
                                                         self.date_from.date,
                                                         self.date_to.date,
                                                         start_idx)
                    url_list.extend(self.__parse_list_page(url_list_page))
                    pbar.update(1)

            makedir(self.fname_news_url_list)
            with open(self.fname_news_url_list, 'wb') as f:
                pk.dump(url_list, f)
        return url_list

    def __parse_comment(self, url_article):
        comments = []

        oid = url_article.split("oid=")[1].split("&")[0]
        aid = url_article.split("aid=")[1]
        page = 1    
        comment_header = {
            'User-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36',
            'referer':url_article}

        while True:
            url_comment_api = 'https://apis.naver.com/commentBox/cbox/web_neo_list_jsonp.json?ticket=news&templateId=default_society&pool=cbox5&_callback=jQuery1707138182064460843_1523512042464&lang=ko&country=&objectId=news'+oid+'%2C'+aid+'&categoryId=&pageSize=20&indexSize=10&groupId=&listType=OBJECT&pageType=more&page='+str(page)+'&refresh=false&sort=FAVORITE' 
            r = requests.get(url_comment_api, headers=comment_header)
            comment_content = BeautifulSoup(r.content,'html.parser')    
            total_comment = str(comment_content).split('comment":')[1].split(',')[0]
            match = re.findall('"contents":"([^\*]*)","userIdNo"', str(comment_content))
            comments.append(match)

            if int(total_comment) <= ((page)*20):
                break
            else : 
                page += 1

        return list(itertools.chain(*comments))

    def __parse_article_page(self, url_article):
        req = urllib.request.Request(url=url_article, headers=self.headers)
        html = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(html, 'lxml')
        time.sleep(self.time_lag)

        try:
            article_title = soup.select('h3[id=articleTitle]')[0].text
        except:
            self._errors.append(url_article)
            article_title = 'NoTitle'

        try:
            article_date = re.sub('\.', '', soup.select('span[class=t11]')[0].text.split(' ')[0])
        except:
            self._errors.append(url_article)
            article_date = 'NoDate'

        try:
            article_category = soup.select('em[class=guide_categorization_item]')[0].text
        except:
            self._errors.append(url_article)
            article_category = 'NoCategory'

        try:
            article_content = soup.select('div[id=articleBodyContents]')[0].text.strip()
        except:
            self._errors.append(url_article)
            article_content = 'NoContent'

        try:
            article_comment_list = self.__parse_comment(url_article)
            article_comment_count = len(article_comment_list)
        except:
            self._errors.append(url_article)
            article_comment_list = ['NoComments']
            article_comment_count = 0

        article_config = {
            'url': url_article,
            'title': article_title,
            'date': article_date,
            'category': article_category,
            'content': article_content,
            'comment_list': article_comment_list,
            'comment_count': article_comment_count
            }
        return Article(**article_config)

    def get_articles(self):
        with open(self.fname_news_url_list, 'rb') as f:
            url_list = pk.load(f)

        if self.do_sampling:
            url_list = random.sample(url_list, self.news_num_samples)
        else:
            pass

        articles = []
        print('=========================================================')
        print('  >>Parsing Articles ...\n\tQuery: {} ({} to {}) ...'.format(self.query.query, self.date_from.date, self.date_to.date))
        with tqdm(total=len(url_list)) as pbar:
            for idx, url in enumerate(url_list):
                if any((get_url_uniq(url) in url_) for url_ in self.news_archive):
                    fname_article_in_archive = [f for f in self.news_archive if get_url_uniq(url) in f][0]
                    with open(fname_article_in_archive, 'rb') as f:
                        article = pk.load(f)
                        article.id = ''
                else:
                    article = self.__parse_article_page(url)

                fname_article = os.path.join(self.fdir_news_corpus_articles, article.date, '{}.pk'.format(article.url_uniq))
                makedir(fname_article)
                with open(fname_article, 'wb') as f:
                    pk.dump(article, f)
                articles.append(article)
                pbar.update(1)
        self.__export_excel(articles)

        if self._errors:
            print('Errors: {}'.format(len(self._errors)))
            makedir(self.fname_errors)
            with open(self.fname_errors, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self._errors))

        return articles

    def __export_excel(self, articles):
        articles_dict = defaultdict(list)
        for article in articles:
            articles_dict['id'].append(article.id)
            articles_dict['date'].append(article.date)
            articles_dict['title'].append(article.title)
            articles_dict['category'].append(article.category)
            articles_dict['content'].append(article.content)
            articles_dict['comment_list'].append(' SEP '.join(article.comment_list))
            articles_dict['comment_count'].append(article.comment_count)
            articles_dict['url'].append(article.url)

        articles_dict_sort = pd.DataFrame(articles_dict).sort_values(by=['date'], axis=0)
        save_df2excel(articles_dict_sort, self.fname_articles_xlsx)

class NewsQuery:
    def __init__(self, query):
        self.query = query

    def __call__(self):
        return quote(self.query.encode('utf-8'))

    def __str__(self):
        return '{}'.format(self.query)

    def __len__(self):
        return len(self.query.split('+'))

class NewsDate:
    def __init__(self, date):
        self.date = date
        self.formatted = self.__convert_date()

    def __call__(self):
        return self.formatted

    def __str__(self):
        return '{}'.format(self.__call__())

    def __convert_date(self):
        try:
            return datetime.strptime(self.date, '%Y%m%d').strftime('%Y.%m.%d')
        except:
            return ''