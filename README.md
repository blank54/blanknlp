# blanknlp
A bunch of python codes to analyze text data in the construction industry.  
Mainly reconstitute the pre-exist python libraries for TM and NLP.

## _Project Information_
- Supported by C!LAB (@Seoul Nat'l Univ.)
- First Release: 2020.02.12.

## _Contributors_
- Seonghyeon Boris Moon (blank54@snu.ac.kr, https://github.com/blank54/)
- Gitaek Lee (lgt0427@snu.ac.kr)
- Taeyeon Chang (jgwoon1838@snu.ac.kr, _a.k.a. Kowoon Chang_)

## _Initialization_ (IMPORTANT)
The user needs _custom.cfg_ file in the workspace. Refer to _sample.cfg_ for the necessary attributes.  
Refer to the following hierarchy.

```
WORKSPACE
    └blanknlp
        └sample.cfg
        └...
    └custom.cfg
    └...
```

- - -

# Data
We provide some pickled data for tutorial.  
The data is uploaded at _'./blanknlp/data/sample.pk'_ and the user can use it via _cfg.fname_docs_sample_.

```python
import os
import pickle as pk

fname_docs_sample = os.path.join(cfg.root, cfg.fname_docs_sample)
with open(fname_docs_sample, 'rb') as f:
    docs = pk.load(f)
```

The _**docs**_ contains 58 documents.  
Each _**doc**_ consists of 100 words with several stopwords removed.

```python
print('# of docs: {}'.format(len(docs)))
for idx, doc in enumerate(docs):
    print('# of words in doc: {}'.format(len(doc)))
    print(doc[:5])
    if idx > 3:
        break
```

- - -

# Preprocessing
## Preprocessing: English
_Not Ready Yet_

## Preprocessing: Korean
_Not Ready Yet_

- - -

# Web Crawling
## Web Crawling: Naver News
>Sourcecode:
>>_web_crawling.py_  
>>_/tutorial/run_web_crawling.py_

A customized class _**NewsCrawler**_ to facilitate the process of web crawling from naver news.  Mainly refer to _**urllib**_ and _**BeautifulSoup**_.
>https://docs.python.org/3/library/urllib.html  
>https://www.crummy.com/software/BeautifulSoup/bs4/doc/

Note that the Naver news platform only provides **4,000 articles** per query.  
The dafault settings of the web crawler are
- **3 seconds** sleep after parsing a url page
- sampling **100 articles** from list page if _**do_sampling**_ is _**True**_  

Import related libraries

```python
# Configuration
from config import Config
with open(FNAME_CUSTOM_CONFIG, 'r') as f: # './blanknlp/custom.cfg'
    cfg = Config(f)

from blanknlp.web_crawling import NewsCrawler
```

Build _**NewsCrawler**_ with _**query**_, _**date_from**_, and _**date_to**_.

```python
# Input Query
query = YOUR_QUERY # '교량+사고+유지관리'
date_from = YOUR_DATE_FROM # '20190701'
date_to = YOUR_DATE_TO # '20190710'

# Build News Crawler
crawling_config = {
    'query': query,
    'date_from': date_from,
    'date_to': date_to
}
news_crawler = NewsCrawler(**crawling_config)
```

The crawling process consists of two stages: parse_list_page(_**.get_url_list()**_) and parse_article_page(_**.get_articles()**_).  
Finally, the crawler parses **url**, **title**, **date**, **category**, **content**, and **comments** from the articles.

```python
# Run Crawling
news_crawler.get_url_list() ## Note: returns list of url_list_page
news_crawler.get_articles() ## Note: returns list of articles
```

As default, the user can explore the crawled data in _.xlsx_ format at _'data/news/articles/YOUR_QUERY/'_.  
Every article is pickled in _**Article**_ class at _'corpus/news/articles/YOUR_QUERY/'_ by date, which allows the user not to access, parse, and save an article that is already exist in the corpus.  
Use the function _**read_articles()**_ to read the articles data.  
It returns a list of _**Article**_. Then the user can use the attributes of each element such as _**url**_, _**title**_, _**date**_, _**content**_, _**comment_list**_.
See _web_crawling.py/Article()_ for more information.  
Note that the content of article commonly starts with a junk text such as _'// flash 오류를 우회하기 위한 함수 추가 function \_flash\_removeCallback() \{\}'_.

```python
from blanknlp.web_crawling import read_articles

articles = read_articles(YOUR_QUERY, YOUR_DATE_FROM, YOUR_DATE_TO) # '교량+사고+유지관리', '20190701', '20190710'

for article in articles[:3]:
    print('Title: {}'.format(article.title))
    print('Date: {}'.format(article.date))
    print('Contents: \n{}...'.format(article.content[100:200])) ## Note: avoid junk text
    print()
```

## Web Crawling: Twitter
_Not Ready Yet_

- - -

# Visualization
## Word Network
>Sourcecode:
>>_visualize.py_  
>>_/tutorial/run_word_network.py_

A customized class _**WordNetwork**_ to facilitate the usage of the python library _**networkx**_.
>https://networkx.github.io/

The default settings are
- word combination weighting based on the **distance** within the sentence
- top **50 words** to be shown in the network
- save the network with filename _'tmp_word_network.png'_ in directory _'./result/word_network/'_.

The user can customize the settings via _config_. See _visualize.py/WordNetwork_ for detail options.  
Import related libraries.
```python
from config import Config
with open(FNAME_CUSTOM_CONFIG, 'r') as f:
    cfg = Config(f)

from blanknlp.visualize import WordNetwork
```

Prepare _**docs**_ and build _**WordNetwork**_.

```python
# Data Import
docs = [['word1', 'word2', ...], ['word3', ...], ...]

# Model Development
wn_config = {
    'docs': docs,
    'top_n': 100, ## Note: top n words to show
    'fname_plt': './result/my_network.png'
}
word_network = WordNetwork(**wn_config)
```

Just _**.network()**_ to draw a word network.

```python
# Draw Network
word_network.network()
```

- - -

# Text Embedding
>Sourcecode:
>>_embedding.py_  
>>_/tutorial/run_embedding.py_

## Text Embedding: TF-IDF
Term Frequency-Inverse Document Frequency (TF-IDF), One of the most simple and general text embedding techniques is provided.  
We utilized _**TfidfVectorizer**_ from _**sklearn**_ library.
>https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html?highlight=tfidf#sklearn.feature_extraction.text.TfidfVectorizer

Import related libraries.

```python
# Configuration
from config import Config
with open(FNAME_CUSTOM_CONFIG, 'r') as f:
    cfg = Config(f)

from blanknlp.function import makedir
from blanknlp.embedding import embedding_tfidf
```

The TF-IDF embedding model requires _**tagged_docs**_, of which format is **list of tuple(id, doc)**.  
The type of _**id**_ is not restricted to _**str**_, but recommended.

```python
# Data Import
docs = ['This is a sentecne', 'This is another sentence', ...]

# Data Preparation
tagged_docs = [(str(idx), doc) for idx, doc in enumerate(docs)]
```

Embedding TF-IDF and save the model. The TF-IDF model is composed of three items (i.e., _**id2idx**_, _**tfidf_matrix**_, _**tfidf_vocab**_).
- _**id2idx**_: a dictionary that returns the **row index of tfidf_matrix** of each **id of doc** in _**tagged_docs**_.
- _**tfidf_matrix**_: a matrix of numeric value (i.e., TF-IDF) with row length of **# of document** and column length of **vocabulary size**.
- _**tfdif_vocab**_: a dictionary of vocabularies used in whole _**tagged_docs**_, which returns the **column index of tfidf_matrix** of each term.

The user should keep these three items to utilize TF-IDF results.

```python
# Embedding TF-IDF
tfidf_model = embedding_tfidf(tagged_docs=tagged_docs)
id2idx, tfidf_matrix, tfidf_vocab = tfidf_model
print('Shape of TF-IDF (# of docs, # of terms): {}'.format(tfidf_matrix.shape))

with open(FNAME_TFIDF_MODEL, 'wb') as f: # os.path.join(cfg.root, cfg.fname_tfidf_model)
    pk.dump(tfidf_model, f)
```

The user can utilize the TF-IDF results as below.  
The sample code assumed that the user needs the tfidf vector of **document id 7**.

```python
# TF-IDF Model Usage
with open(FNAME_TFIDF_MODEL, 'rb') as f: # os.path.join(cfg.root, cfg.fname_tfidf_model)
    tfidf_model = pk.load(f)
    id2idx, tfidf_matrix, tfidf_vocab = tfidf_model

idx = 7
doc_id, doc_text = tagged_docs[idx]
tfidf_vector = tfidf_matrix[id2idx[doc_id],:] ## Note: TF-IDF vector for tagged_docs[idx]
print('TF-IDF ID of Doc[{}]: {}'.format(idx, id2idx[str(idx)]))
print('TF-IDF Vector of Doc[{}]:'.format(id2idx[str(idx)]))
print(tfidf_vector)
print(tfidf_vector.shape)
```

## Text Embedding: Word2Vec
_Not Ready Yet_

## Text Embedding: Doc2Vec
_Not Ready Yet_