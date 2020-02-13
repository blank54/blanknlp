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

- - -
## Initialization
Modify _**root**_ of the _**custom.cfg**_ to your _root directory_ of the project.

```python
# Root
root: 'YOUR_ROOT_DIRECTORY' # '/data/blank54/workspace/my_project/'
```

- - -

## Data
We provide some pickled data for tutorial.  
The users can reach it as below.

```python
import os
import pickle as pk

fname_docs_sample = './blanknlp/data/sample.pk'
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
## Web Crawling
>Sourcecode:
>>_web_crawling.py_  
>>_/test/run_web_crawling.py_

A customized class _**NewsCrawler**_ to facilitate the process of web crawling from naver news.  Mainly refer to _**urllib**_ and _**BeautifulSoup**_.
>https://docs.python.org/3/library/urllib.html  
>https://www.crummy.com/software/BeautifulSoup/bs4/doc/

Note that the **Naver News** platform only provides **4,000 articles** per query.  
The dafault settings of the web crawler are
- **3 seconds** sleep after parsing a url page
- sampling **100** articles from list page if _**do_sampling**_ is _**True**_

Import related libraries
```python
# Configuration
from config import Config
with open(FNAME_YOUR_CRAWLING_CONFIG, 'r') as f:
    cfg = Config(f)

import sys
sys.path.append(cfg.root)
from web_crawling import NewsCrawler
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

Run _**NewsCrawler**_.  
The crawling process consists of two stages: parse_list_page(_**.get_url_list()**_) and parse_article_page(_**.get_articles()**_).  
Finally, the crawler parses **url**, **title**, **date**, **category**, **content**, and **comments** from the articles.

```python
# Run Crawling
news_crawler.get_url_list() ## returns list of url_list_page
news_crawler.get_articles() ## returns list of articles
```

As default, the user can get the crawled data in _.xlsx_ format at _'data/news/articles/YOUR_QUERY/'_.  
Every article is pickled in _**Article**_ class at _'corpus/news/articles/YOUR_QUERY/'_ by date, which allows the user not to access, parse, and save an article that is already exist in the corpus.

- - -

## Word Network
>Sourcecode:
>>_visualize.py_  
>>_/test/run_word_network.py_

A customized class _**WordNetwork**_ to facilitate the usage of the python library _**networkx**_.
>https://networkx.github.io/

The default settings are
- word combination weighting based on the **distance** within the sentence
- top **50 words** to be shown in the network
- save the network with filename _**'tmp_word_network.png'**_ in directory _**'./result/word_network/'**_.

The user can customize the settings via _**config**_. See _visualize.py/WordNetwork_ for detail options.

```python
from config import Config
with open(FNAME_YOUR_CRAWLING_CONFIG, 'r') as f:
    cfg = Config(f)

import sys
sys.path.append(cfg.root)
from visualize import WordNetwork

wn_config = {
    'docs': docs,
    'top_n': 100,
    'fname_plt': './result/my_network.png'
}
word_network = WordNetwork(**wn_config)
```

Just _**.network()**_ to draw a word network.

```python
word_network.network()
```

- - -

## Text Embedding
>Sourcecode:
>>_embedding.py_  
>>_/test/run_embedding.py_

### TF-IDF
Term Frequency-Inverse Document Frequency (TF-IDF), One of the most simple and general text embedding techniques is provided.  
We utilized _**TfidfVectorizer**_ from _**sklearn**_ library.
>https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html?highlight=tfidf#sklearn.feature_extraction.text.TfidfVectorizer

Import related libraries.

```python
# Configuration
from config import Config
with open(FNAME_YOUR_CRAWLING_CONFIG, 'r') as f:
    cfg = Config(f)

import sys
sys.path.append(cfg.root)
from function import makedir
from embedding import embedding_tfidf
```

The TF-IDF embedding model requires _**tagged_docs**_, of which format is **list of tuple(id, doc)**.  
Note that a _**doc**_ is a **list of word** (i.e., [w1, w2, ...]).
```python
# Data Import
with open(FNAME_DOCS, 'rb') as f:
    docs = pk.load(f)

# Data Preparation
tagged_docs = [(str(idx), ' '.join(doc)) for idx, doc in enumerate(docs)]
```

Embedding TF-IDF and save the model. The TF-IDF model is composed of three items (i.e., _**id2idx**_, _**tfidf_matrix**_, _**tfidf_vocab**_).
- **id2idx**: a dictionary that returns the **row index of tfidf_matrix** of each **id of doc** in **tagged_docs**.
- **tfidf_matrix**: a matrix of numeric value (i.e., TF-IDF) with row length of **# of document** and column length of **vocabulary size**.
- **tfdif_vocab**: a dictionary of vocabularies used in whole _**tagged_docs**_, which returns the **column index of tfidf_matrix** of each term.

The user should keep these three items to utilize TF-IDF results.

```python
# Embedding TF-IDF
tfidf_model = embedding_tfidf(tagged_docs=tagged_docs)
id2idx, tfidf_matrix, tfidf_vocab = tfidf_model
print('Shape of TF-IDF (# of docs, # of terms): {}'.format(tfidf_matrix.shape))

with open(FNAME_TFIDF_MODEL, 'wb') as f:
    pk.dump(tfidf_model, f)
```

The user can utilize the TF-IDF results as below.  
The sample code assumed that the user needs the tfidf vector of **document id 7**.

```python
# TF-IDF Model Usage
with open(FNAME_TFIDF_MODEL, 'rb') as f:
    tfidf_model = pk.load(f)
    id2idx, tfidf_matrix, tfidf_vocab = tfidf_model

idx = 7
doc_id, doc_text = tagged_docs[idx]
tfidf_vector = tfidf_matrix[id2idx[doc_id],:] # TF-IDF vector for tagged_docs[idx]
print('TF-IDF ID of Doc[{}]: {}'.format(idx, id2idx[str(idx)]))
print('TF-IDF Vector of Doc[{}]:'.format(id2idx[str(idx)]))
print(tfidf_vector)
print(tfidf_vector.shape)
```