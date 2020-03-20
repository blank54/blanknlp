# blanknlp
A bunch of python codes to analyze text data in the construction industry.  
Mainly reconstitute the pre-exist python libraries for Natural Language Processing (NLP).

## _Project Information_
- Supported by C!LAB (@Seoul Nat'l Univ.)
- First Release: 2020.02.12.

## _Release Note_
- 1.0.0 (2020.02.12.): Initialized
- 1.0.1 (2020.03.20.): Modified an input type of _**NER Model**_.

## _Contributors_
- Seonghyeon Boris Moon (blank54@snu.ac.kr, https://github.com/blank54/)
- Gitaek Lee (lgt0427@snu.ac.kr)
- Taeyeon Chang (jgwoon1838@snu.ac.kr, _a.k.a. Kowoon Chang_)

## _Release Note_
2020.03.17. (TUE)
- The sample data in Korean is uploaded.
- _**DataHandler**_ and _**TextHandler**_ is now released.


## _Initialization_ (IMPORTANT)
The user needs _custom.cfg_ file in the workspace. Refer to _sample.cfg_ for the necessary attributes.  
Refer to the following hierarchy.

```
WORKSPACE
    └blanknlp
        └sample.cfg
        └...
    └custom.cfg
    └YOUR_PYTHON_CODE.py
    └...
```

- - -

# Data
## Sample Data
We provide some pickled data for tutorial at _'./blanknlp/data/'_.  
The user can use it via _cfg.fname_docs_sample_eng_ for English data, or _cfg.fname_docs_sample_kor_ for Korean data.  

```python
import os
import pickle as pk

fname_docs_sample_eng = os.path.join(cfg.root, cfg.fname_docs_sample_eng)
with open(fname_docs_sample_eng, 'rb') as f:
    docs = pk.load(f)
```

## DataHandler
>Sourcecode:
>>_function.py/DataHandler_
The class _**DataHandle**_ wrapped up several useful functions that frequently used during python programming.

```python
from blanknlp.function import DataHandler
data_handler = DataHandler()
```

It covers following methods with straightforward names:
1. _**.makedir()**_:  
  - creates a directory.

```python
fdir = '../YOUR_DIRCTORY/'
fname = '../YOUR_DIRCTORY/FILENAME.tmp'

# If the input is a directory, create it.
data_handler.makedir(path=fdir)

# If the input is a filename, create the mother directory.
data_handler.makedir(path=fname)

# If the directory exists, do nothing.
```

2. _**.export_excel()**_:  
It saves a _**dict**_ or _**DataFrame**_ object in _**.xlsx**_ format.  
Also provides _index_ and _orient_ options of _**pd.DataFrame()**_.

```python
import pandas as pd

data_dict = {}
data_df = pd.DataFrame(data_dict)

# If the input is a dictionary, convert it to DataFrame and save it.
data_handler.export_excel(data=data_dict, fname=YOUR_FNAME)

# If the input is a DataFrame, save it.
data_handler.export_excel(data=data_df, fname=YOUR_FNAME)
```

3. _**.flist_archive()**_:  
It returns every filenames from every sub-directories of the input directory.

```python
fdir = './YOUR_DIRCTORY'
flist = data_handler.flist_archive(fdir)
```

4. _**.f1_score()**_:
It requires precision and recall values and returns a non-biased f1 score.

```python
precision = YOUR_PRECISION_VALUE
recall = YOUR_RECALL_VALUE
f1_score = data_handler.f1_score(p=precision, r=recall)
```

5. _**.get_latest_fpath()**_:
It returns the latest file in the input directory.
NOTE: _not verified yet_.

```python
fpath = './YOUR_DIRCTORY'
fpath_latest = data_handler.get_latest_fpath(fpath)
```

- - -

# Preprocessing
## TextHandler
>Sourcecode:
>>_function.py/TextHandler_

The _**TextHandler**_ provides several useful functions of Natural Language Processing (NLP) to handle the text data. The users might utilize it before preprocess the text.

```python
from blanknlp.function import TextHandler
text_handler = TextHandler()
```

### TextHandler: User Dictionary
The users can modify the dictionaries of the _**TextHandler**_.
The dictionary lists are initialized in './blanknlp/thesaurus/'. Update the list to fit the analysis purpose.
The elements of the list should be separated with the EOL(i.e., \n).

- stopphrase_list.txt  
A _Stopphrase list_ covers a group of unnecessary words, numbers, or characters, which disturbs the meaning of the text. Crawled news articles or table of contents from reports might have ones.  

- synonym_list.txt  
The _Synonym list_ covers terms that represent the same instance but written in different notations.  

- unit_list.txt  
The _Unit list_ covers various unit notations. It can be recognized as a specific version of synonyms for units.  

CAUTION: Save the lists in customized filenames, so not to be overwritten by _git pull_.  
NOTE: The lists are language-independent.

```python
text_handler.fname_stopphrase_list = './YOUR_STOPPHRASE_LIST.txt'
text_handler.fname_synonym_list = './YOUR_SYNONYM_LIST.txt'
text_handler.fname_unit_list = './YOUR_UNIT_LIST.txt'
```

The user can utilize the method _**synonym()**_ as below. Every word of the input text that included in the synonym list would be converted into its synonym.

```python
YOUR_TEXT = 'I am a boy'
YOUR_TEXT_AFTER_SYNONYM = text_handler.synonym(text=YOUR_TEXT)
```

### TextHandler: Word Marking
_Not Ready Yet_

### TextHandler: Cleaning
_Not Ready Yet_

<!-- 
### Text Handling
1. Word Marking
2. Cleaning

 -->
<!-- 
## Preprocessing: English
>Sourcecode:
>>_function.py/Preprocess()_  
>>_/tutorial/run_preprocess.py_

The _**Preprocess**_ provides several functions to preprocess the text data in English.  
Mainly utilizes the python library _**nltk**_ as default (e.g., stopword_removal, stemming, lemmatization), but also supports customized preprocessing using the thesaurus lists at _/thesaurus/_. Modify the thesaurus lists for what you need.  
>https://www.nltk.org/

THe default settings of the _**Preprocess**_ are
- use _**stopwords**_ from _**nltk.corpus**_ for the default stoplist
- use _**LancasterStemmer**_ from _**nltk.stem.lancaster**_
- use _**WordNetLemmatizer**_ from _**nltk.stem**_

Import related libraries

```python
# Configuration
from config import Config
with open(FNAME_CUSTOM_CONFIG, 'r') as f: # './blanknlp/custom.cfg'
    cfg = Config(f)

from blanknlp.function import Preprocess
```

Build _**Preprocess**_ model with config.

```python
pr_config = {
    'do_synonym': True,
    'do_lower': True,
    'do_stop': True,
    'stoplist': 'custom'
}
pr = Preprocess(**pr_config)
```

Import docs and preprocess the data.  
The result (i.e., _**docs_prep**_) is a list of preprocessed _**doc**_ (i.e., a list of sentences that cleaned, synonymed, lowered, and customized stopwords removed).

```python
docs_prep = []
for doc in docs:
    sents = pr.cleaning(doc).split('  ')
    docs_prep.append([pr.stemmize(sent) for sent in sents])
```

For the users who need a particular function of _**Preprocess**_, we provide several methods to be used directly.  
Note that each method overlaps the former result as the level of preprocessing goes deeper. For example, the _**stopword_removal**_ requires _**cleaning**_ and _**tokenize**_ as mandatory, and _**synonym**_, _**lower**_, and _**marking**_ as optional.

```python
for doc in docs:
    sents = pr.cleaning(doc).split('  ')
    for sent in sents:
        sent_prep = pr.synonym(sent)
        # sent_prep = pr.marking(sent)
        sent_prep = pr.tokenize(sent)
        sent_prep = pr.stopword_removal(sent)
        print('BEFORE: {}'.format(sent))
        print('AFTER: {}'.format(sent_prep))
```

## Preprocessing: Korean
_Not Ready Yet_ 
-->

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

Import related libraries.

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
Use the function _**read_articles()**_ to read the articles data, which returns a list of _**Article**_.  
If the user wants every article regardless of its date, just input _**YOUR_QUERY**_ and the function would set the date as default value (0 to end).

```python
from blanknlp.web_crawling import read_articles

articles = read_articles(query=YOUR_QUERY, date_from=YOUR_DATE_FROM, date_to=YOUR_DATE_TO) # '교량+사고+유지관리', '20190701', '20190710'
articles = read_articles(YOUR_QUERY)
```

Then the user can use the attributes of each element such as _**url**_, _**title**_, _**date**_, _**content**_, _**comment_list**_.
See _web_crawling.py/Article()_ for more information.  
Note that the content of article commonly starts with a junk text such as _'// flash 오류를 우회하기 위한 함수 추가 function \_flash\_removeCallback() \{\}'_.

```python
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