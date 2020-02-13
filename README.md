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

## Data
We provide some pickled data for tutorial.
The users can reach it as below.

```python
import os
import pickle as pk

fname_docs_sample = './data/sample.pk'
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

## Word Network
>Sourcecode: _/test/run_word_network.py_

A customized class _**WordNetwork**_ was developed to facilitate the usage of the python library _**networkx**_.

The default settings are
- word combination weighting based on the _**distance**_ within the sentence
- top _**50 words**_ to be shown in the network
- save the network with filename _**'tmp_word_network.png'**_ in directory _**'./result/word_network/'**_.

The user can customize the settings by _**config**_. See _visualize.py/WordNetwork_ for detail options.

```python
import os
from visualize import *

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
>Sourcecode: _/test/run_embedding.py_

### TF-IDF
Term Frequency-Inverse Document Frequency (TF-IDF), One of the most simple and general text embedding techniques is provided. We utilized _**TfidfVectorizer**_ from _**sklearn**_ library.
>https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html?highlight=tfidf#sklearn.feature_extraction.text.TfidfVectorizer

Import related libraries.

```python
# Configuration
import os

import sys
sys.path.append(YOUR_ROOT_DIRECTORY)
from function import makedir
from embedding import embedding_tfidf, embedding_doc2vec
```

The TF-IDF embedding model requires _**tagged_docs**_, of which format is _**list of tuple(id, doc)**_. Note that a _**doc**_ is a _**list of word**_ (i.e., [w1, w2, ...]).
```python
# Data Import
with open(FNAME_DOCS, 'rb') as f:
    docs = pk.load(f)

# Data Preparation
tagged_docs = [(str(idx), ' '.join(doc)) for idx, doc in enumerate(docs)]
```

Embedding TF-IDF and save the model. The TF-IDF model is composed of three items (i.e., _**id2idx**_, _**tfidf_matrix**_, _**tfidf_vocab**_).
- **id2idx**: a dictionary that returns the _**row index of tfidf_matrix**_ of each _**id of doc**_ in _**tagged_docs**_.
- **tfidf_matrix**: a matrix of numeric value (i.e., TF-IDF) with row length of _**# of document**_ and column length of _**vocabulary size**_.
- **tfdif_vocab**: a dictionary of vocabularies used in whole _**tagged_docs**_, which returns the _**column index of tfidf_matrix**_ of each term.

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
The sample code assumed that the user needs the tfidf vector of _**document id 7**_.

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