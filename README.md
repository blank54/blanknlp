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