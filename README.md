# blanknlp
A bunch of python codes to analyze text data in the construction industry.
Mainly reconstitute the pre-exist python libraries for TM and NLP.

## _Project Information_
- Supported by C!LAB (@Seoul Nat'l Univ.)
- First Release: 2020.02.12.

## _Contributors_
Seonghyeon Boris Moon (blank54@snu.ac.kr, https://github.com/blank54/)
Gitaek Lee (lgt0427@snu.ac.kr)
Taeyeon Chang (jgwoon1838@snu.ac.kr, _a.k.a. Kowoon Chang_)

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
>Sourcecode: _/test/word_network.py_

A customized class _**WordNetwork**_ was developed to facilitate the usage of the python library _**networkx**_.
The default setting is _count_option=='dist'_, _save_plt==True_, and _show_plt==True_. The user can customize the setting by _**config**_.

```python
import os
from visualize import *

wn_config = {
    'docs': docs,
    'save_plt': save_plt
}
word_network = WordNetwork(**wn_config)
```