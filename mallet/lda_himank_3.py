# -*- coding: utf-8 -*-
"""lda-himank.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12zx-jsZowxlpoN5yMB7CbRrZWVJVBfMu
"""

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import re
import numpy as np
import pandas as pd
from pprint import pprint
import os
from pathlib import Path
import json
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import csv
import random
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.test.utils import datapath
import spacy
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import json

# filename = 'news_cleaned_2018_02_13.csv'
# train_docs = []
# test_docs = []
# i=0
# with open(filename) as f:
#     reader = csv.reader(f)
#     for row in reader:
#         #ignore if not political and short text
#         if(not row[3] == 'political' or len(row[5]) < 1000):
#             continue
#         if(i >= 5100):
#             break
#         train_docs.append(row[5])
#         i+=1
# train_docs = train_docs[:5000]
# test_docs = test_docs[5000:]

# with open('sampled_te_docs_10k.json', 'r') as outfile:  
#     test_docs = json.load(outfile)
with open('sampled_tr_docs_10k.json', 'r') as outfile:  
    train_docs = json.load(outfile)
    
stop = set(stopwords.words('english'))
lemma = WordNetLemmatizer()
def clean(doc):
    deacc = gensim.utils.simple_preprocess(doc, deacc=True, min_len=4)
    stop_free = [i for i in deacc if i not in stop]
    normalized = [lemma.lemmatize(word) for word in stop_free]
    return normalized
train_docs_clean = [clean(doc) for doc in train_docs]
# test_docs_clean = [clean(doc) for doc in test_docs]

dictionary = corpora.Dictionary(train_docs_clean)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in train_docs_clean]

# test_dictionary = corpora.Dictionary(test_docs_clean) 
# test_doc_term_matrix = [dictionary.doc2bow(text) for text in test_docs_clean]


Lda = gensim.models.wrappers.LdaMallet
# ldamodel = Lda(doc_term_matrix, num_topics=100, id2word = dictionary, passes=50)
ldamodel = Lda('Mallet/bin/mallet', corpus=doc_term_matrix, num_topics=150, id2word = dictionary, optimize_interval=10, iterations=5000, alpha=0.05)
temp_file = "model_10k_5000_topics_150"
ldamodel.save(temp_file)

####### LOAD MODEL
# temp_file = datapath("{}model_LDA_mallet_15k".format(base_path))
# ldamodel = Lda.load("/model")

# print(ldamodel.print_topics(num_words=3))
len(train_docs)

doc_num = 28
print(train_docs[doc_num])

tops = ldamodel[doc_term_matrix[doc_num]]
tops = sorted(tops, key=lambda a:a[1], reverse=True)
for top in tops[:10]:
    print(top[1], ldamodel.show_topic(top[0]))
    print("\n")
    
# tops = ldamodel.get_document_topics(doc_term_matrix[doc_num])
# tops = sorted(tops, key=lambda a:a[1], reverse=True)
# for top in tops:
#     print(top[1], top[0])
#     print(ldamodel.print_topic(top[0]))

# doc_num = 7
# print(test_docs[doc_num])

# tops = ldamodel[test_doc_term_matrix[doc_num]]
# tops = sorted(tops, key=lambda a:a[1], reverse=True)
# for top in tops[:10]:
#     print(top[1], ldamodel.show_topic(top[0]))
#     print("\n")
    
# tops = ldamodel.get_document_topics(doc_term_matrix[doc_num])
# tops = sorted(tops, key=lambda a:a[1], reverse=True)
# for top in tops:
#     print(top[1], top[0])
#     print(ldamodel.print_topic(top[0]))

# test_doc_num = 70

# print(test_docs[test_doc_num])
# tops = ldamodel.get_document_topics(test_doc_term_matrix[test_doc_num])
# tops = sorted(tops, key=lambda a:a[1], reverse=True)
# for top in tops:
#     print(top[1])
#     print(ldamodel.print_topic(top[0]))