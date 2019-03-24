# -*- coding: utf-8 -*-

import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
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



"""
doc_type = 'train'
doc_num = 3092       ## Train doc_num range (0-4999)
 
 OR 

doc_type = 'test'
doc_num = 70         ## Test doc_num range (0-99)
"""
########### CHANGE THIS
doc_type = 'train'
doc_num = 3091
###################






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

with open('test_lda.json', 'r') as outfile:  
    test_docs = json.load(outfile)
with open('train_lda.json', 'r') as outfile:  
    train_docs = json.load(outfile)



stop = set(stopwords.words('english'))
lemma = WordNetLemmatizer()
def clean(doc):
    deacc = gensim.utils.simple_preprocess(doc, deacc=True, min_len=4)
    stop_free = [i for i in deacc if i not in stop]
    normalized = [lemma.lemmatize(word) for word in stop_free]
    return normalized
train_docs_clean = [clean(doc) for doc in train_docs]
test_docs_clean = [clean(doc) for doc in test_docs]

dictionary = corpora.Dictionary(train_docs_clean)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in train_docs_clean]

test_dictionary = corpora.Dictionary(test_docs_clean) 
test_doc_term_matrix = [dictionary.doc2bow(text) for text in test_docs_clean]

Lda = gensim.models.wrappers.LdaMallet

####### SAVE MODEL 
# model_path = '/Users/himankyadav/Downloads/mallet-2.0.8/bin/mallet'
# ldamodel = Lda(model_path, corpus=doc_term_matrix, num_topics=100, id2word = dictionary, optimize_interval=10)
# temp_file = datapath("model_LDA_mallet")
# ldamodel.save(temp_file)

####### LOAD MODEL
temp_file = datapath("model_LDA_mallet")
ldamodel = Lda.load('model_LDA_mallet')

# print(ldamodel.print_topics(num_words=3))
# len(train_docs)


if doc_type=='train':
    print(train_docs[doc_num])
    print("\n")
    tops = ldamodel[doc_term_matrix[doc_num]]
    tops = sorted(tops, key=lambda a:a[1], reverse=True)
    for top in tops[:10]:
        print(top[1], ldamodel.show_topic(top[0]))
        print("\n")
elif doc_type=='test':
    print(test_docs[doc_num])
    print("\n")
    tops = ldamodel[test_doc_term_matrix[doc_num]]
    tops = sorted(tops, key=lambda a:a[1], reverse=True)
    for top in tops[:10]:
        print(top[1], ldamodel.show_topic(top[0]))
        print("\n")


# print(datapath('a'))

# FNN = 'D:/6740/Articles/'
# text_data = []
# train_docs = []
# test_docs = []
# test_text = []
# i = 0
# for filename in os.listdir(FNN):
#     if filename.endswith('.csv') and not 'BuzzFeed_Fake_1-Webpage' in filename:
#         print(filename)
#         with open(FNN + filename, encoding='utf8') as f:
#             reader = csv.reader(f)
#             for row in reader:
#                 if(not row[3] == 'political'):
#                     continue
#                 if(i > 5000):
#                     break
#                 if(not len(row) == 17):
#                     print(row)
#                     continue
#                 text = get_tokens(row[5])
#                 #text_data = []
#                 r = random.random()
#                 if(len(text) > 0 and r > .01):
#                     text_data.append(text)
#                     train_docs.append(row[5])
#                     i = i+1
#                 elif(len(text) > 0):
#                     test_docs.append(text)
#                     test_text.append(row[5])
#                 '''
#                 sentences = d['text'].split("\n\n")
#                 for sent in sentences:
#                     text = get_tokens(sent)
#                     text_data.append(text)
#                 '''
# model = get_model(text_data, 5)

# model.print_topics(num_words=5, num_topics=100)

# #restricted to politics
# dictionary = corpora.Dictionary(text_data) 
# corpus = [dictionary.doc2bow(text) for text in text_data]
# i = 0
# while i < len(corpus):
#     print("\n\n")
#     print(train_docs[i])
#     tops = model.get_document_topics(corpus[i])
#     tops = sorted(tops, key=lambda a:a[1], reverse=True)
#     for top in tops:
#         print(top[1])
#         print(model.print_topic(top[0]))
#     i = i+ 1

# dictionary = corpora.Dictionary(test_docs) 
# corpus = [dictionary.doc2bow(text) for text in test_docs]
# i = 0
# for doc in corpus:
#     print("\n\n")
#     print(test_text[i])
#     tops = model.get_document_topics(doc)
#     for top in tops:
#         print(top[1])
#         print(model.print_topic(top[0]))
#     i = i+ 1

