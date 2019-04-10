# -*- coding: utf-8 -*-
from collections import defaultdict
import json
import csv
import sys
import pprint
from random import shuffle
# from twilio.rest import Client

csv.field_size_limit(sys.maxsize)

# account_sid = ''
# auth_token = ''
# client = Client(account_sid, auth_token)

# def tw_msg(msg):
#     message = client.messages \
#                 .create(
#                      body=msg,
#                      from_='',
#                      to=''
#                  )

# Reading in 10k articles for each category
CATEGORY_COUNT = 15000
TEST_RATIO = 0.01
total_article_count = 11 * CATEGORY_COUNT
article_count = defaultdict(int)
article_count_train = defaultdict(int)
article_count_test = defaultdict(int)
filename = 'news_cleaned_2018_02_13.csv'
train_docs = []
test_docs = []
i=0
with open(filename) as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            document_txt = row[5]
            tag = row[3]
            #stop if all articles max reached
            if(sum(article_count.values()) >= total_article_count):
                break
            #ignore if category max reached or short text or unknown tag
            if(article_count[tag] >= CATEGORY_COUNT or len(document_txt) < 1000 or tag=='unknown' or tag==""):
                continue
            if(article_count[tag] > int(TEST_RATIO*CATEGORY_COUNT)):
                train_docs.append(document_txt)
                article_count_train[tag] += 1
            else:
                test_docs.append(document_txt)
                article_count_test[tag] += 1
            article_count[tag] += 1
            if(article_count[tag] >= CATEGORY_COUNT-10):
                print(tag)
        except:
            print(row)
                
shuffle(train_docs)
shuffle(test_docs)
            
print(len(train_docs))
print(len(test_docs))
pprint.pprint(dict(article_count), width=1)
pprint.pprint(dict(article_count_train), width=1)
pprint.pprint(dict(article_count_test), width=1)

# tw_msg(json.dumps(dict(article_count)))
# tw_msg(json.dumps(dict(article_count_train)))
# tw_msg(json.dumps(dict(article_count_test)))


with open('train_lda.json', 'w') as outfile:  
    json.dump(train_docs, outfile)
with open('test_lda.json', 'w') as outfile:  
    json.dump(test_docs, outfile)


# with open('test_lda.json', 'r') as outfile:  
#     test_docs = json.load(outfile)
# with open('train_lda.json', 'r') as outfile:  
#     train_docs = json.load(outfile)