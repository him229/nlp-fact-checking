# -*- coding: utf-8 -*-
from collections import defaultdict
import json
import csv
import sys
import pprint
from random import shuffle, uniform
from twilio.rest import Client

csv.field_size_limit(sys.maxsize)

account_sid = 'AC4485958b112cad72f2fc674245e62d80'
auth_token = '71b13dc22b69f2d57d407631494561f3'
client = Client(account_sid, auth_token)

def tw_msg(msg):
    message = client.messages \
                .create(
                     body=msg,
                     from_='+17372104735',
                     to='+18722256140'
                 )

# Reading in 10k articles for each category
CATEGORY_COUNT = 15000
TEST_RATIO = 0.01
total_article_count = 8 * CATEGORY_COUNT
rand_thresh_l = 0.05
rand_thresh_h = 0.25
filename = 'news_cleaned_2018_02_13.csv'

article_count = defaultdict(int)
article_count_final = defaultdict(int)
train_docs = []
test_docs = []
i=0

domains = defaultdict(set)
all_docs = defaultdict(list)

with open(filename) as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            tag = row[3]
            document_txt = row[5]
            rand_val = uniform(0,1)
            if(len(document_txt) < 1000 or tag=='unknown' or tag=="" or tag=="satire" or tag=="clickbait" or tag=="junksci" or tag=="rumor"):
                continue
            if((tag=="unreliable" or tag=="hate") and rand_val > rand_thresh_h):
                continue
            if((tag!="unreliable" or tag!="hate") and rand_val > rand_thresh_l):
                continue
            article_count[tag] += 1
            domains[tag].add(row[2])
            all_docs[tag].append(row)
            pprint.pprint(dict(article_count), width=1)
            pprint.pprint(dict(domains), width=1)
            print(row[0])
        except Exception as e:
            print(e)

with open(filename) as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            tag = row[3]
            document_txt = row[5]
            rand_val = uniform(0,1)
            if(len(document_txt) < 1000 or tag=='unknown' or tag=="" or tag=="satire" or tag=="clickbait" or tag=="junksci" or tag=="rumor" or rand_val > rand_thresh):
                continue
            article_count[tag] += 1 
            domains[tag].add(row[2])
            all_docs[tag].append(row)
            pprint.pprint(dict(article_count), width=1)
            pprint.pprint(dict(domains), width=1)
            print(row[0])
        except Exception as e:
            print(e)

with open('data/sampled_all_docs_all.json', 'w') as outfile:  
    json.dump(all_docs, outfile)
    
for k,v in all_docs.items():
    cat_docs = all_docs[k]
    shuffle(cat_docs)
    cat_docs = cat_docs[:CATEGORY_COUNT]
    all_docs[k] = cat_docs
    article_count_final[k] = len(cat_docs)

pprint.pprint(dict(article_count), width=1)
pprint.pprint(dict(article_count_final), width=1)
pprint.pprint(dict(domains), width=1)

tw_msg(json.dumps(dict(article_count)))
tw_msg(json.dumps(dict(article_count_final)))
tw_msg(json.dumps(dict(domains)))

with open('data/sampled_all_docs_15k.json', 'w') as outfile:  
    json.dump(all_docs, outfile)