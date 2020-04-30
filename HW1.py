#!/usr/bin/env python
# coding: utf-8
from argparse import ArgumentParser
from collections import defaultdict
import math
import sys
import os
import time
import csv
import numpy as np
from scipy.sparse import csr_matrix
import xml.etree.ElementTree as ET
from sklearn import preprocessing


parser = ArgumentParser(description='vsmodel')
parser.add_argument('-r', '--relevance_feedback', action='store_true', help="using Rocchio")
parser.add_argument('-i', type=str, default="wm-2020-vsm-model/queries/query-train.xml", dest="query_file")
parser.add_argument('-o', type=str, default="myresults.csv", dest="ranked_list")
parser.add_argument('-m', type=str, default="wm-2020-vsm-model/model/", dest="model_dir")
parser.add_argument('-d', type=str, dest="NTCIR_dir")
args = parser.parse_args()


doc_TF = defaultdict(dict)
doc_freq = defaultdict(int)
doc_len = defaultdict(int)
doc_IDF = defaultdict(float)
vocab_dict = defaultdict(int)
ngram_dict = defaultdict(int)



model_dir = args.model_dir
ranked_list = args.ranked_list
query_file = args.query_file
feedback = args.relevance_feedback



# Create vocab_dict
with open(model_dir + 'vocab.all', 'r') as f:
    vocab = f.read().split('\n') #vocab[word_index] = word
    
    for i, v in enumerate(vocab):
        vocab_dict[v] = i



with open(model_dir + 'file-list', 'r') as f:
    doc_list = f.read().split('\n')
    for i, doc in enumerate(doc_list):
        doc_list[i] = doc.split('/')[-1].lower()
        


doc_id_list = []
term_id_list = []
count_list = []

# Compute TF
with open(model_dir + 'inverted-file', 'r') as f:
    lines = f.readlines()
    i = 0
    term_id = 0
    length = len(lines)
    while i < length:
        
        vocab_id1, vocab_id2, freq = [int(j) for j in lines[i].split(" ")]
        term = vocab[vocab_id1] if vocab_id2 == -1 else vocab[vocab_id1] + vocab[vocab_id2]

        doc_freq[term] = freq
            
        ngram_dict[(vocab_id1, vocab_id2)] = term_id
        
        for k in range(freq):
            doc_id, count = [int(j) for j in lines[i + k + 1].split(" ")]
            doc_TF[term][doc_id] = count
            doc_id_list.append(doc_id)
            term_id_list.append(term_id)
            count_list.append(count)
        
        i += (freq + 1)
        term_id += 1


DOC_TF = csr_matrix((count_list, (doc_id_list, term_id_list)), shape=(len(doc_list), len(ngram_dict)), dtype='float')



DOC_IDF = []

# Compute IDF
for term in list(ngram_dict.keys()):
    idf = math.log((len(doc_list) + 1)/(doc_freq[term]+1))
    doc_IDF[term] = idf
    DOC_IDF.append(idf)

DOC_IDF = np.asarray(DOC_IDF)
DOC_IDF.shape




def compute_query_TF_IDF(query, vocab_dict, ngram_dict, doc_IDF, Wt=2, Wc=1):
    
    title = query['title']
    combine = query['combine']
    TF = np.zeros(len(ngram_dict))
    IDF = np.zeros(len(ngram_dict))
    
    # Process title
    # unigram
    for term in title:
        if term in vocab_dict:
            if (vocab_dict[term], -1) in ngram_dict:
                TF[ngram_dict[(vocab_dict[term], -1)]] += Wt
                IDF[ngram_dict[(vocab_dict[term], -1)]] = doc_IDF[term]
    # bigram
    for i in range(len(title)-1):
        if title[i] in vocab_dict and title[i+1] in vocab_dict and (vocab_dict[title[i]], vocab_dict[title[i+1]]) in ngram_dict:
                TF[ngram_dict[(vocab_dict[title[i]], vocab_dict[title[i+1]])]] += Wt*2
                IDF[ngram_dict[(vocab_dict[title[i]], vocab_dict[title[i+1]])]] = doc_IDF[term]
    
    # Process combine
    # unigram
    for term in combine:
        if term in vocab_dict:
            if (vocab_dict[term], -1) in ngram_dict:
                TF[ngram_dict[(vocab_dict[term], -1)]] += Wc
                IDF[ngram_dict[(vocab_dict[term], -1)]] = doc_IDF[term]
    # bigram
    for i in range(len(combine)-1):
        if combine[i] in vocab_dict and combine[i+1] in vocab_dict and (vocab_dict[combine[i]], vocab_dict[combine[i+1]]) in ngram_dict:
                TF[ngram_dict[(vocab_dict[combine[i]], vocab_dict[combine[i+1]])]] += Wc*2
                IDF[ngram_dict[(vocab_dict[combine[i]], vocab_dict[combine[i+1]])]] = doc_IDF[term]
    
    return TF, IDF




root = ET.parse(query_file).getroot()
topics= root.findall("topic")

query_list = []
query_TF = []
query_IDF = []
query_len = []

for topic in topics:
    query_id = topic.find("number").text.strip().split("ZH")[1]
    title = topic.find("title").text.strip()
    question = topic.find("question").text.strip()[:-1]
    narrative = topic.find("narrative").text.split("。")[0].replace("，","").replace("、","")
    concepts = topic.find("concepts").text.strip().replace("、","")[:-1]
    combine = "".join([question, narrative, concepts])
    combine = combine.replace("查詢","").replace("相關文件內容","").replace("應","").replace("包括","").replace("說明","")

    query = {'query_id':query_id, 'title': title, 'combine': combine}
    query_list.append(query)
    query_len.append(len(combine))

    tf, idf = compute_query_TF_IDF(query, vocab_dict, ngram_dict, doc_IDF)
    query_TF.append(tf) # list of defaultdict
    query_IDF.append(idf)
    
Q_IDF = np.stack(query_IDF).sum(0)
Q_IDF = np.log((len(topics)+1) / (Q_IDF+1))
Q_TF = np.stack(query_TF)
Q_TF = csr_matrix(Q_TF)



def normalize(TF, IDF, k=4, b=0.75):

    doc_length = TF.sum(1)
    avg_length = doc_length.mean()
    
    TF = TF.tocoo()
    

    tmp = TF * (k + 1)
    TF.data += k
    TF.data = tmp.data / TF.data
       
    normalizer = (1 - b + b * doc_length / avg_length) 
    TF.data = TF.data / np.array(normalizer[TF.row]).reshape(len(TF.data),)  

    TF.data *= IDF[TF.col]
    TF = TF.tocsr()
    TF = preprocessing.normalize(TF, norm='l2', axis=1)
    
    return TF




DOC_VEC = normalize(DOC_TF, DOC_IDF)
QUERY_VEC = normalize(Q_TF, DOC_IDF)



if feedback:
    print("Rocchio")
    alpha = 1
    beta = 0.80
    rel_docs = 10
    gamma = 0.15
    nonrel_docs = 10

    nonzero_col = np.unique(QUERY_VEC.indices)
    DOC_VEC = DOC_VEC[:,nonzero_col]
    QUERY_VEC = QUERY_VEC[:,nonzero_col]
    cos = DOC_VEC*(QUERY_VEC.transpose())

    pred = []

    for query_id in range(len(query_list)):
        similarity = []
        for file_id in range(len(doc_list)):
            similarity.append((file_id, cos[file_id, query_id]))
        similarity.sort(key = lambda x: x[1], reverse = True)
        ranking = [(i[0], i[1]) for i in similarity]
        rel_doc_id = [i[0] for i in ranking[:rel_docs]]
        nonrel_doc_id = [i[0] for i in ranking[-nonrel_docs:]]
        for i in rel_doc_id:
            QUERY_VEC[query_id] += (beta / rel_docs * DOC_VEC[i])
        for i in nonrel_doc_id:
            QUERY_VEC[query_id] -= (gamma / nonrel_docs * DOC_VEC[i])
    



nonzero_col = np.unique(QUERY_VEC.indices)
DOC_VEC = DOC_VEC[:,nonzero_col]
QUERY_VEC = QUERY_VEC[:,nonzero_col]

cos = DOC_VEC*(QUERY_VEC.transpose())

pred = []

for query_id in range(len(query_list)):
    similarity = []
    for file_id in range(len(doc_list)):
        similarity.append((file_id, cos[file_id, query_id]))
    similarity.sort(key = lambda x: x[1], reverse = True)
    ranking = [doc_list[i[0]] for i in similarity]
    pred.append(ranking)



with open(ranked_list, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['query_id', 'retrieved_docs'])
    print('Results writing to {}'.format(os.path.join(os.getcwd(), ranked_list)))
    for i, q in enumerate(query_list):
        docs = " ".join(pred[i][:100])
        writer.writerow([q['query_id'], docs])
