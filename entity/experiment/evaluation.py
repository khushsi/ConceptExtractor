# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import os,sys
import numpy as np
from scipy import spatial
from entity.rank_metric import  ndcg_at_k, mean_average_precision
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import csv
import queue as Q

__author__ = "Khushboo Thaker"
__email__ = "kmt81@pitt.edu"

f_textbook_textbook = 'data/textbook_textbook.txt'
f_textbook_wiki = 'data/textbook_wiki.csv'
f_textbook_pub = 'data/textbook_pub.csv'
BASE_FOLDER = 'data/keyphrase_output/'

DIRECT_VECTOR = 'directvectors'
WORD_VECTOR = 'wordvectors'
CONCEPT_VECTOR = 'conceptvectors'

types = [DIRECT_VECTOR,WORD_VECTOR,CONCEPT_VECTOR]

def readConceptVectors(filepath):
    text = []
    with open(filepath, 'r') as irfile:
        for line in irfile.readlines():
            line = line.split(",")[0]
            text.append(line)
    return text


def readWordVectors(filepath):
    text = []
    with open(filepath, 'r') as irfile:
        for line in irfile.readlines():
            line = line.split(",")[0]
            # line = ' '.join(line.lower().split(" "))
            text.append(line)
    return ' '.join(text)

def direct_word_cosine(text1, text2):
    vectorizer = TfidfVectorizer(analyzer='word', stop_words=[])
    tfidf = vectorizer.fit_transform([text1, text2])

    return(1.0 - (((tfidf * tfidf.T).A)[0,1]))

def NDCG(results):
    ndcg1 = 0
    ndcg3 = 0
    for result in results:
        ndcg1 += ndcg_at_k(result,1,method=1)
        ndcg3 += ndcg_at_k(result, 3, method=1)
    ndcg1 = ndcg1/len(results)
    ndcg3 = ndcg3/len(results)

    return str(ndcg1) + " " + str(ndcg3)

class TestDataset:
    def __init__(self, *args, **kwargs):
        self.metric =  mean_average_precision
        self.name = "Textbook - ?"
        self.testmap={}
        self.kvalues = None
        self.startswith = ['iir-','iir-']






def LoadVectors(testdataset, folder,type=DIRECT_VECTOR):
    fileobj = {}
    file2obj = {}
    for filename in os.listdir(folder):
        if filename.startswith(testdataset.bookname[0]):
            filenamek = filename.split(".")[0].lower().replace(" ","_")
            # print(filenamek)
            if type == DIRECT_VECTOR:
                fileobj[filenamek] = pickle.load(open(folder + filename, 'rb'))
            elif type == WORD_VECTOR:
                fileobj[filenamek] = readWordVectors(folder + filename)


    for filename in os.listdir(folder):
        if filename.startswith(testdataset.bookname[1]):
            filenamek = filename.split(".")[0].lower().replace(" ","_")
            # print(filenamek)
            if type == DIRECT_VECTOR:
                file2obj[filenamek] = pickle.load(open(folder + filename, 'rb'))
            elif type == WORD_VECTOR:
                file2obj[filenamek] = readWordVectors(folder + filename)

    scorevector = {}
    prediction = {}
    results = []
    for doc in fileobj.keys():
        scorevector[doc] = Q.PriorityQueue()
        for wiki in file2obj.keys():
            if type == DIRECT_VECTOR:
                scorevector[doc].put((round(spatial.distance.cosine(fileobj[doc], file2obj[wiki]),8), wiki))
            elif type == WORD_VECTOR:
                scorevector[doc].put((round(direct_word_cosine(fileobj[doc], file2obj[wiki]), 8), wiki))
        prediction[doc] = [scorevector[doc].get(0)[1] for x in range(scorevector[doc].qsize())]
        # print(prediction)

        if(doc in testdataset.testmap.keys()):
            # print(doc,"present")
            tempresult = []
            # print(testdataset.testmap[doc])
            for pred in prediction[doc]:
                if pred in testdataset.testmap[doc]:
                    tempresult.append(testdataset.testmap[doc][pred])
                else:
                    tempresult.append(0)
            # print(tempresult)
            results.append(tempresult)

    score = testdataset.metric(results)

    return fileobj,file2obj, prediction,results,score




if __name__ == '__main__':
    print('Run experiment for %s' % "vector space model")
    ########################## Load the X-y pairs #############################
    print('Load the X-y pairs')

    import pickle


## define datasets

    t2t_dataset = TestDataset()
    t2t_dataset.metric = NDCG
    t2t_dataset.name = "Textbook - Textbook"
    t2t_dataset.bookname = ['iir-', 'mir-']
    reader = csv.reader(open(f_textbook_textbook, 'r'), delimiter="\t")

    for row in reader:
        t2t_dataset.testmap[row[0]] = {}
        for val in row[1].split(","):
            text = val.split(":")[0].replace(".txt.phrases", "")
            score = float(val.split(":")[1])
            t2t_dataset.testmap[row[0]].update({text: score})

    score = 1
    t2w_dataset = TestDataset()
    t2w_dataset.metric = mean_average_precision
    t2w_dataset.name = "Textbook - wikipedia"
    reader = csv.reader(open(f_textbook_wiki, 'r'), delimiter=",")
    t2w_dataset.bookname = ['iirtest-', 'wikitest-']
    for row in reader:
        t2w_dataset.testmap[row[0]] = {}
        for val in row[1].split(","):
            text = val.split(":")[0].replace(".txt.phrases", "")
            t2w_dataset.testmap[row[0]].update({text: score})

    t2p_dataset = TestDataset()
    t2p_dataset.metric = mean_average_precision
    t2p_dataset.name = "Textbook - publications"
    reader = csv.reader(open(f_textbook_pub, 'r'), delimiter=",")
    t2p_dataset.bookname = ['chapterwiseiir-', 'iirbookpubs-']

    for row in reader:
        t2p_dataset.testmap[row[0]] = {}
        for val in row[2].split(","):
            text = val.split(":")[0].replace(".txt.phrases", "")
            t2p_dataset.testmap[row[0]].update({text: score})

    test_datasets = [t2p_dataset, t2t_dataset, t2w_dataset]

    concept_folders = {}
    concept_folders[DIRECT_VECTOR] = ['Doc2Vec']#['tLDA200','tsLDA200','tLDA250','tsLDA250','LDA200','LDA210','LDA220','LDA230','LDA240','LDA250','LDA260','LDA270','LDA280','LDA290','LDA300','LDA400','LDA500','UNIGRAM']
    # concept_folders[WORD_VECTOR] = ['greedy-acm','greedy-wiki']
    # concept_folders[CONCEPT_VECTOR] = ['TFIDFNP']

    for dataset in test_datasets:
        for type in concept_folders.keys():
            for ir_folder in concept_folders[type]:

                fileobj,file2obj,prediction, results,score = LoadVectors(dataset,BASE_FOLDER+ir_folder+"/",type=type)
                print(dataset.name, type, ir_folder, score)


