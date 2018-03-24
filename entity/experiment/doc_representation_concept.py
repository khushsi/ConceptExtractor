# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import os,sys
import numpy as np

from entity.config import Config
from entity.data import DataProvider
from entity.rank_metric import mean_average_precision
from scipy import spatial


__author__ = "Khushboo Thaker"
__email__ = "kmt81@pitt.edu"


tags_of_interest = [ "Data_pre-processing" , "Document_classification" , "Document_clustering" , "Evaluation_measures_(information_retrieval)"  , "Language_model" , "Probabilistic_relevance_model" , "Query_understanding" , "Relevance_feedback" , "Search_data_structure" , "Search_engine_indexing"  ,  "Tf–idf"]


# model_name = ['doc2vec','attr']
model_dirs = { 'doc2vec':'prodx_doc2vec'}#, 'attr':'attribute-argumented_model.freq=10'}




if __name__ == '__main__':
    # print('Run experiment for %s' % model_name)
    ########################## Load the X-y pairs #############################
    print('Load the X-y pairs')

    for model_name in model_dirs:
        args = sys.argv
        args = [args[0], model_dirs[model_name], "prod", "200", "1"]
        flag = args[1]
        n_processer = int(args[4])
        conf = Config(flag, args[2], int(args[3]))
        print(flag)

        dp = DataProvider(conf)
        doc_embed = np.load(conf.path_doc_npy+'.npy')
        print(len(dp.idx2prod))

        irfolders=["TFIDF3"]

        combineresults={}

        for ir_directory in irfolders:
            print(ir_directory ,end=" ")
            ir_directory =  "data/keyphrase_output/" + ir_directory +"/"
            fileobj = {}
            wikiobj = {}
            fcount={}

            for filename in os.listdir(ir_directory):

                if filename.endswith('phrases') and filename.startswith('iirtest-'):
                    print(filenamek)
                    filenamek = filename.split(".")[0].replace(" ","_")
                    fileobj[filenamek] = np.zeros(200)
                    fcount[filenamek]=0
                    with open(ir_directory+filename,'r') as irfile:
                        # print(irfile)
                        for line in irfile.readlines():
                            line = line.split(",")[0]
                            concept = ' '.join(line.split(" "))
                            # print(concept)
                            if concept in dp.idx2prod :
                                print("found",concept)
                                idx = dp.idx2prod.tolist().index(concept)
                                fcount[filenamek] += 1
                                fileobj[filenamek] += doc_embed[0][idx]

                if filename.startswith("wikitest-"):
                    filenamek = filename.split(".")[0].lower().replace(" ","_")
                    wikiobj[filenamek] = np.zeros(200)
                    fcount[filenamek] = 0
                    print(filenamek)
                    with open(ir_directory+filename,'r') as irfile:
                        for line in irfile.readlines():
                            line = line.split(",")[0]
                            concept = ' '.join(line.split(" "))
                            if(concept in dp.idx2prod):
                                fcount[filenamek] += 1
                                idx = dp.idx2prod.tolist().index(concept)
                                wikiobj[filenamek] += doc_embed[0][idx]
            docwikiscore = {}
            import queue as Q

            old_err_state = np.seterr(divide='raise')
            ignored_states = np.seterr(**old_err_state)

            for doc in wikiobj.keys():
                if(fcount[doc] > 0) :
                    wikiobj[doc] = np.divide(wikiobj[doc] , np.ones(200) * fcount[doc])
            #
            for doc in fileobj.keys():
                if(fcount[doc] > 0) :
                    fileobj[doc] = np.divide(fileobj[doc] , np.ones(200) * fcount[doc])

            for doc in fileobj.keys():
                docwikiscore[doc] = Q.PriorityQueue()
                for wiki in wikiobj.keys():
                    docwikiscore[doc].put((spatial.distance.cosine(fileobj[doc],wikiobj[wiki]) ,wiki))


            wikiannotations = 'data/textbook_wiki.csv'
            file = open(wikiannotations,'r')
            import csv
            reader = csv.reader(file)
            true = {}
            for row in reader:
                results=row[1].lower().split(",")
                true[row[0]]=list(map(str.strip, results))

            # print(true)

            prediction = {}
            for doc in fileobj.keys():

                prediction[doc] = []
                sizeofpred = docwikiscore[doc].qsize()
                for i in range(sizeofpred):
                    prediction[doc].append(docwikiscore[doc].get(0)[1])


            # print(prediction)

            results=[]
            for keyd in prediction.keys():
                # print(keyd)
                # print(true[keyd])
                tempresult=[]
                if keyd in true.keys():
                    pcount = 0
                    for pred in prediction[keyd]:
                        # print(pred)
                        pcount = pcount + 1
                        if keyd not in combineresults.keys():
                            combineresults[keyd] = []
                        combineresults[keyd].append((pred, pcount))

                        if(pred in true[keyd]):
                            tempresult.append(1)
                        else:
                            tempresult.append(0)
                    results.append(tempresult)

            # print(results)

            print(mean_average_precision(results))