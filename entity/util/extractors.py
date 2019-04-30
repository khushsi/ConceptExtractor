import heapq
from entity.util.config import config
import csv
import pandas as pd
import os

def extract_top_kcs(doc2concepts, output_dir,topk=-1,threshold = 0.0,define_kc=False):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    topicnames = []
    columns = ['concept','weight']
    list_con = []

    for id in doc2concepts:
        if define_kc:
            if len(topicnames) == 0:
                topicnames = ["kc" + str(i) for i in range(len(doc2concepts[id]))]

        if define_kc:
            temptokens = list(zip(doc2concepts[id],topicnames))
        else:
            temptokens = doc2concepts[id]

        temptokens1 = []
        for (x, y) in temptokens:
            if x > threshold:
                temptokens1.append((round(x,4),y))
        if topk == -1 or len(temptokens1) < topk:
            topklen = len(temptokens1)
        else:
            topklen = topk

        tokindex = heapq.nlargest(topklen, temptokens1)
        filew = open(output_dir+config.dir_sep+str(id)+config.file_ext,'w')
        csvwriter = csv.writer(filew)
        csvwriter.writerow(columns)

        for (x,y) in tokindex:
            csvwriter.writerow([y,x])
            list_con.append([id,y,x])

        filew.close()

    df = pd.DataFrame(list_con,columns=['item_id','concept','weight'])
    return

