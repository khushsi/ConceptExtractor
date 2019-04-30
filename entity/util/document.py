import csv
from nltk.tokenize import sent_tokenize
import pandas as pd
import sys

class Document:
    def __init__(self, *args, **kwargs):
        self.sentences = []
        self.npchunks = []
        self.type = "general"
        self.id = args[0]
        self.text = args[1]
        self.type= args[0].split("-")[0]
        self.otherfields = {}
        if len(args) > 2:
            self.otherfields=args[2]

        sen_list = sent_tokenize(self.text)

        for sen in sen_list:
            self.sentences.append(sen)
            # print(sen)
        self.no_sent = len(self.sentences)

    def __str__(self):
        return '%s\t%s' % (self.id, self.text)


def load_document(path,booknames=[],textfield=['text'],idfield="id",otherfields=[],idelimiter=',',booknamefield='bookname'):
    print('Start loading documents from %s' % path)
    doc_list = []
    df = pd.read_csv(path,header=0,delimiter=idelimiter)

    for index,row in df.iterrows():
        if(len(booknames) == 0  or ( row[booknamefield].startswith(tuple(booknames)))):
            # print(row[idfield])
            text = [ str(row[field]) for field in textfield]
            text = text  + [" "]
            otherfields_dict = { column:row[column] for column in df.columns if (column not in textfield and column != idfield and column in otherfields)}
            doc = Document(str(row[idfield]),' '.join(text),otherfields_dict)
            doc_list.append(doc)
    return doc_list




