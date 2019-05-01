from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import itertools
import heapq
from entity.util import nlp,common as cm
import nltk
import pickle
import spacy

#nlp_spacy = spacy.load('en')
nlp_spacy = spacy.load('en_core_web_sm')

class TFIDFExtractor():
    '''
    Word2Vec_features
    '''
    def __init__(self, documents,ngram = (1,3) ,mindf = 1):
        pdocuments = nlp.preprocessed_docs(documents)
        self.raw_documents = documents
        self.docs = [ ' '.join(doc.tokens) for doc in pdocuments]
        self.maxdf = max(len(documents) * 0.90,mindf)
        self.mindf = mindf
        self.ngram = ngram
        self.model = None
        self.trained = False

    def train(self):
        print("TFIDF trained for " + str(self.ngram))

        tf = TfidfVectorizer(analyzer='word', ngram_range=self.ngram,min_df=self.mindf,max_df=self.maxdf )
        self.matrix = tf.fit(self.docs)
        self.model = tf
        self.i2w = tf.get_feature_names()
        self.trained = True

    def save_model(self,filename):
        pickle.dump(self, open(filename,'wb'))


    def get_Topics(self,topicdocs):
        ptopicdocs = nlp.preprocessed_docs(topicdocs)
        docs = [' '.join(doc.tokens) for doc in ptopicdocs]
        if not  self.trained == True:
            self.train()
        matrix = self.model.transform(docs).todense()
        topic_dic = {}
        i = 0
        for doci in ptopicdocs:

            temptokens = zip(matrix[i].tolist()[0], itertools.count())
            temptokens1 = []
            for (x, y) in temptokens:
                if x > 0.0  :
                    temptokens1.append((x, self.i2w[y]))

            topic_dic[doci.id] = temptokens1

            i += 1

        return topic_dic




    def npchunk(self,doc):
        npchunklist = []
        for sen in doc:
            ichunklist = list(nlp_spacy(sen).noun_chunks)
            ichunklist = [ nlp.preprocessText(str(ichunk.text)) for ichunk in ichunklist]
            ichunklist = [ichunk for ichunk in ichunklist if len(ichunk) > 0]
            # ichunklistt = [' '.join(ichunk)  for ichunk in ichunklist if len(ichunk) <= 3 and len(ichunk) > 0]
            for ichunk in ichunklist:
                if len(ichunk) <= 3  and len(ichunk) >0 :
                   npchunklist.append(' '.join(ichunk))
                elif len(ichunk) > 3:
                    newchunks = nltk.ngrams(ichunk,3)
                    for nc in newchunks:
                        npchunklist.append(' '.join(nc))

        return list(set(npchunklist))

    def get_Topics_npFilter(self,topicdocs):
        ptopicdocs = nlp.preprocessed_docs(topicdocs)
        docs = [' '.join(doc.tokens) for doc in ptopicdocs]

        if not  self.trained == True:
            self.train()
        matrix = self.model.transform(docs).todense()
        topic_dic = {}
        i = 0
        for doci in ptopicdocs:
            chunks = self.npchunk(doci.sentences)
            temptokens = zip(matrix[i].tolist()[0], itertools.count())
            temptokens1 = []
            tfidf_dic={}
            for (x, y) in temptokens:
                if x > 0.0 :
                    tfidf_dic[self.i2w[y]] = x

            for chunk in chunks:
                if chunk in tfidf_dic:
                    temptokens1.append((tfidf_dic[chunk],' '.join(nlp.preprocessText(chunk,stemming=False,stopwords_removal=False))))

            topic_dic[doci.id] = temptokens1
            i+=1

        return topic_dic

    def get_Topics_goldFilter(self,topicdocs):
        gconcepts = cm.getConcepts()
        ptopicdocs = nlp.preprocessed_docs(topicdocs)
        docs = [' '.join(doc.tokens) for doc in ptopicdocs]

        if not  self.trained == True:
            self.train()
        matrix = self.model.transform(docs).todense()
        topic_dic = {}
        i = 0
        for doci in ptopicdocs:
            temptokens = zip(matrix[i].tolist()[0], itertools.count())
            temptokens1 = []
            tfidf_dic={}
            for (x, y) in temptokens:
                if x > 0.0 :
                    tfidf_dic[self.i2w[y]] = x

            for chunk in gconcepts:
                if chunk in tfidf_dic:
                    temptokens1.append((tfidf_dic[chunk],chunk))
            if len(temptokens1) == 0:
                temptokens1.append((1,"dummy"))
            topic_dic[doci.id] = temptokens1
            i+=1

        return topic_dic


    def get_Topics_listFilter(self,topicdocs,concept_list_file,gold_concept_file):
        gconcepts = cm.getConcepts(gold_concept_file)
        ptopicdocs = nlp.preprocessed_docs(topicdocs)
        docs = [' '.join(doc.tokens) for doc in ptopicdocs]

        if not  self.trained == True:
            self.train()
        matrix = self.model.transform(docs).todense()
        topic_dic = {}
        i = 0
        for doci in ptopicdocs:
            temptokens = zip(matrix[i].tolist()[0], itertools.count())
            temptokens1 = []
            tfidf_dic={}
            for (x, y) in temptokens:
                if x > 0.0 :
                    tfidf_dic[self.i2w[y]] = x

            for chunk in gconcepts:
                if chunk in tfidf_dic:
                    temptokens1.append((tfidf_dic[chunk],chunk))
            if len(temptokens1) == 0:
                temptokens1.append((1,"dummy"))
            topic_dic[doci.id] = temptokens1
            i+=1

        return topic_dic

    def get_Topics_partialSectionFilter(self,topicdocs,section_wise_folder):
        gconcepts,sectionwise_concepts = getPartialSectionConcepts(section_wise_folder)
        ptopicdocs = nlp.preprocessed_docs(topicdocs)
        docs = [' '.join(doc.tokens) for doc in ptopicdocs]

        if not  self.trained == True:
           self.train()
        matrix = self.model.transform(docs).todense()
        topic_dic = {}
        i = 0
        for doci in ptopicdocs:
           temptokens = zip(matrix[i].tolist()[0], itertools.count())
           temptokens1 = []
           tfidf_dic={}
           for (x, y) in temptokens:
               if x > 0.0 :
                   tfidf_dic[self.i2w[y]] = x

           for chunk in gconcepts:
               if chunk in tfidf_dic:
                   temptokens1.append((tfidf_dic[chunk],chunk))
           if len(temptokens1) == 0:
               temptokens1.append((1,"dummy"))
           topic_dic[doci.id] = temptokens1
           i+=1

        return topic_dic

def getGlobalngrams(grams,documents,threshold):

    singlecorpus = ""
    for doc in documents:
        singlecorpus += ' '+ doc.text + '\n'


    ncorpus = ' '.join(nlp.preprocessText(singlecorpus))
    tf = TfidfVectorizer(analyzer='word', ngram_range=grams, stop_words=nlp.stopwords)
    tfidf_matrix = tf.fit_transform([ncorpus])
    feature_names = tf.get_feature_names()
    doc = tfidf_matrix.todense()
    temptokens = zip(doc.tolist()[0], itertools.count())
    temptokens = [(x, y) for (x, y) in temptokens if x > threshold]
    tokindex = heapq.nlargest(len(temptokens), temptokens)
    global1grams = dict([(feature_names[y],x) for (x, y) in tokindex ])
    topindex = [ (feature_names[y],x)  for (x,y) in tokindex ]
    f = open('data/file'+str(grams[0])+".txt",'w')
    for key in global1grams:
        f.write(key+","+global1grams[key]+"\n")


    return  global1grams,topindex