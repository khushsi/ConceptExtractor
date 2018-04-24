# encoding: utf-8
from __future__ import generators
from collections import Counter
import pickle
import os
import spacy
import re
import csv
import marisa_trie
import nltk
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import sys
from nltk.stem.porter import PorterStemmer
import heapq
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.tokenize import sent_tokenize
import gensim
from gensim import corpora
from gensim.matutils import sparse2full
from collections import defaultdict
import numpy as np
import pandas as pd

IR_CORPUS = 'data/iirmirbook.tsv'
KC_CORPUS = 'data/conceptdocs.csv'


nlp = spacy.load('en')
IS_STEM=False
REMOVE_STOPWORDS=True

print("default stemming :",IS_STEM)
print("default stopword removal :",REMOVE_STOPWORDS)
stemmer = PorterStemmer()

def stem(word):
    stemword = word.strip()
    if(len(stemword) > 3):
        stemword = stemmer.stem(stemword)

    return stemword

def multiwordstem(word_list ):
    for i in range(len(word_list)):
        word_list[i] = stem(word_list[i])
    return ' '.join(word_list)

def preprocessToken(text):
    return re.sub(r'\W+|\d+', '', text.strip().lower())

def preprocessText(text,stemming=True,stopwords_removal=True):
    # print(text)
    text = re.sub("[ ]{1,}",r' ',text)
    text = re.sub(r'\W+|\d+', ' ', text.strip().lower())
    tokens = [token.strip()  for token in text.split(" ")]
    tokens = [token for token in tokens if len(token) > 1]
    if stopwords_removal:
        tokens = [token for token in tokens if token not in stopwords]
    if stemming:
        tokens = [stem(token) for token in tokens ]

    tokens = [token.strip() for token in tokens if len(token.strip()) > 1]
    return tokens

class KeywordList:
    def __init__(self, name):
        self.name = name
        self.wordlist = WORDLIST_PATH[name]
        print (name)
        self.triepath = TRIE_CACHE_DIR+name+'_trie_dict.cache'
        self.trie = self.load_trie(self.triepath)

    def load_trie(self, trie_cache_file):
        '''
        Load a prebuilt tree from file or create a new one
        :return:
        '''
        trie = None

        if os.path.isfile(trie_cache_file):
            print('Start loading trie from %s' % trie_cache_file)
            with open(trie_cache_file, 'rb') as f:
                trie = pickle.load(f)
        else:
            print('Trie not found, creating %s' % trie_cache_file)
            count = 0
            listwords = []
            dict_files = [self.wordlist]
            for dict_file in dict_files:
                print(dict_file)
                file = open(dict_file, 'r')
                for line in file:
                    print(line)
                    tokens = preprocessText(line,stemming=IS_STEM,stopwords_removal=REMOVE_STOPWORDS)
                    if(len(tokens)>0):
                        listwords.append(tokens)

            trie = MyTrie(listwords)
            with open(trie_cache_file, 'wb') as f:
                pickle.dump(trie, f)
        return trie

def KnuthMorrisPratt(text, pattern):

    '''Yields all starting positions of copies of the pattern in the text.
Calling conventions are similar to string.find, but its arguments can be
lists or iterators, not just strings, it returns all matches, not just
the first one, and it does not need the whole text in memory at once.
Whenever it yields, it will have read the text exactly up to and including
the match that caused the yield.'''

    # allow indexing into pattern and protect against change during yield
    pattern = list(pattern)

    # build table of shift amounts
    shifts = [1] * (len(pattern) + 1)
    shift = 1
    for pos in range(len(pattern)):
        while shift <= pos and pattern[pos] != pattern[pos-shift]:
            shift += shifts[pos-shift]
        shifts[pos+1] = shift

    # do the actual search
    startPos = 0
    matchLen = 0
    for c in text:
        while matchLen == len(pattern) or \
              matchLen >= 0 and pattern[matchLen] != c:
            startPos += shifts[matchLen]
            matchLen -= shifts[matchLen]
        matchLen += 1
        if matchLen == len(pattern):
            yield startPos

__author__ = 'Memray'
'''
A self-implenmented trie keyword matcher
'''
#KEYWORD_LIST_PATH = '/home/memray/Project/acm/ACMParser/resource/data/phrases/'
KEYWORD_LIST_PATH = 'data/keyphrase/wordlist/'
ACL_KEYWORD_PATH = KEYWORD_LIST_PATH + 'acl_keywords.txt'
ACM_KEYWORD_PATH = KEYWORD_LIST_PATH + 'acm_keywords_168940.txt'
MICROSOFT_KEYWORD_PATH = KEYWORD_LIST_PATH + 'microsoft_keywords.txt'
WIKI_KEYWORD_PATH = KEYWORD_LIST_PATH + 'wikipedia_14778209.txt'
WIKI_LINKS_PATH = KEYWORD_LIST_PATH + 'link.txt'
WIKI_SECTION_PATH = KEYWORD_LIST_PATH + 'section.txt'
WIKI_ITALICS_PATH = KEYWORD_LIST_PATH + 'italics.txt'
WIKI_MERGEALL_PATH = KEYWORD_LIST_PATH + 'mergeall.final.txt'

WORDLIST_DIR = 'data/keyphrase/wordlist/'
TRIE_CACHE_DIR = 'data/keyphrase/extracted_keyword/'
WORDLIST_PATH = {'greedy-wiki':WORDLIST_DIR+'wikipedia_14778209.txt',
                 'greedy-acm':WORDLIST_DIR+'acm_keywords_168940.txt',
                 'wikilink': WORDLIST_DIR + 'link.txt',
                 'wikiitalic': WORDLIST_DIR + 'italics.txt',
                 'wikisection': WORDLIST_DIR + 'section.txt',
                 'wikimergeall': WORDLIST_DIR + 'mergeall.final.txt',
                 'wikiacm': WORDLIST_DIR + 'wikiacm.txt',
                 'wikitemp':WORDLIST_DIR + 'wiki_123'
                 }
    #'greedy-wiki':WORDLIST_DIR+'wikipedia_14778209.txt',
dict_files = [ACM_KEYWORD_PATH]

OUTPUT_DIR      = 'keyphrase_output/'

KEYWORD_ONLY    = OUTPUT_DIR + 'keyword_only.txt'
WIKI_ONLY       = OUTPUT_DIR + 'wiki_only.txt'
TOP_TFIDF       = OUTPUT_DIR + 'tfidf_100.txt'
WIKI_links       = OUTPUT_DIR + 'wiki_only_list.txt'
WIKI_italics       = OUTPUT_DIR + 'wiki_only_italics.txt'
WIKI_section       = OUTPUT_DIR + 'wiki_only_section.txt'
WIKI_merge       = OUTPUT_DIR + 'wiki_only_merge.txt'




def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

import string

# def isEnglish(s):
#     return s.translate(None, string.punctuation).isalnum()
STOPWORD_PATH = 'data/stopword/stopword_en.txt'


def load_stopwords(sfile=STOPWORD_PATH):

    dict = set()
    file = open(sfile, 'r')
    for line in file:
        dict.add(line.lower().strip())
    return dict

stopwords = load_stopwords()

stopwords_min = load_stopwords('data/stopword/stopword_min.txt')

def isInChunk(word,chunklist):
    wordlist = word.split(" ")
    if word in chunklist:
        return True
    if len(wordlist) > 1:
        for chunk in chunklist:
            listchunk = chunk.split(" ")
            for s in KnuthMorrisPratt(listchunk,wordlist):
                return True
    return False

def isAlreadyPresent(word,presentlist):
    # print(presentlist)
    # print(word)
    for chunk in presentlist:
        listchunk = chunk[0].split(" ")
        for s in KnuthMorrisPratt(listchunk,word.split(" ")):
            # print(word)
            return True
    return False



class MyTrie:
    """
    Implement a static trie with  search, and startsWith methods.
    """
    def __init__(self,words):
        newlist = self.maketrie(words)
        self.nodes = marisa_trie.Trie(newlist)

    # Inserts a phrase into the trie.
    def maketrie(self, words):
        makelist = []
        for word  in words:
            current_word = ' '.join(word)
            makelist.append(current_word)
        return makelist

    # Returns if the word is in the trie.
    def search(self, words):
        if( words in self.nodes ):
            return True
        else:
            return False

    # Scan a sentence and find any ngram that appears in the sentence
    def scan(self, sentence, min_length=1, max_length=3):
        keyword_list = []
        tokens = preprocessText(sentence,stemming=IS_STEM,stopwords_removal=REMOVE_STOPWORDS)

        ngrams = []
        for i in range(min_length, max_length+1):
            ngrams += nltk.ngrams(tokens, i)

        for ngram in ngrams:
            if(self.search(' '.join(ngram))):
                keyword_list.append(' '.join(ngram))

        return keyword_list

class Word2VecExtractor():
    '''
    Word2Vec_features
    '''
    def __init__(self, pretrained_model = 'google'):
        # Load Google/GloVe's pre-trained Word2Vec model.
        if pretrained_model == 'google':
            self.pretrained_w2v_path = "/Users/memray/Data/glove/GoogleNews-vectors-negative300.bin"
        else:
            self.pretrained_w2v_path = "/Users/memray/Data/glove/glove.6B.300d.w2v.txt"

        self.vector_length       = 300

        print('Loading %s W2V model...' % pretrained_model)
        if self.pretrained_w2v_path.endswith('bin'):
            self.model = gensim.models.KeyedVectors.load_word2vec_format(self.pretrained_w2v_path, binary=True).wv
        else:
            self.model = gensim.models.KeyedVectors.load_word2vec_format(self.pretrained_w2v_path, binary=False).wv

    def vectorize(self, x):
        '''
        for a sentence x
        :param x:
        :return:
        '''
        vec         = np.average([self.model.wv[w] for w in x if w in self.model.wv], axis=0)
        if vec.any() == None or vec.any() == 'nan' or vec.shape != (self.vector_length, ):
            vec = np.zeros((self.vector_length, )).reshape(-1,1)
        return vec

    def extract(self, testset, OUTPUT_FOL=None):
        if not os.path.exists(OUTPUT_FOL):
            os.makedirs(OUTPUT_FOL)

        for doc in testset:
            tokens = preprocessText(doc.text, stemming=IS_STEM, stopwords_removal=REMOVE_STOPWORDS)
            vec    = self.vectorize(tokens).tolist()
            with open(os.path.join(OUTPUT_FOL, doc.id + ".txt.phrases"), 'wb') as f_:
                pickle.dump(vec, f_)

class Doc2VecExtractor():
    '''
    Doc2Vec_features
    '''
    def __init__(self, ldocuments, model_path=None, output_path=None, pretrained_model = 'google'):
        self.doc2idx_dict = {}
        self.model_path  = model_path

        if not os.path.exists(model_path[:model_path.rfind(os.path.sep)]):
            os.makedirs(model_path[:model_path.rfind(os.path.sep)])

        self.output_path = output_path
        if pretrained_model == 'google':
            self.pretrained_w2v_path = "/Users/memray/Data/glove/GoogleNews-vectors-negative300.bin"
        else:
            self.pretrained_w2v_path = "/Users/memray/Data/glove/glove.6B.300d.w2v.txt"

        if model_path != None and os.path.exists(model_path):
            print('Loading existing model: %s' % model_path)
            self.d2v_model  = Doc2Vec.load(model_path)
        else:
            print('Training new model and exporting to %s' % model_path)
            self.d2v_model  = self.train_D2V(ldocuments)

    def get_id(self, doc_str):
        '''
        Given a str, return its ID
        :param doc_str:
        :return:
        '''
        return self.doc2idx_dict[doc_str]

    def train_D2V(self, ldocuments):
        '''
        Load or train Doc2Vec
        '''
        document_dict = {}
        id2num_dict  = {}
        documents = []
        for doc in ldocuments:
            doc_num = len(document_dict)
            id2num_dict[doc.id] = doc_num

            words  = preprocessText(doc.text, stemming=IS_STEM, stopwords_removal=REMOVE_STOPWORDS)
            tagged_doc = TaggedDocument(words=words, tags=[doc_num])
            document_dict[doc.id] = (doc_num, tagged_doc)

            documents.append(tagged_doc)

        # d2v_model = Doc2Vec(size=self.config['d2v_vector_length'], window=self.config['d2v_window_size'], min_count=self.config['d2v_min_count'], workers=4, alpha=0.025, min_alpha=0.025) # use fixed documents rate
        d2v_model = Doc2Vec(size=300, window=5, min_count=3, workers=10,iter=30)
        d2v_model.build_vocab(documents)
        if self.pretrained_w2v_path:
            if self.pretrained_w2v_path.endswith('bin'):
                d2v_model.intersect_word2vec_format(self.pretrained_w2v_path, binary=True)
            else:
                d2v_model.intersect_word2vec_format(self.pretrained_w2v_path, binary=False)

        # for epoch in range(20):
        # print('D2V training epoch = %d' % epoch)
        d2v_model.train(documents, total_examples=len(documents))
            # d2v_model.alpha -= 0.002  # decrease the learning rate
            # d2v_model.min_alpha = d2v_model.alpha  # fix the learning rate, no decay

        # store the model to mmap-able files
        d2v_model.save(self.model_path)

        if self.output_path != None:
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)

            for doc.id, (doc_num, _) in document_dict.items():
                with open(os.path.join(self.output_path, doc.id + ".txt.phrases"), 'wb') as f_:
                    pickle.dump(d2v_model.docvecs[doc_num].tolist(), f_)

        return d2v_model

    def vectorize(self, tokens):
        '''
        for a sentence x
        :param x:
        :return:
        '''
        return self.d2v_model.infer_vector(tokens)

    def extract_d2v(self, testset, OUTPUT_FOL=None):
        if not os.path.exists(OUTPUT_FOL):
            os.makedirs(OUTPUT_FOL)

        for doc in testset:
            tokens = preprocessText(doc.text, stemming=IS_STEM, stopwords_removal=REMOVE_STOPWORDS)
            vec    = self.vectorize(tokens).tolist()
            with open(os.path.join(OUTPUT_FOL, doc.id + ".txt.phrases"), 'wb') as f_:
                pickle.dump(vec, f_)

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


def load_document(path,booknames=['iir-']):
    print('Start loading documents from %s' % path)
    doc_list = []
    file = open(path, 'r',encoding='utf-8', errors='ignore')

    with file as tsv:
        csv.field_size_limit(sys.maxsize)
        tsvin = csv.reader(file, delimiter=',')
        for row in tsvin:
            if row[0].startswith(tuple(booknames)):
                doc = Document(row[0].strip(),row[1].strip())
                doc_list.append(doc)
    return doc_list

def load_document_allfields(path,booknames=[],textfield=['text'],idfield="id",otherfields=[]):
    print('Start loading documents from %s' % path)
    doc_list = []
    file = open(path, 'r',encoding='utf-8', errors='ignore')
    df = pd.read_csv(path,header=0)

    for index,row in df.iterrows():
        if(len(booknames) == 0  or ( row[idfield].startswith(tuple(booknames)))):
            # print(row[idfield])
            text = [ str(row[field]) for field in textfield]
            text = text  + [" "]
            otherfields_dict = { column:row[column] for column in df.columns if (column not in textfield and column != idfield and column in otherfields)}
            doc = Document(str(row[idfield]),' '.join(text),otherfields_dict)
            doc_list.append(doc)
    return doc_list


def load_documenttsv(path,booknames=['iir-']):
    print('Start loading documents from %s' % path)
    doc_list = []
    file = open(path, 'r',encoding='utf-8', errors='ignore')

    with file as tsv:
        csv.field_size_limit(sys.maxsize)
        tsvin = csv.reader(file, delimiter='\t')
        for row in tsvin:
            if row[0].startswith(tuple(booknames)):
                doc = Document(row[0].strip(),row[1].strip())
                doc_list.append(doc)
    return doc_list

def getGlobalngrams(grams,documents,threshold):

    singlecorpus = ""
    for doc in documents:
        singlecorpus += ' '+ doc.text + '\n'


    ncorpus = ' '.join(preprocessText(singlecorpus,stemming=IS_STEM,stopwords_removal=REMOVE_STOPWORDS))
    tf = TfidfVectorizer(analyzer='word', ngram_range=grams, stop_words=stopwords)
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

def extract_np_high_tfidf_words( documents, top_k=200, ngram=(1,1), OUTPUT_FOL='TFIDF',is_global=True):
    '''
    Return the top K 1-gram terms according to TF-IDF
    Load corpus and convert to Dictionary and Corpus of gensim
    :param corpus_path
    :param num_feature, indicate how many terms you wanna retain, not useful now
    :return:
    '''

    if not os.path.exists(OUTPUT_DIR + OUTPUT_FOL):
        os.makedirs(OUTPUT_DIR + OUTPUT_FOL)

    texts = [[preprocessText(sen,stemming=False,stopwords_removal=False)  for sen in document.sentences] for document in documents]
    corpus = [' '.join(preprocessText(document.text,stemming=IS_STEM,stopwords_removal=REMOVE_STOPWORDS)) for document in documents]

    npchunkcorpus = []
    npdocumentcorpus = {}

    iDoc = 0
    for text in texts:
        npdocumentcorpus[iDoc] = []
        for sen in text:
            ichunklist = list(nlp(' '.join(sen)).noun_chunks)
            npdocumentcorpus[iDoc] += ichunklist
            npchunkcorpus.append(ichunklist)
        iDoc += 1


    top_k_list = {}


    # chunkn=set()
    # for textnp in npchunkcorpus:
    #     for chunk in textnp:
    #         chunklisti = ' '.join([tok.lower() for tok in str(chunk).split(" ") if tok not in stopwords])

    for iDoc in npdocumentcorpus.keys():
        textnp = npdocumentcorpus[iDoc]
        chunklisti = []
        documents[iDoc].npchunks = []
        for chunk in textnp:
            chunklisti.append(' '.join(preprocessToken(str(chunk.text))))
        documents[iDoc].npchunks += chunklisti



    tf = TfidfVectorizer(analyzer='word', ngram_range=ngram,stop_words=stopwords,min_df=2)


    tfidf_matrix = tf.fit_transform(corpus)
    feature_names = tf.get_feature_names()

    doc_id=0

    for doc in tfidf_matrix.todense():
        temptokens = zip(doc.tolist()[0], itertools.count())
        temptokens1=[]
        for (x, y) in temptokens:
            stemy = feature_names[y]
            if x > 0.001:
                temptokens1.append((x,y))

        tokindex = heapq.nlargest(len(temptokens1), temptokens1)

        top_k_list[documents[doc_id].id] = []
        for (x, y) in tokindex:
            top_k_list[documents[doc_id].id].append((feature_names[y], x))
            # if isInChunk(feature_names[y],set(documents[doc_id].npchunks)) and not isAlreadyPresent(feature_names[y],top_k_list[documents[doc_id].id]) and len(feature_names[y]) > 2 :
            #     top_k_list[documents[doc_id].id].append((feature_names[y],x) )

        doc_id += 1

    for doc in documents:

        f = open(OUTPUT_DIR + OUTPUT_FOL + "/" + doc.id.replace("\\","_") + ".txt.phrases", 'w')
        writeformat = [ str(x)+","+str(round(y,4)) for (x,y) in top_k_list[doc.id]]
        f.write('\n'.join(writeformat[0:top_k]))
        f.write('\n')
        f.close()






def extract_vectors( documents,  ngram=(1,1), OUTPUT_FOL='UNIGRAMS'):
    '''
    Return the top K 1-gram terms according to TF-IDF
    Load corpus and convert to Dictionary and Corpus of gensim
    :param corpus_path
    :param num_feature, indicate how many terms you wanna retain, not useful now
    :return:
    '''

    if not os.path.exists(OUTPUT_DIR + OUTPUT_FOL):
        os.makedirs(OUTPUT_DIR + OUTPUT_FOL)

    corpus = [' '.join(preprocessText(document.text,stemming=IS_STEM,stopwords_removal=REMOVE_STOPWORDS)) for document in documents]


    tf = TfidfVectorizer(analyzer='word', ngram_range=ngram,min_df=3)
    tfidf_matrix = tf.fit_transform(corpus)

    doc_id=0

    for doc in tfidf_matrix.todense():
        f = open(OUTPUT_DIR + OUTPUT_FOL + "/" + documents[doc_id].id.replace("\\","_") + ".txt.phrases", 'wb')
        pickle.dump(doc.tolist()[0],f)
        doc_id += 1
        f.close()


def extract_unigrams( documents,  ngram=3, OUTPUT_FOL='UNIGRAMS'):
    '''
    Return the top K 1-gram terms according to TF-IDF
    Load corpus and convert to Dictionary and Corpus of gensim
    :param corpus_path
    :param num_feature, indicate how many terms you wanna retain, not useful now
    :return:
    '''

    if not os.path.exists(OUTPUT_DIR + OUTPUT_FOL):
        os.makedirs(OUTPUT_DIR + OUTPUT_FOL)


    for doc in documents:
        l_ngrams = []
        tokens = preprocessText(doc.text,stemming=IS_STEM,stopwords_removal=REMOVE_STOPWORDS)
        for i in range(0,ngram):
            l_ngrams += nltk.ngrams(tokens,n=i+1)

        tokens = set()
        for token in l_ngrams:
            tokens.add(' '.join(list(token)))

        f = open(OUTPUT_DIR + OUTPUT_FOL + "/" + doc.id.replace("\\","_") + ".txt.phrases", 'w')
        f.write('\n'.join(tokens))
        f.close()



def extract_author_keywords(keyword_trie, documents, OUTPUT_FOL):
    '''
    Return all the matching keywords according to the given keyword list (ACM/ACL/MS/Wiki/IR-glossary)
    :param keyword_trie:
    :param documents:
    :return:
    '''

    OUTPUT_FOLN = OUTPUT_DIR + OUTPUT_FOL
    if not os.path.exists(OUTPUT_FOLN):
        os.makedirs(OUTPUT_FOLN)

    print('Extracting keywords basing on wordlist, output to %s' % os.path.abspath(OUTPUT_FOLN))
    for doc in documents:

        f = open(OUTPUT_FOLN+'//'+doc.id+".txt.phrases",'w')
        keyword_list = keyword_trie.scan(doc.text)
        keyword_dict = Counter(keyword_list)
        f.write('\n'.join(keyword_dict.keys()))
        f.close()


def extract_lda(documents,testset,OUTPUT_FOL="wLDA",topics_n = 200):
    '''
    Extract the top K highest probability concepts according to the LLDA result
    :param documents:
    :param top_k:
    :return:
    '''
    # export_to_files(documents)
    doc_set=[]
    for doc in documents:
        doc_set.append(doc.text)

    tok_set=[]
    tok_dict={}
    test_dict = {}

    for doc in documents:
        tokens = preprocessText(doc.text,stemming=IS_STEM,stopwords_removal=REMOVE_STOPWORDS)
        tok_set.append(tokens)
        tok_dict[doc.id] = tokens

    for doc in testset:
        tokens = preprocessText(doc.text,stemming=IS_STEM,stopwords_removal=REMOVE_STOPWORDS)
        tok_set.append(tokens)
        test_dict[doc.id] = tokens

    dictionary = corpora.Dictionary(tok_set)
    corpus = [dictionary.doc2bow(text) for text in tok_set]

    OUTPUT_FOLN = OUTPUT_DIR + OUTPUT_FOL
    if not os.path.exists(OUTPUT_FOLN):
        os.makedirs(OUTPUT_FOLN)

    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=topics_n, id2word = dictionary)
    pickle.dump(ldamodel, open('data/ldagensim'+str(topics_n),'wb'))

    for doci in documents:
        doc = tok_dict[doci.id]
        f = open( OUTPUT_FOLN + "/" + doci.id + ".txt.phrases", 'wb')
        topics = sparse2full(ldamodel[dictionary.doc2bow(doc)], topics_n).tolist()

        pickle.dump(topics,f)
        f.close()

    for doci in testset:
        doc = test_dict[doci.id]
        f = open( OUTPUT_FOLN + "/" + doci.id + ".txt.phrases", 'wb')
        topics = sparse2full(ldamodel[dictionary.doc2bow(doc)], topics_n).tolist()

        pickle.dump(topics,f)
        f.close()
def extract_high_tfidf_words( documents, top_k=200, ngram=(1,1), OUTPUT_FOL='TFIDF'):
    '''
    Return the top K 1-gram terms according to TF-IDF
    Load corpus and convert to Dictionary and Corpus of gensim
    :param corpus_path
    :param num_feature, indicate how many terms you wanna retain, not useful now
    :return:
    '''

    if not os.path.exists(OUTPUT_DIR + OUTPUT_FOL):
        os.makedirs(OUTPUT_DIR + OUTPUT_FOL)

    texts = [' '.join(preprocessText(document.text,stemming=IS_STEM,stopwords_removal=REMOVE_STOPWORDS))  for document in documents]

    #### Create Scikitlearn corpus
    top_k_list = {}


    tf = TfidfVectorizer(analyzer='word', ngram_range=ngram,stop_words=stopwords,min_df=2,max_df=1000)
    tfidf_matrix = tf.fit_transform(texts)
    feature_names = tf.get_feature_names()

    doc_id=0

    for doc in tfidf_matrix.todense():
        temptokens = zip(doc.tolist()[0], itertools.count())
        temptokens1=[]
        for (x, y) in temptokens:
            stemy = feature_names[y]
            if x > 0.0:
                temptokens1.append((x,y))

        tokindex = heapq.nlargest(len(temptokens1), temptokens1)

        top_k_list[documents[doc_id].id] = []
        for (x, y) in tokindex:
            top_k_list[documents[doc_id].id].append((feature_names[y],x) )
        doc_id += 1



    for doc in documents:

        # output_file.write('{0}\t{1}\n'.format(doc.id, ','.join(top_k_list[doc.id])))
        f = open(OUTPUT_DIR + OUTPUT_FOL + "/" + doc.id.replace("\\","_") + ".txt.phrases", 'w')
        writeformat = [ str(x)+","+str(round(y,4)) for (x,y) in top_k_list[doc.id]]
        f.write('\n'.join(writeformat[0:top_k]))
        # f.write('\n'.join(top_k_list[doc.id][0:top_k]))
        f.write('\n')
        f.close()

def extract_top_kcs_sm_from_vector( KC_FOLDER,  OUTPUT_FILE='TFIDFQtext'):
    '''
    Return the top K 1-gram terms according to TF-IDF
    Load corpus and convert to Dictionary and Corpus of gensim
    :param corpus_path
    :param num_feature, indicate how many terms you wanna retain, not useful now
    :return:
    '''
    lenVector = 0
    topicnames = []
    columns = ["id","term","weight"]
    file_csv = open(OUTPUT_DIR + OUTPUT_FILE , 'w')
    csvwriter = csv.writer(file_csv, delimiter=',')
    csvwriter.writerow(columns)


    for filename in os.listdir(OUTPUT_DIR+KC_FOLDER):
        kcVector = pickle.load(open(OUTPUT_DIR+KC_FOLDER +"/"+ filename,'rb'))
        print(len(kcVector))
        if( lenVector == 0):
            lenVector = len(kcVector)
            topicnames = [ "kc"+str(i) for i in range(lenVector)]

        temptokens = zip(kcVector,topicnames)
        temptokens1 = []
        for (x, y) in temptokens:
            if x > 0.0:
                temptokens1.append((x,y))

        tokindex = heapq.nlargest(len(temptokens1), temptokens1)

        for (x,y) in tokindex:
            csvwriter.writerow([filename.replace(".txt.phrases",""),y,x])
    #
    # write_to_file=True
    # if write_to_file == True:
    #     f = open(OUTPUT_DIR+OUTPUT_FOL+".csv", "w")
    #     f.write("questionid,term,tf,doc_nbterms,nb_docs,df,idf,tfidf\n")
    #
    #     nb_docs =len(documents)
    #     doc_id=0
    #     ctf_model = CountVectorizer(analyzer='word', ngram_range=ngram, stop_words=stopwords, min_df=1)
    #     ctfidf_matrix = ctf_model.fit_transform(texts)
    #     ctfidf_matrixdense = ctfidf_matrix.todense()
    #     cfeature_names = ctf_model.get_feature_names()
    #
    #     for doc in tfidf_matrix.todense():
    #         print(doc_id)
    #         temptokens = zip(doc.tolist()[0], itertools.count())
    #         cmat = ctfidf_matrixdense[doc_id].tolist()[0]
    #         doc_nbterms = sum(cmat)
    #         temptokens1=[]
    #         for (x, y) in temptokens:
    #             if x > 0:
    #                 cindex = cfeature_names.index(feature_names[y])
    #                 f_td = cmat[cindex]
    #                 df = len(ctfidf_matrix[:, cindex].data)
    #                 idf = round(np.log(nb_docs / df), 8)
    #                 z=x
    #                 temptokens1.append((z, x, y,f_td,doc_nbterms,nb_docs,df,idf))
    #
    #         tokindex = heapq.nlargest(len(temptokens1), temptokens1)
    #         # print(tokindex)
    #
    #         top_k_list[documents[doc_id].id] = []
    #         for (x, y,z,f_td,doc_nbterms,nb_docs,df,idf) in tokindex:
    #             top_k_list[documents[doc_id].id].append((feature_names[z],x,y,f_td,doc_nbterms,nb_docs,df,idf) )
    #
    #         if len(top_k_list[documents[doc_id].id]) == 0:
    #             for (x, y, z, f_td, doc_nbterms, nb_docs, df, idf) in tokindex:
    #                 top_k_list[documents[doc_id].id].append((feature_names[z], x, y, f_td, doc_nbterms, nb_docs, df, idf))
    #
    #         if len(top_k_list[documents[doc_id].id]) < 2 and len(tokindex) > 0:
    #             print("here",documents[doc_id].id)
    #             for i in range(len(tokindex)):
    #                 ( x, y,z, f_td, doc_nbterms, nb_docs, df, idf) = tokindex[i]
    #                 print(doc_id)
    #                 print(documents[doc_id].id)
    #                 print(tokindex)
    #                 print( ( feature_names[z], y,x, f_td, doc_nbterms, nb_docs, df, idf))
    #                 top_k_list[documents[doc_id].id].append((feature_names[z],x,y,f_td,doc_nbterms,nb_docs,df,idf) )
    #         doc_id += 1
    #
    #
    #
    #     for doc in documents:
    #         for (x,y,z,f_td,doc_nbterms,nb_docs,df,idf) in top_k_list[doc.id]:
    #             f.write(str(doc.id) + ","  + x + "," + str(f_td) + "," + str(doc_nbterms) + "," + str(nb_docs) + "," + str(df) + "," + str(idf) + "," + str(round(z,5)) + "\n")



        # f.write('\n')
        # print("writing over")
        # f.close()

# def extract_high_tfidf_words_sm( documents, ngram=(1,1), OUTPUT_FOL='TFIDFQtext'):
#     '''
#     Return the top K 1-gram terms according to TF-IDF
#     Load corpus and convert to Dictionary and Corpus of gensim
#     :param corpus_path
#     :param num_feature, indicate how many terms you wanna retain, not useful now
#     :return:
#     '''
#
#     if not os.path.exists(OUTPUT_DIR + OUTPUT_FOL):
#         os.makedirs(OUTPUT_DIR + OUTPUT_FOL)
#
#     texts = [' '.join(preprocessText(document.text,stemming=IS_STEM,stopwords_removal=REMOVE_STOPWORDS))  for document in documents]
#     print(texts)
#     texts = ['keywords' if len(text) < 2 else text for text in texts ]
#     print(texts)
#     #### Create Scikitlearn corpus
#     top_k_list = {}
#
#     tf = TfidfVectorizer(analyzer='word', ngram_range=ngram,stop_words=stopwords,min_df=2,max_df=1000)
#     tfidf_matrix = tf.fit_transform(texts)
#     feature_names = tf.get_feature_names()
#
#     doc_id=0
#
#     for doc in tfidf_matrix.todense():
#         temptokens = zip(doc.tolist()[0], itertools.count())
#         temptokens1=[]
#         for (x, y) in temptokens:
#             stemy = feature_names[y]
#             if x > 0.0:
#                 temptokens1.append((x,y))
#
#         tokindex = heapq.nlargest(len(temptokens1), temptokens1)
#
#         top_k_list[documents[doc_id].id] = []
#         for (x, y) in tokindex:
#             top_k_list[documents[doc_id].id].append((feature_names[y],x) )
#         doc_id += 1
#
#     write_to_file=True
#     if write_to_file == True:
#         f = open(OUTPUT_DIR+OUTPUT_FOL+".csv", "w")
#         f.write("questionid,term,tf,doc_nbterms,nb_docs,df,idf,tfidf\n")
#
#         nb_docs =len(documents)
#         doc_id=0
#         ctf_model = CountVectorizer(analyzer='word', ngram_range=ngram, stop_words=stopwords, min_df=1)
#         ctfidf_matrix = ctf_model.fit_transform(texts)
#         ctfidf_matrixdense = ctfidf_matrix.todense()
#         cfeature_names = ctf_model.get_feature_names()
#
#         for doc in tfidf_matrix.todense():
#             print(doc_id)
#             temptokens = zip(doc.tolist()[0], itertools.count())
#             cmat = ctfidf_matrixdense[doc_id].tolist()[0]
#             doc_nbterms = sum(cmat)
#             temptokens1=[]
#             for (x, y) in temptokens:
#                 if x > 0:
#                     cindex = cfeature_names.index(feature_names[y])
#                     f_td = cmat[cindex]
#                     df = len(ctfidf_matrix[:, cindex].data)
#                     idf = round(np.log(nb_docs / df), 8)
#                     z=x
#                     temptokens1.append((z, x, y,f_td,doc_nbterms,nb_docs,df,idf))
#
#             tokindex = heapq.nlargest(len(temptokens1), temptokens1)
#             # print(tokindex)
#
#             top_k_list[documents[doc_id].id] = []
#             for (x, y,z,f_td,doc_nbterms,nb_docs,df,idf) in tokindex:
#                 top_k_list[documents[doc_id].id].append((feature_names[z],x,y,f_td,doc_nbterms,nb_docs,df,idf) )
#
#             if len(top_k_list[documents[doc_id].id]) == 0:
#                 for (x, y, z, f_td, doc_nbterms, nb_docs, df, idf) in tokindex:
#                     top_k_list[documents[doc_id].id].append((feature_names[z], x, y, f_td, doc_nbterms, nb_docs, df, idf))
#
#             if len(top_k_list[documents[doc_id].id]) < 2 and len(tokindex) > 0:
#                 print("here",documents[doc_id].id)
#                 for i in range(len(tokindex)):
#                     ( x, y,z, f_td, doc_nbterms, nb_docs, df, idf) = tokindex[i]
#                     print(doc_id)
#                     print(documents[doc_id].id)
#                     print(tokindex)
#                     print( ( feature_names[z], y,x, f_td, doc_nbterms, nb_docs, df, idf))
#                     top_k_list[documents[doc_id].id].append((feature_names[z],x,y,f_td,doc_nbterms,nb_docs,df,idf) )
#             doc_id += 1
#
#
#
#         for doc in documents:
#             for (x,y,z,f_td,doc_nbterms,nb_docs,df,idf) in top_k_list[doc.id]:
#                 f.write(str(doc.id) + ","  + x + "," + str(f_td) + "," + str(doc_nbterms) + "," + str(nb_docs) + "," + str(df) + "," + str(idf) + "," + str(round(z,5)) + "\n")
#
#
#
#         f.write('\n')
#         print("writing over")
#         f.close()
def extract_tfidf_kcs_sm( documents, ngram=(1,1), OUTPUT_FOL='TFIDFQtext'):
    '''
    Return the top K 1-gram terms according to TF-IDF
    Load corpus and convert to Dictionary and Corpus of gensim
    :param corpus_path
    :param num_feature, indicate how many terms you wanna retain, not useful now
    :return:
    '''


    texts = [' '.join(preprocessText(document.text,stemming=True,stopwords_removal=True)).replace("nan"," ")  for document in documents]

    #### Create Scikitlearn corpus
    top_k_list = {}

    tf = TfidfVectorizer(analyzer='word', ngram_range=ngram,min_df=1)
    tfidf_matrix = tf.fit_transform(texts)
    feature_names = tf.get_feature_names()

    ctf_model = CountVectorizer(analyzer='word', ngram_range=ngram,  min_df=1)
    ctfidf_matrix = ctf_model.fit_transform(texts)
    ctfidf_matrixdense = ctfidf_matrix.todense()
    cfeature_names = ctf_model.get_feature_names()

    # doc_id=0

    # for doc in tfidf_matrix.todense():
    #     temptokens = zip(doc.tolist()[0], itertools.count())
    #     temptokens1=[]
    #     for (x, y) in temptokens:
    #         stemy = feature_names[y]
    #         if x > 0.0:
    #             temptokens1.append((x,y))
    #
    #     tokindex = heapq.nlargest(len(temptokens1), temptokens1)
    #
    #     top_k_list[documents[doc_id].id] = []
    #     for (x, y) in tokindex:
    #         top_k_list[documents[doc_id].id].append((feature_names[y],x) )
    #     doc_id += 1
    othercolums = documents[0].otherfields.keys()
    columns = ["id","term","tf","doc_nbterms","nb_docs","df","idf","tfidf"] + list(othercolums)


    file_csv = open(OUTPUT_DIR+OUTPUT_FOL+".csv", 'w')
    csvwriter = csv.writer(file_csv, delimiter=',')
    csvwriter.writerow(columns)
    nb_docs =len(documents)
    doc_id=0


    for doc in tfidf_matrix.todense():
        print(doc_id)
        temptokens = zip(doc.tolist()[0], itertools.count())
        cmat = ctfidf_matrixdense[doc_id].tolist()[0]
        doc_nbterms = sum(cmat)
        temptokens1=[]
        for (x, y) in temptokens:
            if x > 0:
                cindex = cfeature_names.index(feature_names[y])
                f_td = cmat[cindex]
                df = len(ctfidf_matrix[:, cindex].data)
                idf = round(np.log(nb_docs / df), 8)
                z=x
                temptokens1.append((z, x, y,f_td,doc_nbterms,nb_docs,df,idf))

        tokindex = heapq.nlargest(len(temptokens1), temptokens1)
        # print(tokindex)

        top_k_list[documents[doc_id].id] = []
        for (x, y,z,f_td,doc_nbterms,nb_docs,df,idf) in tokindex:
            feature_dict = {}
            feature_dict = {"term":feature_names[z],"tf":x,"tf":f_td,"doc_nbterms":doc_nbterms,"nb_docs":nb_docs,"df":df,"idf":idf,"tfidf":y,"id":documents[doc_id].id}
            for column in columns:
                if(column in documents[doc_id].otherfields.keys()):
                    feature_dict[column] = documents[doc_id].otherfields[column]

            top_k_list[documents[doc_id].id].append(feature_dict )
            row_new = [feature_dict[column] for column in columns]
            csvwriter.writerow(row_new)
            # print(row_new)

        # if len(top_k_list[documents[doc_id].id]) == 0:
        #     for (x, y, z, f_td, doc_nbterms, nb_docs, df, idf) in tokindex:
        #         top_k_list[documents[doc_id].id].append((feature_names[z], x, y, f_td, doc_nbterms, nb_docs, df, idf))
        #
        # if len(top_k_list[documents[doc_id].id]) < 2 and len(tokindex) > 0:
        #     print("here",documents[doc_id].id)
        #     for i in range(len(tokindex)):
        #         ( x, y,z, f_td, doc_nbterms, nb_docs, df, idf) = tokindex[i]
        #         print(doc_id)
        #         print(documents[doc_id].id)
        #         print(tokindex)
        #         print( ( feature_names[z], y,x, f_td, doc_nbterms, nb_docs, df, idf))
        #         top_k_list[documents[doc_id].id].append((feature_names[z],x,y,f_td,doc_nbterms,nb_docs,df,idf) )
        doc_id += 1

    print("writing over")

if __name__=='__main__':

    keyword_trie = None
    word_list = [ 'greedy-wiki']#,'gredy-acm']

    # listbooks = ['irv-','issr-','mir-','iir-','foa-','zhai-','iirbookpubs-','seirip-','chapterwiseiir-','wiki-','wikitest-','iirtest-']
    # listbooks = [ 'iir-',  'wikitest-']
    # listbooks_test = ['chapterwiseiir-']
    llistbooks = ['iir-', 'mir-', 'foa-']
    documents = load_documenttsv(IR_CORPUS,llistbooks)



    # # Code For NP Chunks
    print("NP Chunks ")
    # extract_np_high_tfidf_words( documents, top_k=5, ngram=(1,1), OUTPUT_FOL='TFIDFNP115')
    # extract_np_high_tfidf_words(documents, top_k=5, ngram=(2, 2), OUTPUT_FOL='TFIDFNP225')
    # extract_np_high_tfidf_words(documents, top_k=5, ngram=(3, 3), OUTPUT_FOL='TFIDFNP335')

    # extract_np_high_tfidf_words(documents, top_k=30, ngram=(1, 3), OUTPUT_FOL='TFIDFNP30')

    # documentstest = load_document(IR_CORPUS,listbooks_test)

    # kl = KeywordList('greedy-acm')
    # keyword_trie = kl.trie
    # extract_author_keywords(keyword_trie, documents, 'greedy-acm')
    #


    # kl = None
    # kl = KeywordList('greedy-wiki')
    # keyword_trie = kl.trie
    # extract_author_keywords(keyword_trie, documents, 'greedy-wiki')

    # llistbooks = ['iir-', 'mir-', 'foa-']
    # llistbooks_test = ['chapterwiseiir', 'wikitest-', 'mir-', 'iir-', 'iirbookpubs-','iirtest-']

    # Code for Doc2Vec
    # ldocuments = load_document(IR_CORPUS, llistbooks)
    # ldocumentstest = load_document(IR_CORPUS, llistbooks_test)
    # model_name = 'Doc2Vec(stemming=False, stopwords_removal=True, pretrained=Google)'
    # d2v = Doc2VecExtractor(ldocuments, model_path='model/%s/doc2vec_ir.model' % model_name, pretrained_model = 'google')
    # d2v.extract_d2v(testset=ldocumentstest, OUTPUT_FOL="data/keyphrase_output/%s/" % model_name)

    # Code for Word2Vec
    # ldocuments = load_document(IR_CORPUS, llistbooks)
    # ldocumentstest = load_document(IR_CORPUS, llistbooks_test)
    # w2v = Word2VecExtractor(pretrained_model = 'google')
    # w2v.extract(testset=ldocuments, OUTPUT_FOL="data/keyphrase_output/Word2Vec-Google/")
    # w2v.extract(testset=ldocumentstest, OUTPUT_FOL="data/keyphrase_output/Word2Vec-Google/")

    ## Code For LDA
    # llistbooks = ['irv-', 'issr-', 'foa-', 'zhai-', 'seirip-', 'wiki-', 'sigir']
    # llistbooks_test = ['chapterwiseiir', 'wikitest-', 'mir-', 'iir-', 'iirbookpubs-','iirtest-']
    #
    # ldocuments = load_document(IR_CORPUS, llistbooks)
    # ldocumentstest = load_document(IR_CORPUS, llistbooks_test)
    # for i in range(200, 350, 50):
    #     extract_lda(ldocuments, OUTPUT_FOL="LDA" + str(i), topics_n=i, testset=ldocumentstest)
    #
    #
    #
    # # # Code For extracting VSM
    # # print("VSM")
    # # print("VSM")
    # # extract_vectors(documents,  ngram=(1, 1), OUTPUT_FOL='UNIGRAM')
    # #
    # # Code For Extract Unigrams
    # print("UNIGRAMS ")
    # extract_unigrams(documents,  ngram=1, OUTPUT_FOL='UNIGRAMS')
    #
    # ## Code For Extract Unigrams
    # print("NGRAMS ")
    # extract_unigrams(documents,  ngram=3, OUTPUT_FOL='NGRAMS')
    # #
    #
    # print("HIGH NP TFIDF WORDS")
    # for i in range(1,4):
    #     extract_np_high_tfidf_words( documents, top_k=5, ngram=(i,i), OUTPUT_FOL='TFIDFNP'+str(i))
    #
    # print("HIGH TFIDF WORDS")
    # for i in range(1,4):
    #     extract_high_tfidf_words( documents, top_k=5, ngram=(i,i), OUTPUT_FOL='TFIDF'+str(i))
    # extract_high_tfidf_words(documents, top_k=10, ngram=(1, 1), OUTPUT_FOL='TFIDF13',write_to_file=True)
    # print("Global NGRAMS")
    # getGlobalngrams(grams=(2,2),documents=documents,threshold=0.01)
    # getGlobalngrams(grams=(3,3), documents=documents, threshold=0.01)
    #
    # print("NGRAMS ")
    # listbooks = ['mir-','iir-','iirbookpubs-','chapterwiseiir-','wikitest-','iirtest-']
    # documents = load_document(IR_CORPUS,listbooks)
    #
    # extract_unigrams(documents,  ngram=3, OUTPUT_FOL='13NGRAMS')

    # QS_CORPUS = 'data/readingcircleCorpus.csv'
    QS_CORPUS = 'data/QuizInfo - quiz2text.csv'
    # PAGE_CORPUS = 'data/pagewise_corpus_ir.csv'
    listbooks = ['']
    # documents = load_document(PAGE_CORPUS,listbooks)
    # documents = load_document_allfields(QS_CORPUS, listbooks,textfield=['question','choice_all'],idfield="que_id",otherfields=['quiz_id','quiz_session','quiz_order','que_order','Type'])
    # extract_tfidf_kcs_sm(documents,  ngram=(1, 1), OUTPUT_FOL='TFIDF.all.quiz2text')

    PAGE_CORPUS = 'data/pagewise_corpus_ir_sui.csv'
    # documents = load_document_allfields(PAGE_CORPUS, booknames=['iir','sui'],textfield=['text'],idfield="page",otherfields=[])
    # extract_tfidf_kcs_sm(documents, ngram=(1, 1), OUTPUT_FOL='TFIDF.all.page2text_ir_sui')


    documents = load_document_allfields(PAGE_CORPUS, booknames=[], textfield=['text'], idfield="page",
                                        otherfields=[])

    quizdocuments = load_document_allfields(QS_CORPUS, booknames=[], textfield=['question','choice_all'], idfield="id",
                                        otherfields=[])

    # extract_lda(documents,documents,OUTPUT_FOL="LDA_17FALL",topics_n=200)
    extract_lda(documents, quizdocuments, OUTPUT_FOL="LDA_17FALL", topics_n=200)

    # extract_top_kcs_sm_from_vector(KC_FOLDER="LDA_17FALL",OUTPUT_FILE="LDA.irbook.pagewise.weight.csv",booknames=[''])

    # READING_CIRCLE_PAGE_CORPUS = 'data/redingcircle_pagewise_corpus_ir.csv'
    # documents = load_document_allfields(READING_CIRCLE_PAGE_CORPUS, listbooks, textfield=['text'], idfield="page", otherfields=[])
    # extract_tfidf_kcs_sm(documents, ngram=(1, 1), OUTPUT_FOL='TFIDF.all.page2text.readingcircle.16SpringFall')