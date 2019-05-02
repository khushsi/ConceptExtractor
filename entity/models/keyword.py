from entity.util.config import config
import os, pickle
from entity.util.trie import MyTrie
from entity.util import nlp
from collections import Counter
import itertools

class KeywordList:
    def __init__(self, name,wordlist_path,):
        self.name = name
        self.wordlist = wordlist_path
        self.triepath = config.TRIE_CACHE_DIR+name+'_trie_dict.cache'
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
                file = open(dict_file, 'r', encoding='utf8')
                for line in file:
                    tokens = nlp.preprocessText(line)
                    if(len(tokens)>0):
                        listwords.append(tokens)

            trie = MyTrie(listwords)
            with open(trie_cache_file, 'wb') as f:
                pickle.dump(trie, f)
        return trie




    def get_Topics(self,topicdocs):


        ptopicdocs = nlp.preprocessed_docs(topicdocs)

        topic_dic = {}
        for doci in ptopicdocs:

            keyword_list = self.trie.scan(doci.text)
            keyword_dict = Counter(keyword_list)

            topic_dic[doci.id] = [ key for key in keyword_dict.keys()]

        return topic_dic


    def get_Topics_secondFilter(self,doc2concepts):

        topic_dic = {}
        for doci in doc2concepts:
            topic_dic[doci] = []

            for weight,concept in doc2concepts[doci]:
                keyword = self.trie.scan_list([concept])
                if len(keyword) > 0:
                    topic_dic[doci].append((weight,concept))

        return topic_dic