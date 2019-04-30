from entity.util import nlp
import nltk
# import  marisa_trie as trie
import pygtrie as trie

class MyTrie:
    """
    Implement a static trie with  search, and startsWith methods.
    """
    def __init__(self,words):

        self.nodes = trie.Trie()
        self.maketrie(words)
    # Inserts a phrase into the trie.
    def maketrie(self, words):
        for word  in words:
            current_word = ' '.join(word)
            self.nodes[current_word] = current_word

    # Returns if the word is in the trie.
    def search(self, words):
        if( words in self.nodes ):
            return True
        else:
            return False

    # Scan a sentence and find any ngram that appears in the sentence
    def scan(self, sentence, min_length=1, max_length=5):
        keyword_list = []
        tokens = nlp.preprocessText(sentence)

        ngrams = []
        for i in range(min_length, max_length+1):
            ngrams += nltk.ngrams(tokens, i)

        for ngram in ngrams:
            if(self.search(' '.join(ngram))):
                keyword_list.append(' '.join(ngram))

        return keyword_list


    def scan_list(self, wordlist):
        keyword_list = []

        ngrams = wordlist

        for ngram in ngrams:
            if(self.search(ngram)):
                keyword_list.append(' '.join(ngram))

        return keyword_list

