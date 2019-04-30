import re
from nltk.stem.porter import PorterStemmer
import spacy
from entity.util.config import config


nlp = spacy.load('en')
stemmer = PorterStemmer()

STOPWORD_PATH = 'data/stopword/stopword_en.txt'

def load_stopwords(sfile = config.STOPWORD_PATH):

    dict = set()
    file = open(sfile, 'r')
    for line in file:
        dict.add(line.lower().strip())
    return dict



def stem(text):
    word_list = text.split(" ")
    for i in range(len(word_list)):
        word_list[i] = stemmer.stem(word_list[i])
    return ' '.join(word_list)


def stemList(word_list):
    for i in range(len(word_list)):
        word_list[i] = stem(word_list[i])
    return ' '.join(word_list)


def preprocessToken(text):
    return re.sub(r'\W+|\d+', '', text.strip().lower())


def preprocessText(text,stemming=config.IS_STEM,stopwords_removal=config.REMOVE_STOPWORDS):
    # print(text)
    text = re.sub("[ ]{1,}",r' ',text)
    text = re.sub(r'\W+|\d+', ' ', text.strip().lower())
    tokens = [token.strip()  for token in text.split(" ")]
    tokens = [token for token in tokens if len(token) > 1]
    if stopwords_removal:
        stopwords = load_stopwords()
        tokens = [token for token in tokens if token not in stopwords]
    if stemming:
        tokens = [stem(token) for token in tokens ]

    tokens = [token.strip() for token in tokens if len(token.strip()) > 1]
    return tokens

def preprocessed_docs(documents):

    for doc in documents:
        tokens = preprocessText(doc.text)
        doc.tokens = tokens

    return documents

# Got this function from internet on some forum
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


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


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
