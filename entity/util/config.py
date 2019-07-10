

class config:
    IR_CORPUS = 'data/iirmirbook.tsv'
    KC_CORPUS = 'data/conceptdocs.csv'
    IS_STEM=True
    REMOVE_STOPWORDS=True
    STOPWORD_PATH = 'data/stopword/stopword_en.txt'
    TRIE_CACHE_DIR = 'data/triecache/'
    REFRESH_CACHE = False

    #
    dir_sep = "/"
    file_ext =".txt"
    # Models

    # Concepts
    TFIDF = 'tfidf'
    TFIDFNP = 'tfidfnp'#noun phrases that are ranked by tfidf
    NGRAMS = 'ngrams' #all ngrams
    LIST_FILTER = 'list_filter' #TFIDF ngrams + filter (need to specify which filter)
    WIKI_FILTER = 'wiki_filter_np' #unused (but should have been TFIDF noun phrase + filter) also need to create


    # LIST
    wiki_list = 'data/wordlist/wikipedia_14778209.txt'
    irbook_glossary_list = 'data/wordlist/irbook_glossary_707.txt'
    irbook_expert_list = 'data/wordlist/expert_vocab_for_ir_book.txt'
    expert_vote_list = 'data/wordlist/expert_vocab_IRBook_Format_Vote.txt'
    Remove_Prev_models = True
    acm_list = 'data/wordlist/acm_keywords_168940.txt'
    #expert_list = 'data/wordlist/IRexpert.txt'

    # STEMMERS
    PORTER = 'PORTER'
    LANCASTER = 'LANCASTER'
    SNOWBALL = 'SNOWBALL'
    LEMMATIZER_WORDNET = 'LEMMATIZER_WORDNET' #NOTE: NEED TO DOWNLOAD WORDNET FROM NLTK

    #DOWNLOADING WORDNET:
    #python -c "import nltk; nltk.download('wordnet')"

    #Additional info:
    #Set up spacy by running:
    #python -m spacy download en
    #python -c "import spacy; spacy.load('en_core_web_sm')"
