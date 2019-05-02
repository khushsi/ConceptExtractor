

class config:
    IR_CORPUS = 'data/iirmirbook.tsv'
    KC_CORPUS = 'data/conceptdocs.csv'
    IS_STEM=True
    REMOVE_STOPWORDS=True
    STOPWORD_PATH = 'data/stopword/stopword_en.txt'
    TRIE_CACHE_DIR = 'data/triecache/'

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
    Remove_Prev_models = True
    acm_list = 'data/wordlist/acm_keywords_168940.txt'
