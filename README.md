This project creates concepts with different formats

TFIDF
TFIDFNP
LDA
greedy wiki
greedy acm
etc.

Also it has possiblity to learn representation for these concepts using

Concept2Vec
Doc2Vec
Word2Vec
DocTag2Vec
Joint Learning

---


For using concept_extractor
-  You have to have a text file  with text in : IR_CORPUS = 'add the path to your file here'
Uncomment the below line
- # print("HIGH NP TFIDF WORDS")
    # for i in range(1,4):
    #     extract_np_high_tfidf_words( documents, top_k=5, ngram=(i,i), OUTPUT_FOL='TFIDFNP'+str(i))