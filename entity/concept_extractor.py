# encoding: utf-8
from __future__ import generators
import os
from entity.util.config import config
from entity.util import document as dc
from entity.util import common as cm
from entity.models import tfidf,keyword
from entity.util.extractors import extract_top_kcs


if __name__=='__main__':

    BOOK_CORPUS = 'data/trialSyllabus/syllabusCSV.csv'
    listbooks = []

    concept = config.LIST_FILTER #WHICH FILTER
    keyword_list_path = config.irbook_expert_list #WHICH KEYWORD LIST TO FILTER


    bookdocs_16 = dc.load_document(BOOK_CORPUS,
                                          booknames=[],
                                          textfield=['text'],
                                          idfield="docid",
                                          otherfields=['bookname'],
                                          booknamefield='bookname',
                                   )

    train_docs = bookdocs_16
    extract_docs = bookdocs_16
    model_dir = "model/"
    concept_dir = "concepts/EXPERT_LEMMATIZER_WORDNET/"
    Define_kc = False
    no_of_topics = -1 #TOP K

    if concept == config.TFIDFNP:

        ptfidf =  model_dir +"m_"+ concept +".pickle"
        outdir =concept_dir + config.dir_sep + concept

        if no_of_topics == -1:
            outdir += "all" + config.file_ext
        else:
            outdir += "top" + str(no_of_topics) + config.file_ext

        model = None

        if not os.path.exists(ptfidf) or config.Remove_Prev_models:
            model = tfidf.TFIDFExtractor(train_docs,ngram=(1,5),mindf=1)
            model.train()
            model.save_model(ptfidf)
        else:
            print(" Model Exists : "+ptfidf)

        model = cm.load_model(ptfidf)

        doc2concepts = model.get_Topics_npFilter(extract_docs)

        df = extract_top_kcs(doc2concepts=doc2concepts,output_dir=outdir,define_kc=False,topk=no_of_topics)

    if concept == config.TFIDF:

        ptfidf =  model_dir +"m_"+ concept +".pickle"
        outdir =concept_dir + config.dir_sep + concept

        if no_of_topics == -1:
            outdir += "all" + config.file_ext
        else:
            outdir += "top" + str(no_of_topics) + config.file_ext

        model = None

        if not os.path.exists(ptfidf)  or config.Remove_Prev_models:
            model = tfidf.TFIDFExtractor(train_docs,ngram=(1,5),mindf=1)
            model.train()
            model.save_model(ptfidf)
        else:
            print(" Model Exists : "+ptfidf)

        model = cm.load_model(ptfidf)

        doc2concepts = model.get_Topics(extract_docs)

        df = extract_top_kcs(doc2concepts=doc2concepts,output_dir=outdir,define_kc=False,topk=no_of_topics)

    if concept == config.NGRAMS:

        ptfidf =  model_dir +"m_"+ concept +".pickle"
        outdir =concept_dir + config.dir_sep + concept

        if no_of_topics == -1:
            outdir += "all" + config.file_ext
        else:
            outdir += "top" + str(no_of_topics) + config.file_ext

        model = None

        if not os.path.exists(ptfidf)  or config.Remove_Prev_models:
            model = tfidf.TFIDFExtractor(train_docs,ngram=(1,5),mindf=1)
            model.train()
            model.save_model(ptfidf)
        else:
            print(" Model Exists : "+ptfidf)

        model = cm.load_model(ptfidf)
        doc2concepts = model.get_Topics(extract_docs)
        df = extract_top_kcs(doc2concepts=doc2concepts,output_dir=outdir,define_kc=False,topk=-1)

    if concept == config.LIST_FILTER:
        ingram = (1,5)
        name = (keyword_list_path.split('/')[-1]).split('.')[0] #hardcoded filename for triecache
        ptfidf =  model_dir +"m_"+ concept+"ngram"+str(ingram[0])+"_"+str(ingram[1])+".pickle"
        outdir =concept_dir + config.dir_sep + concept+"_"+name

        if no_of_topics == -1:
            outdir += "all"
        else:
            outdir += "top" + str(no_of_topics)

        model = None

        if not os.path.exists(ptfidf)  or config.Remove_Prev_models:
            model = tfidf.TFIDFExtractor(train_docs,ngram=ingram)
            model.train()
            model.save_model(ptfidf)
        else:
            print(" Model Exists : "+ptfidf)

        model = cm.load_model(ptfidf)
        keyword_list = keyword.KeywordList(name,keyword_list_path)
        doc2concepts_ngram = model.get_Topics(extract_docs)
        doc2concepts = keyword_list.get_Topics_secondFilter(doc2concepts_ngram) #FILTER


        df = extract_top_kcs(doc2concepts=doc2concepts,output_dir=outdir,define_kc=False,topk=no_of_topics)
        # df = extract_top_kcs_sm(doc2concepts=doc2concepts,define_kc=False,topk=no_of_topics)
        # df.to_csv(outdir,index=None)