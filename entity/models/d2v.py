import gensim
import numpy as np
import os
import pickle
from entity.util import nlp
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

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

            words  = nlp.preprocessText(doc.text)
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
            tokens = nlp.preprocessText(doc.text)
            vec    = self.vectorize(tokens).tolist()
            with open(os.path.join(OUTPUT_FOL, doc.id + ".txt.phrases"), 'wb') as f_:
                pickle.dump(vec, f_)