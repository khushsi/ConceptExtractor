import gensim
import numpy as np
import os
import pickle
from entity.util import nlp

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
            tokens = nlp.preprocessText(doc.text)
            vec    = self.vectorize(tokens).tolist()
            with open(os.path.join(OUTPUT_FOL, doc.id + ".txt.phrases"), 'wb') as f_:
                pickle.dump(vec, f_)