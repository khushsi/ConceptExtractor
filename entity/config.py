import os

import logging

import time, datetime
import sys

from enum import Enum

class TrainType(Enum):
    train_product = 0
    train_tag = 1

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.value == other.value


    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value


class Config:
    def init_logging(self, logfile):
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                      datefmt='%m/%d/%Y %H:%M:%S')
        fh = logging.FileHandler(logfile)
        # ch = logging.StreamHandler()
        ch = logging.StreamHandler(sys.stdout)

        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # fh.setLevel(logging.INFO)
        # ch.setLevel(logging.INFO)
        logging.getLogger().addHandler(ch)
        logging.getLogger().addHandler(fh)
        self.level = logging.getLogger().setLevel(logging.INFO)

        return logging

    def __init__(self, flag, train_type, dim_item):
        home = os.environ["HOME"]
        if train_type == "prod":
            self.train_type = TrainType.train_product
        elif train_type == "tag":
            self.train_type = TrainType.train_tag
        self.flag = flag

        self.i = 150
        self.tf_cutoff = self.i  # use 100, #(freq>100)=22.5k, #(freq>200)=16.5k, #(freq>500)=10.6k
        self.timemark  = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
        # for data
        # self.path_data = "".join([home, "/Data/yelp_bkp/output/review_processed_rest_interestword_DEC22.txt"])
        # self.path_data = "".join([home, "/Data/yelp_s/output/review_processed_rest_interestword_Jan7_alltrue_nostem.txt"])
        self.path_data = "".join([home, "/Data/yelp_s/output/review_processed_rest_interestword_20170425_freq=100.txt"])
        # self.path_data = "".join([home, "/Data/yelp_s/sample.txt"])
        # self.path_embed = "".join([home, "/Data/glove/glove.processed.840B.300d.txt"])
        self.path_embed     = "".join([home, "/Data/glove/glove.6B.200d.txt"])
        self.path_raw_data  = "".join([home, "/Data/yelp_s/output/raw_review_restaurant.json"])
        self.path_raw_data_dict  = "".join([home, "/Data/yelp_s/output/raw_review_restaurant_dictioary.json"])

        self.path_log = "".join([home, "/Data/yelp_s/model/training.%s_%s.%s.log" % (self.flag, self.flag, self.timemark)])
        if not os.path.exists(home+'/Data/yelp_s/model/'):
            os.makedirs(home+'/Data/yelp_s/model/')
        self.logger = self.init_logging(self.path_log)

        # for Gensim setting
        self.path_gensim_tf = 30 # set as 30, #(freq>30)=50k, #(freq>1)=226k,
        self.processed_document_path = home + "/Data/yelp_s/output/" + 'restaurant_processed_document.txt'
        self.path_review_pairs = home + "/Data/yelp_s/output/" + 'restaurant_review_pairs_freq=%d.txt' % self.tf_cutoff

        self.path_gensim_dict = home + "/Data/yelp_s/output/gensim_yelp_review_freq=%d.dict" % self.path_gensim_tf
        self.path_gensim_corpus = home + "/Data/yelp_s/output/gensim_yelp_review_freq=%d.corpus" % self.path_gensim_tf
        # Gensim LDA
        self.lda_model_path = home + '/Data/yelp_s/model/lda/yelp_restaurant_review.lda'
        self.path_lda_doc_vector = home + '/Data/yelp_s/model/lda/yelp_10class_doc-lda_vectors.pkl'
        # Gensim TfIdf
        self.tfidf_model_path = home + '/Data/yelp_s/model/tfidf/yelp_restaurant_review.tfidf'
        self.path_tfidf_doc_vector = home + '/Data/yelp_s/model/tfidf/yelp_10class_doc-tfidf_vectors.pkl'

        # for model
        self.dim_word = 200
        self.dim_item = dim_item
        self.neg_trials = 100

        self.path_weight = "".join([home, "/Data/yelp_s/model/chk_",self.flag , "/weight"])
        if not os.path.exists(os.path.dirname(self.path_weight)):
            os.makedirs(os.path.dirname(self.path_weight))
        self.path_checkpoint = "".join([home, "/Data/yelp_s/model/chk_", self.flag, "/checkpointweights.hdf5"])
        if not os.path.exists(os.path.dirname(self.path_checkpoint)):
            os.makedirs(os.path.dirname(self.path_checkpoint))
        self.path_npy = "".join([home, "/Data/yelp_s/model/npy/"])
        if not os.path.exists(os.path.dirname(self.path_npy)):
            os.makedirs(os.path.dirname(self.path_npy))
        self.batch_size = 300000
        self.n_epoch = 700
        # self.sample_per_epoch = 19135900ccccc
        self.sample_per_epoch = 12500000
        # self.sample_per_epoch = 500000

        # for framework
        # for save
        self.path_doc_npy = "".join([home, "/Data/yelp_s/model/chk_",self.flag,"/doc"])
        self.path_word_npy = "".join([home, "/Data/yelp_s/model/chk_",self.flag,"/word"])
        self.path_tag_npy = "".join([home, "/Data/yelp_s/model/chk_",self.flag,"/tag"])
        self.path_model_npy = "".join([home, "/Data/yelp_s/model/chk_",self.flag,"/model"])

        # generate in the evaluate/eva_product.py
        self.path_doc_w2c = "".join([home, "/Data/yelp_s/model/chk_",self.flag,"/doc.txt"])
        self.path_word_w2c = "".join([home, "/Data/yelp_s/model/chk_",self.flag,"/word.txt"])
        if not os.path.exists(os.path.dirname(self.path_doc_npy)):
            os.makedirs(os.path.dirname(self.path_doc_npy))
        if not os.path.exists(os.path.dirname(self.path_word_npy)):
            os.makedirs(os.path.dirname(self.path_word_npy))
        if not os.path.exists(os.path.dirname(self.path_tag_npy)):
            os.makedirs(os.path.dirname(self.path_tag_npy))
        if not os.path.exists(os.path.dirname(self.path_model_npy)):
            os.makedirs(os.path.dirname(self.path_model_npy))
        if not os.path.exists(os.path.dirname(self.path_doc_w2c)):
            os.mkdir(os.path.dirname(self.path_doc_w2c))
        if not os.path.exists(os.path.dirname(self.path_word_w2c)):
            os.mkdir(os.path.dirname(self.path_word_w2c))

        # self.path_logs = "".join([home, "/Data/yelp_s/model/log/", self.flag, ".log"])
        # if not os.path.exists(os.path.dirname(self.path_logs)):
        #     os.mkdir(os.path.dirname(self.path_logs))

