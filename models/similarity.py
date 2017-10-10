import os, numpy, pickle
from gensim import corpora, models, similarities
from gensim.matutils import Dense2Corpus, dense2vec, cossim

from models.transformer import *


class SimilarityIndex():
    def __init__(self, seqs, filepath, min_freq=1, use_tfidf=True, use_lsi=False, n_lsi_dim=500):
        self.min_freq = min_freq
        self.use_tfidf = use_tfidf
        self.use_lsi = use_lsi
        self.n_lsi_dim = n_lsi_dim
        self.filepath = filepath

        if not os.path.isdir(self.filepath):
            os.mkdir(self.filepath)

        self.lexicon_filepath = self.filepath + "/lexicon"

        if os.path.exists(self.lexicon_filepath):
            self.load_lexicon()
        else:
            self.make_lexicon(seqs)

        self.min_freq = min(self.lexicon.dfs.values())

        if self.use_tfidf:
            self.tfidf_filepath = self.filepath + "/tfidf"
            if os.path.exists(self.tfidf_filepath):
                self.load_tfidf_model()
            else:
                self.make_tfidf_model(seqs)

        if self.use_lsi:
            self.index_dir_filepath = self.filepath + "/lsi-index"
            self.lsi_filepath = self.filepath + "/lsi"
            if os.path.exists(self.lsi_filepath):
                self.load_lsi_model()
            else:
                self.make_lsi_model(seqs)
            self.n_lsi_dim = self.lsi_model.num_topics
        else:
            self.index_dir_filepath = self.filepath + "/index"

        self.index_filepath = self.index_dir_filepath + "/index"
        if os.path.exists(self.index_filepath):
            self.load_index()
        else:
            if not os.path.isdir(self.index_dir_filepath):
                os.mkdir(self.index_dir_filepath)
            self.make_index(seqs)

    def load_lexicon(self):
        print "loading lexicon from", self.lexicon_filepath
        self.lexicon = corpora.Dictionary.load(self.lexicon_filepath)

    def make_lexicon(self, seqs):
        self.lexicon = corpora.Dictionary(tokenize(seq) for seq in seqs)
        self.lexicon.filter_extremes(no_below=self.min_freq)
        print "generated lexicon of", len(self.lexicon.keys()), "words with frequency >=", self.min_freq
        self.lexicon.compactify()
        self.lexicon.save(self.lexicon_filepath)
        print "saved lexicon to", self.lexicon_filepath

    def load_tfidf_model(self):
        print "loading tfidf from", self.tfidf_filepath
        self.tfidf_model = models.TfidfModel.load(self.tfidf_filepath, mmap='r')

    def make_tfidf_model(self, seqs):
        self.tfidf_model = models.TfidfModel((self.lexicon.doc2bow(tokenize(seq)) for seq in seqs))
        self.tfidf_model.save(self.tfidf_filepath)
        print "saved tfidf to", self.tfidf_filepath

    def load_lsi_model(self):
        print "loading lsi model from", self.lsi_filepath
        self.lsi_model = models.LsiModel.load(self.lsi_filepath, mmap='r')

    def make_lsi_model(self, seqs):
        if self.use_tfidf:
            seqs = (self.tfidf_model[self.lexicon.doc2bow(tokenize(seq))] for seq in seqs)
        else:
            seqs = (self.lexicon.doc2bow(tokenize(seq)) for seq in seqs)
        self.lsi_model = models.LsiModel(seqs, num_topics=self.n_lsi_dim, id2word=self.lexicon)
        self.lsi_model.save(self.lsi_filepath)
        print "saved lsi model to", self.lsi_filepath
    
    def load_index(self):
        print "loading index from", self.index_filepath
        self.index = similarities.Similarity.load(self.index_filepath, mmap='r')
    
    def make_index(self, seqs):
        print "building index for sequences"
        #import pdb;pdb.set_trace()
        if self.use_lsi:
            if self.use_tfidf:
                seqs = (self.lsi_model[self.tfidf_model[self.lexicon.doc2bow(tokenize(seq))]] for seq in seqs)
            else:
                seqs = (self.lsi_model[self.lexicon.doc2bow(tokenize(seq))] for seq in seqs)
            num_features = self.lsi_model.num_topics
        else:
            if self.use_tfidf:
                seqs = (self.tfidf_model[self.lexicon.doc2bow(tokenize(seq))] for seq in seqs)
            else:
                seqs = (self.lexicon.doc2bow(tokenize(seq)) for seq in seqs)
            num_features = len(self.lexicon.keys())

        self.index = similarities.Similarity(output_prefix=self.index_filepath, corpus=None, num_features=num_features)
        self.index.save(self.index_filepath)
        self.index.add_documents(seqs)
        self.index.save(self.index_filepath)
        print "saved index to", self.index_filepath

    def get_sim_seq_idxs(self, seqs, n_best=5):
        if type(seqs) in (unicode, str):
            seqs = [seqs]
        #import pdb;pdb.set_trace()
        self.index.num_best = n_best
        if self.use_tfidf:
            seqs = [self.tfidf_model[self.lexicon.doc2bow(tokenize(seq))] for seq in seqs]
        else:
            seqs = [self.lexicon.doc2bow(tokenize(seq)) for seq in seqs]
        if self.use_lsi:
            seqs = self.lsi_model[seqs]
        sim_seqs = self.index[seqs]
        sim_idxs = [[sim_seq[0] for sim_seq in n_best_seqs] for n_best_seqs in sim_seqs]
        sim_idxs = numpy.array([idxs + [numpy.nan] * (n_best - len(idxs)) for idxs in sim_idxs]) # in some weird cases, fewer than n_best sequence IDs will be returned
        sim_scores = [[sim_seq[1] for sim_seq in n_best_seqs] for n_best_seqs in sim_seqs]
        sim_scores = numpy.array([scores + [numpy.nan] * (n_best - len(scores)) for scores in sim_scores]) # in some weird cases, fewer than n_best sequence IDs will be returned
        return sim_idxs, sim_scores

    @classmethod
    def load(cls, filepath, use_tfidf=True, use_lsi=False):
        sim_index = SimilarityIndex(seqs=None, filepath=filepath, use_tfidf=use_tfidf, use_lsi=use_lsi)
        return sim_index


