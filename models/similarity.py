import os, numpy, pickle
from gensim import corpora, models, similarities
from gensim.matutils import Dense2Corpus, dense2vec, cossim

from models.transformer import segment_and_tokenize, tokenize

import story_corpus
reload(story_corpus)
from story_corpus import *


class SequenceYielder():
    #retrieves full stories one by one
    def __init__(self, seq_ids, db_filepath):
        self.seq_ids = seq_ids
        self.db_filepath = db_filepath
        if not self.db_filepath:
            self.db_filepath = parsed_db_filepath #db defaults to main stories db if name not given
    def __iter__(self):
        #retrieve from stories table
        for seq_idx, seq_id in enumerate(self.seq_ids):
            seq = get_seqs(seq_id, self.db_filepath)
            if seq_idx % 50000 == 0:
                print "retrieved", seq_idx, "sequences..."
            #assert(seq)
            yield seq

            
# class KeywordIndex():
#     def __init__(self, keywords):
#         self.keywords = keywords
#         print "building keyword index"
#         self.keyword_dict = corpora.Dictionary([self.keywords])
#         self.keyword_doc = self.keyword_dict.doc2bow(self.keywords)
#         self.index = similarities.MatrixSimilarity(corpus=[self.keyword_doc])
#     def get_scores(self, docs):
#         docs = [self.keyword_dict.doc2bow(segment_and_tokenize(doc)) for doc in docs]
#         scores = self.index[docs].flatten()
#         #score = cossim(self.keyword_doc, docs)
#         return scores


class SimilarityIndex():
    def __init__(self, seqs, dirname, fileprefix, min_freq=2, use_tfidf=False, use_lsi=False, n_lsi_dim=500):
        self.min_freq = min_freq
        self.use_tfidf = use_tfidf
        self.use_lsi = use_lsi
        self.n_lsi_dim = n_lsi_dim
        self.dirname = dirname
        self.fileprefix = fileprefix

        if not os.path.isdir(self.dirname):
            os.mkdir(self.dirname)

        self.lexicon_filepath = self.dirname + "/" + self.fileprefix + ".lexicon"

        if os.path.exists(self.lexicon_filepath):
            self.load_lexicon()
        else:
            self.make_lexicon(seqs)

        if self.use_tfidf:
            self.tfidf_filepath = self.dirname + "/" + self.fileprefix + ".tfidf"
            if os.path.exists(self.tfidf_filepath):
                self.load_tfidf_model()
            else:
                self.make_tfidf_model(seqs)

        if self.use_lsi:
            self.index_dirname = self.dirname + "/" + self.fileprefix + "-lsi-index"
            self.lsi_filepath = self.dirname + "/" + self.fileprefix + ".lsi"
            if os.path.exists(self.lsi_filepath):
                self.load_lsi_model()
            else:
                self.make_lsi_model(seqs)
        else:
            self.index_dirname = self.dirname + "/" + self.fileprefix + "-index"

        self.index_filepath = self.index_dirname + "/" + self.fileprefix + ".index"
        if os.path.exists(self.index_filepath):
            self.load_index()
        else:
            if not os.path.isdir(self.index_dirname):
                os.mkdir(self.index_dirname)
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
        #self.lsi_model = models.RpModel((self.lexicon.doc2bow(tokenize(seq)) for seq in seqs), id2word=self.lexicon, num_topics=500)
        self.lsi_model.save(self.lsi_filepath)
        print "saved lsi model to", self.lsi_filepath
    
    def load_index(self):
        print "loading index from", self.index_filepath
        self.index = similarities.Similarity.load(self.index_filepath, mmap='r')
        for shard in self.index.shards:#something weird happened with the naming of the BOW index, and need to update directory name in shards
            shard.dirname = self.index_dirname
    
    def make_index(self, seqs):
        print "building index for story sequences"
        #import pdb;pdb.set_trace()
        if not os.path.isdir(self.index_dirname):
            os.mkdir(self.index_dirname)
        if self.use_lsi:
            if self.use_tfidf:
                seqs = (self.lsi_model[self.tfidf_model[self.lexicon.doc2bow(tokenize(seq))]] for seq in seqs)
            else:
                seqs = (self.lsi_model[self.lexicon.doc2bow(tokenize(seq))] for seq in seqs)
            num_features = self.lsi_model.num_topics
        else:
            seqs = (self.tfidf_model[self.lexicon.doc2bow(tokenize(seq))] for seq in seqs)
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
            seqs = [self.tfidf_model[self.lexicon.doc2bow(segment_and_tokenize(seq))] for seq in seqs]
        else:
            seqs = [self.lexicon.doc2bow(segment_and_tokenize(seq)) for seq in seqs]
        if self.use_lsi:
            seqs = self.lsi_model[seqs]
        sim_seqs = self.index[seqs]
        sim_idxs = numpy.array([[sim_seq[0] for sim_seq in n_best_seqs] for n_best_seqs in sim_seqs])
        scores = numpy.array([[sim_seq[1] for sim_seq in n_best_seqs] for n_best_seqs in sim_seqs])
        if len(sim_idxs) == 1:
            sim_idxs = sim_idxs[0]
            scores = scores[0]
        return sim_idxs, scores

class RetrievalIndex():
    '''Creates a similarity index for stories or story segments from the given database based on IDs,
    then retrieves similar sequences by finding IDs in db'''

    def __init__(self, dirname, fileprefix, seq_ids=None, min_freq=5, use_tfidf=False, use_lsi=False, n_lsi_dim=500):
        self.seq_ids = seq_ids
        self.dirname = dirname
        self.fileprefix = fileprefix
        self.db_filepath = self.dirname + "/" + self.fileprefix + ".db"
        self.ids_filepath = self.dirname + "/" + self.fileprefix + ".ids"

        if os.path.exists(self.ids_filepath):
            self.load_ids() #IDs may be associated with stories or with story segments
        else:
            assert(self.seq_ids is not None) #IDs must be provided if ID file doesn't exist
            self.save_ids(self.seq_ids)

        seqs = SequenceYielder(self.seq_ids, self.db_filepath)
        self.sim_index = SimilarityIndex(seqs=seqs, dirname=self.dirname, fileprefix=self.fileprefix, 
                                        min_freq=min_freq, use_tfidf=use_tfidf, use_lsi=use_lsi, n_lsi_dim=n_lsi_dim)

    def load_ids(self):
        print "loading sequence IDs from", self.ids_filepath
        with open(self.ids_filepath, 'rb') as path:
            #self.seq_ids = numpy.array([int(seq_id.strip()) for seq_id in path.readlines()])
            self.seq_ids = pickle.load(path)

    def save_ids(self, ids):
        with open(self.ids_filepath, 'wb') as path:
            #path.write("\n".join(map(str, self.seq_ids)))
            pickle.dump(self.seq_ids, path)
        print "saved sequence IDs to", self.ids_filepath

    def get_similar_ids(self, seqs, n_best=100):
        '''get most similar sequences to given sequences by db IDs'''
        if type(seqs) in (unicode, str):
            seqs = [seqs]
        sim_idxs, scores = self.sim_index.get_sim_seq_idxs(seqs=seqs, n_best=n_best)
        if type(sim_idxs[0]) in (list, numpy.ndarray):
            sim_db_ids = [[self.seq_ids[idx] for idx in idxs] for idxs in sim_idxs]
        else:
            sim_db_ids = [self.seq_ids[idx] for idx in sim_idxs]
        sim_db_ids = numpy.array(sim_db_ids)
        if len(sim_db_ids) == 1:
            sim_db_ids = sim_db_ids[0]
        return sim_db_ids

