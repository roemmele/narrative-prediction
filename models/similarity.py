import os, numpy
from gensim import corpora, models, similarities
from gensim.matutils import Dense2Corpus, dense2vec

from models.transformer import segment_and_tokenize, tokenize

class StoryYielder():
    #retrieves full stories one by one
    def __init__(self, story_ids):
        self.story_ids = story_ids
    def __iter__(self):
        #retrieve from stories table
        for story_id in self.story_ids:
            story = get_stories(story_id)
            assert(story)
            yield story
    

class SequenceYielder():
    #retrieves story segments one by one
    def __init__(self, n_sent, story_ids, db_filepath):
        self.n_sent = n_sent
        self.story_ids = story_ids
        self.db_filepath = db_filepath
        self.n_seq = make_seqs_table(self.story_ids, n_sent, self.db_filepath)
    def __iter__(self):
        #retrieve from seqs table - sqlite3 ids start at 1
        for seq_id in range(1, self.n_seq + 1):
            #import pdb;pdb.set_trace()
            #retrieve next - in sqlite ids start at 1
            seq = get_seqs(seq_id, self.db_filepath)
            assert(seq)
            yield seq


class SimilarityIndex():
    def __init__(self, filepath, story_ids=None, stories=None, n_sent=None, min_freq=2):
        '''either retreive stories from story ids in db, or index given stories'''
        self.n_sent = n_sent
        self.story_ids = story_ids
        self.stories = stories
        self.min_freq = min_freq
        self.filepath = filepath
        self.name = filepath.split("/")[-1]
        if self.filepath[-1] != "/":
            self.filepath += "/"
        if os.path.isdir(self.filepath):
            #try to load existing index if name given
            self.load_index()
        else:
            assert(self.story_ids or self.stories)
            #assert(self.stories is None if self.story_ids)
            #assert(self.stories is None if self.n_sent)
            #if index by this name doesn't exist, create new index
            os.mkdir(self.filepath)
            if not self.stories:
                #retrieve stories from db
                if self.n_sent:
                    #import pdb;pdb.set_trace()
                    #stories are sequences of self.n_sent sentences
                    stories = SequenceYielder(self.n_sent, self.story_ids,
                                                  db_filepath=self.filepath + self.name + ".seqs.db")
                    #print "building index for", stories.n_seq, "story sequences of", self.n_sent, "sentences each"
                else:
                    stories = StoryYielder(self.story_ids)
            self.make_index(stories)
            self.save_index()
    
    def load_index(self):
        print "loading index", self.name
        if os.path.isdir(self.filepath + self.name + ".story_ids"):
            with open(self.filepath + self.name + ".story_ids", 'rb') as f:
                self.story_ids = pickle.load(f)
        self.lexicon = corpora.Dictionary.load(self.filepath + self.name + ".lexicon")
        #self.model = models.TfidfModel.load(self.filepath + self.name + ".model")
        self.index = similarities.MatrixSimilarity.load(self.filepath + self.name + ".index")
                   
    def save_index(self):
        if self.story_ids:
            with open(self.filepath + self.name + ".story_ids", 'wb') as f:
                pickle.dump(self.story_ids, f)
        self.lexicon.save(self.filepath + self.name + ".lexicon")
        #self.model.save(self.filepath + self.name + ".model")
        self.index.save(self.filepath + self.name + ".index")

        print "saved index to", self.name, "folder"
                   
    
    def make_index(self, seqs):
        print "building index for", len([seq for seq in seqs]), "story sequences"
        self.lexicon = corpora.Dictionary([tokenize(seq) for seq in seqs])
        self.lexicon.filter_extremes(no_below=self.min_freq)
        print "generated lexicon of", len(self.lexicon.keys()), "words with frequency >=", self.min_freq
        self.lexicon.compactify()
        #import pdb;pdb.set_trace()
        corpus = [self.lexicon.doc2bow(tokenize(seq)) for seq in seqs]
        #self.model = models.TfidfModel(corpus, id2word=self.lexicon, normalize=True)
        #self.index = similarities.MatrixSimilarity(self.model[corpus])
        self.index = similarities.MatrixSimilarity(corpus)


    def get_similar_seqs(self, seqs, n_best=1):
        #import pdb;pdb.set_trace()
        seqs = [self.lexicon.doc2bow(segment_and_tokenize(seq)) for seq in seqs]
        #scores = self.index[self.model[seq]]
        scores = self.index[seqs]
        best_ids = numpy.argsort(scores, axis=1)#[::-1]
        if self.n_sent:
            #use sequence ids for retrieval from seq db - sqlite ids start at 1
            best_ids = list(best_ids[:, -n_best:] + 1)
            seqs = get_seqs(seq_ids=best_ids, db_filepath=self.filepath + self.name + ".seqs.db")
        elif self.story_ids:
            #use story ids for retrieval from story db
            best_ids = [self.story_ids[id] for id in best_ids[:, -n_best:]]
            seqs = get_stories(story_ids=best_ids)
        elif self.stories:
            #stories already loaded in memory
            seqs = [[self.stories[id] for id in ids] for ids in best_ids[:, -n_best:]]
        return best_ids[:, -n_best:], seqs