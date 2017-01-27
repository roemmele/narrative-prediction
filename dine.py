
# coding: utf-8

# In[10]:

from __future__ import unicode_literals
import MySQLdb, json, numpy, inspect, sys, copy, timeit, six, spacy, pickle, os
#from segtok.segmenter import split_single
#from segtok.tokenizer import word_tokenizer
from sklearn.base import BaseEstimator, clone
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.cross_validation import train_test_split, LeaveOneOut, KFold, cross_val_predict
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, RepeatVector, TimeDistributed, Lambda, Masking, merge, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop, SGD, Adagrad, Adam
from keras import backend as K
from seq2seq.models import Seq2Seq

# sys.path.append("../ROC")
# import roc_pmi
# reload(roc_pmi)
# from roc_pmi import *
# sys.path.append("../AvMaxSim")
# import similarity_score as sim_score

rng = numpy.random.RandomState(0)
numpy.set_printoptions(precision=3)

sys.setrecursionlimit(5000)


# In[3]:

def segment(text):
    #return [sentence for sentence in split_single(text.strip())]
    return [sentence.string.strip() for sentence in encoder(text).sents]

def tokenize(sentence):
    #return [word.lower() for word in word_tokenizer(sentence)]
    return [word.lower_ for word in encoder(sentence) if word.string.strip()]

def segment_and_tokenize(text):
    #import pdb;pdb.set_trace()
    #return [word for sentence in segment(text) for word in tokenize(sentence)]
    return [word.lower_ for sentence in encoder(text).sents 
                            for word in sentence if word.string.strip()]

def replace_entities(tok_seq, ents):
    #input is sequence already tokenized into words
    #if rep strings given, replace entity word (e.g. "PERSON") with different string (e.g. "PERSON" > "I")
    return [ents[word] if word in ents else word for word in tok_seq]
    #tok_seq = [rep_strings[word] if word in rep_strings else word for word in tok_seq]
    #return tok_seq

def get_entities(text):
    return {ent.string.strip().lower(): ent.label_ for ent in encoder(text).ents}

def connectDB(host, user, passwd, db_name):
    return MySQLdb.connect(host=host, user=user, passwd=passwd, db=db_name)


# In[4]:

# #load spacy model for nlp tools
encoder = spacy.load('en')


# In[5]:

all_page_ids = [112, 113, 114, 126, 128, 129, 133, 134, 135, 136, 137, 138, 139, 140, 141, 143, 145, 146,                147, 148, 149, 150, 159, 160, 163, 165, 167, 174, 176, 178]

#db credentials
host = "saysomething.ict.usc.edu"
user = "dine_user" 
passwd = "notasecret"
db_name = "dine"

def get_page(page_id):
    db = connectDB(host, user, passwd, db_name)
    cursor = db.cursor()
    cursor.execute("SELECT json FROM page WHERE id = %s", (page_id,))
    page = cursor.fetchone()[0]
    page = json.loads(page)
    page_text = page["text"]
    outcome_set = [outcome["text"] for outcome in page["outcomes"]]
    input_examples = [outcome["examples"] for outcome in page["outcomes"]]
    cursor.close()
    db.close()
    proxy_inputs, proxy_outcomes = map_proxy_inputs(input_examples, outcome_set)
    return page_text, outcome_set, proxy_inputs, proxy_outcomes

def get_all_page_ids():
    db = connectDB(host, user, passwd, db_name)
    cursor = db.cursor()
    cursor.execute("SELECT id FROM page")
    page_ids = [page_id[0] for page_id in cursor.fetchall()]
    cursor.close()
    db.close()
    return page_ids

def get_project_page_ids(project):
    db = connectDB(host, user, passwd, db_name)
    cursor = db.cursor()
    cursor.execute("SELECT id FROM page WHERE project = %s", (project,))
    page_ids = [id[0] for id in cursor.fetchall()]
    cursor.close()
    db.close()
    return page_ids

def map_proxy_inputs(input_examples, outcome_set):
    '''pair authors' example inputs with their designated outcomes'''
    in_seqs = []
    y = []
    for out_index, out_seq in enumerate(outcome_set):
        for example in input_examples[out_index]:
            in_seqs.append(example + ".")
            y.append(out_index)
    assert(len(in_seqs) == len(y))
    return in_seqs, numpy.array(y)

def abbrev_seq(sequence):
    #keep only first sentence in sequence
    return list(split_single(sequence))[0]


def get_interactions(page_id, crowd_only=False):
    '''return user inputs for this page id along with the outcomes shown to the user
    as well as the annotated (gold) outcomes (if annotations exist)'''
    db = connectDB(host, user, passwd, db_name)
    cursor = db.cursor()
    cursor.execute("SELECT interaction.user_input, interaction.ordered_outcomes, annotation.outcome                    FROM page                    LEFT JOIN interaction ON page.id = interaction.page_id                    LEFT JOIN annotation ON interaction.id = annotation.interaction_id                    WHERE page.id = %s                    AND interaction.task IN %s                    AND ((annotation.annotator = page.author AND annotation.class = \"outcome\")                            OR annotation.id IS NULL)                    AND interaction.ordered_outcomes IS NOT NULL AND NOT interaction.ordered_outcomes = \"\"                    ORDER BY interaction.id", (page_id, tuple(["crowd"]) 
                                                        if crowd_only else tuple(["page", "crowd"])))
    interactions = cursor.fetchall()
    cursor.close()
    db.close()
    #filter non-ascii characters from user inputs
    user_inputs = [unicode(interaction[0], errors="ignore") for interaction in interactions]
    #list of outcomes ordered by score 
    top_outcomes = numpy.array([json.loads(interaction[1])[0] for interaction in interactions], dtype=int)
    annotated_outcomes = [interaction[2] for interaction in interactions]
    if None not in annotated_outcomes:
        annotated_outcomes = numpy.array(annotated_outcomes, dtype=int)
    assert(len(user_inputs) == len(top_outcomes) and len(user_inputs) == len(annotated_outcomes))
    return user_inputs, top_outcomes, annotated_outcomes

def map_input_outcome_seqs(inputs, outcome_seqs):
    assert(len(inputs) == len(outcome_seqs))
    assert(None not in outcome_seqs)
    input_outcome_seqs = []
    for input, outcome in zip(inputs, outcome_seqs):
        #add end of sentence marker (period) to input if not already there
        #import pdb.pdb_set_trace()
        if input[-1] not in [".", "?", "!"]:
            input += "."
        input_outcome_seqs.append(input + " " + outcome)
    return input_outcome_seqs


# In[2]:

def check_y_choices(X, y_choices):
    #check if y_choices is single set or if there are different choices for each input
    if type(y_choices) == list or len(y_choices.shape) > 2:
        #different set of choices for each input
        assert(len(X) == len(y_choices)) 
    else:
        #different set of choices for each input
        y_choices = numpy.repeat(y_choices[None,:,:], len(X), axis=0)
    return y_choices

def show_predictions(X, y, prob_y=None, y_choices=None):
    #show model predictions compared with gold
    y_choices = check_y_choices(X, y_choices)
    for x, y_, prob_y_, choices in zip(X, y, prob_y, y_choices):
        print "INPUT:", x
        pred_y = numpy.argmax(prob_y_)
        for index, choice in enumerate(choices):
            if index == pred_y:
                #predicted choice
                choice = ">" + choice
            if index == y_:
                #correct choice
                choice = "*" + choice
            print choice, "({:.6f})".format(prob_y_[index])
        print "\n"
    return 

def load_pipeline(filepath, embeddings=None):            
    #load saved models
    transformer = pickle.load(open(filepath + '/transformer.pkl', 'rb'))
    #if embeddings are given, add them to transformer (they don't come pickled with transformer)
    transformer.embeddings = embeddings
    #load keras model itself
    classifier = pickle.load(open(filepath + '/classifier.pkl', 'rb'))
    classifier.model = load_model(filepath + '/classifier.h5')
    #load entire classifier object
    return transformer, classifier


# In[51]:

class SavedModel():
    def save(self):
        #import pdb;pdb.set_trace()
        
        #save model
        if not os.path.isdir(self.filepath):
            os.mkdir(self.filepath)
        
        self.model.save(self.filepath + '/classifier.h5')
        pickle.dump(self, open(self.filepath + '/classifier.pkl', 'wb'))
        
    def __getstate__(self):
        #import pdb;pdb.set_trace()
        return dict((k, v) for (k, v) in self.__dict__.iteritems() if k != 'model')


# In[19]:

class SequenceTransformer(FunctionTransformer):
    def __init__(self, lexicon=None, min_freq=1, max_length=None, extra_text=[], 
                 pad_seq=False, verbose=1, unk_word="<UNK>", embeddings=None, replace_ents=True, 
                 embed_y=False, reduce_emb_mode=None,  copy_input_to_output=False, filepath=None):
        #import pdb;pdb.set_trace()
        FunctionTransformer.__init__(self)
        self.lexicon = lexicon
        self.unk_word = unk_word #string representation for unknown words in lexicon
        #use existing word embeddings if given
        self.embeddings = embeddings
        self.pad_seq = pad_seq
        if lexicon:
            #use existing lexicon
            self.lexicon = lexicon
            self.lexicon_size = len(self.lexicon)
            #insert entry for empty timeslot in lexicon lookup
            self.lexicon_lookup = [None] + [word for index, word in 
                                    sorted([(index, word) for word, index in self.lexicon.items()])]
            assert(len(self.lexicon_lookup) == self.lexicon_size + 1)
            
        self.min_freq = min_freq
        self.max_length = max_length
        #page text includes additional words that should be included in lexicon
        if type(extra_text) is str or type(extra_text) is unicode:
            extra_text = [extra_text]
        self.extra_text = extra_text
        self.verbose = verbose
        #specify if y_seqs should be converted to embeddings like input seqs
        if embed_y:
            assert(self.embeddings is not None)
        self.embed_y = embed_y
        #specify if named entities should be replaced with generic labels
        self.replace_ents = replace_ents
        #specify if embeddings should be combined across sequence (e.g. take mean, sum)
        if reduce_emb_mode:
            assert(self.embeddings is not None)
            assert(not self.pad_seq)
        self.reduce_emb_mode = reduce_emb_mode
        self.copy_input_to_output = copy_input_to_output
        if verbose and self.replace_ents:
            print "filter named entities = True"
        self.filepath = filepath
        
    def make_lexicon(self, text=[]):
        #import pdb;pdb.set_trace()
        word_counts = {}
        for sequence in list(text):
            #first get named entities
            words = segment_and_tokenize(text=sequence)

            if self.replace_ents:
                #reduce vocab by mapping all named entities to entity labels (e.g. "PERSON")
                ents = get_entities(sequence)
                if ents:
                    words = replace_entities(words, ents)

            for word in words:
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1

        words, counts = zip(*word_counts.items())
        counts = numpy.array(counts)

        #compute num words with count >= min_word_frequency
        lexicon_size = numpy.sum(counts >= self.min_freq)

        #get indices of lexicon words sorted by their count;
        #words that occur less often than the frequency threshold will be removed
        sorted_word_indices = numpy.argsort(counts)[::-1]

        lexicon = {}

        #add unknown word
        lexicon[self.unk_word] = 1

        '''convert counts in lexicon to indices - start indices at 2
        lower indices are reserved for unknown words (1) and empty timeslots (0)'''
        for index, word_index in enumerate(sorted_word_indices[:lexicon_size]):
            lexicon[words[word_index]] = index + 2

        return lexicon
    
    def fit(self, X, y_seqs=None):
        #import pdb;pdb.set_trace()
        #add words from outcomes and page to lexicon in addition to user inputs
        X = self.format_seqs(seqs=X, unravel=True)
        #X = self.unravel_seqs(seqs=X)
        if y_seqs is not None:
            y_seqs = self.format_seqs(seqs=y_seqs, unravel=True)
#             y_seqs = self.unravel_seqs(seqs=y_seqs)
            #add outcome sequences to lexicon
            self.extra_text += set(y_seqs)
        lexicon_text = X + self.extra_text
        self.lexicon = self.make_lexicon(text=lexicon_text)
        self.lexicon_size = len(self.lexicon)
        #insert entry for empty timeslot in lexicon lookup
        self.lexicon_lookup = [None] + [word for index, word in 
                                sorted([(index, word) for word, index in self.lexicon.items()])]
        assert(len(self.lexicon_lookup) == self.lexicon_size + 1)
        #import pdb;pdb.set_trace()
        if self.pad_seq:
            self.set_max_length(seqs=lexicon_text)
        if self.filepath:
            #if filepath given, save transformer
            self.save()
        if self.verbose:
            print "generated lexicon of", self.lexicon_size, "words with frequency >=", self.min_freq
    
    def format_seqs(self, seqs, unravel=False):
        #get input and output into standard format
        if isinstance(seqs, (str, unicode)):
            #input is single string, put inside list
            seqs = [seqs]
            
        if isinstance(seqs[0], (str, unicode)):
            #put each string inside tuple
            seqs = [[seq] for seq in seqs]
        
        assert(type(seqs[0]) in [list, tuple])
        
        if unravel:
            seqs = [sent for seq in seqs for sent in seq]
        
        return seqs
    
    def lookup_eos(self, eos_markers=[".", "?", "!"]):
        #get indices of end of sentence markers (needed for generating with language model)
        eos_idxs = [self.lexicon[marker] for marker in eos_markers if marker in self.lexicon]
        return eos_idxs
        
    def fit_transform(self, X, y_seqs=None, **fit_params):
        #import pdb;pdb.set_trace()
        rnn_params = {}
        #if lexicon given, model is already fit
        if not self.lexicon:     
            #generate lexicon if not given
            self.fit(X, y_seqs)
        rnn_params['lexicon_size'] =  self.lexicon_size
        rnn_params['max_length'] = self.max_length
        if self.embeddings is not None:
            rnn_params['embedded_input'] = True
        #import pdb;pdb.set_trace()
        X, y_seqs = self.transform(X, y_seqs)
        return X, y_seqs, rnn_params
    
    def text_to_nums(self, seqs):
        #import pdb;pdb.set_trace()
        encoded_seqs = []
        for seq in seqs:
            encoded_seq = segment_and_tokenize(seq)
            if self.replace_ents:
                #import pdb;pdb.set_trace()
                #map recognized named entities to entity labels (e.g. "PERSON")
                ents = get_entities(seq)
                if ents:
                    encoded_seq = replace_entities(encoded_seq, ents)
            encoded_seq = [self.lexicon[word] if word in self.lexicon else 1
                   for word in encoded_seq]
            encoded_seqs.append(encoded_seq)
        assert(len(seqs) == len(encoded_seqs))
        return encoded_seqs
    
    def decode_seqs(self, seqs):
        if type(seqs[0]) not in (list, numpy.ndarray, tuple):
            seqs = [seqs]
        decoded_seqs = []
        #transform numerical seq back intro string
        for seq in seqs:
            seq = [self.lexicon_lookup[word] if self.lexicon_lookup[word] else "None" for word in seq]
            seq = " ".join(seq)
            decoded_seqs.append(seq)
        if len(decoded_seqs) == 1:
            decoded_seqs = decoded_seqs[0]
        return decoded_seqs
            
    def embed_seqs(self, seqs):
        #import pdb;pdb.set_trace()
        #convert word indices to vectors
        embedded_seqs = []
        for seq in seqs:
            #convert to vectors rather than indices - if word not in lexicon represent with all zeros
            seq = [self.embeddings[self.lexicon_lookup[word]]
                   if self.lexicon_lookup[word] in self.embeddings.vocab
                    else numpy.zeros((self.embeddings.vector_size,))
                   for word in seq]
            seq = numpy.array(seq)
            embedded_seqs.append(seq)
        return embedded_seqs
    
    def reduce_embs(self, seqs):
        #import pdb;pdb.set_trace()
        #combine embeddings of each sequence by averaging or summing them
        if self.reduce_emb_mode == 'mean':
            #only average non-zero embeddings
#             seqs = [[sent[sent.sum(axis=1) != 0] for sent in seq] for seq in seqs]
#             seqs = [[sent if sent.size else numpy.zeros((1, self.n_embedding_nodes)) 
#                      for sent in seq] for seq in seqs]
            seqs = numpy.array([[numpy.mean(sent, axis=0) for sent in seq] for seq in seqs])
        elif self.reduce_emb_mode == 'sum':
            seqs = numpy.array([[numpy.sum(sent, axis=0) for sent in seq] for seq in seqs])
        return seqs
    
    def set_max_length(self, seqs):
        self.max_length = max([len(segment_and_tokenize(seq)) for seq in seqs])
        if self.verbose:
            print self.max_length, "words in longest sequence"
            
    def pad_nums(self, seqs):
        #import pdb;pdb.set_trace()
        seqs = [pad_sequences(sequences=seq, maxlen=self.max_length, padding='post')
               for seq in seqs]
                
        seqs = numpy.array(seqs)
        return seqs
    
    def pad_embeddings(self, seqs):
        #import pdb;pdb.set_trace()
        assert(type(seqs[0]) is list and len(seqs[0][0].shape) == 2)
        #input sequences are a list of sentences
        seqs = [numpy.array([numpy.append(sent, numpy.zeros((self.max_length - len(sent), 
                self.embeddings.vector_size)), axis=0)
                for sent in seq]) for seq in seqs]
        
        seqs = numpy.array(seqs)
        return seqs
    
    def remove_extra_dim(self, seqs):
        #if seqs have an extra dimension of one, flatten it
        if len(seqs[0]) == 1:
            if type(seqs) is numpy.ndarray:
                seqs = seqs[:, 0]
            else:
                seqs = [sent for seq in seqs for sent in seq]
        return seqs
        
    def transform(self, X, y_seqs=None):
        #import pdb;pdb.set_trace()
        if y_seqs is not None:
            y_seqs = self.format_seqs(seqs=y_seqs)
            y_seqs = [self.text_to_nums(seqs=seqs) for seqs in y_seqs]
            if self.embeddings and self.embed_y:
                y_seqs = [self.embed_seqs(seqs=seqs) for seqs in y_seqs]
                if self.reduce_emb_mode:
                    y_seqs = self.reduce_embs(y_seqs)
                if self.pad_seq:
                    y_seqs = self.pad_embeddings(seqs=y_seqs)
            else:
                if self.pad_seq:
                    y_seqs = self.pad_nums(seqs=y_seqs)
            assert(len(X) == len(y_seqs))
            y_seqs = self.remove_extra_dim(seqs=y_seqs)

        X = self.format_seqs(seqs=X)
        X = [self.text_to_nums(seqs=seqs) for seqs in X]
        if self.copy_input_to_output:
            assert(y_seqs is None)
            y_seqs = X
            if self.pad_seq:
                y_seqs = self.pad_nums(seqs=y_seqs)
            y_seqs = self.remove_extra_dim(seqs=y_seqs)
        if self.embeddings:
            X = [self.embed_seqs(seqs=seqs) for seqs in X]
            #combine embeddings if specified
            if self.reduce_emb_mode:
                X = self.reduce_embs(seqs=X)
            if self.pad_seq:
                #import pdb;pdb.set_trace()
                X = self.pad_embeddings(seqs=X)
        else:
            if self.pad_seq:
                X = self.pad_nums(seqs=X)
        X = self.remove_extra_dim(seqs=X)
                 
        return X, y_seqs
    
    def save(self):
        #save transformer to file
        if not os.path.isdir(self.filepath):
            os.mkdir(self.filepath)

        pickle.dump(self, open(self.filepath + '/transformer.pkl', 'wb'))
        
    def __getstate__(self):
        #don't save embeddings
        return dict((k, v) for (k, v) in self.__dict__.iteritems() if k != 'embeddings')
    


# In[50]:

class RNNLM(KerasClassifier, SavedModel):
    def __call__(self, lexicon_size, n_timesteps=None, n_embedding_nodes=300, n_hidden_nodes=250, n_hidden_layers=1,
                 embeddings=None, batch_size=1, max_length=None, verbose=1, filepath=None,
                 optimizer='Adam', lr=0.001, clipvalue=5.0, decay=1e-6):
        
        self.lexicon_size = lexicon_size
        self.n_embedding_nodes = n_embedding_nodes
        self.n_hidden_nodes = n_hidden_nodes
        self.n_hidden_layers = n_hidden_layers
        self.batch_size = batch_size
        self.max_length = max_length
        self.verbose = verbose
        self.n_timesteps = n_timesteps
        self.filepath = filepath
        self.embeddings = embeddings
        self.optimizer = optimizer
        self.lr = lr
        self.clipvalue = clipvalue
        self.decay = decay
        
        #create model
        model = self.create_model(self.n_timesteps, self.batch_size)

        return model
    
    def create_model(self, n_timesteps=20, batch_size=1):
        
        model = Sequential()
        
        if self.embeddings is None:
            model.add(Embedding(self.lexicon_size + 1, self.n_embedding_nodes,
                                batch_input_shape=(self.batch_size, self.n_timesteps), mask_zero=True))
        for layer_num in xrange(self.n_hidden_layers):
            model.add(GRU(self.n_hidden_nodes, 
                          batch_input_shape=(self.batch_size, self.n_timesteps, self.n_embedding_nodes),
                          return_sequences=True, stateful=True))
        
        model.add(TimeDistributed(Dense(self.lexicon_size + 1, activation="softmax")))
        
        #select optimizer and compile
        model.compile(loss="sparse_categorical_crossentropy", 
                      optimizer=eval(self.optimizer)(clipvalue=self.clipvalue, lr=self.lr, decay=self.decay))
        if self.verbose:
            print "CREATED RNNLM: embedding layer nodes = {}, hidden layers = {}, "                     "hidden layer nodes = {}, optimizer = {} with lr = {}, "                     "clipvalue = {}, and decay = {}".format(
                    self.n_embedding_nodes, self.n_hidden_layers, self.n_hidden_nodes, 
                    self.optimizer, self.clipvalue, self.lr, self.decay)
                
        return model
    
    def prep_batch(self, batch_x, batch_y):#, step_index):
        assert(len(batch_y) == len(batch_y))
        #filter timesteps that are empty for all sequences in batch
#         if batch_y is not None:
#             batch_len = numpy.sum(numpy.sum(batch_y, axis=0) > 0)
#             batch_y = batch_y[:, :batch_len]
#         else:
#             batch_len = numpy.sum(numpy.sum(batch_x, axis=0) > 0)
#         batch_x = batch_x[:, :batch_len]
        if len(batch_x) < self.batch_size:
            #too few sequences for batch, so add extra rows
            batch_padding = numpy.zeros((self.batch_size - len(batch_x), batch_x.shape[1]))
            if batch_y is not None:
                batch_y = numpy.append(batch_y, batch_padding, axis=0)
            if self.embeddings is not None:
                #adjust X padding for embeddings
                batch_padding = numpy.zeros((self.batch_size - len(batch_x),
                                             batch_x.shape[1], self.n_embedding_nodes))
            batch_x = numpy.append(batch_x, batch_padding, axis=0)
        
#         if batch_x.shape[1] - step_index <= self.n_timesteps:
#             #pad timesteps
#             #import pdb;pdb.set_trace()
#             step_padding = numpy.zeros((batch_x.shape[0],
#                                         self.n_timesteps - (batch_x.shape[1] - step_index) + 1))# + 1))
#             if batch_y is not None:
#                 batch_y = numpy.append(batch_y, step_padding, axis=1)
#             if self.embedded_input:
#                 step_padding = numpy.zeros((batch_x.shape[0],
#                                             self.n_timesteps - (batch_x.shape[1] - step_index) + 1,# + 1,
#                                             self.n_embedding_nodes))
#             batch_x = numpy.append(batch_x, step_padding, axis=1)
        
        if batch_y is not None:
            batch_y = batch_y[:, 1:]#[:, step_index:step_index + self.n_timesteps]
        else:
            batch_y = batch_x[:, 1:]#[:, step_index: step_index + self.n_timesteps]
        batch_x = batch_x[:, :-1]#[:, step_index:step_index + self.n_timesteps]
        
        #keras is weird and makes you add 1 dimension to output
        batch_y = numpy.expand_dims(batch_y, -1)
        assert(batch_x.size > 0 and batch_y.size > 0)

        return batch_x, batch_y
    
    def sort_seqs(self, X, y):
        #sort by descending length
        if self.embeddings is not None:
            sorted_idxs = numpy.argsort((y > 0).sum(axis=-1))[::-1]
        else:
            sorted_idxs = numpy.argsort((X > 0).sum(axis=-1))[::-1]
        X = X[sorted_idxs]
        if y is not None:
            y = y[sorted_idxs]
        return X, y
    
    def prepend_eos(self, seq):
        seq = [[seq[sent_idx - 1][-1]] + sent if sent_idx > 0 else sent
                       for sent_idx, sent in enumerate(seq)]
        return seq
    
    def sample_words(self, p_next_word, n_samples):
        next_words = []
        while len(next_words) < n_samples:
            next_word = numpy.random.choice(a=p_next_word.shape[-1], p=p_next_word)
            if next_word not in next_words:
                next_words.append(next_word)
        return numpy.array(next_words)

    def fit_epoch(self, X, y, rnn_params, **kwargs):
        
        if not hasattr(self, 'model'):
            self.sk_params.update(rnn_params)
            #infer if input is embedded
            #import pdb;pdb.set_trace()
            if 'embeddings' in self.sk_params and self.sk_params['embeddings'] is not None:
                assert(y is not None)
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
            self.start_time = timeit.default_timer()
            self.epoch = 0
            if self.verbose:
                print("training RNNLM on {} sequences with batch size = {}".format(len(X), self.batch_size))
        
        train_losses = []

        if self.batch_size == 1:
            #process sequences one at a time, one sentence at a time
            for seq_idx, seq in enumerate(X):
                if type(seq[0]) not in (list, numpy.ndarray, tuple):
                    seq = [seq]
                seq = self.prepend_eos(seq)
                if y is not None:
                    y_seq = y[seq_idx]
                for sent_idx, sent in enumerate(seq):
                    #import pdb;pdb.set_trace()
                    sent_x = numpy.array(sent)
                    if y is not None:
                        sent_y = numpy.array(y_seq[sent_idx])
                    else:
                        sent_y = sent_x
                    sent_x = sent_x[None, :-1]
                    sent_y = sent_y[None, 1:]
                    sent_y = numpy.expand_dims(sent_y, axis=-1)
                    assert(sent_x.size > 0 and sent_y.size > 0)
                    assert(len(sent_x) == len(sent_y))
                    train_loss = self.model.train_on_batch(x=sent_x, y=sent_y)
                    train_losses.append(train_loss)
                self.model.reset_states()
                if (seq_idx + 1) % 1000 == 0:
                    print("processed {}/{} sequences in epoch {}, loss = {:.3f} ({:.3f}m)...".format(seq_idx + 1, 
                        len(X), self.epoch + 1, numpy.mean(train_losses), 
                        (timeit.default_timer() - self.start_time) / 60))
            
#             else:
#                 #ensure input is padded (i.e. numpy array instead of list)
#                 assert(type(X) is numpy.ndarray)
#                 #sort sequences by length
#                 #X, y = self.sort_seqs(X, y)
#                 for batch_index in range(0, len(X), self.batch_size):
#                     for sent_index in range(X.shape[1]):
# #                         for step_index in xrange(0, X.shape[1] - 1, self.n_timesteps):
#                         batch_x = numpy.array(X[batch_index:batch_index + self.batch_size, sent_index])
#                         if y is not None:
#                             batch_y = numpy.array(y[batch_index:batch_index + self.batch_size, sent_index])
#                         else:
#                             batch_y = None
#                         batch_x, batch_y = self.prep_batch(batch_x, batch_y)#, step_index)
#                         train_loss = self.model.train_on_batch(x=batch_x, y=batch_y)
#                         train_losses.append(train_loss)
#                     self.model.reset_states()                    
#                     if batch_index and batch_index % 1000 == 0:
#                         print "completed", batch_index, "/ batches in epoch", epoch + 1, "..."
            if self.filepath:
                #save model after each epoch if filepath given
                self.save()
            self.epoch += 1
            if self.verbose:
                print("epoch {} loss: {:.3f} ({:.3f}m)".format(self.epoch, numpy.mean(train_losses),
                                           (timeit.default_timer() - self.start_time) / 60))
    
    def pred_next_words(self, context_seq, sent, mode='max', n_best=1, temp=1.0):
        #use grid search to predict next word given current best predicted sequences
        #import pdb;pdb.set_trace()
        assert(mode == 'max' or mode == 'random')
        if sent:
            seq = context_seq + [sent]
        else:
            seq = context_seq

        for sent_idx, sent in enumerate(seq):
            p_next_word = self.pred_model.predict_on_batch(x=numpy.array(sent)[None, :])[0][-1]
        assert(len(p_next_word.shape) == 1)
        
        if mode == 'random':
            #import pdb;pdb.set_trace()
            p_next_word = numpy.log(p_next_word) / temp
            p_next_word = numpy.exp(p_next_word) / numpy.sum(numpy.exp(p_next_word))
            next_words = self.sample_words(p_next_word, n_best)
        else:
            next_words = numpy.argsort(p_next_word)[::-1][:n_best]
        self.model.reset_states()
        return next_words

    def extend_sent(self, sent, words):
        #extend sequence with each predicted word
        new_sents = []
        for word in words:
            new_sent = sent + [word]
            new_sents.append(new_sent)
        return new_sents
    
    def embed_sent(self, sent):
        embedded_sent = []
        for word in sent:
            #convert last predicted word to embedding
            if self.lexicon_lookup[word] in self.embeddings:
                #next_word = embeddings[lexicon_lookup[next_word]]
                embedded_sent.append(self.embeddings[self.lexicon_lookup[word]])
            else:
                embedded_sent.append(numpy.zeros((self.n_embedding_nodes)))
        return embedded_sent
    
    def pred_sents(self, context_seq, sents, mode, n_best, temp, eos_markers):
        #import pdb;pdb.set_trace()
        new_sents = []
        if not sents:
            sents = [[]]
        for sent in sents:
            if sent:
                if self.check_if_eos(sent, eos_markers):#, lexicon_lookup, )
                    #reached end of sentence marker in generated sentence, so stop generating
                    new_sents.append(sent)
                    continue
            if self.embeddings is not None:
                embedded_sent = self.embed_sent(sent)#, embeddings, lexicon_lookup)
                next_words = self.pred_next_words(context_seq, embedded_sent, mode, n_best, temp)
            else:
                next_words = self.pred_next_words(context_seq, sent, mode, n_best, temp)
            ext_sents = self.extend_sent(sent=sent, words=next_words)
            new_sents.extend(ext_sents)
        #import pdb;pdb.set_trace()
        new_sents, p_sents = self.get_best_sents(context_seq, new_sents, n_best)
        return new_sents, p_sents
    
    def check_if_eos(self, sent, eos_markers):
        #check if an end-of-sentence marker has been generated for this sentence
        if sent[-1] == 0 or sent[-1] in eos_markers:
            return True
        return False
    
    def get_best_sents(self, context_seq, sents, n_best):#, embeddings, lexicon_lookup):
        
        p_sents = []
        
        for sent in sents:      
            #read context to get representation
            p_lst_cont_sent = [self.pred_model.predict_on_batch(x=numpy.array(seq)[None, :])[:, -1]
                               for seq in context_seq][-1]
            if self.embeddings is not None:
                embedded_sent = self.embed_sent(sent)#, embeddings, lexicon_lookup)
                p_sent = self.pred_model.predict_on_batch(x=numpy.array(embedded_sent)[None, :])[0][:-1]
            else:
                p_sent = self.pred_model.predict_on_batch(x=numpy.array(sent)[None, :])[0][:-1]
            p_sent = numpy.append(p_lst_cont_sent, p_sent, axis=0)
            p_sent = p_sent[numpy.arange(len(sent)), sent]#[-1]
            p_sent = numpy.sum(numpy.log(p_sent))
            p_sents.append(p_sent)
            self.model.reset_states()
        
        best_idxs = numpy.argsort(numpy.array(p_sents))[::-1][:n_best]
        best_sents = [sents[idx] for idx in best_idxs]
        #return probs of best sents as well as sents
        p_sents = numpy.array(p_sents)[best_idxs]
        
        return best_sents, p_sents
        
        
    def predict(self, X, y=None, n_words=20, mode='max', n_best=1, temp=1.0, eos_markers=[], **kwargs):
        #generate a new word in a given sequence

        if not hasattr(self, 'pred_model'):
            if self.batch_size > 1:
                #if model uses batch training, create a duplicate model with batch size 1 for prediction
                self.pred_model = self.create_model(lexicon_size=self.lexicon_size,
                                                    n_embedding_nodes=self.n_embedding_nodes, 
                                                    n_hidden_nodes=self.n_hidden_nodes,
                                                    n_hidden_layers=self.n_hidden_layers, 
                                                    batch_size=1)
                #set weights of prediction model
                self.pred_model.set_weights(self.model.get_weights())
            else:
                self.pred_model = self.model
                
        pred_sents = []
        p_pred_sents = []
        for seq in X:
            if type(seq[0]) not in [list, tuple, numpy.ndarray]:
                seq = [seq]
                
            sents = None
            for idx in range(n_words):
                sents, p_sents = self.pred_sents(seq, sents, mode, n_best, temp, eos_markers)
                if numpy.all([self.check_if_eos(sent, eos_markers) for sent in sents]):
                    #all generated sentences have end-of-sentence markers
                    break
                    
            if len(sents) == 1:
                sents = sents[0]
                p_sents = p_sents[0]
            pred_sents.append(sents)
            p_pred_sents.append(p_sents)
        return pred_sents, p_pred_sents


class RNNLMClassifier(KerasClassifier):
    def __call__(self, lexicon_size, init_weights=None, max_length=None):
        lm = RNNLM(lexicon_size=lexicon_size, n_timesteps=max_length)
        if init_weights:
            lm.model.set_weights(init_weights)
        return lm
    def fit(self, X, y, **kwargs):
        #sklearn pipeline won't transform y; so x is two arrays, with y as second
        #import pdb;pdb.set_trace()
        #get number of input nodes
        #self.sk_params.update({"lexicon_size": max(max(X[0]))})
        self.model = self.__call__(**self.filter_sk_params(self.__call__))
        n_epochs = kwargs["n_epochs"]
        self.model.train(seqs=X, n_epochs=n_epochs)
    def set_outcomes(self, outcome_set):
#         embed_dim = self.model.layers[0].input_dim
#         self.outcome_set = [[word if word < embed_dim else 1 for word in sequence] for outcome in outcome_set]
        self.outcome_set = outcome_set
    def predict(self, X, **kwargs):
        #import pdb; pdb.set_trace()
        assert(self.outcome_set is not None)
        #if X has any word indexes greater than the number of embedding dimensions, treat them as unknown words (1)
        pred_outcomes = []
        for x in X:
            probs = []
            for outcome in self.outcome_set:
                outcome_prob = self.model.model.predict_proba(x=numpy.array(x + outcome)[None, :-1], verbose=0)[0]
                outcome_prob = outcome_prob[numpy.arange(len(outcome_prob)), numpy.array(x + outcome)[1:]]
                outcome_prob = numpy.sum(numpy.log(outcome_prob))
                probs.append(outcome_prob)
            pred_outcome = numpy.argmax(numpy.array(probs))
            pred_outcomes.append(pred_outcome)
        return pred_outcomes
        


# In[359]:

class RNNPipeline(Pipeline):
    #sklearn pipeline won't pass extra parameters other than input data between steps
    def _pre_transform(self, X, y_seqs=None, **fit_params):
        fit_params_steps = dict((step, {}) for step, _ in self.steps)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Xt = X
        for name, transform in self.steps[:-1]:
            if hasattr(transform, "fit_transform"):
                Xt, y_seqs, rnn_params = transform.fit_transform(Xt, y_seqs, **fit_params_steps[name])
            else:
                Xt, y_seqs, rnn_params = transform.fit(Xt, y_seqs, **fit_params_steps[name]).transform(Xt, y_seqs)
        return Xt, y_seqs, rnn_params, fit_params_steps[self.steps[-1][0]]
    def fit(self, X, y=None, y_seqs=None, **fit_params):
        if self.steps[-1][-1].__class__.__name__ == 'RNNLM' and            self.steps[0][-1].__class__.__name__ == 'SequenceTransformer' and            self.steps[0][-1].get_params()['embeddings'] is not None:
            #import pdb;pdb.set_trace()
            #if this is a language model, no explicit output; input will be copied to output before embedding,
            #if input is embedded
            self.steps[0][-1].set_params(copy_input_to_output = True)
        if self.steps[-1][-1].__class__.__name__ in ['Seq2SeqClassifier', 'MergeSeqClassifier',
                                                      'RNNLMClassifier', 'RNNLM']:
            Xt, y, rnn_params, fit_params = self._pre_transform(X, y_seqs, **fit_params)

        else:
            Xt, _, rnn_params, fit_params = self._pre_transform(X, y_seqs, **fit_params)
        if self.steps[-1][-1].__class__.__name__ == 'RNNLM':
            self.steps[-1][-1].fit_epoch(Xt, y, rnn_params, **fit_params)
        else:
            self.steps[-1][-1].fit(Xt, y, rnn_params, **fit_params)
        return self
    def predict(self, X, y_choices=None, **kwargs):
        #if choice sequences given, predict sequence from this list
        #import pdb;pdb.set_trace()
        Xt = X
        for name, transform in self.steps[:-1]:
            Xt, y_choices = transform.transform(Xt, y_choices)
        if y_choices is not None:
            return self.steps[-1][-1].predict(Xt, y_choices, **kwargs)
        else:
            return self.steps[-1][-1].predict(Xt, **kwargs)
        


# In[322]:

class RNNClassifier(KerasClassifier):
    def __call__(self, lexicon_size, n_outcomes, input_to_outcome_set=None, 
                 max_length=None, emb_weights=None, layer1_weights=None):
        model = Sequential()
        model.add(Embedding(output_dim=100, input_dim=lexicon_size + 1,
                            input_length=max_length, mask_zero=True, name='embedding'))
        #model.add(GRU(output_dim=200, return_sequences=True, name='recurrent1'))
        model.add(GRU(output_dim=200, return_sequences=False, name='recurrent2'))
        model.add(Dense(output_dim=n_outcomes, activation='softmax', name='output'))
        if emb_weights is not None:
            #initialize weights with lm weights
            model.layers[0].set_weights(emb_weights) #set embeddings
        if layer1_weights is not None:
            model.layers[1].set_weights(layer1_weights) #set recurrent layer 1         
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    def fit(self, X, y, rnn_params, **kwargs):
        #import pdb;pdb.set_trace()
        self.sk_params.update(rnn_params)
        super(RNNClassifier, self).fit(X, y, **kwargs)
    def predict(self, X, y_choices=None, **kwargs):
        #import pdb;pdb.set_trace()
        if "input_to_outcome_set" in self.sk_params and self.sk_params["input_to_outcome_set"] is not None:
            input_to_outcome_set = self.sk_params["input_to_outcome_set"]
            #predict from specific outcome set for each input
            pred_y = []
            prob_y = self.model.predict(X, **kwargs)
            for prob_y_, outcome_choices in zip(prob_y, input_to_outcome_set):
                prob_y_ = prob_y_[outcome_choices]
                pred_y_ = outcome_choices[numpy.argmax(prob_y_)]
                pred_y.append(pred_y_)
            return pred_y
        else:
            return super(RNNClassifier, self).predict(X, **kwargs)
        

class MLPClassifier(KerasClassifier):
    def __call__(self, lexicon_size, n_outcomes, input_to_outcome_set=None):
        
        self.n_outcomes = n_outcomes
        self.lexicon_size = lexicon_size
        
        model = Sequential()
        model.add(Dense(output_dim=200, input_dim=self.lexicon_size, activation='tanh', name='hidden1'))
        #model.add(Dense(output_dim=200, input_dim=lexicon_size, activation='tanh', name='hidden2'))
        model.add(Dense(output_dim=n_outcomes, activation='softmax', name='output'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])
        return model
    def fit(self, X, y, **kwargs):
        #get number of input nodes
        self.sk_params.update(lexicon_size = X.shape[1])
        #keras doesn't handle sparse matrices
        X = X.toarray()
        super(MLPClassifier, self).fit(X, y, **kwargs)
    def predict(self, X, **kwargs):
        #keras doesn't handle sparse matrices  
        X = X.toarray()
    
        if "input_to_outcome_set" in self.sk_params and self.sk_params["input_to_outcome_set"] is not None:
            #import pdb; pdb.set_trace()
            input_to_outcome_set = self.sk_params["input_to_outcome_set"]
            #predict from specific outcome set for each input
            pred_y = []
            prob_y = self.model.predict(X, **kwargs)
            for prob_y_, outcome_choices in zip(prob_y, input_to_outcome_set):
                prob_y_ = prob_y_[outcome_choices]
                pred_y_ = outcome_choices[numpy.argmax(prob_y_)]
                pred_y.append(pred_y_)
            return pred_y

        else:
            return super(MLPClassifier, self).predict(X, **kwargs)
    
class Seq2SeqClassifier(KerasClassifier):
    def __call__(self, lexicon_size, max_length, batch_size=None, stateful=False,
                 n_encoding_layers=1, n_decoding_layers=1, 
                 n_embedding_nodes=100, n_hidden_nodes=250, verbose=1, embedded_input=False):
        
        self.stateful = stateful
        self.embedded_input = embedded_input
        self.batch_size = batch_size
        self.max_length = max_length
        self.lexicon_size = lexicon_size
        self.n_embedding_nodes = n_embedding_nodes
        self.n_encoding_layers = n_encoding_layers
        self.n_decoding_layers = n_decoding_layers
        self.n_hidden_nodes = n_hidden_nodes
        self.verbose = verbose
        
        #import pdb;pdb.set_trace()
        model = Sequential()
        
        if not self.embedded_input:
            embedding = Embedding(batch_input_shape=(self.batch_size, self.max_length), 
                                  input_dim=self.lexicon_size + 1,
                                  output_dim=self.n_embedding_nodes, 
                                  mask_zero=True, name='embedding')
            model.add(embedding)

        encoded_input = GRU(batch_input_shape=(self.batch_size, self.max_length, self.n_embedding_nodes),
                            input_length = self.max_length,
                            input_dim = self.n_embedding_nodes,
                            output_dim=self.n_hidden_nodes, return_sequences=False, 
                            name='encoded_input1', stateful=self.stateful)
        model.add(encoded_input)

        repeat_layer = RepeatVector(self.max_length, name="repeat_layer")
        model.add(repeat_layer)
        
        encoded_outcome = GRU(self.n_hidden_nodes, return_sequences=True, name='encoded_outcome1',
                              stateful=self.stateful)#(repeat_layer)
        model.add(encoded_outcome)

        outcome_seq = TimeDistributed(Dense(output_dim=self.lexicon_size + 1, activation='softmax', 
                            name='outcome_seq'))#(encoded_outcome)
        model.add(outcome_seq)

        optimizer = "rmsprop"
        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        
        if self.verbose:
            print "CREATED Sequence2Sequence model: embedding layer sizes:", self.n_embedding_nodes, ",",            self.n_encoding_layers, "encoding layers with size:", self.n_hidden_nodes, ",",            self.n_decoding_layers, "decoding layers with size:", self.n_hidden_nodes, ",",            "optimizer:", optimizer, ", batch_size:", self.batch_size, ", stateful =", self.stateful
        return model
    
    def fit(self, X, y, rnn_params=None, **kwargs):
        #import pdb;pdb.set_trace()
        #y are sequences rather than classes here
        self.sk_params.update(rnn_params)
        if "embedded_input" in self.sk_params and self.sk_params["embedded_input"]:
            max_length = X.shape[-2]
        else:
            max_length = X.shape[-1]
        self.sk_params.update({"max_length": max_length})
        #import pdb;pdb.set_trace()
        patience = 2
        n_lossless_iters = 0
        if "stateful" in self.sk_params and self.sk_params["stateful"]:
            #import pdb;pdb.set_trace()
            #carry over state between batches
            assert(len(X.shape) >= 3)
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
            nb_epoch = self.sk_params["nb_epoch"]
            n_batches = int(numpy.ceil(len(X) * 1. / self.batch_size))
            #import pdb;pdb.set_trace()
            if self.verbose:
                print "training stateful Seq2Seq on", len(X), "sequences for", nb_epoch, "epochs..."
                print n_batches, "batches with", self.batch_size, "sequences per batch"
            start_time = timeit.default_timer()
            min_loss = numpy.inf
            for epoch in range(nb_epoch):
                train_losses = []
                for batch_index in range(0, len(X), self.batch_size):
                    batch_num = batch_index / self.batch_size + 1
                    for sent_index in range(X.shape[1]):
                        batch_X = X[batch_index:batch_index + self.batch_size, sent_index]
                        batch_y = y[batch_index:batch_index + self.batch_size, sent_index]
                        assert(len(batch_X) == len(batch_y))
                        if len(batch_X) < self.batch_size:
                            #too few sequences for batch, so add extra rows
                            batch_X = numpy.append(batch_X, numpy.zeros((self.batch_size - len(batch_X),) 
                                                                    + batch_X.shape[1:]), axis=0)
                            batch_y = numpy.append(batch_y, numpy.zeros((self.batch_size - len(batch_y),
                                                                     batch_y.shape[-1])), axis=0)
                        train_loss = self.model.train_on_batch(x=batch_X, y=batch_y[:, :, None])
                        train_losses.append(train_loss)
                    if batch_num % 100 == 0:
                        print "completed", batch_num, "/", n_batches, "batches in epoch", epoch + 1, "..."
                            
                    self.model.reset_states()
                
                if self.verbose:
                    print("epoch {}/{} loss: {:.3f} ({:.3f}m)".format(epoch + 1, nb_epoch, 
                                                                      numpy.mean(train_losses),
                                                                      (timeit.default_timer() - start_time) / 60))
                if numpy.mean(train_losses) < min_loss:
                    n_lossless_iters = 0  
                    min_loss = numpy.mean(train_losses)
                else:
                    n_lossless_iters += 1
                    if n_lossless_iters == patience:
                        #loss hasn't decreased after waiting number of patience iterations, so stop
                        print "stopping early"
                        break
                     
        else:
            #import pdb;pdb.set_trace()
            #assert(len(X.shape) == 2)
            #regular fit function works for non-stateful models
            early_stop = EarlyStopping(monitor='train_loss', patience=patience, verbose=0, mode='auto')
            super(Seq2SeqClassifier, self).fit(X, y=y[:, :, None], callbacks=[early_stop], **kwargs)
        
    def predict(self, X, y_choices, **kwargs):
        #check if y_choices is single set or if there are different choices for each input
        y_choices = check_y_choices(X, y_choices)
        
        if self.verbose:
            print "predicting outputs for", len(X), "sequences..."
            
        if self.stateful:
            #iterate through sentences as input-output
            #import pdb;pdb.set_trace()
            probs_y = []
            for batch_index in range(0, len(X), self.batch_size):
                for sent_index in range(X.shape[1]):
                    batch_X = X[batch_index:batch_index + self.batch_size, sent_index]
                    if len(batch_X) < self.batch_size:
                        #too few sequences for batch, so add extra rows
                        #import pdb;pdb.set_trace()
                        batch_X = numpy.append(batch_X, numpy.zeros((self.batch_size - len(batch_X),) 
                                                                    + batch_X.shape[1:]), axis=0)
                        assert(len(batch_X) == self.batch_size)
                    probs_next_sent = self.model.predict_on_batch(batch_X)
                #then reduce batch again if it has empty rows 
                if len(X) - batch_index < self.batch_size:
                    #import pdb;pdb.set_trace()
                    batch_X = batch_X[:len(X) - batch_index]
                batch_choices = y_choices[batch_index:batch_index + self.batch_size]
                assert(len(batch_X) == len(batch_choices))
                batch_probs_y = []
                for choice_index in range(batch_choices.shape[1]):
                    #import pdb;pdb.set_trace()
                    #evaluate each choice based on predicted probabilites from most recent sentence
                    batch_choice = batch_choices[:, choice_index]
                    probs_choice = probs_next_sent[numpy.arange(len(batch_choice))[:, None],
                                            numpy.arange(batch_choice.shape[-1]), batch_choice]
                    #have to iterate through instances because each is different length
                    probs_choice = [prob_choice[choice > 0][-1] for choice, prob_choice in 
                                                                     zip(batch_choice, probs_choice)]
                    batch_probs_y.append(probs_choice)
                batch_probs_y = numpy.stack(batch_probs_y, axis=1)
                probs_y.append(batch_probs_y)                
                self.model.reset_states()
            probs_y = numpy.concatenate(probs_y)
            #import pdb;pdb.set_trace()
        
        else:
            probs_y = []
            #import pdb;pdb.set_trace()
            for x, choices in zip(X, y_choices):       
                probs = []
                for choice in choices:
                    prob = self.model.predict(x[None, :], **kwargs)
                    prob = prob[:, numpy.arange(len(choice)), choice]
                    prob = prob[0, choice > 0][-1]
                    #prob = numpy.sum(numpy.log(prob))
                    #prob = numpy.sum(numpy.log(prob))
                    probs.append(prob)
                probs_y.append(numpy.array(probs))
            probs_y = numpy.array(probs_y)
        
        assert(len(probs_y) == len(X))
        #return prob for each choice for each input
        return probs_y
        
        
class MergeSeqClassifier(KerasClassifier):
    def __call__(self, lexicon_size, outcome_set, max_length, n_encoding_layers=1, n_decoding_layers=1, 
                 n_embedding_nodes=100, n_hidden_nodes=200, batch_size=1, verbose=1):
        
        self.lexicon_size = lexicon_size
        self.max_length = max_length
        self.outcome_set = outcome_set
        self.n_embedding_nodes = n_embedding_nodes
        self.n_encoding_layers = n_encoding_layers
        self.n_decoding_layers = n_decoding_layers
        self.n_hidden_nodes = n_hidden_nodes
        self.batch_size = batch_size
        self.verbose = verbose

        #create model
        user_input = Input(shape=(self.max_length,), dtype='int32', name="user_input")
        embedding = Embedding(input_length=self.max_length,
                              input_dim=self.lexicon_size + 1, 
                              output_dim=self.n_embedding_nodes, mask_zero=True, name='input_embedding')
        embedded_input = embedding(user_input)
        encoded_input = GRU(self.n_hidden_nodes, return_sequences=False, name='encoded_input1')(embedded_input)
        
        outcome_seq = Input(shape=(self.max_length,), dtype='int32', name="outcome_seq")
        embedded_outcome = embedding(outcome_seq)
        encoded_outcome = GRU(self.n_hidden_nodes, return_sequences=False, name='encoded_outcome1')(embedded_outcome)
        input_outcome_seq = merge([encoded_input, encoded_outcome], mode='concat', 
                                  concat_axis=-1, name='input_outcome_seq')     
        outcome = Dense(output_dim=len(self.outcome_set), activation='softmax', name='outcome')(input_outcome_seq)
        model = Model(input=[user_input, outcome_seq], output=[outcome])
        model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=["accuracy"])      
        if self.verbose:
            print "CREATED MergeSequence model: embedding layer sizes =", self.n_embedding_nodes, ",",            self.n_encoding_layers, "encoding layers with size", self.n_hidden_nodes, ",",            self.n_decoding_layers, "decoding layers with size", self.n_hidden_nodes, ",",            "lexicon size = ", self.lexicon_size, ",",            "batch size = ", self.batch_size
        return model
    def fit(self, X, y, **kwargs):
        #import pdb;pdb.set_trace()
        if not self.sk_params["outcome_set"]:
            assert("outcome_set" in kwargs)
            self.sk_params.update(kwargs)
            
        #outcome_set[y] are outcome sequences
        super(MergeSeqClassifier, self).fit([X, self.sk_params['outcome_set'][y]], y) #**kwargs)
        #hidden = self.get_sequence.predict(x=[X, numpy.repeat(self.outcome_set[0][None, :], len(X), axis=0)])
        
    def predict(self, X, **kwargs):
        #import pdb;pdb.set_trace()
        
        max_probs = []
        for outcome, outcome_seq in enumerate(self.outcome_set):
            #get probs for this outcome
            probs = self.model.predict(x=[X, numpy.repeat(outcome_seq[None, :], len(X), axis=0)])[:, outcome]
            max_probs.append(probs)
        y = numpy.argmax(numpy.stack(max_probs, axis=1), axis=1)
        return y
    


# In[53]:

class SeqBinaryClassifier(KerasClassifier, SavedModel):
    def __call__(self, lexicon_size, context_size, max_length=None, n_embedding_nodes=300, batch_size=None, 
                 n_hidden_layers=1, n_hidden_nodes=200, verbose=1, embedded_input=False, optimizer='RMSprop',
                 filepath=None):
        
        #import pdb;pdb.set_trace()
        #self.stateful = stateful
        self.batch_size = batch_size
        self.lexicon_size = lexicon_size
        self.max_length = max_length
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_nodes = n_hidden_nodes
        self.n_embedding_nodes = n_embedding_nodes
        self.context_size = context_size
        self.verbose = verbose
        self.embedded_input = embedded_input
        self.filepath = filepath
        
        #mean_layer = Lambda(lambda x: K.mean(x, axis=-2), output_shape=(self.n_embedding_nodes,))
        
        if not self.embedded_input:
            context_input_layer = Input(batch_shape=(self.batch_size, self.max_length), 
                                        dtype='int32', name="context_input_layer")
            seq_input_layer = Input(batch_shape=(self.batch_size, self.max_length), 
                                    dtype='int32', name="seq_input_layer")
            embedding_layer = Embedding(#batch_input_shape=(self.batch_size, self.max_length),
                                        input_dim = self.lexicon_size + 1,
                                        output_dim=self.n_embedding_nodes,
                                        name='embedding')#mask_zero=True,
            embedded_context = embedding_layer(context_input_layer)
            embedded_seq = embedding_layer(seq_input_layer)
            mean_context = mean_layer(embedded_context)
            mean_seq = mean_layer(embedded_seq)
            
            merge_layer = merge([mean_context, mean_seq], mode='concat', concat_axis=-1, 
                    output_shape=(self.n_embedding_nodes * 2,))
        else:
#             context_input_layer = Input(batch_shape=(self.batch_size, self.max_length, self.n_embedding_nodes), 
#                                         dtype='int32', name="context_input_layer")
#             seq_input_layer = Input(batch_shape=(self.batch_size, self.max_length, self.n_embedding_nodes), 
#                                     dtype='int32', name="seq_input_layer")
            context_input_layer = Input(batch_shape=(self.batch_size, self.context_size, self.n_embedding_nodes), 
                                        name="context_input_layer")
#             mean_context_layer = mean_layer(context_input_layer)
            seq_input_layer = Input(batch_shape=(self.batch_size, 1, self.n_embedding_nodes), 
                                    name="seq_input_layer")
#             mean_seq = mean_layer(seq_input_layer)

#             merge_layer = merge([context_input_layer, seq_input_layer], mode='concat', concat_axis=-1, 
#                                 output_shape=(self.n_embedding_nodes * 2,))
            merge_layer = merge([context_input_layer, seq_input_layer], mode='concat', concat_axis=-2,
                                        output_shape=(self.context_size + 1, self.n_embedding_nodes))
    
        #reshape_layer = Reshape((2, self.n_embedding_nodes))(merge_layer)

        hidden_layer1 = GRU(output_dim=self.n_hidden_nodes, return_sequences=False,
                              name='hidden_layer1', stateful=False)(merge_layer)#(reshape_layer)

        pred_layer = Dense(output_dim=1, activation='sigmoid', name='pred_layer')(hidden_layer1)

        model = Model(input=[context_input_layer, seq_input_layer], output=pred_layer)
            
            #         mask_layer = Masking(mask_value=0., input_shape=(None, self.max_length, self.n_embedding_nodes))
#         model.add(mask_layer)

        model.compile(loss="binary_crossentropy", optimizer=eval(optimizer)(), metrics=["accuracy"])
        
        if self.verbose:
            print "CREATED SeqBinary model: embedding layer nodes = {}, "                     "hidden layers = {}, hidden layer nodes = {}, "                     "optimizer = {}, batch size = {}".format(self.n_embedding_nodes, self.n_hidden_layers, 
                                         self.n_hidden_nodes, optimizer, self.batch_size)
        
        return model
    
    def fit(self, X, y, rnn_params=None, **kwargs):      

        self.sk_params.update(rnn_params)
    
        if hasattr(self, 'model'):
            #if model has already been created, continue training with this new data
            kwargs = copy.deepcopy(self.filter_sk_params(Sequential.fit))
            #sentences up to last are context, last sentence is ending to be judged as correct
            self.model.fit([X[:,:-1], X[:,-1:]], y, **kwargs)
        
        else:    
            patience = 2

            #early_stop = EarlyStopping(monitor='train_loss', patience=patience, verbose=0, mode='auto')
            super(SeqBinaryClassifier, self).fit([X[:,:-1], X[:,-1:]], y, **kwargs)
        
        #save model if filepath given
        if self.filepath:
            self.save()  

    def predict(self, X, y_choices, **kwargs):
        #import pdb;pdb.set_trace()
        
        probs_y = []
        for choice_idx in range(y_choices.shape[1]):
            choices = y_choices[:, choice_idx]
            probs = self.model.predict([X, choices[:, None]])[:,-1]
            probs_y.append(probs)
        probs_y = numpy.stack(probs_y, axis=1)
        assert(len(probs_y) == len(X))
            
        return probs_y


# In[9]:

class SequenceVectorizer(CountVectorizer):
    def fit(self, X, y_seqs):
        return super(SequenceVectorizer, self).fit(X + y_seqs)
    def transform(self, X, y_seqs):
        X = super(SequenceVectorizer, self).transform(X)
        #import pdb;pdb.set_trace()
        if type(y_seqs[0]) in [list, tuple]:
            #more than one sequence per instance (e.g. set of choices to predict from)
            y_seqs = [seq for seqs in y_seqs for seq in seqs]
            y_seqs = super(SequenceVectorizer, self).transform(y_seqs)
            y_seqs = y_seqs.toarray().reshape(X.shape[0], -1, X.shape[1])
        else:
            #one-to-one mapping between X and y_seqs
            assert(isinstance(y_seqs[0], (str, unicode)))
            y_seqs = super(SequenceVectorizer, self).transform(y_seqs)
        return X, y_seqs

class AutoencoderPipeline(Pipeline):
    #sklearn pipeline won't pass extra parameters other than input data between steps
    def _pre_transform(self, X, y_seqs=None, **fit_params):
        fit_params_steps = dict((step, {}) for step, _ in self.steps)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Xt = X
        for name, transform in self.steps[:-1]:
            Xt, y_seqs = transform.fit(Xt, y_seqs, **fit_params_steps[name]).transform(Xt, y_seqs)
        return Xt, y_seqs, fit_params_steps[self.steps[-1][0]]
    def fit(self, X, y=None, y_seqs=None, **fit_params):
        #import pdb;pdb.set_trace()
        Xt, y, fit_params = self._pre_transform(X, y_seqs, **fit_params)
        self.steps[-1][-1].fit(Xt, y, **fit_params)
        return self
    def predict(self, X, y_choices=None):
        #check if y_choices is single set or if there are different choices for each input
        
        #import pdb;pdb.set_trace()
        Xt = X
        for name, transform in self.steps[:-1]:
            Xt, y_choices = transform.transform(Xt, y_choices)
        if y_choices is not None:
            return self.steps[-1][-1].predict(Xt, y_choices)
        else:
            return self.steps[-1][-1].predict(Xt)

class Autoencoder(KerasClassifier):
    def __call__(self, lexicon_size, verbose=1):
        self.lexicon_size = lexicon_size
        self.verbose = verbose
        
        model = Sequential()
        model.add(Dense(batch_input_shape=(None, self.lexicon_size), output_dim=200, 
                        activation='relu', name='hidden1'))
        model.add(Dense(output_dim=200, activation='relu', name='hidden2'))
        #model.add(Dense(output_dim=200, input_dim=lexicon_size, activation='tanh', name='hidden2'))
        model.add(Dense(output_dim=self.lexicon_size, activation='sigmoid', name='output'))
        #model.add(Dense(output_dim=n_outcomes, activation='softmax', name='output'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
        return model
    def fit(self, X, y, **kwargs):
        #get number of input nodes
        self.sk_params.update(lexicon_size = X.shape[1])
        #keras doesn't handle sparse matrices
        X = X.toarray()
        y = y.toarray()
        #import pdb;pdb.set_trace()
        super(Autoencoder, self).fit(X, y, **kwargs)
    def predict(self, X, y_choices, **kwargs):
        #keras doesn't handle sparse matrices 
        if type(X) not in [list, tuple, numpy.ndarray]:
            X = X.toarray()
        if type(y_choices) not in [list, tuple, numpy.ndarray]:
            y_choices = y_choices.toarray()
        #import pdb;pdb.set_trace()
        y_choices = check_y_choices(X, y_choices)
        probs_y = []
        for x, choices in zip(X, y_choices):       
            probs = []
            for choice in choices:
                choice = numpy.where(choice > 0)[0]
                prob = super(Autoencoder, self).predict_proba(x[None, :], verbose=0, **kwargs)
                prob = prob[0, choice]
                prob = numpy.sum(numpy.log(prob))
                probs.append(prob)
            probs_y.append(numpy.array(probs))
        #return prob for each choice for each input
        return numpy.array(probs_y)


# In[10]:

def get_train_test_splits(user_inputs, true_outcomes, proxy_inputs, 
                         proxy_outcomes, data_mode):
    
    n_nonproxy_inputs = len(user_inputs)
    
    if data_mode == "proxy-only" or data_mode == "annot-proxy":
        
        #add proxy data to training set; be sure to append proxy data to train only on previous user input indices
        user_inputs = user_inputs + proxy_inputs
        true_outcomes = numpy.concatenate((true_outcomes, proxy_outcomes))
        assert(len(user_inputs) == len(true_outcomes))
        
    if data_mode == "proxy-only":
        #train on proxy only, test on all annotated
        train_test_splits = numpy.arange(len(user_inputs))
        train_test_splits = [(train_test_splits[-len(proxy_inputs):], train_test_splits[:-len(proxy_inputs)])]
        
    else:
        
        assert(data_mode == "annot-proxy" or data_mode == "annot-only")
        #kfold training-testing framework on annotated data
        train_test_splits = KFold(n=n_nonproxy_inputs, n_folds=4, shuffle=True)

    
    if data_mode == "annot-proxy":
    
        #add proxy indices to each train-test split so every training set will contain all proxy data
        train_test_splits = [(numpy.concatenate((train_indices, 
                                             numpy.arange(n_nonproxy_inputs, 
                                                          len(user_inputs), 1))), test_indices) 
                             for train_indices, test_indices in train_test_splits]
        
    assert(numpy.all([len(train_indices) + len(test_indices) == len(user_inputs) 
                      for train_indices, test_indices in train_test_splits]))
    assert(numpy.all([numpy.all(numpy.sort(numpy.concatenate((train_indices, test_indices))) 
            == numpy.arange(len(train_indices) + len(test_indices))) 
                      for train_indices, test_indices in train_test_splits]))
    assert(numpy.all([numpy.all(numpy.in1d(train_indices, test_indices, assume_unique=True, invert=True)) 
                      for train_indices, test_indices in train_test_splits]))
    assert(numpy.all([len(set(test_indices)) == len(test_indices) for _, test_indices in train_test_splits]))

    return (user_inputs, true_outcomes, train_test_splits)


# In[11]:

def train_and_evaluate(model, x, y, train_test_splits, y_choices=None):
    #models is a dict where keys are model names

    #import pdb;pdb.set_trace()
    #seq_to_y = set(zip(y, y_seqs))
#     assert(len(y_to_seq) == max(y) + 1)
    pred_outcomes = []
    true_outcomes = []
    for train_indices, test_indices in train_test_splits:

        #create new unfit model
        new_model = copy.deepcopy(model)

#         if name == "RNNinitLMClassifier":

#             '''train new LM for RNNInitLM
#             train on proxy data + training instances in this fold'''
#             input_outcome_seqs.extend(map_input_outcome_seqs(inputs=train_x, 
#                                     outcome_seqs=[abbrev_seq(outcome_set[outcome]) for outcome in train_y]))
#             lm_transformer = SequenceTransformer(pad_seq=True)
#             lm_transformer.fit(X=input_outcome_seqs)
#             lm = RNNLM(transformer=lm_transformer, batch_size=20)
#             #import pdb;pdb.set_trace()
#             lm.train(seqs=input_outcome_seqs, n_epochs=2)

#             gen_seqs = lm.generate(seed_seqs=["I"], n_words=25)
#             print("\n".join(gen_seqs))

#             lm_emb_weights = lm.model.layers[0].get_weights()
#             lm_layer1_weights = lm.model.layers[1].get_weights()

#             import pdb;pdb.set_trace()
#             new_model.set_params(transformer__lexicon = lm_transformer.lexicon,
#                                  classifier__lexicon_size = lm_transformer.lexicon_size,
#                                  classifier__emb_weights = lm_emb_weights,
#                                  classifier__layer1_weights = lm_layer1_weights)

#         elif name == "Seq2SeqClassifier":

#             new_model.named_steps["transformer"].fit(X=train_x, y_seqs=[abbrev_seq(outcome_set[outcome]) 
#                                                                        for outcome in train_y])
#             new_model.set_params(classifier__lexicon_size = new_model.named_steps["transformer"].lexicon_size,
#                             classifier__max_length = new_model.named_steps["transformer"].max_length,
#                             classifier__outcome_set = new_model.named_steps["transformer"].transform(
#                                                         [abbrev_seq(outcome) for outcome in outcome_set]))
        #import pdb;pdb.set_trace()
        if new_model.__class__.__name__ == "RNNPipeline":
            new_model.fit(X=[x[index] for index in train_indices],
                y=y[train_indices], y_seqs=[y_choices[y[index]] for index in train_indices])
            pred_y = new_model.predict(X=[x[index] for index in test_indices], y_choices=y_choices)
        else:
            #pipeline for non-sequential data (e.g. linear and mlp models)
            new_model.fit(x=user_inputs,y=true_outcomes,
                          y_choices=[abbrev_seq(outcome) for outcome in outcome_set])
            pred_y = new_model.predict(X=[x[index] for index in test_indices])
        pred_outcomes.extend(pred_y)
        true_outcomes.extend([y[index] for index in test_indices])

    accuracy = metrics.accuracy_score(y_true=true_outcomes,
                                      y_pred=pred_outcomes)
    f1_score = metrics.f1_score(y_true=true_outcomes, 
                                y_pred=pred_outcomes,
                                labels=numpy.arange(max(true_outcomes) + 1), average='weighted')

    return accuracy, f1_score


# In[31]:

if __name__ == "__main__":    
    model = sim_score.load_model("../AvMaxSim/vectors.narrative")
    if __name__ == "__main__":
        accuracies = []
        f1_scores = []
        for page_id in all_page_ids:
            #get data for current page
            page_text, outcome_set, proxy_inputs, proxy_outcomes = get_page(page_id=page_id)
            user_inputs, top_outcomes, true_outcomes = get_interactions(page_id=page_id, crowd_only=True)      

            #specify which data used for training: "annot-only", "annot-proxy", or "proxy-only"
            #import pdb;pdb.set_trace()
            user_inputs, true_outcomes, train_test_splits = get_train_test_splits(user_inputs=user_inputs,
                                                                                  true_outcomes=true_outcomes,
                                                                                  proxy_inputs=proxy_inputs,
                                                                                  proxy_outcomes=proxy_outcomes,
                                                                                  data_mode="annot-proxy")

            pred_outcomes = []
            gold_outcomes = []
            for train_indices, test_indices in train_test_splits:
                #import pdb;pdb.set_trace()

                #y_choices = model.named_steps["transformer"].transform([abbrev_seq(outcome) for outcome in outcome_set])
                #import pdb;pdb.set_trace()
    #             prob_y = model.predict(X=[user_inputs[index] for index in test_indices],
    #                                   y_choices=[abbrev_seq(outcome) for outcome in outcome_set])
                prob_y = predict_avemax(model=model, X=[user_inputs[index] for index in test_indices],
                                        y_choices=[abbrev_seq(outcome) for outcome in outcome_set])
                #import pdb;pdb.set_trace()
                pred_y = numpy.argmax(prob_y, axis=1)
                pred_outcomes.extend(pred_y)
                gold_outcomes.extend([true_outcomes[index] for index in test_indices])

            accuracy = metrics.accuracy_score(y_true=gold_outcomes,
                                              y_pred=pred_outcomes)
            accuracies.append(accuracy)
            f1_score = metrics.f1_score(y_true=gold_outcomes, 
                                        y_pred=pred_outcomes,
                                        labels=numpy.arange(max(gold_outcomes) + 1), average='weighted')
            f1_scores.append(f1_score)

            print("{:<20}{:<10}{:<10.3f}{:<10.3f}".format("AveMax", page_id, accuracy, f1_score))
            
    print("{:<20}{:<10}{:<10.3f}{:<10.3f}".format("AveMax", "Mean", numpy.mean(accuracies), numpy.mean(f1_scores)))


# In[826]:

# if __name__ == "__main__":
# accuracies = {}
# f1_scores = {}

# print("{:<20}{:<10}{:<10}{:<10}".format("classifier", "page ID", "accuracy", "f1_score"))
# for page_id in all_page_ids:
    
# #     input_outcome_seqs = []
# #     for page_id_ in all_page_ids:
# #         if page_id_ == page_id:
# #             #don't train on current page's data yet - will use training data alone below
# #             continue
# #         page_text, outcome_set, proxy_inputs, proxy_outcomes = get_page(page_id=page_id_)
# #         user_inputs, top_outcomes, annot_outcomes = get_interactions(page_id=page_id_, crowd_only=True)
# #         outcome_seqs = [abbrev_seq(outcome_set[outcome]) for outcome in #shorten outcome sequences to first sentence
# #                         numpy.concatenate((proxy_outcomes, annot_outcomes))]
# #         #import pdb;pdb.set_trace()
# #         input_outcome_seqs.extend(map_input_outcome_seqs(proxy_inputs + user_inputs, outcome_seqs))

    
#     #get data for current page
#     page_text, outcome_set, proxy_inputs, proxy_outcomes = get_page(page_id=page_id)
#     user_inputs, top_outcomes, true_outcomes = get_interactions(page_id=page_id, crowd_only=True)      

#     #specify which data used for training: "annot-only", "annot-proxy", or "proxy-only"
#     #import pdb;pdb.set_trace()
#     user_inputs, true_outcomes, train_test_splits = get_train_test_splits(user_inputs=user_inputs,
#                                                                           true_outcomes=true_outcomes,
#                                                                           proxy_inputs=proxy_inputs,
#                                                                           proxy_outcomes=proxy_outcomes,
#                                                                           data_mode="annot-proxy")  
        
    
#     models = {#"LinearModel": 
# #           Pipeline(steps=[("transformer", CountVectorizer(tokenizer=lambda x:segment_and_tokenize(x))),
# #                           ("classifier", LogisticRegression(random_state=rng))]),
# #           "MLPClassifier": 
# #           Pipeline(steps=[("transformer", CountVectorizer(tokenizer=lambda x:segment_and_tokenize(x))),
# #                           ("classifier", MLPClassifier(batch_size=20, n_outcomes=len(outcome_set), 
# #                                                        verbose=0, nb_epoch=100))]),
# #           "RNNClassifier":
# #           RNNPipeline(steps=[("transformer", SequenceTransformer(pad_seq=True)),
# #                           ("classifier",  RNNClassifier(batch_size=20, n_outcomes=len(outcome_set), verbose=1, 
# #                                                         nb_epoch=100))])}
    
            
# #               "Seq2SeqClassifier":
# #             RNNPipeline(steps=[("transformer", SequenceTransformer(pad_seq=True)),
# #                       ("classifier", Seq2SeqClassifier(batch_size=20, nb_epoch=100, verbose=0))])}
        
            
# #               "RNNinitLMClassifier":
# #               Pipeline(steps=[("transformer", SequenceTransformer(lexicon=None, pad_seq=True)),
# #                               ("classifier", RNNClassifier(verbose=0, n_outcomes=len(outcome_set), lexicon_size=None,
# #                                                            nb_epoch=100, emb_weights=None, layer1_weights=None))])
#     if not accuracies:
#         accuracies = {name:[] for name in models}
#         f1_scores = {name:[] for name in models} 
    
#     for name, model in models.items():
        
        
#         accuracy, f1_score = train_and_evaluate(model=model, x=user_inputs,
#                                                 y=true_outcomes,
#                                                 y_choices=[abbrev_seq(outcome) for outcome in outcome_set],
#                                                 train_test_splits=train_test_splits)
        
#         accuracies[name].append(accuracy)
#         f1_scores[name].append(f1_score)

#         print("{:<20}{:<10}{:<10.3f}{:<10.3f}".format(name, page_id, accuracy, f1_score))

# print("\n")
# for name in accuracies:
#     print("{:<20}{:<10}{:<10.3f}{:<10.3f}".format(name, "Mean", numpy.mean(accuracies[name]), 
#                                                 numpy.mean(f1_scores[name])))


# In[824]:

# if __name__ == "__main__":
# user_inputs = []
# page_text = []
# outcome_set = []
# proxy_inputs = []
# proxy_outcomes = []
# top_outcomes = []
# annot_outcomes = []
# input_to_outcome_set = [] #map each user input to its outcome set
# n_total_outcomes = 0

# for page_id in all_page_ids:
    
#     #get data for this page
#     page_text_, outcome_set_, proxy_inputs_, proxy_outcomes_ = get_page(page_id=page_id)
#     user_inputs_, top_outcomes_, annot_outcomes_ = get_interactions(page_id=page_id, crowd_only=True)
#     page_text.extend(page_text_)
#     proxy_inputs.extend(proxy_inputs_)
#     user_inputs.extend(user_inputs_)
    
#     #adjust outcome numbers in top_outcomes and annot_outcomes to account for outcomes from other pages
#     new_outcome_indices = numpy.arange(n_total_outcomes, n_total_outcomes + len(outcome_set_), 1)
#     input_to_outcome_set.extend([new_outcome_indices] * len(user_inputs_))
    
#     outcome_set.append(outcome_set_)
#     n_total_outcomes += len(outcome_set_)
    
#     proxy_outcomes_ = new_outcome_indices[proxy_outcomes_]
#     top_outcomes_ = new_outcome_indices[top_outcomes_]
#     annot_outcomes_ = new_outcome_indices[annot_outcomes_]
        
#     proxy_outcomes.extend(proxy_outcomes_)
#     top_outcomes.extend(top_outcomes_)
#     annot_outcomes.extend(annot_outcomes_)     

#     assert(numpy.all(numpy.array(list(set(proxy_outcomes))) == numpy.arange(n_total_outcomes)))
#     assert(max(proxy_outcomes) == n_total_outcomes - 1)
#     assert(max(top_outcomes) <= n_total_outcomes - 1)
#     assert(max(annot_outcomes) <= n_total_outcomes - 1)
#     assert(len(proxy_inputs) == len(proxy_outcomes))
#     assert(len(user_inputs) == len(top_outcomes) == len(annot_outcomes) == len(input_to_outcome_set))
    
# proxy_outcomes = numpy.array(proxy_outcomes)
# top_outcomes = numpy.array(top_outcomes)
# annot_outcomes = numpy.array(annot_outcomes)
# input_to_outcome_set = numpy.array(input_to_outcome_set)

# train_test_splits = KFold(n=len(user_inputs), n_folds=4, shuffle=True)

# models = {"MLPClassifierAll": Pipeline(steps=[("transformer", 
#                                                CountVectorizer(tokenizer=lambda x:segment_and_tokenize(x))),
#                         ("classifier", MLPClassifier(n_outcomes=n_total_outcomes, verbose=0, nb_epoch=200))]),
#           "RNNClassifierAll": Pipeline(steps=[("transformer", SequenceTransformer(pad_seq=True)),
#                       ("classifier",RNNClassifier(n_outcomes = n_total_outcomes, verbose=0,  nb_epoch=200))])}

# accuracies = {name:[] for name in models}
# f1_scores = {name:[] for name in models}
# for name, model in models.items():

#     pred_outcomes = []
#     for train_indices, test_indices in train_test_splits:

#         #add proxy data to training set
#         train_x = proxy_inputs + [user_inputs[index] for index in train_indices]
#         train_y = numpy.concatenate((proxy_outcomes, annot_outcomes[train_indices]))

#         #import pdb;pdb.set_trace() 
#         model.fit(X=train_x, y=train_y)

#         test_x = [user_inputs[index] for index in test_indices]
#         test_y = annot_outcomes[test_indices]

#         #import pdb;pdb.set_trace()
#         #sklearn pipeline won't allow parameters to be passed to predict fn
#         model.set_params(classifier__input_to_outcome_set=input_to_outcome_set[test_indices])
#         pred_y = model.predict(X=test_x)
#         pred_outcomes.extend(pred_y)

#     true_outcomes = [annot_outcomes[index]
#                      for index in numpy.concatenate([test_indices for train_indices, test_indices
#                                                                          in train_test_splits])]
    
#     assert(len(pred_outcomes) == len(true_outcomes))
#     accuracy = metrics.accuracy_score(y_true=true_outcomes,
#                                       y_pred=pred_outcomes)
#     accuracies[name].append(accuracy)
#     f1_score = metrics.f1_score(y_true=true_outcomes, 
#                                 y_pred=pred_outcomes,
#                                 labels=numpy.arange(len(outcome_set)), average='weighted')
#     f1_scores[name].append(f1_score)

#     print("{:<20}{:<10.3f}{:<10.3f}".format(name, accuracy, f1_score))

# # print("\n")
# # print("{:<20}{:<10}{:<10.3f}{:<10.3f}".format(name, "Mean", numpy.mean(accuracies), 
# #                                             numpy.mean(f1_scores)))


# In[859]:

# #train lm on all pages
# input_outcome_seqs = []
# for page_id in all_page_ids[:1]:
#     import pdb;pdb.set_trace()
#     page_text, outcome_set, proxy_inputs, proxy_outcomes = get_page(page_id=page_id)
#     user_inputs, top_outcomes, annot_outcomes = get_interactions(page_id=page_id, crowd_only=True)
#     outcome_seqs = [abbrev_seq(outcome_set[outcome]) for outcome in #shorten outcome sequences to first sentence
#                     numpy.concatenate((proxy_outcomes, annot_outcomes))]
#     input_outcome_seqs.extend(map_input_outcome_seqs(proxy_inputs + user_inputs, outcome_seqs))
    
# lm_transformer = SequenceTransformer(pad_seq=True) #y_is_seq=True)
# lm_transformer.fit(X=input_outcome_seqs)
# lm = RNNLM(transformer=lm_transformer, batch_size=20)
# #import pdb;pdb.set_trace()
# lm.train(seqs=input_outcome_seqs, n_epochs=2)


# In[39]:

# page_ids = get_project_pages(project="R") #get Runner pages
# #for some reason R is associated with a few other pages not in Runner, remove them
# page_ids = [page_id for page_id in page_ids if page_id not in [97, 99, 143]]
# #page_text, input_examples, outcome_set, next_page_id = get_page(page_id=page_id)
# page_text = [get_page(page_id=page_id)[0] for page_id in page_ids]
# transformer = SequenceTransformer(pad_seq=False)#, y_is_seq=True)
# sequences = transformer.fit_transform(X=page_text)


# In[335]:

# page_id = 112
# page_text, input_examples, outcome_set = get_page(page_id=page_id)
# #represent outcome sequences by first sentence only
# abbr_outcome_set = [list(split_single(outcome))[0] for outcome in outcome_set]
# user_inputs, annotated_outcomes = get_annotated_interactions(page_id=page_id)
# author_inputs, author_outcomes = map_author_inputs(input_examples=input_examples, outcome_set=outcome_set)
# inputs = user_inputs + author_inputs
# outcomes = numpy.concatenate((annotated_outcomes, author_outcomes))
# outcome_seqs = [outcome_set[outcome] for outcome in outcomes]
# input_outcome_seqs = [input + " " + outcome for input, outcome in zip(inputs, outcome_seqs)]
# input_outcome_seqs

