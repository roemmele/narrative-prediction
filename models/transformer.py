import numpy, os, spacy, pickle
from sklearn.preprocessing import FunctionTransformer
from keras.preprocessing.sequence import pad_sequences

#load spacy model for nlp tools
encoder = spacy.load('en')

def segment(text):
    return [sentence.string.strip() for sentence in encoder(text).sents]

def tokenize(sentence):
    return [word.lower_ for word in encoder(sentence) if word.string.strip()]

def segment_and_tokenize(text):
    #import pdb;pdb.set_trace()
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

def load_transformer(filepath, embeddings=None):
    #load saved models
    with open(filepath + '/transformer.pkl', 'rb') as f:
        transformer = pickle.load(f)
    transformer.word_embeddings = embeddings
    return transformer

class SequenceTransformer(FunctionTransformer):
    def __init__(self, lexicon=None, min_freq=1, max_length=None, extra_text=[], 
                 pad_seq=False, verbose=1, unk_word="<UNK>", word_embeddings=None, sent_encoder=None,
                 replace_ents=False, embed_y=False, reduce_emb_mode=None, copy_input_to_output=False, 
                 filepath=None):
        #import pdb;pdb.set_trace()
        FunctionTransformer.__init__(self)
        self.unk_word = unk_word #string representation for unknown words in lexicon
        #use existing word embeddings if given
        self.word_embeddings = word_embeddings
        self.sent_encoder = sent_encoder
        if self.word_embeddings is not None:
            if type(self.word_embeddings) is not dict:
                #convert Word2Vec embeddings to dict
                self.word_embeddings = {word:self.word_embeddings[word] for word in self.word_embeddings.vocab}
            self.n_embedding_nodes = len(self.word_embeddings.values()[0])
        self.pad_seq = pad_seq
        self.lexicon = lexicon
        if self.lexicon:
            #use existing lexicon
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
            assert(self.word_embeddings is not None or self.sent_encoder is not None)
        self.embed_y = embed_y
        #specify if named entities should be replaced with generic labels
        self.replace_ents = replace_ents
        #specify if embeddings should be combined across sequence (e.g. take mean, sum)
        # if reduce_emb_mode:
        #     assert(self.word_embeddings is not None)
        #     assert(not self.pad_seq)
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
        if not self.lexicon:    #transformer is already fit is lexicon exists
            X = self.format_seqs(seqs=X, unravel=True)
            if y_seqs is not None:
                y_seqs = self.format_seqs(seqs=y_seqs, unravel=True)
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
        self.fit(X, y_seqs)
        rnn_params['lexicon_size'] =  self.lexicon_size
        rnn_params['max_length'] = self.max_length
        if self.word_embeddings is not None or self.sent_encoder is not None:
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
            
    def embed_words(self, seqs):
        #import pdb;pdb.set_trace()
        #convert word indices to vectors
        embedded_seqs = []
        for seq in seqs:
            #convert to vectors rather than indices - if word not in lexicon represent with all zeros
            seq = [self.word_embeddings[self.lexicon_lookup[word]]
                   if self.lexicon_lookup[word] in self.word_embeddings
                    else numpy.zeros((self.n_embedding_nodes))
                   for word in seq]
            seq = numpy.array(seq)
            embedded_seqs.append(seq)
        return embedded_seqs

    def encode_sents(self, seqs):
        #convert sentences to vectors
        encoded_seqs = []
        for seq in seqs:
            # if self.sent_encoder.__class__.__name__ == 'Sequential':
            #     #keras encoder
            seq = self.sent_encoder.predict(numpy.array(seq)[None])[0][-1]
            # elif self.sent_encoder.__class__.__name__ == 'dict':
            #     #skipthoughts encoder
            #     seq = skipthoughts.encode(sent_encoder, seq)
            encoded_seqs.append(seq)
            self.sent_encoder.reset_states()
        encoded_seqs = numpy.array(encoded_seqs)
        return encoded_seqs
    
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
        seqs = [pad_sequences(sequences=seq, maxlen=self.max_length)
               for seq in seqs]
                
        seqs = numpy.array(seqs)
        return seqs

    def pad_encoded_sents(self):
        import pdb;pdb.set_trace()
        seqs = [numpy.array([numpy.append(sent, numpy.zeros((self.max_length - len(sent), 
                self.lm_classifier.encoder_model.layers[-1].output_dim)), axis=0)
                for sent in seq]) for seq in seqs]
        
        seqs = numpy.array(seqs)
        return seqs

    def pad_embeddings(self, seqs):
        #import pdb;pdb.set_trace()
        assert(type(seqs[0]) is list and len(seqs[0][0].shape) == 2)
        #input sequences are a list of sentences
        seqs = [numpy.array([numpy.append(sent, numpy.zeros((self.max_length - len(sent), 
                self.word_embeddings.n_embedding_nodes)), axis=0)
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
            if self.sent_encoder and self.embed_y:
                y_seqs = numpy.array([self.encode_sents(seqs=seqs) for seqs in y_seqs])
                if self.pad_seq:
                    y_seqs = self.pad_embeddings(seqs=y_seqs)
            elif self.word_embeddings and self.embed_y:
                y_seqs = [self.embed_words(seqs=seqs) for seqs in y_seqs]
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
        if self.sent_encoder:
            X = numpy.array([self.encode_sents(seqs=seqs) for seqs in X])
            if self.pad_seq:
                #import pdb;pdb.set_trace()
                X = self.pad_encoded_sents(seqs=X)
        elif self.word_embeddings:
            X = [self.embed_words(seqs=seqs) for seqs in X]
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

        with open(self.filepath + '/transformer.pkl', 'wb') as f:
            pickle.dump(self, f)
        
    def __getstate__(self):
        #don't save embeddings
        state = dict((k, v) for (k, v) in self.__dict__.iteritems() if k not in ('word_embeddings', 'sent_encoder'))
        state.update({'word_embeddings': None, 'sent_encoder': None})
        return state

