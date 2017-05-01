import numpy, os, spacy, pickle, sys, re
from sklearn.preprocessing import FunctionTransformer
from keras.preprocessing.sequence import pad_sequences

sys.path.append('skip-thoughts-master')
sys.path.append('../skip-thoughts-master')
import skipthoughts
reload(skipthoughts)
from training import vocab
from training import train as encoder_train
from training import tools as encoder_tools
reload(encoder_tools)

#load spacy model for nlp tools
encoder = spacy.load('en')

def segment(text):
    return [sentence.string.strip() for sentence in encoder(text).sents]

def tokenize(text):
    return [word.lower_ for word in encoder(text)]# if word.string.strip()]

def segment_and_tokenize(text):
    #import pdb;pdb.set_trace()
    return [word.lower_ for sentence in encoder(text).sents 
                            for word in sentence if word.string.strip()]

def replace_entities(tok_seq, ents):
    #input is sequence already tokenized into words
    #if rep strings given, replace entity word (e.g. "PERSON") with different string (e.g. "PERSON" > "I")
    return [ents[word] if word in ents else word for word in tok_seq]

def get_entities(text):
    return [(ent.string.strip(), ent.label_) for ent in encoder(text).ents]

def load_transformer(filepath, embeddings=None):
    #load saved models
    with open(filepath + '/transformer.pkl', 'rb') as f:
        transformer = pickle.load(f)
    transformer.word_embeddings = embeddings
    return transformer

class SkipthoughtsTransformer():
    def __init__(self, encoder_module, encoder, encoder_dim, verbose=True):
        self.encoder_module = encoder_module
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.verbose = verbose
    def fit(self, X, y_seqs=None):
        #model is already fit
        return
    def transform(self, X, y_seqs=None, X_filepath=None, y_seqs_filepath=None):
        if y_seqs is not None:
            y_seqs = self.encode(y_seqs, y_seqs_filepath)
        X = self.encode(X, X_filepath)
        return X, y_seqs
    def encode(self, seqs, filepath=None):
        #import pdb;pdb.set_trace()
        n_seqs = len(seqs)
        if type(seqs[0]) in (list, tuple):
            seq_length = numpy.array([len(seq) for seq in seqs])
            if numpy.all(seq_length[0] == seq_length[1:]):
                #every sequence has the same length
                seq_length = seq_length[0]
            #flatten seqs
            seqs = [sent for seq in seqs for sent in seq]
        else:
            seq_length = 1
        seqs_shape = (len(seqs), self.encoder_dim)
        if filepath:
            encoded_seqs = numpy.memmap(filepath, dtype='float64',
                                        mode='w+', shape=seqs_shape)
        else:
            encoded_seqs = numpy.zeros(seqs_shape)

        chunk_size = 500000
        for seq_idx in range(0, len(seqs), chunk_size):
            #memory errors if encoding a large number of stories
            encoded_seqs[seq_idx:seq_idx + chs_size] = self.encoder_module.encode(self.encoder, 
                                                                                    seqs[seq_idx:seq_idx + chunk_size], verbose=self.verbose)
        
        if type(seq_length) not in (list, tuple, numpy.ndarray) and seq_length > 1:
            encoded_seqs = encoded_seqs.reshape(n_seqs, seq_length, self.encoder_dim)
        else:
            #different lengths per sequence
            idxs = [numpy.sum(seq_length[:idx]) for idx in range(len(seq_length))] + [None] #add -1 for last entry
            encoded_seqs = [encoded_seqs[idxs[start]:idxs[start+1]] for start in range(len(idxs) - 1)]
            
        return encoded_seqs

def load_skipthoughts_transformer(filepath='../skip-thoughts-master', word_embeddings='../ROC/AvMaxSim/vectors', n_nodes=4800, pretrained=True, verbose=True):
    if pretrained:
        #filepaths are hard-coded for pre-trained skipthought model
        encoder_module = skipthoughts
        sent_encoder = encoder_module.load_model(path_to_models=filepath)

    else:
        encoder_module = encoder_tools
        sent_encoder = encoder_module.load_model(embed_map=word_embeddings, 
                                                 path_to_model=filepath + '/encoder', 
                                                 path_to_dictionary=filepath + '/lexicon')
        
    transformer = SkipthoughtsTransformer(encoder_module=encoder_module, 
                                          encoder=sent_encoder,
                                          encoder_dim=n_nodes, verbose=verbose)

    print 'loaded skipthoughts encoder from', filepath

    return transformer

def load_seqs(filepath, memmap=False, shape=None):
    if memmap:
        #file was saved as memmap
        seqs = numpy.memmap(filepath, dtype='float64', mode='r', shape=shape)
    else:
        seqs = numpy.load(filepath, mmap_mode='r')
    print "loaded sequences from filepath", filepath
    return seqs


class SequenceTransformer(FunctionTransformer):
    def __init__(self, lexicon=None, min_freq=1, max_lexicon_size=500000, max_length=None, 
                 pad_seq=False, verbose=1, unk_word=u"<UNK>", word_embeddings=None, sent_encoder=None,
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
        self.max_lexicon_size = max_lexicon_size
        self.max_length = max_length
        self.verbose = verbose
        #specify if y_seqs should be converted to embeddings like input seqs
        if embed_y:
            assert(self.word_embeddings is not None or self.sent_encoder is not None)
        self.embed_y = embed_y
        self.replace_ents = replace_ents #specify if named entities should be replaced with generic labels
        self.reduce_emb_mode = reduce_emb_mode #specify if embeddings should be combined across sequence (e.g. take mean, sum)
        self.copy_input_to_output = copy_input_to_output
        if verbose and self.replace_ents:
            print "filter named entities = True"
        if filepath and not os.path.isdir(filepath):
            os.mkdir(filepath)
        self.filepath = filepath

    def make_lexicon(self, seqs):

        if self.lexicon is None:    #transformer has been fit before if lexicon exists
            self.word_counts = {}

        self.lexicon = {}
        self.lexicon[self.unk_word] = 1
        #import pdb;pdb.set_trace()
        #word_counts = {}
        for seq in seqs:
            #first get named entities
            words = tokenize(text=seq)

            if self.replace_ents:
                #reduce vocab by mapping all named entities to entity labels (e.g. "PERSON")
                ents = get_entities(seq)
                if ents:
                    words = replace_entities(words, ents)

            for word in words:
                if word not in self.word_counts:
                    self.word_counts[word] = 1
                else:
                    self.word_counts[word] += 1

        for word, count in self.word_counts.items():
            if count >= self.min_freq:
                self.lexicon[word] = max(self.lexicon.values()) + 1

        self.lexicon_size = len(self.lexicon.keys())
        self.lexicon_lookup = [None] + [word for index, word in 
                                sorted([(index, word) for word, index in self.lexicon.items()])] #insert entry for empty timeslot in lexicon lookup
        assert(len(self.lexicon_lookup) == self.lexicon_size + 1)

    
    def fit(self, X, y_seqs=None):
        #import pdb;pdb.set_trace()
        #add words from outcomes and page to lexicon in addition to user inputs
        X = self.format_seqs(seqs=X, unravel=True)
        if y_seqs is not None:
            y_seqs = self.format_seqs(seqs=y_seqs, unravel=True)
            X += set(y_seqs) #add outcome sequences to lexicon
        self.make_lexicon(seqs=X)
        #import pdb;pdb.set_trace()
        if self.pad_seq:
            self.set_max_length(seqs=X)
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
    
    def lookup_eos(self, eos_tokens=[".", "?", "!"]):
        #get indices of end of sentence markers (needed for generating with language model)
        eos_idxs = [self.lexicon[token] for token in eos_tokens if token in self.lexicon]
        return eos_idxs
        
    def fit_transform(self, X, y_seqs=None):
        #import pdb;pdb.set_trace()
        self.fit(X, y_seqs)
        # rnn_params['lexicon_size'] =  self.lexicon_size
        # rnn_params['max_length'] = self.max_length
        # if self.word_embeddings is not None or self.sent_encoder is not None:
        #     rnn_params['embedded_input'] = True
        #import pdb;pdb.set_trace()
        X, y_seqs = self.transform(X, y_seqs)
        return X, y_seqs
    
    def text_to_nums(self, seqs):
        #import pdb;pdb.set_trace()
        encoded_seqs = []
        for seq in seqs:
            encoded_seq = tokenize(seq)
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

    def detokenize_sent(self, sent, eos_tokens=['.', '?', '!'], cap_tokens=[]):
        '''use simple rules for transforming list of tokens back into string
        cap tokens is optional list of words that should be capitalized'''
        #detok_sents = []

        # for (sent_idx, sent) in enumerate(sents): #capitalize named entities
        detok_sent = sent
        for token_idx, token in enumerate(detok_sent):
            for len_idx in range(4, 0, -1):
                token = " ".join([token[0].upper() + token[1:] for token in detok_sent[token_idx:token_idx + len_idx]])
                ent = get_entities(token)
                if ent:
                    ent, ent_type = ent[0]
                    if ent == token and ent_type in ('PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'FACILITY', 'EVENT', 'WORK OF ART', 'LANGUAGE'): #if cap tokens is given, only capitalize tokens that are found in this list
                        if not cap_tokens or (cap_tokens and ent.lower() in cap_tokens):
                            detok_sent[token_idx:token_idx + len_idx] = ent.split()
                            break

        # if type(detok_sent) in (tuple, list):
        detok_sent = " ".join(detok_sent)

        #capitalize first-person "I" pronoun
        detok_sent = re.sub("i ", "I ", detok_sent)
        # detok_sent = re.sub(" i ", " I ", detok_sent)

        #rules for contractions
        detok_sent = re.sub(" n\'t ", "n\'t ", detok_sent)
        detok_sent = re.sub(" \'d ", "\'d ", detok_sent)
        detok_sent = re.sub(" \'s ", "\'s ", detok_sent)
        detok_sent = re.sub(" \'ve ", "\'ve ", detok_sent)
        detok_sent = re.sub(" \'ll ", "\'ll ", detok_sent)
        detok_sent = re.sub(" \'m ", "\'m ", detok_sent)
        detok_sent = re.sub(" \'re ", "\'re ", detok_sent)

        #rules for formatting punctuation
        detok_sent = re.sub(" \.", ".", detok_sent)
        detok_sent = re.sub(" \!", "!", detok_sent)
        detok_sent = re.sub(" \?", "?", detok_sent)
        detok_sent = re.sub(" ,", ",", detok_sent)
        detok_sent = re.sub(" \- ", "-", detok_sent)
        detok_sent = re.sub(" :", ":", detok_sent)
        detok_sent = re.sub(" ;", ";", detok_sent)
        detok_sent = re.sub("\$ ", "$", detok_sent)

        punc_pairs = {"\'": "\'", "\"": "\"", "(": ")", "[": "]"} #map each opening puncutation mark to closing mark
        open_punc = None
        for (char_idx, char) in enumerate(detok_sent): #check for quotes and parenthesis
            if char in punc_pairs:
                open_punc = char
                if char_idx + 1 == len(detok_sent):
                    #sent ends in unmatched punc
                    detok_sent = detok_sent[:char_idx-1] + char
                elif detok_sent[char_idx + 1] == " ":
                    detok_sent = detok_sent[:char_idx] + char + detok_sent[char_idx+1:]
            elif open_punc and char is punc_pairs[open_punc]: #end quote/parenthesis
                if char_idx > 0 and detok_sent[char_idx-1] == " ":
                    detok_sent = detok_sent[:char_idx-1] + char + detok_sent[char_idx:]
                    open_punc = None

        detok_sent = detok_sent.strip()
        detok_sent = detok_sent[0].upper() + detok_sent[1:]
        while len(detok_sent) > 1 and detok_sent[-1] in ("\"", "\'", "-", ",", ":", ";"):
            detok_sent = detok_sent[:-1] #if sentence ends with punctuation that is not end of sentence punctuation, remove it
        if detok_sent[-1] not in eos_tokens:
            detok_sent += "." #if sentence doesn't end in punctuation, add a period as default
        #detok_sents.append(detok_sent)

        return detok_sent
    
    def decode_seqs(self, seqs, eos_tokens=['.', '?', '!'], cap_tokens=[]):
        if type(seqs[0]) not in (list, numpy.ndarray, tuple):
            seqs = [seqs]
        decoded_seqs = []
        #transform numerical seq back intro string
        for seq in seqs:
            seq = [self.lexicon_lookup[word] if self.lexicon_lookup[word] else "None" for word in seq]
            #seq = " ".join(seq)
            seq = self.detokenize_sent(sent=seq, eos_tokens=eos_tokens, cap_tokens=cap_tokens)#, named_ents=named_ents) #detokenize; pass a list of words that should be capitalized
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
            seqs = numpy.array([[numpy.mean(sent, axis=0) for sent in seq] for seq in seqs])
        elif self.reduce_emb_mode == 'sum':
            seqs = numpy.array([[numpy.sum(sent, axis=0) for sent in seq] for seq in seqs])
        return seqs
    
    def set_max_length(self, seqs):
        self.max_length = max([len(tokenize(seq)) for seq in seqs])
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
        
    def transform(self, X, y_seqs=None, X_filepath=None, y_seqs_filepath=None):
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

            if y_seqs_filepath:
                numpy.save(y_seqs_filepath, y_seqs)

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

        if X_filepath:
            numpy.save(X_filepath, X)
                 
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

