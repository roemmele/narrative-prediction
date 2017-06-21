import numpy, os, spacy, pickle, sys, re, random
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
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

rng = numpy.random.RandomState(0)

def segment(seq):
    seq = [sent.string.strip() for sent in encoder(seq).sents]
    return seq

def tokenize(seq, lowercase=True, recognize_ents=False):
    seq = encoder(seq)
    if recognize_ents: #merge named entities into single tokens
        ent_start_idxs = {ent.start:ent for ent in seq.ents if ent.string.strip()}
        #combine each ent into a single token; this is pretty hard to read, but it works
        seq = [ent_start_idxs[word_idx] if word_idx in ent_start_idxs else word
                for word_idx, word in enumerate(seq) 
                    if (not word.ent_type_ or word_idx in ent_start_idxs)]
    if lowercase: #don't lowercase if token is an entity (entities will be of type span instead of token; or will be prefixed with 'ENT_' if already transformed to types)
        seq = [word.string.strip().lower() if (type(word) == spacy.tokens.token.Token and not word.string.startswith('ENT_')) else word.string.strip() for word in seq]
    else:
        seq = [word.string.strip() for word in seq]
    seq = [word for word in seq if word] #make sure no empty strings returned
    return seq


def ent_counts_to_probs(ent_counts, min_freq=1):
    ent_counts = {ent_type:{ent:count for ent,count in ent_counts[ent_type].items() #filter by frequency
                        if count >= min_freq} for ent_type in ent_counts}
    sum_ent_counts = {ent_type:sum(counts.values()) for ent_type, counts in ent_counts.items()}
    ent_probs = {ent_type:{ent:count * 1. / sum_ent_counts[ent_type] for ent,count in ent_counts[ent_type].items()}
                                                                                         for ent_type in ent_counts}
    return ent_probs

def get_ents(seq, include_ent_types=('PERSON','NORP','ORG','GPE'), recognize_gender=False,
            gender_filenames={'FEMALE':'female_names.pkl', 'MALE':'male_names.pkl'}):
    '''return dict of all entities in seq mapped to their entity types, optionally labeled with gender for PERSON entities'''

    if recognize_gender:
        names_gender = {}
        for gender, filename in gender_filenames.items():
            with open(filename) as f:
                names_gender[gender] = pickle.load(f)
    ents = {}
    for ent in encoder(seq).ents:
        ent_type = ent.label_
        if ent_type in include_ent_types:
            ent = ent.string.strip()
            if ent: #not sure why, but whitespace can be detected as an ent, so need to check for this
                ents[ent] = [ent_type]
                if recognize_gender and ent_type == 'PERSON': #append gender to entity type
                    detected_gender = None
                    for gender, names in names_gender.items():
                        if ent.split()[0] in names: #person may have multiple tokens in name, just check first name
                            if detected_gender: #name is associated with both genders, so omit it
                                detected_gender = None
                            else:
                                detected_gender = gender
                    if detected_gender:
                        ents[ent].append(detected_gender)
                ents[ent] = "_".join(ents[ent])
    return ents


def number_ents(ents):
    '''return dict of all entities in seq mapped to their entity types, 
    with numerical suffixes to distinguish entities of the same type'''
    ent_type_counts = {}
    num_ents = {}
    #print ents
    for ent, ent_type in ents.items():
        #print num_ents
        #print ent, ent_type
        coref_ent = [num_ent for num_ent in num_ents if num_ent.split()[0] == ent.split()[0] and ents[num_ent] == ent_type] #treat ents with same first word as co-referring
        if coref_ent:
            num_ents[ent] = num_ents[coref_ent[0]]
        else:
            ent_type = ent_type.split("_")
            if ent_type[0] in ent_type_counts:
                ent_type_counts[ent_type[0]] += 1
            else:
                ent_type_counts[ent_type[0]] = 1
            num_ents[ent] = ent_type
            num_ents[ent].insert(1, str(ent_type_counts[ent_type[0]] - 1)) #insert number id after entity type (and before tag, if it exists)
            num_ents[ent] = "_".join(num_ents[ent])
    return num_ents


def adapt_seq_ents(seq, ents={}, sub_ent_probs={}):

    if type(seq) in (str, unicode):
        seq = tokenize(seq, lowercase=False)

    ents = {ent_type:ent for ent,ent_type in ents.items()} #reverse ents so that types map to names
    adapted_seq_ents = {"_".join(token.split("_")[1:]):None for token in seq if token.startswith('ENT_')}

    if not adapted_seq_ents:
        return seq

    for seq_ent_type in {ent_type:adapted_ent for ent_type,adapted_ent in adapted_seq_ents.items() if not adapted_ent}:
        if seq_ent_type in ents:
            adapted_seq_ents[seq_ent_type] = ents[seq_ent_type]
            del ents[seq_ent_type]

    if ents:
        for seq_ent_type in {ent_type:adapted_ent for ent_type,adapted_ent in adapted_seq_ents.items() if not adapted_ent}:
            for ent_type,ent in ents.items():
                if seq_ent_type.split("_")[0] in ent_type.split("_")[0]:
                    #import pdb;pdb.set_trace()
                    adapted_seq_ents[seq_ent_type] = ents[ent_type]
                    del ents[ent_type]
                    break

    for seq_ent_type in {ent_type:adapted_ent for ent_type,adapted_ent in adapted_seq_ents.items() if not adapted_ent}:
        #import pdb;pdb.set_trace()
        if seq_ent_type.split("_")[0] in sub_ent_probs:
            sub_ents, sub_probs = zip(*sub_ent_probs[seq_ent_type.split("_")[0]].items())
            rand_ent_idx = rng.choice(len(sub_ents), p=numpy.array(sub_probs))
            adapted_seq_ents[seq_ent_type] = sub_ents[rand_ent_idx]

    for seq_ent_type in {ent_type:adapted_ent for ent_type,adapted_ent in adapted_seq_ents.items() if not adapted_ent}:
        #import pdb;pdb.set_trace()
        adapted_seq_ents[seq_ent_type] = u'<UNK>' #as last resort, replace entity with UNK token

    seq = [adapted_seq_ents["_" .join(token.split("_")[1:])] if "_" .join(token.split("_")[1:]) in adapted_seq_ents else token for token in seq]
    return seq

def detokenize_seq(seq, ents=[]):
    '''use simple rules for transforming list of tokens back into string
    ents is optional list of words (named entities) that should be capitalized'''
    if type(seq) in (str, unicode):
        seq = tokenize(seq, lowercase=False)
    seq = [sent.split() for sent in segment(" ".join(seq))] #split sequence into sentences
    #print seq
    detok_seq = []
    for sent_idx, sent in enumerate(seq):

        assert(type(sent) in (list,tuple))

        if ents:
            token_idx = 0
            while token_idx < len(sent): #capitalize all tokens that appear in cap_ents
                for ent in ents:
                    ent = ent.split()
                    if sent[token_idx:token_idx + len(ent)] == [token.lower() for token in ent]:
                        # import pdb;pdb.set_trace()
                        sent[token_idx:token_idx + len(ent)] = list(ent)
                        token_idx += len(ent) - 1
                        break
                token_idx += 1


        # if type(detok_sent) in (tuple, list):
        detok_sent = " ".join(sent)

        #capitalize first-person "I" pronoun
        detok_sent = re.sub(" i ", " I ", detok_sent)

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
        detok_sent = re.sub("\' \'", "\'\'", detok_sent)
        detok_sent = re.sub("\` \`", "\`\`", detok_sent)

        #replace repeated single quotes with double quotation mark.
        detok_sent = re.sub("\'\'", "\"", detok_sent)
        detok_sent = re.sub("\`\`", "\"", detok_sent)

        #filter repetitive characters
        #detok_sent = re.sub("(\'\s*){3,}", "\'\'", detok_sent)
        #detok_sent = re.sub("(\`\s*){3,}", "\`\`", detok_sent)
        detok_sent = re.sub("([\"\']\s*){2,}", "\" ", detok_sent)
        #detok_sent = re.sub("(\.\s*){4,}", "...", detok_sent)


        punc_pairs = {"\'": "\'","\'": "\'", "`": "\'", "\"": "\"", "(": ")", "[": "]"} #map each opening puncutation mark to closing mark
        open_punc = []
        char_idx = 0
        while char_idx < len(detok_sent): #check for quotes and parenthesis
            char = detok_sent[char_idx]
            if open_punc and char == punc_pairs[open_punc[-1]]: #end quote/parenthesis
                if char_idx > 0 and detok_sent[char_idx-1] == " ":
                    detok_sent = detok_sent[:char_idx-1] + detok_sent[char_idx:]
                    open_punc.pop()
            elif char in punc_pairs:
                if char_idx < len(detok_sent) - 1 and detok_sent[char_idx + 1] == " ":
                    open_punc.append(char)
                    detok_sent = detok_sent[:char_idx + 1] + detok_sent[char_idx+2:]
            if char_idx < len(detok_sent) and detok_sent[char_idx] == char:
                char_idx += 1

        detok_sent = detok_sent.strip()
        for char_idx, char in enumerate(detok_sent): #capitalize first alphabetic character
            if char.isalpha():
                detok_sent = detok_sent[:char_idx + 1].upper() + detok_sent[char_idx + 1:]
                break
        detok_seq.append(detok_sent)

    detok_seq = " ".join(detok_seq)
    #print detok_seq
    return detok_seq

def filter_gen_seq(seq, n_sents=1):
    '''given a generated sequence, filter so that only the first n_sents are included in final generated sequence'''
    seq = [sent.split() for sent in segment(" ".join(seq))] #split into sentences
    seq = [word for sent in seq[:n_sents] for word in sent]
    return seq


def load_transformer(filepath, embeddings=None):
    #load saved models
    if not os.path.exists(filepath + "/transformer.pkl"):
        return None
    with open(filepath + '/transformer.pkl', 'rb') as f:
        transformer = pickle.load(f)
    transformer.word_embeddings = embeddings
    print 'loaded transformer with', transformer.lexicon_size, 'words from', str(filepath) + '/transformer.pkl'
    return transformer


class SkipthoughtsTransformer():
    def __init__(self, encoder_module=skipthoughts, encoder=None, encoder_dim=4800, verbose=True):
        self.encoder_module = encoder_module
        if not encoder:
            encoder = self.encoder_module.load_model("skip-thoughts-master")
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.verbose = verbose
    def fit(self, seqs):
    # def fit(self, X, y_seqs=None):
        #model is already fit
        return
    #def transform(self, X, y_seqs=None, X_filepath=None, y_seqs_filepath=None):
    def transform(self, seqs, seqs_filepath=None):
        # if y_seqs is not None:
        #     y_seqs = self.encode(y_seqs, y_seqs_filepath)
        #X = self.encode(X, X_filepath)
        seqs = self.encode(seqs, seqs_filepath)
        #return X, y_seqs
        return seqs
    def encode(self, seqs, seqs_filepath=None):
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
        if seqs_filepath:
            encoded_seqs = numpy.memmap(seqs_filepath, dtype='float64',
                                        mode='w+', shape=seqs_shape)
        else:
            encoded_seqs = numpy.zeros(seqs_shape)

        chunk_size = 500000
        for seq_idx in xrange(0, len(seqs), chunk_size):
            #memory errors if encoding a large number of stories
            encoded_seqs[seq_idx:seq_idx + chunk_size] = self.encoder_module.encode(self.encoder, 
                                                                                    seqs[seq_idx:seq_idx + chunk_size], verbose=self.verbose)
        
        if type(seq_length) in (list, tuple, numpy.ndarray):
            #different lengths per sequence
            idxs = [numpy.sum(seq_length[:idx]) for idx in xrange(len(seq_length))] + [None] #add -1 for last entry
            encoded_seqs = [encoded_seqs[idxs[start]:idxs[start+1]] for start in xrange(len(idxs) - 1)]
        else:
            encoded_seqs = encoded_seqs.reshape(n_seqs, seq_length, self.encoder_dim)
            
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




def resave_transformer_nosklearn(old_transformer):
    import pdb;pdb.set_trace()
    new_transformer = SequenceTransformer()
    #new_transformer.unk_word = old_transformer.unk_word
    new_transformer.pad_seq = old_transformer.pad_seq
    new_transformer.lexicon = old_transformer.lexicon
    new_transformer.lexicon_size = old_transformer.lexicon_size
    new_transformer.lexicon_lookup = old_transformer.lexicon_lookup
    if hasattr(old_transformer, 'word_counts'):
        new_transformer.word_counts = old_transformer.word_counts
    else:
        new_transformer.word_counts = {}
    new_transformer.min_freq = old_transformer.min_freq
    new_transformer.max_length = old_transformer.max_length
    new_transformer.verbose = old_transformer.verbose
    #new_transformer.embed_y = old_transformer.embed_y
    new_transformer.generalize_ents = old_transformer.generalize_ents
    new_transformer.reduce_emb_mode = old_transformer.reduce_emb_mode
    new_transformer.copy_input_to_output = old_transformer.copy_input_to_output
    new_transformer.filepath = old_transformer.filepath
    new_transformer.save()



class SequenceTransformer():#FunctionTransformer
    def __init__(self, min_freq=1, max_length=None, pad_seq=False, lexicon=[],
                    verbose=1, unk_word=u"<UNK>", word_embeddings=None, generalize_ents=False,
                    reduce_emb_mode=None, copy_input_to_output=False, filepath=None): #embed_y=False, 
        #FunctionTransformer.__init__(self)
        self.unk_word = unk_word #string representation for unknown words in lexicon
        #use existing word embeddings if given
        self.word_embeddings = word_embeddings
        if self.word_embeddings is not None:
            if type(self.word_embeddings) is not dict:
                #convert Word2Vec embeddings to dict
                self.word_embeddings = {word:self.word_embeddings[word] for word in self.word_embeddings.vocab}
            self.n_embedding_nodes = len(self.word_embeddings.values()[0])
        self.pad_seq = pad_seq
        self.lexicon = lexicon
        self.word_counts = {}
        self.min_freq = min_freq
        self.max_length = max_length
        self.verbose = verbose
        #specify if y_seqs should be converted to embeddings like input seqs
        # if embed_y:
        #     assert(self.word_embeddings is not None)
        #self.embed_y = embed_y
        self.generalize_ents = generalize_ents #specify if named entities should be replaced with generic labels
        self.ent_counts = {}
        self.reduce_emb_mode = reduce_emb_mode #specify if embeddings should be combined across sequence (e.g. take mean, sum)
        #self.copy_input_to_output = copy_input_to_output
        if self.verbose:
            print "created transformer with minimum word frequency =", self.min_freq, ", generalize named entities =", self.generalize_ents
        # if filepath and not os.path.isdir(filepath):
        #     os.mkdir(filepath)
        self.filepath = filepath

    def make_lexicon(self, seqs):

        self.lexicon = {} #regenerate lexicon everytime this function is called; word_counts will persist between calls
        self.lexicon[self.unk_word] = 1
        for seq in seqs:
            #first get named entities
            if self.generalize_ents:
                #reduce vocab by mapping all named entities to entity labels (e.g. "PERSON_0")
                for ent, ent_type in get_ents(seq).items(): #build a dictionary of entities that can be substituted when a generated entity isn't resolved
                    if ent_type not in self.ent_counts:
                        self.ent_counts[ent_type] = {}
                    if ent not in self.ent_counts[ent_type]:
                        self.ent_counts[ent_type][ent] = 1
                    else:
                        self.ent_counts[ent_type][ent] += 1
                seq = self.replace_ents_in_seqs([seq])[0]
            # else:
            seq = tokenize(seq)

            for word in seq:
                if word not in self.word_counts:
                    self.word_counts[word] = 1
                else:
                    self.word_counts[word] += 1

        for word, count in self.word_counts.items():
            if count >= self.min_freq or (self.generalize_ents and word.startswith("ENT_")): #if word is an entity, automatically include it in vocab; otherwise include word if it occurs at least min_freq times
                self.lexicon[word] = max(self.lexicon.values()) + 1

        self.lexicon_size = len(self.lexicon.keys())
        self.lexicon_lookup = [None] + [word for index, word in 
                                sorted([(index, word) for word, index in self.lexicon.items()])] #insert entry for empty timeslot in lexicon lookup
        assert(len(self.lexicon_lookup) == self.lexicon_size + 1)
        if self.filepath: #if filepath given, save transformer
            self.save()
        if self.verbose:
            print "added lexicon of", self.lexicon_size, "words with frequency >=", self.min_freq
    
    # def fit(self, seqs):
    #     #seqs = self.format_seqs(seqs=seqs, unravel=True)
    #     if not self.lexicon: #lexicon already given
    #         self.make_lexicon(seqs=seqs)


    #     if self.pad_seq:
    #         self.set_max_length(seqs=seqs)
    #     if self.filepath: #if filepath given, save transformer
    #         self.save()

    
    # def format_seqs(self, seqs, unravel=False):
    #     #get input and output into standard format
    #     if isinstance(seqs, (str, unicode)):
    #         #input is single string, put inside list
    #         seqs = [seqs]
            
    #     if isinstance(seqs[0], (str, unicode)):
    #         #put each string inside tuple
    #         seqs = [[seq] for seq in seqs]
        
    #     assert(type(seqs[0]) in [list, tuple])
        
    #     if unravel:
    #         seqs = [sent for seq in seqs for sent in seq]
        
    #     return seqs
    
    # def lookup_eos(self, eos_tokens=[".", "?", "!"]):
    #     #get indices of end of sentence markers (needed for generating with language model)
    #     eos_idxs = [self.lexicon[token] for token in eos_tokens if token in self.lexicon]
    #     return eos_idxs
        
    def fit_transform(self, seqs):
        self.fit(seqs)
        seqs = self.transform(seqs)
        return seqs

    def replace_ents_in_seqs(self, seqs):
        '''extract entities from seqs and replace them with their entity types'''
        replaced_seqs = []
        for seq in seqs:
            ents = number_ents(get_ents(seq))
            seq = tokenize(seq, lowercase=False, recognize_ents=True)
            seq = ['ENT_' + ents[word] if word in ents else word for word in seq]
            seq = " ".join(seq)
            replaced_seqs.append(seq)
        assert(len(seqs) == len(replaced_seqs))
        return replaced_seqs

    def tok_seq_to_nums(self, seq):
        seq = [self.lexicon[word] if word in self.lexicon else 1 if word else 0
                for word in seq] #map each token in list of tokens to an index; if word is None, replace with 0; if word is not None but not in lexicon, replace with 1
        return seq
    
    def text_to_nums(self, seqs):
        '''tokenize string sequences and convert to list of word indices'''
        #import pdb;pdb.set_trace()
        encoded_seqs = []
        for seq in seqs:
            # if self.generalize_ents: #reduce vocab by mapping all named entities to entity labels (e.g. "PERSON_0")
            #     seq = self.replace_ents(seq)
            seq = tokenize(seq)
            seq = self.tok_seq_to_nums(seq)
            encoded_seqs.append(seq)
        assert(len(seqs) == len(encoded_seqs))
        return encoded_seqs

    def num_seqs_to_counts(self, seqs):
        '''takes sequences of word indices as input and returns word count vectors'''
        count_vecs = []
        for seq in seqs:
            count_vec = numpy.bincount(numpy.array(seq), minlength=self.lexicon_size + 1)
            count_vecs.append(count_vec)
        count_vecs = numpy.array(count_vecs)
        count_vecs[:,0] = 0 #don't include 0s in vector (0's are words that are not part of context)
        return count_vecs
    
    def decode_num_seqs(self, seqs, n_sents_per_seq=None, detokenize=False, ents=[], capitalize_ents=False, adapt_ents=False):
        if type(seqs[0]) not in (list, numpy.ndarray, tuple):
            seqs = [seqs]
        decoded_seqs = []
        #transform numerical seq back into string
        for seq_idx, seq in enumerate(seqs):
            #import pdb;pdb.set_trace()
            seq = [self.lexicon_lookup[word] if self.lexicon_lookup[word] else self.unk_word for word in seq]
            if n_sents_per_seq: #if filter_n_sents is a number, filter generated sequence to only the first N=filter_n_sents sentences
                seq = filter_gen_seq(seq, n_sents=n_sents_per_seq)
            if ents and adapt_ents: #replace generated entities with those given in ents
                seq = adapt_seq_ents(seq, ents=ents[seq_idx], sub_ent_probs=ent_counts_to_probs(self.ent_counts))
            if detokenize: #apply rules for transforming token list into formatted sequence
                if ents and capitalize_ents:
                    seq = detokenize_seq(seq, ents=ents[seq_idx])#, named_ents=named_ents) #detokenize; pass a list of words that should be capitalized
                else:
                    seq = detokenize_seq(seq, ents=[])
            else:
                seq = " ".join(seq) #otherwise just join tokens with whitespace between each
            decoded_seqs.append(seq)
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

    # def pad_encoded_sents(self):
    #     import pdb;pdb.set_trace()
    #     seqs = [numpy.array([numpy.append(sent, numpy.zeros((self.max_length - len(sent), 
    #             self.lm_classifier.encoder_model.layers[-1].output_dim)), axis=0)
    #             for sent in seq]) for seq in seqs]
        
    #     seqs = numpy.array(seqs)
    #     return seqs

    def pad_embeddings(self, seqs):
        #import pdb;pdb.set_trace()
        assert(type(seqs[0]) is list and len(seqs[0][0].shape) == 2)
        #input sequences are a list of sentences
        seqs = [numpy.array([numpy.append(sent, numpy.zeros((self.max_length - len(sent),
                self.word_embeddings.n_embedding_nodes)), axis=0)
                for sent in seq]) for seq in seqs]
        
        seqs = numpy.array(seqs)
        return seqs
    
    # def remove_extra_dim(self, seqs):
    #     #if seqs have an extra dimension of one, flatten it
    #     if len(seqs[0]) == 1:
    #         if type(seqs) is numpy.ndarray:
    #             seqs = seqs[:, 0]
    #         else:
    #             seqs = [sent for seq in seqs for sent in seq]
    #     return seqs
        
    # def transform(self, seqs, seqs_filepath=None):
    #     #seqs = self.format_seqs(seqs=seqs)
    #     if self.generalize_ents:
    #         seqs = self.replace_ents_in_seqs(seqs)
    #     #seqs = [self.text_to_nums(seqs=seqs_) for seqs_ in seqs]
    #     seqs = self.text_to_nums(seqs)
    #     if self.word_embeddings:
    #         seqs = [self.embed_words(seqs=seqs_) for seqs_ in seqs]
    #         #combine embeddings if specified
    #         if self.reduce_emb_mode:
    #             seqs = self.reduce_embs(seqs=seqs)
    #         if self.pad_seq:
    #             #import pdb;pdb.set_trace()
    #             seqs = self.pad_embeddings(seqs=seqs)
    #     else:
    #         if self.pad_seq:
    #             seqs = self.pad_nums(seqs=seqs)
        #seqs = self.remove_extra_dim(seqs=seqs)

        # if seqs_filepath:
        #     numpy.save(seqs_filepath, seqs)
                 
        return seqs

    def seqs_to_feature_words(self, seqs, include_pos=('NOUN', 'PRON', 'PROPN')):
        '''input is sequences of where each sequence is a list of tokens where entities have already been replaced, if applicable;
        extract feature words from sequence; feature words are either named entities (i.e. has prefix 'ENT_') or whose pos tag is in include_pos;
        output will have sequence same length as original sequence but with None in places where word is not a context word'''

        feature_seqs = []
        for seq in seqs:
            seq_pos = [word.pos_ for word in encoder(seq)]
            seq = tokenize(seq)
            feature_seq = []
            for word, pos in zip(seq, seq_pos):
                if word.startswith('ENT_') or pos in include_pos:
                    feature_seq.append(word)
                else:
                    feature_seq.append(None)
            assert(len(feature_seq) == len(seq))
            feature_seqs.append(feature_seq)
        return feature_seqs
    
    def save(self):
        #save transformer to file
        if not os.path.isdir(self.filepath):
            os.mkdir(self.filepath)

        with open(self.filepath + '/transformer.pkl', 'wb') as f:
            pickle.dump(self, f)

        print "Saved transformer to", self.filepath + '/transformer.pkl'
        
    def __getstate__(self):
        #don't save embeddings
        state = dict((k, v) for (k, v) in self.__dict__.iteritems() if k not in ('word_embeddings'))
        state.update({'word_embeddings': None})
        return state

# class CountTransformer(SequenceTransformer):
#     '''transform sequences and then sequences into count (bag-of-words) vectors'''
#     # def fit_transform(self, seqs, lexicon=None):
#     #     self.fit(seqs)
#     #     return self.transform(seqs)
#     def transform(self, seqs):




