from __future__ import print_function
import timeit, numpy, pickle, os, copy
import theano
import theano.tensor as T
from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras.layers.merge import *
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional
from keras.optimizers import RMSprop, SGD, Adagrad, Adam
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
from scipy.spatial.distance import cosine

rng = numpy.random.RandomState(0)
theano_rng = T.shared_randomstreams.RandomStreams(123)


class SavedModel():
    def save(self):
        if not os.path.isdir(self.filepath):
            os.mkdir(self.filepath)
        
        self.model.save(self.filepath + '/classifier.h5')
        with open(self.filepath + '/classifier.pkl', 'wb') as f:
            pickle.dump(self, f)
        print("Saved", self.__class__.__name__, "to", self.filepath)
        
    def __getstate__(self):
        #don't save model itself with classifier object, it'll be saved separately as .h5 file
        attrs = self.__dict__.copy()
        if 'model' in attrs:
            del attrs['model']
        if 'pred_model' in attrs:
            del attrs['pred_model']
        if 'eval_model' in attrs:
            del attrs['eval_model']
        if 'encoder_model' in attrs:
            del attrs['encoder_model']
        if 'sample_words' in attrs:
            del attrs['sample_words']
        return attrs

    @classmethod
    def load(cls, filepath, custom_objects=None):
        with open(filepath + '/classifier.pkl', 'rb') as f:
            classifier = pickle.load(f)
        classifier.model = load_model(filepath + '/classifier.h5')
        print('loaded classifier from', filepath + '/classifier.pkl')
        return classifier

def get_seq_batch(seqs, batch_size=None, padding='pre', max_length=None, n_timesteps=None):
    if not max_length:
        max_length = max([len(seq) for seq in seqs])
    if n_timesteps:
        #if timesteps given, make sure length is divisible by it
        max_length = int(numpy.ceil(max_length * 1. / n_timesteps)) * n_timesteps + 1
    seqs = pad_sequences(sequences=seqs, maxlen=max_length, padding=padding)
    if batch_size and len(seqs) < batch_size:
        #import pdb;pdb.set_trace()
        #too few sequences for batch, so add extra rows
        batch_padding = numpy.zeros((batch_size - len(seqs), seqs.shape[1]), dtype='int64')
        seqs = numpy.append(seqs, batch_padding, axis=0)
    return seqs

def get_vector_batch(seqs, vector_length, binary_values=False):
    #import pdb;pdb.set_trace()
    '''takes sequences of word indices as input and returns word count vectors'''
    batch = []
    for seq in seqs:
        seq = numpy.bincount(numpy.array(seq), minlength=vector_length)
        batch.append(seq)
    batch = numpy.array(batch)
    batch[:,0] = 0
    if binary_values:
        batch[batch > 0] = 1 #store 1 if word occurred rather than total count
    return batch

def get_batch_features(features, batch_size=None):
    if batch_size and len(features) < batch_size:
        #too few sequences for batch, so add extra rows
        batch_padding = numpy.zeros((batch_size - len(features), features.shape[1]), dtype='int64')
        features = numpy.append(features, batch_padding, axis=0)
    return features

def get_sort_order(seqs):
    #return indices that will sort sequences by length
    lengths = [len(seq) for seq in seqs]
    sorted_idxs = numpy.argsort(lengths)
    return sorted_idxs

def batch_seqs_to_list(batch_seqs, len_batch, batch_size):
    '''convert sequences from padded array back to list'''
    if len_batch < batch_size:
        batch_seqs = batch_seqs[:len_batch] #remove padding if batch was padded
    batch_seqs = batch_seqs.tolist()
    return batch_seqs

class LogisticRegressionClassifier():
    def __init__(self, n_output_classes, batch_size=100, verbose=True):
        
        self.n_output_classes = n_output_classes
        self.batch_size = batch_size
        self.verbose = verbose
        self.n_input_nodes = None
        
    def create_model(self):
        model = Sequential()
        model.add(Dense(output_dim=self.n_output_classes, input_dim=self.n_input_nodes, activation='sigmoid', name='output_layer'))
        model.add(Activation('softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, seqs, labels, n_input_nodes, n_epochs=1):
        if not hasattr(self, 'model'):
            assert(n_input_nodes is not None)
            self.n_input_nodes = n_input_nodes
            self.model = self.create_model()
            if self.verbose:
                print("Created model", self.__class__.__name__, ":", self.__dict__)
        if type(seqs) in (list, tuple): #input may already be transformered into matrix, if not, transform
            seqs = get_vector_batch(seqs, vector_length=self.n_input_nodes)
        self.model.fit(seqs, labels, batch_size=self.batch_size, nb_epoch=n_epochs, verbose=self.verbose)

    def predict(self, seqs):
        if type(seqs) in (list, tuple):
            seqs = get_vector_batch(seqs, self.n_input_nodes)
        probs = self.model.predict(seqs, batch_size=self.batch_size)
        return probs



class MLPClassifier():
    def __init__(self, n_output_classes, n_hidden_nodes=200, n_hidden_layers=1, batch_size=100, verbose=True):
        
        self.n_output_classes = n_output_classes
        self.n_hidden_nodes = n_hidden_nodes
        self.n_hidden_layers = n_hidden_layers
        self.batch_size = batch_size
        self.verbose = verbose
        self.n_input_nodes = None
        
    def create_model(self):
        model = Sequential()
        for layer_num in range(self.n_hidden_layers):
            if layer_num == 0:
                model.add(Dense(output_dim=self.n_hidden_nodes, input_dim=self.n_input_nodes, activation='tanh', name='hidden_layer' + str(layer_num + 1)))
            else:
                model.add(Dense(output_dim=self.n_hidden_nodes, activation='tanh', name='hidden_layer' + str(layer_num + 1)))
        model.add(Dense(output_dim=self.n_output_classes, activation='softmax', name='output_layer'))      
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, seqs, labels, n_input_nodes, n_epochs=1):
        if not hasattr(self, 'model'):
            assert(n_input_nodes is not None)
            self.n_input_nodes = n_input_nodes
            self.model = self.create_model()
            if self.verbose:
                print("Created model", self.__class__.__name__, ":", self.__dict__)
        if type(seqs) in (list, tuple): #input may already be transformered into matrix, if not, transform
            seqs = get_vector_batch(seqs, vector_length=self.n_input_nodes) #pad seqs
        self.model.fit(seqs, labels, batch_size=self.batch_size, nb_epoch=n_epochs, verbose=self.verbose)

    def predict(self, seqs):
        if type(seqs) in (list, tuple): #input may already be transformered into matrix, if not, transform
            seqs = get_vector_batch(seqs, self.n_input_nodes) #pad seqs
        probs = self.model.predict(seqs, batch_size=self.batch_size)
        return probs


class RNNClassifier():
    def __init__(self,  n_output_classes, n_embedding_nodes=100, n_hidden_nodes=200, n_hidden_layers=1, batch_size=100, verbose=True):#, emb_weights=None, layer1_weights=None):
        self.n_output_classes = n_output_classes
        self.n_embedding_nodes = n_embedding_nodes
        self.n_hidden_nodes = n_hidden_nodes
        self.n_hidden_layers = n_hidden_layers
        self.batch_size = batch_size
        self.verbose = verbose
        self.lexicon_size = None
        self.n_timesteps = None
    def create_model(self):
        model = Sequential()
        model.add(Embedding(output_dim=self.n_embedding_nodes, input_dim=self.lexicon_size + 1,
                            input_length=self.n_timesteps, mask_zero=True, name='embedding_layer'))
        for layer_num in range(self.n_hidden_layers):
            if layer_num == self.n_hidden_layers - 1:
                return_sequences = False
            else: #add extra hidden layers
                return_sequences = True
            model.add(GRU(output_dim=self.n_hidden_nodes, return_sequences=return_sequences, name='hidden_layer' + str(layer_num + 1)))
        model.add(Dense(output_dim=self.n_output_classes, activation='softmax', name='output_layer'))
        # if emb_weights is not None:
        #     #initialize weights with lm weights
        #     model.layers[0].set_weights(emb_weights) #set embeddings
        # if layer1_weights is not None:
        #     model.layers[1].set_weights(layer1_weights) #set recurrent layer 1         
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    def fit(self, seqs, labels, lexicon_size, n_timesteps=None, n_epochs=1):
        #import pdb;pdb.set_trace()
        if not hasattr(self, 'model'):
            assert(lexicon_size is not None)
            self.lexicon_size = lexicon_size
            self.n_timesteps = n_timesteps
            if not self.n_timesteps:
                self.n_timesteps = max([len(seq) for seq in seqs]) #if n_timesteps not given, set it to length of longest sequence
            self.model = self.create_model()
            if self.verbose:
                print("Created model", self.__class__.__name__, ":", self.__dict__)
        seqs = get_seq_batch(seqs, max_length=self.n_timesteps) #pad seqs
        self.model.fit(seqs, labels, batch_size=self.batch_size, nb_epoch=n_epochs, verbose=self.verbose)
    def predict(self, seqs):
        #import pdb;pdb.set_trace()
        seqs = get_seq_batch(seqs, max_length=self.n_timesteps) #pad seqs
        probs = self.model.predict(seqs, batch_size=self.batch_size, verbose=self.verbose)
        return probs


class RNNLM(SavedModel):
    def __init__(self, use_features=False, use_pos=False, lexicon_size=None, n_pos_tags=None, n_timesteps=15, n_embedding_nodes=300, n_pos_embedding_nodes=25,
                n_pos_nodes=100, n_feature_nodes=100, n_hidden_nodes=250, n_hidden_layers=1, embeddings=None, batch_size=1, verbose=1, filepath=None, optimizer='Adam',
                lr=0.001, clipvalue=5.0, decay=1e-6):
        
        self.lexicon_size = lexicon_size
        self.n_pos_tags = n_pos_tags
        self.n_embedding_nodes = n_embedding_nodes
        self.n_pos_embedding_nodes = n_pos_embedding_nodes
        self.n_pos_nodes = n_pos_nodes
        self.n_feature_nodes = n_feature_nodes
        self.n_hidden_nodes = n_hidden_nodes
        self.n_hidden_layers = n_hidden_layers
        self.batch_size = batch_size
        self.verbose = verbose
        self.n_timesteps = n_timesteps
        self.filepath = filepath
        self.embeddings = embeddings
        self.optimizer = optimizer
        self.lr = lr
        self.clipvalue = clipvalue
        self.decay = decay
        self.use_features = use_features
        self.use_pos = use_pos
        self.sample_words = None

        if self.verbose:
            print("Created model", self.__class__.__name__, ":", self.__dict__)
    
    def create_model(self, n_timesteps=None, batch_size=1, include_pred_layer=True):

        input_layers = []

        seq_input_layer = Input(batch_shape=(batch_size, n_timesteps), name="seq_input_layer")
        input_layers.append(seq_input_layer)

        seq_embedding_layer = Embedding(input_dim=self.lexicon_size + 1, 
                                        output_dim=self.n_embedding_nodes, mask_zero=True, name='seq_embedding_layer')(seq_input_layer)

        for layer_num in range(self.n_hidden_layers):
            if layer_num == 0:
                seq_hidden_layer = GRU(output_dim=self.n_hidden_nodes, return_sequences=True, stateful=True, name='seq_hidden_layer1')(seq_embedding_layer)
            else: #add extra hidden layers
                seq_hidden_layer = GRU(output_dim=self.n_hidden_nodes, return_sequences=True, stateful=True, name='seq_hidden_layer' + str(layer_num + 1))(seq_hidden_layer)

        if self.use_pos:
            pos_input_layer = Input(batch_shape=(batch_size, n_timesteps), name="pos_input_layer")
            input_layers.append(pos_input_layer)

            pos_embedding_layer = Embedding(input_dim=self.n_pos_tags + 1,
                                            output_dim=self.n_pos_embedding_nodes, mask_zero=True, name='pos_embedding_layer')(pos_input_layer)

            pos_hidden_layer = GRU(output_dim=self.n_pos_nodes, return_sequences=True, stateful=True, name='pos_hidden_layer')(pos_embedding_layer)

            seq_hidden_layer = merge([seq_hidden_layer, pos_hidden_layer], mode='concat', concat_axis=-1, name='pos_merge_hidden_layer')

        if self.use_features:
            feature_input_layer = Input(batch_shape=(batch_size, self.lexicon_size + 1), name="feature_input_layer")
            input_layers.append(feature_input_layer)
            feature_hidden_layer = Dense(output_dim=self.n_feature_nodes, activation='sigmoid', name='feature_hidden_layer')(feature_input_layer)
            feature_hidden_layer = RepeatVector(n_timesteps)(feature_hidden_layer)

            seq_hidden_layer = merge([seq_hidden_layer, feature_hidden_layer], mode='concat', concat_axis=-1, name='feature_merge_hidden_layer')

        output_layers = []
        if include_pred_layer:
            pred_layer = TimeDistributed(Dense(self.lexicon_size + 1, activation="softmax", name='pred_layer'))(seq_hidden_layer)
            output_layers.append(pred_layer)
            if self.use_pos:
                pred_pos_layer = TimeDistributed(Dense(self.n_pos_tags + 1, activation="softmax", name='pred_pos_layer'))(seq_hidden_layer)
                output_layers.append(pred_pos_layer)

        model = Model(input=input_layers, output=output_layers)
            
        #select optimizer and compile
        model.compile(loss="sparse_categorical_crossentropy", 
                      optimizer=eval(self.optimizer)(clipvalue=self.clipvalue, lr=self.lr, decay=self.decay))
                
        return model

    def fit(self, seqs, pos_seqs=None, feature_vecs=None, lexicon_size=None):
        
        if not hasattr(self, 'model'):
            assert(lexicon_size is not None)
            self.lexicon_size = lexicon_size
            self.model = self.create_model(n_timesteps=self.n_timesteps, batch_size=self.batch_size)
            self.start_time = timeit.default_timer()

        if self.verbose:
            print("training RNNLM on {} sequences with batch size = {}".format(len(seqs), self.batch_size))
        
        train_losses = []

        assert(type(seqs[0][0]) not in (list, tuple, numpy.ndarray))
        sorted_idxs = get_sort_order(seqs)
        seqs = [seqs[idx] for idx in sorted_idxs] #sort seqs by length
        if self.use_pos:
            assert(len(seqs) == len(pos_seqs))
            # assert(numpy.all(numpy.array(len(seq) == len(pos_seq)) for seq, pos_seq in zip(seqs, pos_seqs)))
            pos_seqs = [pos_seqs[idx] for idx in sorted_idxs]
        if self.use_features:
            assert(len(seqs) == len(feature_vecs))
            feature_vecs = feature_vecs[sorted_idxs]
        for batch_index in range(0, len(seqs), self.batch_size):
            batch_inputs = []
            batch_outputs = []
            batch = get_seq_batch(seqs=seqs[batch_index:batch_index + self.batch_size], batch_size=self.batch_size, n_timesteps=self.n_timesteps) #prep batch
            if self.use_pos:
                batch_pos = get_seq_batch(seqs=pos_seqs[batch_index:batch_index + self.batch_size], batch_size=self.batch_size, n_timesteps=self.n_timesteps)
            if self.use_features:
                batch_features = get_batch_features(features=feature_vecs[batch_index:batch_index + self.batch_size], batch_size=self.batch_size)
                batch_inputs.append(batch_features)
            for step_index in range(0, batch.shape[-1] - 1, self.n_timesteps):
                batch_x = batch[:, step_index:step_index + self.n_timesteps]
                if not numpy.sum(batch_x):
                    train_loss = 0
                    continue #batch is all zeros, skip
                batch_y = batch[:, step_index + 1:step_index + self.n_timesteps + 1, None]
                batch_inputs = [batch_x]
                batch_outputs = [batch_y]
                if self.use_pos:
                    batch_pos_x = batch_pos[:, step_index:step_index + self.n_timesteps]
                    batch_pos_y = batch_pos[:, step_index + 1:step_index + self.n_timesteps + 1, None]
                    if not numpy.sum(batch_pos_x):
                        import pdb;pdb.set_trace()
                    batch_inputs.append(batch_pos_x)
                    batch_outputs.append(batch_pos_y)
                if self.use_features:
                    batch_inputs.append(batch_features)
                train_loss = self.model.train_on_batch(x=batch_inputs, y=batch_outputs)
                train_losses.append(train_loss)
            self.model.reset_states()                
            if batch_index and batch_index % 5000 == 0:
                print("processed {} sequences, loss: {:.3f} ({:.3f}m)...".format(batch_index, numpy.mean(train_losses),
                                                                                (timeit.default_timer() - self.start_time) / 60))

        if self.filepath:
            self.save() #save model if filepath given
        if self.verbose:
            print("loss: {:.3f} ({:.3f}m)".format(numpy.mean(train_losses),
                                                (timeit.default_timer() - self.start_time) / 60))

    def get_batch_p_next_words(self, words, pos=None, features=None):
        #self.check_pred_model(len(words))
        batch_inputs = [words[:, None]]
        if self.use_pos:
            assert(pos is not None)
            batch_inputs.append(pos[:, None])
        if self.use_features:
            assert(features is not None)
            batch_inputs.append(features)
        if self.use_pos:
            p_next_words, _ = self.pred_model.predict_on_batch(x=batch_inputs)
            p_next_words = p_next_words[:, -1]
        else:
            p_next_words = self.pred_model.predict_on_batch(x=batch_inputs)[:, -1]
        return p_next_words

    def pred_batch_next_words(self, p_next_words, mode='max', n_best=1, temp=1.0, prevent_unk=True):

        def sample_word(p_next_word):
            word = theano_rng.choice(size=(n_best,), a=T.arange(p_next_word.shape[0]), replace=True, p=p_next_word, dtype='int64')
            return word

        def init_sample_words(temp):
            #initilize theano function for random sampling
            Temp = T.scalar()
            P_Next_Words = T.matrix('p_next_words')#, dtype='float64')
            P_Adj_Next_Words = T.nnet.softmax(T.log(P_Next_Words) / Temp)
            Next_Words, Updates = theano.scan(fn=sample_word, sequences=P_Adj_Next_Words)
            sample_words = theano.function([P_Next_Words, Temp], Next_Words, updates=Updates)#, allow_input_downcast=True)
            return sample_words

        if prevent_unk: #prevent model from generating unknown words by redistributing probability; assumes indices 0 and 1 are unknown words (0s are padding, 1 is explicit unknown word)
            #import pdb;pdb.set_trace()
            p_padding = p_next_words[:,0]
            p_unk = p_next_words[:,1]
            added_p = ((p_padding + p_unk) / p_next_words[:,2:].shape[-1])[:,None]
            p_next_words[:,2:] = p_next_words[:,2:] + added_p
            p_next_words[:,0] = 0.0
            p_next_words[:,1] = 0.0

        if mode == 'random':
            #numpy is too slow at random sampling, so use theano
            if not hasattr(self, 'sample_words') or not self.sample_words:
                self.sample_words = init_sample_words(temp)
            next_words = self.sample_words(p_next_words, temp)
        else:
            next_words = numpy.argmax(p_next_words, axis=1)[:,None]

        p_next_words = p_next_words[numpy.arange(len(p_next_words))[:,None], next_words]

        #for right now samples will always be size 1
        next_words = next_words[:, 0]
        p_next_words = p_next_words[:, 0]
        return next_words, p_next_words

    def extend_seq(self, seq, words):
        #extend sequence with each predicted word
        new_seqs = []
        for word in words:
            new_seq = seq + [word]
            new_seqs.append(new_seq)
        return new_seqs

    def check_if_null(self, seq):
        if seq[-1] == 0:
            return True
        return False

    def check_pred_model(self, batch_size):
        '''check if predictor (generation) model exists; if not, create it; n_timesteps will always be 1 since generating one word at a time'''

        if not hasattr(self, 'pred_model') or batch_size != self.pred_model.layers[0].batch_input_shape[0]:
            self.pred_model = self.create_model(batch_size=batch_size, n_timesteps=1)
            if self.verbose:
                print("created predictor model")

        self.pred_model.set_weights(self.model.get_weights()) #transfer weights from trained model
        
    def predict(self, seqs, feature_vecs=None, max_length=35, mode='max', batch_size=1, n_best=1, temp=1.0, prevent_unk=True):
        '''this function cannot be used if use_pos == True; use predict_with_pos() in pipeline class instead'''

        pred_seqs = []

        for batch_index in range(0, len(seqs), batch_size):
            batch_features = None
            batch_seqs = get_seq_batch(seqs=seqs[batch_index:batch_index + batch_size], batch_size=batch_size) #prep batch
            if self.use_features:
                batch_features = get_batch_features(features=feature_vecs[batch_index:batch_index + batch_size], batch_size=batch_size)

            self.read_batch(seqs=batch_seqs, features=batch_features) #batch_pos[:,:-1] if self.use_pos else None,

            batch_pred_seqs = numpy.zeros((batch_size, max_length), dtype='int64')

            # p_next_words = init_p_next_words
            p_next_words = self.get_batch_p_next_words(words=batch_seqs[:,-1], features=batch_features) #pos=batch_pos[:,-1] if self.use_pos else None

            for idx in range(max_length): #now predict
                next_words, p_next_words = self.pred_batch_next_words(p_next_words, mode, n_best, temp, prevent_unk)
                batch_pred_seqs[:, idx] = next_words
                p_next_words = self.get_batch_p_next_words(words=batch_pred_seqs[:, idx], features=batch_features)

            self.pred_model.reset_states()

            batch_pred_seqs = batch_seqs_to_list(batch_seqs=batch_pred_seqs, len_batch=len(seqs[batch_index:batch_index + batch_size]), batch_size=batch_size)
            pred_seqs.extend(batch_pred_seqs)

            if batch_index and batch_index % 1000 == 0:
                print("generated new sequences for {}/{} inputs...".format(batch_index, len(seqs)))

        return pred_seqs

    def read_batch(self, seqs, pos=None, features=None):
        '''will read all words in sequence up until last word'''

        self.check_pred_model(len(seqs))

        p_next_words = numpy.zeros((seqs.shape[0], seqs.shape[-1] - 1))

        for idx in range(seqs.shape[-1] - 1): #read in given sequence from which to predict
            p_next_words[:, idx] = self.get_batch_p_next_words(words=seqs[:, idx], pos=pos[:, idx] 
                                                            if self.use_pos else None, features=features)[numpy.arange(len(seqs)), seqs[:, idx+1]]

        return p_next_words

    def get_probs(self, seqs, pos_seqs=None, feature_vecs=None, batch_size=1, return_word_probs=False):
        '''return log probabilities computed by model for each word in each sequence'''

        probs = []

        for batch_index in range(0, len(seqs), batch_size):
            batch_features = None
            batch_pos = None
            batch_seqs = get_seq_batch(seqs=seqs[batch_index:batch_index + batch_size], batch_size=batch_size) #prep batch
            if self.use_pos:
                batch_pos = get_seq_batch(seqs=pos_seqs[batch_index:batch_index + batch_size], batch_size=batch_size)
            if self.use_features:
                batch_features = get_batch_features(features=feature_vecs[batch_index:batch_index + batch_size], batch_size=batch_size)

            p_next_words = self.read_batch(seqs=batch_seqs, pos=batch_pos if self.use_pos else None, features=batch_features)
            p_next_words = numpy.log(p_next_words)
            len_seqs = [len(seq) for seq in seqs[batch_index:batch_index + batch_size]]
            #remove padding from each sequence and before computing mean
            p_next_words = [p_next_words_[-len_seq:] for p_next_words_, len_seq in zip(p_next_words, len_seqs)]
            probs.extend(p_next_words)

            self.pred_model.reset_states()

            if batch_index and batch_index % 1000 == 0:
                print("computed probabilities for {}/{} sequences...".format(batch_index, len(seqs)))

        if not return_word_probs: #return overall probability of sequence instead of probs for each word
            probs = numpy.array([numpy.sum(prob_words) for prob_words in probs])

        return probs


    def create_encoder(self):

        self.encoder_model = self.create_model(n_timesteps=None, batch_size=1, pred_layer=False)
        self.encoder_model.set_weights(self.model.get_weights()[:-2])
        if self.verbose:
            print("created encoder model")

    def get_embeddings(self):

        embeddings = self.model.get_weights()[0]
        return embeddings

class MLPBinaryClassifier(SavedModel):
    def __init__(self, n_embedding_nodes=300, batch_size=100, 
                 n_hidden_layers=1, n_hidden_nodes=500, verbose=1, optimizer='Adam',
                 filepath=None, use_dropout=False):
        
        self.batch_size = batch_size
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_nodes = n_hidden_nodes
        self.n_embedding_nodes = n_embedding_nodes
        self.verbose = verbose
        self.use_dropout = use_dropout
        self.optimizer = optimizer
        self.filepath = filepath
        self.lexicon_size = None
        self.embedded_input = None

    def create_model(self):

        # if self.embedded_input:
        #     seq1_layer = Input(shape=(self.n_timesteps, self.n_embedding_nodes), name="seq1_layer")
        #     seq2_layer = Input(shape=(self.n_timesteps, self.n_embedding_nodes), name="seq2_layer")
        #     mask_layer = Masking(mask_value=0.0, name='mask_layer')
        #     seq1_mask_layer = mask_layer(seq1_layer)
        #     seq2_mask_layer = mask_layer(seq2_layer)
        #     prev_seq1_layer = seq1_mask_layer
        #     prev_seq2_layer = seq2_mask_layer
        # else:
        #emb_layer = Embedding(self.lexicon_size + 1, self.n_embedding_nodes, mask_zero=True, name='emb_layer')
        seq1_layer = Input(shape=(self.lexicon_size + 1,), name="seq1_layer")
        #seq1_emb_layer = emb_layer(seq1_layer)
        seq2_layer = Input(shape=(self.lexicon_size + 1,), name="seq2_layer")
        #seq2_emb_layer = emb_layer(seq2_layer)
        prev_seq1_layer = seq1_layer
        prev_seq2_layer = seq2_layer

        for layer_idx in range(self.n_hidden_layers):
            # if layer_idx == self.n_hidden_layers - 1:
            #     return_sequences = False
            seq1_hidden_layer = Dense(output_dim=self.n_hidden_nodes, name='seq1_hidden_layer', activation='tanh')(prev_seq1_layer)
            seq2_hidden_layer = Dense(output_dim=self.n_hidden_nodes, name='seq2_hidden_layer', activation='tanh')(prev_seq2_layer)

        merge_layer = merge([seq1_hidden_layer, seq2_hidden_layer], mode='concat', concat_axis=-1, name='merge_layer')
        dense_layer = Dense(output_dim=self.n_hidden_nodes, name='dense_layer', activation='tanh')(merge_layer)
        pred_layer = Dense(output_dim=1, name='pred_layer', activation='sigmoid')(dense_layer)
        model = Model(input=[seq1_layer, seq2_layer], output=pred_layer)
        model.compile(loss='binary_crossentropy', optimizer='adam')#, metrics=['accuracy'])
        return model

    # def get_batch(self, seqs):
    #     '''takes sequences of word indices as input and returns word count vectors'''
    #     batch = []
    #     for seq in seqs:
    #         seq = numpy.bincount(numpy.array(seq), minlength=self.lexicon_size + 1)
    #         batch.append(seq)
    #     batch = numpy.array(batch)
    #     batch[:,0] = 0
    #     return batch
    
    def fit(self, seqs1, seqs2, labels, lexicon_size=None, embedded_input=False, n_epochs=1):

        if not hasattr(self, 'model'):
            # self.embedded_input = embedded_input
            # if not self.embedded_input:
            assert(lexicon_size is not None)
            self.lexicon_size = lexicon_size
            self.model = self.create_model()
            print("Created model", self.__class__.__name__, ":", self.__dict__)

        assert(len(seqs1) == len(seqs2))

        #self.model.fit(x=[seqs1, seqs2], y=labels, batch_size=self.batch_size, nb_epoch=n_epochs)

        # shuffle instances
        random_idxs = rng.permutation(len(seqs1))
        seqs1 = [seqs1[idx] for idx in random_idxs]
        seqs2 = [seqs2[idx] for idx in random_idxs]
        labels = labels[random_idxs]

        for epoch in range(n_epochs):
            losses = []
            print("EPOCH:", epoch + 1)
            for batch_idx in range(0, len(seqs1), self.batch_size):
                batch_seqs1 = get_vector_batch(seqs1[batch_idx:batch_idx+self.batch_size], vector_length=self.lexicon_size + 1)
                batch_seqs2 = get_vector_batch(seqs2[batch_idx:batch_idx+self.batch_size], vector_length=self.lexicon_size + 1)
                batch_labels = labels[batch_idx:batch_idx+self.batch_size]
                losses.append(self.model.train_on_batch([batch_seqs1, batch_seqs2], batch_labels))
                if batch_idx and batch_idx % (self.batch_size * 1000) == 0:
                    print("loss: {:.3f}, accuracy: {:.3f}".format(numpy.mean(numpy.array(losses)[:,0]), numpy.mean(numpy.array(losses)[:,1])))
            print("loss: {:.3f}, accuracy: {:.3f}".format(numpy.mean(numpy.array(losses)[:,0]), numpy.mean(numpy.array(losses)[:,1])))

        if self.filepath:
            self.save()

    def predict(self, seq1, seq2):
        '''return score for likelihood of output seq given input seq'''

        prob = self.model.predict([seq1, seq2])[0][0]
        return prob

class RNNBinaryClassifier(SavedModel):
    def __init__(self, n_embedding_nodes=300, batch_size=100, embedded_input=True, n_input_sents=1,
                 n_hidden_layers=1, n_hidden_nodes=500, filepath=None):#, use_dropout=False):
        
        self.batch_size = batch_size
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_nodes = n_hidden_nodes
        self.n_embedding_nodes = n_embedding_nodes
        # self.verbose = verbose
        # self.use_dropout = use_dropout
        # self.optimizer = optimizer
        self.filepath = filepath
        self.embedded_input = embedded_input
        self.n_input_sents = n_input_sents
        # self.n_timesteps = None
        self.lexicon_size = None

    def create_model(self, ranking=False, use_dropout=False):

        ####################CURRENT METHOD######################
        context_input_layer = Input(batch_shape=(self.batch_size, self.n_input_sents, self.n_embedding_nodes), name="context_input_layer")

        seq_input_layer = Input(batch_shape=(self.batch_size, 1, self.n_embedding_nodes), name="seq_input_layer")

        merge_layer = merge([context_input_layer, seq_input_layer], mode='concat', concat_axis=-2, name='merge_layer')

        mask_layer = Masking(mask_value=0.0, input_shape=(self.n_input_sents + 1, self.n_embedding_nodes))(merge_layer)

        hidden_layer = GRU(output_dim=self.n_hidden_nodes, return_sequences=False, stateful=False, name='context_hidden_layer')(mask_layer)#(merge_layer)

        pred_layer = Dense(output_dim=1, activation='sigmoid', name='pred_layer')(hidden_layer)

        model = Model(input=[context_input_layer, seq_input_layer], output=pred_layer)
        ##########################################################

        ####################ALTERNATIVE METHOD######################
        # context_input_layer = Input(batch_shape=(self.batch_size, self.context_size, self.n_embedding_nodes), name="context_input_layer")

        # context_hidden_layer = GRU(output_dim=self.n_hidden_nodes, return_sequences=False, stateful=False, name='context_hidden_layer')(context_input_layer)

        # seq_input_layer = Input(batch_shape=(self.batch_size, self.n_embedding_nodes), name="seq_input_layer")

        # seq_hidden_layer = Dense(output_dim=self.n_hidden_nodes, activation='tanh', name='seq_hidden_layer')(seq_input_layer)

        # merge_layer = merge([context_hidden_layer, seq_hidden_layer], mode='dot', concat_axis=-1, name='merge_layer')

        # sigmoid_layer = Activation('sigmoid')(merge_layer)

        # self.model = Model(input=[context_input_layer, seq_input_layer], output=sigmoid_layer)
        ##########################################################

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def fit(self, seqs1, seqs2, labels, lexicon_size=None, n_epochs=1, save_to_filepath=False):

        if not hasattr(self, 'model'):
            if not self.embedded_input:
                assert(lexicon_size is not None)
                self.lexicon_size = lexicon_size
            self.model = self.create_model()
            print("Created model", self.__class__.__name__, ":", self.__dict__)

        assert(len(seqs1) == len(seqs2) == len(labels))

        for epoch in range(n_epochs):
            losses = []
            if n_epochs > 1:
                print("EPOCH:", epoch + 1)
            for batch_idx in range(0, len(seqs1), self.batch_size):
                batch_seqs1 = numpy.array(seqs1[batch_idx:batch_idx+self.batch_size])
                batch_seqs2 = numpy.array(seqs2[batch_idx:batch_idx+self.batch_size])
                batch_labels = labels[batch_idx:batch_idx+self.batch_size]
                losses.append(self.model.train_on_batch(x=[batch_seqs1, batch_seqs2], y=batch_labels))
                if batch_idx and batch_idx % (self.batch_size * 1000) == 0:
                    print("loss: {:.7f}".format(numpy.mean(numpy.array(losses))))
            print("loss: {:.7f}".format(numpy.mean(numpy.array(losses))))

            if save_to_filepath:
                self.save()

    def predict(self, seq1, seq2):
        '''return score for likelihood of seq1 given seq2'''

        prob = self.model.predict(x=[numpy.array(seq1)[None], numpy.array(seq2)[None]])[0][0]
        return prob

    @classmethod
    def load(cls, filepath):
        with open(filepath + '/classifier.pkl', 'rb') as f:
            classifier = pickle.load(f)
        classifier.model = load_model(filepath + '/classifier.h5', custom_objects={'ranking_loss':classifier.ranking_loss})
        print('loaded classifier from', filepath + '/classifier.pkl')
        return classifier

class MLPLM(SavedModel):
    def __init__(self, n_timesteps, lexicon_size=None, n_embedding_nodes=300, n_hidden_nodes=500, n_hidden_layers=1, 
                 embeddings=None, batch_size=1, verbose=1, filepath=None, optimizer='Adam', lr=0.001, clipvalue=5.0, decay=1e-6):
        
        self.n_timesteps = n_timesteps
        self.lexicon_size = lexicon_size
        self.n_embedding_nodes = n_embedding_nodes
        self.n_hidden_nodes = n_hidden_nodes
        self.n_hidden_layers = n_hidden_layers
        self.batch_size = batch_size
        self.verbose = verbose
        self.filepath = filepath
        self.embeddings = embeddings
        self.optimizer = optimizer
        self.lr = lr
        self.clipvalue = clipvalue
        self.decay = decay
        self.sample_words = None

        if self.verbose:
            print("CREATED MLPLM: embedding layer nodes = {}, hidden layers = {}, " \
                    "hidden layer nodes = {}, optimizer = {} with lr = {}, " \
                    "clipvalue = {}, and decay = {}".format(
                    self.n_embedding_nodes, self.n_hidden_layers, self.n_hidden_nodes, 
                    self.optimizer, self.lr, self.clipvalue, self.decay))
    
    def create_model(self, n_timesteps, batch_size=1, pred_layer=True):
        
        model = Sequential()
        
        # if self.embeddings is None:
        model.add(Embedding(self.lexicon_size + 1, self.n_embedding_nodes,
                            batch_input_shape=(batch_size, n_timesteps)))#, mask_zero=True))

        model.add(Reshape((self.n_embedding_nodes * n_timesteps,)))

        for layer_num in range(self.n_hidden_layers):
            model.add(Dense(self.n_hidden_nodes, batch_input_shape=(batch_size, n_timesteps, self.n_embedding_nodes), activation='tanh'))

        if pred_layer: 
            model.add(Dense(self.lexicon_size + 1, activation="softmax"))
            
        #select optimizer and compile
        model.compile(loss="sparse_categorical_crossentropy", 
                      optimizer=eval(self.optimizer)(clipvalue=self.clipvalue, lr=self.lr, decay=self.decay))
                
        return model

    def fit(self, seqs, lexicon_size=None, n_epochs=5):
        if not hasattr(self, 'model'):
            assert(lexicon_size is not None)
            self.lexicon_size = lexicon_size
            self.model = self.create_model(n_timesteps=self.n_timesteps, batch_size=self.batch_size)
            self.start_time = timeit.default_timer()

        assert(type(seqs[0][0]) not in [list, tuple, numpy.ndarray])

        X = [[seq[idx:idx+self.n_timesteps+1] for idx in range(len(seq) - self.n_timesteps)] for seq in seqs]
        X = numpy.array([ngram for seq in X for ngram in seq])
        y = X[:, -1][:, None]
        X = X[:, :-1]

        train_loss = self.model.fit(X, y, nb_epoch=n_epochs, batch_size=self.batch_size)

        if self.filepath:
            #save model after each epoch if filepath given
            self.save()
        #self.epoch += 1
        if self.verbose:
            print("loss: {:.3f} ({:.3f}m)".format(numpy.mean(train_loss.history['loss']),
                                       (timeit.default_timer() - self.start_time) / 60))

    def pred_next_words(self, p_next_words, mode='max', n_best=1, temp=1.0, prevent_unk=True):

        def sample_word(p_next_word):
            word = theano_rng.choice(size=(n_best,), a=T.arange(p_next_word.shape[0]), replace=True, p=p_next_word, dtype='int64')
            return word

        def init_sample_words(temp):
            #initilize theano function for random sampling
            P_Next_Words = T.matrix('p_next_words')#, dtype='float64')
            P_Adj_Next_Words = T.nnet.softmax(T.log(P_Next_Words) / temp)
            Next_Words, Updates = theano.scan(fn=sample_word, sequences=P_Adj_Next_Words)
            sample_words = theano.function([P_Next_Words], Next_Words, updates=Updates)#, allow_input_downcast=True)
            return sample_words

        if prevent_unk: #prevent model from generating unknown words by redistributing probability; assumes index 1 is prob of unknown word
            #import pdb;pdb.set_trace()
            p_unk = p_next_words[:,1]
            added_p = (p_unk / p_next_words[:,2:].shape[-1])[:,None]
            p_next_words[:,2:] = p_next_words[:,2:] + added_p
            p_next_words[:,1] = 0.0

        if mode == 'random':
            #numpy is too slow at random sampling, so use theano
            if not hasattr(self, 'sample_words') or not self.sample_words:
                self.sample_words = init_sample_words(temp)
            next_words = self.sample_words(p_next_words)
        else:
            next_words = numpy.argmax(p_next_words, axis=1)[:, None]

        return next_words

    def predict(self, seqs, max_length=35, mode='max', batch_size=1, n_best=1, temp=1.0, prevent_unk=True):

        assert(type(seqs[0][0]) not in [list, tuple, numpy.ndarray])

        X = [[seq[idx:idx+self.n_timesteps] for idx in range(len(seq) - self.n_timesteps + 1)] for seq in seqs]
        X = numpy.array([seq[-1] for seq in X]) #only predict from last ngram in each sequence

        for idx in range(max_length):
            p_next_words = self.model.predict(X[:, -self.n_timesteps:], batch_size=batch_size)
            next_words = self.pred_next_words(p_next_words, mode, n_best, temp, prevent_unk)
            X = numpy.append(X, next_words, axis=1)

        X = X[:, self.n_timesteps:]
        pred_seqs = list(X)

        return pred_seqs

    def get_probs(self, seqs):#, batch_size=1):
        '''compute log prob of given sequences'''
        probs = []

        assert(type(seqs[0][0]) not in [list, tuple, numpy.ndarray])
        
        len_seqs = [len(seq) for seq in seqs]

        X = [[seq[idx:idx+self.n_timesteps+1] for idx in range(len(seq) - self.n_timesteps)] for seq in seqs]
        X = numpy.array([ngram for seq in X for ngram in seq])
        y = X[:, -1][:, None]
        X = X[:, :-1]

        p_next_words = self.model.predict(X)[numpy.arange(len(X)), y[:,0]]
        idx = 0
        for len_seq in len_seqs: #reshape probs back into sequences to get mean log prob for each sequence
            #p_next_words_ = numpy.mean(numpy.log(p_next_words[idx:idx+len_seq-self.n_timesteps]))
            p_next_words_ = p_next_words[idx:idx+len_seq-self.n_timesteps]
            idx += len_seq-self.n_timesteps
            probs.append(p_next_words_)
        #probs = numpy.array(probs)
        return probs

    
class EncoderDecoder(SavedModel):
    def __init__(self, n_embedding_nodes=300, n_hidden_nodes=500, recurrent=False, batch_size=100, filepath=None, verbose=True):
        
        self.n_embedding_nodes = n_embedding_nodes
        self.n_hidden_nodes = n_hidden_nodes
        self.filepath = filepath
        self.recurrent = recurrent
        self.verbose = verbose
        self.batch_size = batch_size
        self.n_timesteps = None
        self.lexicon_size = None

    def create_model(self):

        if self.recurrent: #use sequence-to-sequence (RNN) model
            encoder_inputs = Input(shape=(self.n_timesteps,), name='input_seq_layer')
            emb_layer = Embedding(self.lexicon_size + 1, self.n_embedding_nodes, mask_zero=True, name='emb_seq_layer')
            emb_encoder_inputs = emb_layer(encoder_inputs)
            encoder = GRU(self.n_hidden_nodes, name='encoded_seq_layer', return_state=True)
            encoder_outputs, state_h = encoder(emb_encoder_inputs)
            decoder_inputs = Input(shape=(self.n_timesteps,))
            emb_decoder_inputs = emb_layer(decoder_inputs)
            decoder_gru = GRU(self.n_hidden_nodes, name='decoded_seq_layer', return_sequences=True)
            decoder_outputs = decoder_gru(emb_decoder_inputs, initial_state=state_h)
            decoder_dense = Dense(self.lexicon_size + 1, name='output_seq_layer', activation='softmax')
            decoder_outputs = decoder_dense(decoder_outputs)
            model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
            model.compile(loss="sparse_categorical_crossentropy", optimizer='adam')

        else: #flat encoder-decoder (no recurrent layer)

            input_seq_layer = Input(shape=(self.lexicon_size + 1,), name="input_seq_layer")
            output_seq_layer = Dense(output_dim=self.lexicon_size + 1, activation='sigmoid', name='output_seq_layer')(input_seq_layer)
            model = Model(input=input_seq_layer, output=output_seq_layer)
            model.compile(loss="binary_crossentropy", optimizer='adam')

        return model
    
    def fit(self, seqs1, seqs2, n_timesteps=None, lexicon_size=None, n_epochs=1, save_to_filepath=False):

        if not hasattr(self, 'model'):
            assert(lexicon_size is not None)
            self.lexicon_size = lexicon_size
            self.n_timesteps = n_timesteps
            if self.recurrent and not self.n_timesteps:
                self.n_timesteps = max([len(seq) for seq in seqs1 + seqs2]) #if n_timesteps not given, set it to length of longest sequence
            self.model = self.create_model()
            if self.verbose:
                print("Created model", self.__class__.__name__, ":", self.__dict__)

        assert(len(seqs1) == len(seqs2))

        for epoch in range(n_epochs):
            losses = []
            if n_epochs > 1:
                if self.verbose:
                    print("EPOCH:", epoch + 1)
            for batch_idx in range(0, len(seqs1), self.batch_size):
                if self.recurrent:
                    batch_seqs1 = get_seq_batch(seqs1[batch_idx:batch_idx+self.batch_size], max_length=self.n_timesteps)
                    batch_seqs2 = get_seq_batch(seqs2[batch_idx:batch_idx+self.batch_size], padding='post', max_length=self.n_timesteps)
                    batch_seqs2 = numpy.insert(batch_seqs2, 0, numpy.zeros(len(batch_seqs2)), axis=-1) #prepend zeros (not sure if this is necessary)
                    losses.append(self.model.train_on_batch(x=[batch_seqs1, batch_seqs2[:,:-1]], y=batch_seqs2[:,1:,None]))
                else:
                    batch_seqs1 = get_vector_batch(seqs1[batch_idx:batch_idx+self.batch_size], vector_length=self.lexicon_size+1)
                    batch_seqs2 = get_vector_batch(seqs2[batch_idx:batch_idx+self.batch_size], vector_length=self.lexicon_size+1)
                    losses.append(self.model.train_on_batch(x=batch_seqs1, y=batch_seqs2))
                if batch_idx and batch_idx % (self.batch_size * 1000) == 0:
                    if self.verbose:
                        print("loss: {:.7f}".format(numpy.mean(numpy.array(losses))))
            if self.verbose:
                print("loss: {:.7f}".format(numpy.mean(numpy.array(losses))))
            if save_to_filepath and self.filepath:
                self.save()
        
    def predict(self, seq1, seq2, pred_method='multiply'):

        if self.recurrent:
            seq1 = get_seq_batch([seq1], max_length=self.n_timesteps)
            seq2 = get_seq_batch([seq2], padding='post', max_length=self.n_timesteps)
            seq2 = numpy.insert(seq2, 0, numpy.zeros(len(seq2)), axis=-1) #prepend zeros (not sure if this is necessary)
            probs = self.model.predict_on_batch([seq1, seq2[:,:-1]])[0]
            probs = probs[numpy.arange(self.n_timesteps), seq2[:,1:]]
            probs = probs[seq2[:,1:] > 0]
        else:
            seq1 = get_vector_batch([seq1], vector_length=self.lexicon_size+1)
            probs = self.model.predict_on_batch(seq1)[0]
            seq2 = get_vector_batch([seq2], vector_length=self.lexicon_size+1)
            probs = probs[seq2[0].astype('bool')]

        prob = numpy.sum(numpy.log(probs))
        return prob

    def get_most_probable_words(self, seq1, top_n_words=10, unigram_probs=None):

        if self.flat_input:
            if self.embedded_input:
                seq1 = seq1[None]
            else:
                seq1 = get_vector_batch([seq1], vector_length=self.lexicon_size+1)
        else:
            seq1 = get_seq_batch([seq1], max_length=self.n_timesteps)

        probs = self.model.predict_on_batch(seq1)[0]
        if unigram_probs is not None: #discount probabilities by unigram frequency if given
            probs = probs / unigram_probs ** 0.66
            probs[numpy.isinf(probs)] = 0.0 #replace inf
        most_probable_words = numpy.argsort(probs)[::-1][:top_n_words]
        probs = probs[most_probable_words] #filter to return only probs for most probable words

        return most_probable_words, probs


        
class EmbeddingSimilarity(SavedModel):
    def predict(self, seq1, seq2):
        seq1 = seq1 + 1e-8
        seq2 = seq2 + 1e-8 #smooth to avoid NaN
        score = 1 - cosine(seq1, seq2)
        return score


class CausalEmbeddings(SavedModel):
    def __init__(self, n_embedding_nodes=300, n_hidden_nodes=500, batch_size=100, filepath=None):
        self.lexicon_size = None
        self.n_embedding_nodes = n_embedding_nodes
        self.n_hidden_nodes = n_hidden_nodes
        self.batch_size = batch_size
        self.filepath = filepath

    def create_model(self):

        if self.embedded_input:
            cause_word_layer = Input(shape=(self.n_embedding_nodes,), name="cause_word_layer") 
            effect_word_layer = Input(shape=(self.n_embedding_nodes,), name="effect_word_layer")
            cause_emb_layer = Dense(output_dim=self.n_embedding_nodes, name='cause_emb_layer', activation='tanh')(cause_word_layer)
            effect_emb_layer = Dense(output_dim=self.n_embedding_nodes, name='effect_emb_layer', activation='tanh')(effect_word_layer)
        else:
            cause_word_layer = Input(shape=(1,), name="cause_word_layer")
            effect_word_layer = Input(shape=(1,), name="effect_word_layer")
            cause_emb_layer = Embedding(self.lexicon_size + 1, self.n_embedding_nodes, name='cause_emb_layer')(cause_word_layer)
            effect_emb_layer = Embedding(self.lexicon_size + 1, self.n_embedding_nodes, name='effect_emb_layer')(effect_word_layer)
            flatten_layer = Flatten(name='flatten_layer')
            cause_emb_layer = flatten_layer(cause_emb_layer)
            effect_emb_layer = flatten_layer(effect_emb_layer)


        merge_layer = merge([cause_emb_layer, effect_emb_layer], mode='concat', concat_axis=-1, name='merge_layer')
        dense_layer = Dense(output_dim=self.n_hidden_nodes, name='dense_layer', activation='tanh')(merge_layer)
        pred_layer = Dense(output_dim=1, name='pred_layer', activation='sigmoid')(dense_layer)
        model = Model(input=[cause_word_layer, effect_word_layer], output=pred_layer)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def fit(self, cause_words, effect_words, labels, lexicon_size=None, embedded_input=False, n_epochs=1):

        if not hasattr(self, 'model'):
            self.embedded_input = embedded_input
            if not self.embedded_input:
                assert(lexicon_size is not None)
                self.lexicon_size = lexicon_size
            self.model = self.create_model()
            print("Created model", self.__class__.__name__, ":", self.__dict__)

        self.model.fit(x=[cause_words[:, None], effect_words[:, None]], y=labels, batch_size=self.batch_size, nb_epoch=n_epochs)

        if self.filepath:
            self.save()

    def predict(self, cause_words, effect_words):

        probs = self.model.predict(x=[cause_words, effect_words])
        return probs


