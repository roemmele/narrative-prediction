from __future__ import print_function
import timeit, numpy, pickle, os, copy
import theano
import theano.tensor as T
#from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.optimizers import RMSprop, SGD, Adagrad, Adam
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K

theano_rng = T.shared_randomstreams.RandomStreams(123)


class SavedModel():
    def save(self):
        #import pdb;pdb.set_trace()
        
        #save model
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
    def load(cls, filepath):
        with open(filepath + '/classifier.pkl', 'rb') as f:
            classifier = pickle.load(f)
        classifier.model = load_model(filepath + '/classifier.h5')
        print('loaded classifier from', filepath + '/classifier.pkl')
        return classifier

def get_batch(seqs, batch_size=None, padding='pre', max_length=None, n_timesteps=None):
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


class RNNClassifier():
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

def resave_classifier_nosklearn(old_classifier):
    import pdb;pdb.set_trace()
    new_classifier = RNNLM()
    new_classifier.lexicon_size = old_classifier.lexicon_size
    new_classifier.n_embedding_nodes = old_classifier.n_embedding_nodes
    new_classifier.n_hidden_nodes = old_classifier.n_hidden_nodes
    new_classifier.n_hidden_layers = old_classifier.n_hidden_layers
    new_classifier.batch_size = old_classifier.batch_size
    new_classifier.max_length = old_classifier.max_length
    new_classifier.verbose = old_classifier.verbose
    new_classifier.n_timesteps = old_classifier.n_timesteps
    new_classifier.filepath = old_classifier.filepath
    new_classifier.embeddings = old_classifier.embeddings
    new_classifier.optimizer = old_classifier.optimizer
    new_classifier.lr = old_classifier.lr
    new_classifier.clipvalue = old_classifier.clipvalue
    new_classifier.decay = old_classifier.decay
    #new_classifier.separate_context = old_classifier.separate_context
    #new_classifier.max_seq_length = old_classifier.max_sent_length
    #new_classifier.sample_words = old_classifier.sample_words
    #new_classifier.model = old_classifier.model
    #new_classifier.start_time = old_classifier.start_time
    #ew_classifier.epoch = old_classifier.epoch
    with open(new_classifier.filepath + '/classifier.pkl', 'wb') as f:
        pickle.dump(new_classifier, f)
    #new_classifier.save()


class RNNLM(SavedModel):#KerasClassifier
    def __init__(self, use_features=False, use_pos=False, lexicon_size=None, n_pos_tags=None, n_timesteps=None, n_embedding_nodes=300, n_pos_embedding_nodes=25,
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
            batch = get_batch(seqs=seqs[batch_index:batch_index + self.batch_size], batch_size=self.batch_size, n_timesteps=self.n_timesteps) #prep batch
            if self.use_pos:
                batch_pos = get_batch(seqs=pos_seqs[batch_index:batch_index + self.batch_size], batch_size=self.batch_size, n_timesteps=self.n_timesteps)
            if self.use_features:
                batch_features = get_batch_features(features=feature_vecs[batch_index:batch_index + self.batch_size], batch_size=self.batch_size)
                batch_inputs.append(batch_features)
            for step_index in range(0, batch.shape[-1] - 1, self.n_timesteps):
                batch_x = batch[:, step_index:step_index + self.n_timesteps]
                if not numpy.sum(batch_x):
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
        self.check_pred_model(len(words))
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
            next_words = self.sample_words(p_next_words, temp)
            #next_words = numpy.array([self.sample_word(p_next_word, n_best, temp)[0] for p_next_word in p_next_words])
        else:
            #next_words = numpy.array([numpy.argmax(p_next_word) for p_next_word in p_next_words])
            next_words = numpy.argmax(p_next_words, axis=1)[:,None]

        #p_next_words = p_next_words[numpy.arange(len(p_next_words)), next_words]
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
    
    # def get_best_sents(self, seqs, p_sents, n_best):
        
    #     best_idxs = numpy.argsort(numpy.array(p_sents))[::-1][:n_best]
    #     best_sents = [sents[idx] for idx in best_idxs]
    #     #return probs of best sents as well as sents
    #     p_sents = numpy.array(p_sents)[best_idxs]
        
    #     return best_sents, p_sents

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
            # batch_pos = None
            batch_seqs = get_batch(seqs=seqs[batch_index:batch_index + batch_size], batch_size=batch_size) #prep batch
            # if self.use_pos:
            #     batch_pos = get_batch(seqs=pos_seqs[batch_index:batch_index + batch_size], batch_size=batch_size)
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

    def get_probs(self, seqs, pos_seqs=None, feature_vecs=None, batch_size=1):
        '''return log probabilities computed by model for each word in each sequence'''

        probs = []

        for batch_index in range(0, len(seqs), batch_size):
            batch_features = None
            batch_pos = None
            batch_seqs = get_batch(seqs=seqs[batch_index:batch_index + batch_size], batch_size=batch_size) #prep batch
            if self.use_pos:
                batch_pos = get_batch(seqs=pos_seqs[batch_index:batch_index + batch_size], batch_size=batch_size)
            if self.use_features:
                batch_features = get_batch_features(features=feature_vecs[batch_index:batch_index + batch_size], batch_size=batch_size)

            p_next_words = self.read_batch(seqs=batch_seqs, pos=batch_pos if self.use_pos else None, features=batch_features)
            #p_next_words = numpy.log(p_next_words)
            len_seqs = [len(seq) for seq in seqs[batch_index:batch_index + batch_size]]
            #remove padding from each sequence and before computing mean
            #p_next_words = numpy.array([numpy.mean(p_next_words_[-len_seq:]) for p_next_words_, len_seq in zip(p_next_words, len_seqs)])
            p_next_words = [p_next_words_[-len_seq:] for p_next_words_, len_seq in zip(p_next_words, len_seqs)]
            probs.extend(p_next_words)

            self.pred_model.reset_states()

            if batch_index and batch_index % 1000 == 0:
                print("computed probabilities for {}/{} sequences...".format(batch_index, len(seqs)))

        #probs = numpy.array(probs)

        return probs


    def create_encoder(self):

        self.encoder_model = self.create_model(n_timesteps=None, batch_size=1, pred_layer=False)
        self.encoder_model.set_weights(self.model.get_weights()[:-2])
        if self.verbose:
            print("created encoder model")

    def get_embeddings(self):

        embeddings = self.model.get_weights()[0]
        return embeddings


class FeatureRNNLM(RNNLM):
    def __init__(self, n_feature_nodes=100, **params):
        self.n_feature_nodes = n_feature_nodes
        RNNLM.__init__(self, **params)

    def create_model(self, n_timesteps=1, batch_size=1, include_pred_layer=True):

        seq_input_layer = Input(batch_shape=(batch_size, n_timesteps), name="seq_input_layer")
        seq_embedding_layer = Embedding(input_dim=self.lexicon_size + 1, 
                                        output_dim=self.n_embedding_nodes, mask_zero=True, name='seq_embedding_layer')(seq_input_layer)

        for layer_num in range(self.n_hidden_layers):
            if layer_num == 0:
                seq_hidden_layer = GRU(output_dim=self.n_hidden_nodes, return_sequences=True, stateful=True, name='seq_hidden_layer1')(seq_embedding_layer)
            else: #add extra hidden layers
                seq_hidden_layer = GRU(output_dim=self.n_hidden_nodes, return_sequences=True, stateful=True, name='seq_hidden_layer' + str(layer_num + 1))(seq_hidden_layer)

        feature_input_layer = Input(batch_shape=(batch_size, self.lexicon_size + 1), name="feature_input_layer")
        feature_hidden_layer = Dense(output_dim=self.n_feature_nodes, activation='sigmoid', name='feature_hidden_layer')(feature_input_layer)

        #context_hidden_layer = Reshape((1, self.n_hidden_nodes))(context_hidden_layer)
        feature_hidden_layer = RepeatVector(n_timesteps)(feature_hidden_layer)

        merge_hidden_layer = merge([seq_hidden_layer, feature_hidden_layer], mode='concat', concat_axis=-1, name='merge_hidden_layer')

        if include_pred_layer:
            pred_layer = TimeDistributed(Dense(self.lexicon_size + 1, activation="softmax"))(merge_hidden_layer)

        model = Model(input=[seq_input_layer, feature_input_layer], output=pred_layer)
            
        #select optimizer and compile
        model.compile(loss="sparse_categorical_crossentropy", 
                      optimizer=eval(self.optimizer)(clipvalue=self.clipvalue, lr=self.lr, decay=self.decay))
                
        return model

    def sort_seqs(self, seqs, feature_vecs):
        #sort by descending length
        lengths = [len(seq) for seq in seqs]
        sorted_idxs = numpy.argsort(lengths)#[::-1]
        seqs = [seqs[idx] for idx in sorted_idxs]
        feature_vecs = feature_vecs[sorted_idxs]
        return seqs, feature_vecs

    def fit(self, seqs, feature_vecs, lexicon_size=None):#, context_size=None):

        #import pdb;pdb.set_trace()
        
        if not hasattr(self, 'model'):
            self.lexicon_size = lexicon_size
            self.model = self.create_model(n_timesteps=self.n_timesteps, batch_size=self.batch_size)
            self.start_time = timeit.default_timer()

        if self.verbose:
            print("training FeatureRNNLM on {} sequences with batch size = {}".format(len(seqs), self.batch_size))
        
        train_losses = []

        # assert(type(seqs[0][0]) not in (list, tuple, numpy.ndarray))
        assert(len(seqs) == len(feature_vecs))
        #sort seqs by length
        seqs, feature_vecs = self.sort_seqs(seqs, feature_vecs)
        for batch_index in range(0, len(seqs), self.batch_size):
            #prep batch
            batch = get_batch(seqs=seqs[batch_index:batch_index + self.batch_size], batch_size=self.batch_size, n_timesteps=self.n_timesteps)
            batch_features = get_batch_features(features=feature_vecs[batch_index:batch_index + self.batch_size], batch_size=self.batch_size)
            for step_index in range(0, batch.shape[-1] - 1, self.n_timesteps):
                batch_x = batch[:, step_index:step_index + self.n_timesteps]
                if not numpy.sum(batch_x): #batch is all zeros, skip
                    continue
                batch_y = batch[:, step_index + 1:step_index + self.n_timesteps + 1, None]
                train_loss = self.model.train_on_batch(x=[batch_x, batch_features], y=batch_y)
            train_losses.append(train_loss)
            self.model.reset_states()                
            if batch_index and batch_index % 5000 == 0:
                print("processed {} sequences, loss: {:.3f} ({:.3f}m)...".format(batch_index, numpy.mean(train_losses),
                                                                                (timeit.default_timer() - self.start_time) / 60))

        if self.filepath:
            #save model if filepath given
            self.save()
        if self.verbose:
            print("loss: {:.3f} ({:.3f}m)".format(numpy.mean(train_losses),
                                                (timeit.default_timer() - self.start_time) / 60))

    def get_batch_p_next_words(self, words, features):
        p_next_words = self.pred_model.predict_on_batch(x=[words[:, None], features])[:, -1]
        return p_next_words

    def predict(self, seqs, feature_vecs, max_length=35, mode='max', batch_size=1, n_best=1, temp=1.0, prevent_unk=True):

        #import pdb;pdb.set_trace()

        if not hasattr(self, 'pred_model') or batch_size != self.pred_model.input_shape[0][0]:
            self.pred_model = self.create_model(batch_size=batch_size, n_timesteps=1)

            #set weights of prediction model
            if self.verbose:
                print("created predictor model")

        self.pred_model.set_weights(self.model.get_weights())

        pred_seqs = []

        for batch_index in range(0, len(seqs), batch_size):
            #prep batch
            batch = get_batch(seqs=seqs[batch_index:batch_index + batch_size], batch_size=batch_size)
            batch_features = get_batch_features(features=feature_vecs[batch_index:batch_index + batch_size], batch_size=batch_size)
            #batch_features = numpy.insert(batch_features, 0, numpy.zeros(len(batch_features)), axis=1) #prepend column of zeros for processing first word of sequence (where no context exists)
            
            batch_pred_seqs = numpy.zeros((batch_size, max_length), dtype='int64')

            for idx in range(batch.shape[-1]): #read in given sequence from which to predict
                p_next_words = self.get_batch_p_next_words(words=batch[:, idx], features=batch_features)

            for idx in range(max_length): #now predict
                next_words, p_next_words = self.pred_batch_next_words(p_next_words, mode, n_best, temp, prevent_unk)
                batch_pred_seqs[:, idx] = next_words
                #batch_features = numpy.concatenate([batch_features, batch_pred_seqs[:, idx:idx+1]], axis=1) #add new words to context vector
                p_next_words = self.get_batch_p_next_words(words=batch_pred_seqs[:, idx], features=batch_features)
            self.pred_model.reset_states()

            if len(seqs[batch_index:batch_index + batch_size]) < batch_size:
                #remove padding if batch was padded
                batch_pred_seqs = batch_pred_seqs[:len(seqs[batch_index:batch_index + batch_size])]
            batch_pred_seqs = batch_pred_seqs.tolist()

            pred_seqs.extend(batch_pred_seqs)

            if batch_index and batch_index % 1000 == 0:
                print("generated new sequences for {}/{} inputs...".format(batch_index, len(seqs)))

        return pred_seqs


class SeqBinaryClassifier(SavedModel):#, KerasClassifier):
    def __init__(self, context_size, lexicon_size=None, n_embedding_nodes=300, batch_size=100, 
                 n_hidden_layers=1, n_hidden_nodes=500, verbose=1, embedded_input=True, optimizer='RMSprop', clipvalue=numpy.inf,
                 filepath=None, use_dropout=False, pairs=False): #max_length=None, , save_freq=20
        
        self.batch_size = batch_size
        self.lexicon_size = lexicon_size
        #self.max_length = max_length
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_nodes = n_hidden_nodes
        self.n_embedding_nodes = n_embedding_nodes
        self.context_size = context_size
        self.verbose = verbose
        self.embedded_input = embedded_input
        self.use_dropout = use_dropout
        self.clipvalue = clipvalue
        self.optimizer = optimizer
        #self.update_num = 0
        #self.save_freq = save_freq
        self.filepath = filepath

        ####################CURRENT METHOD######################
        # context_input_layer = Input(batch_shape=(self.batch_size, self.context_size, self.n_embedding_nodes), name="context_input_layer")

        # seq_input_layer = Input(batch_shape=(self.batch_size, self.n_embedding_nodes), name="seq_input_layer")

        # reshape_seq_layer = Reshape((1, self.n_embedding_nodes))(seq_input_layer)

        # merge_layer = merge([context_input_layer, reshape_seq_layer], mode='concat', concat_axis=-2, name='merge_layer')

        # mask_layer = Masking(mask_value=0.0, input_shape=(self.context_size + 1, self.n_embedding_nodes))(merge_layer)

        # hidden_layer = GRU(output_dim=self.n_hidden_nodes, return_sequences=False, stateful=False, name='context_hidden_layer')(mask_layer)#(merge_layer)

        # pred_layer = Dense(output_dim=1, activation='sigmoid', name='pred_layer')(hidden_layer)

        # self.model = Model(input=[context_input_layer, seq_input_layer], output=pred_layer)
        ##########################################################

        ####################ALTERNATIVE METHOD######################
        context_input_layer = Input(batch_shape=(self.batch_size, self.context_size, self.n_embedding_nodes), name="context_input_layer")

        context_hidden_layer = GRU(output_dim=self.n_hidden_nodes, return_sequences=False, stateful=False, name='context_hidden_layer')(context_input_layer)

        seq_input_layer = Input(batch_shape=(self.batch_size, self.n_embedding_nodes), name="seq_input_layer")

        seq_hidden_layer = Dense(output_dim=self.n_hidden_nodes, activation='tanh', name='seq_hidden_layer')(seq_input_layer)

        merge_layer = merge([context_hidden_layer, seq_hidden_layer], mode='dot', concat_axis=-1, name='merge_layer')

        sigmoid_layer = Activation('sigmoid')(merge_layer)

        self.model = Model(input=[context_input_layer, seq_input_layer], output=sigmoid_layer)
        ##########################################################

        # if self.use_dropout:
        #     merge_layer = Dropout(p=0.25)(merge_layer)

        # for layer_num in range(self.n_hidden_layers):

        #     if layer_num + 1 == self.n_hidden_layers:
        #         return_sequences = False
        #     else:
        #         return_sequences = True

        #     if layer_num == 0:
        #         if self.use_dropout:
        #             merge_layer = Dropout(p=0.5)(merge_layer)
        #         seq_hidden_layer = GRU(output_dim=self.n_hidden_nodes, return_sequences=return_sequences, stateful=False)(merge_layer)
        #     else:
        #         if self.use_dropout:
        #             hidden_layer = Dropout(p=0.5)(hidden_layer)
        #         seq_hidden_layer = GRU(output_dim=self.n_hidden_nodes, return_sequences=return_sequences, stateful=False)(hidden_layer)


        optimizer = eval(self.optimizer)(clipvalue=self.clipvalue)

        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        if self.verbose:
            print("Created model", self.__class__.__name__, ":", self.__dict__)
        
    
    def fit(self, input_seqs, output_seqs, labels):
    
        #sentences up to last are context, last sentence is ending to be judged as correct
        history = self.model.fit([input_seqs, output_seqs], labels, nb_epoch=1, batch_size=self.batch_size)
        
        print("loss: {:.3f}, accuracy: {:.3f}".format(history.history['loss'][0], history.history['acc'][0]))

        if self.filepath:
            self.save()

    def predict(self, input_seq, output_seq):
        '''return score for likelihood of output seq given input seq'''

        prob = self.model.predict([input_seq[None], output_seq[None]])[0]
        return prob

class RNNBinaryClassifier(SavedModel):#, KerasClassifier):
    def __init__(self, n_timesteps, lexicon_size=None, n_embedding_nodes=300, batch_size=100, 
                 n_hidden_layers=1, n_hidden_nodes=500, verbose=1, optimizer='Adam', clipvalue=numpy.inf,
                 filepath=None, use_dropout=False):
        
        self.batch_size = batch_size
        self.lexicon_size = lexicon_size
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_nodes = n_hidden_nodes
        self.n_embedding_nodes = n_embedding_nodes
        self.n_timesteps = n_timesteps
        self.verbose = verbose
        self.use_dropout = use_dropout
        self.clipvalue = clipvalue
        self.optimizer = optimizer
        self.filepath = filepath

    def create_model(self):

        emb_layer = Embedding(self.lexicon_size + 1, self.n_embedding_nodes, mask_zero=True, name='emb_layer')

        cause_layer = Input(shape=(self.n_timesteps,), name="cause_layer")
        cause_emb_layer = emb_layer(cause_layer)
        effect_layer = Input(shape=(self.n_timesteps,), name="effect_layer")
        effect_emb_layer = emb_layer(effect_layer)

        cause_hidden_layer = GRU(output_dim=self.n_hidden_nodes, return_sequences=False, 
                                 stateful=False, name='cause_hidden_layer')(cause_emb_layer)
        effect_hidden_layer = GRU(output_dim=self.n_hidden_nodes, return_sequences=False, 
                                 stateful=False, name='effect_hidden_layer')(effect_emb_layer)

        merge_layer = merge([cause_hidden_layer, effect_hidden_layer], mode='dot', concat_axis=-1, name='merge_layer')
        #flatten_layer = Flatten(name='flatten_layer')(merge_layer)
        sigmoid_layer = Activation('sigmoid')(merge_layer)
        model = Model(input=[cause_layer, effect_layer], output=sigmoid_layer)
        model.compile(loss='binary_crossentropy', optimizer='adam')

        return model
    
    def fit(self, input_seqs, output_seqs, labels, lexicon_size=None):

        if not hasattr(self, 'model'):
            assert(lexicon_size is not None)
            self.lexicon_size = lexicon_size
            self.model = self.create_model()
            print("Created model", self.__class__.__name__, ":", self.__dict__)
    
        self.model.fit([input_seqs, output_seqs], labels, nb_epoch=1, batch_size=self.batch_size)

        if self.filepath:
            self.save()

    def predict(self, input_seq, output_seq):
        '''return score for likelihood of output seq given input seq'''

        prob = self.model.predict([input_seq[None], output_seq[None]])[0]
        return prob


class Autoencoder():
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



# class MLPClassifier(KerasClassifier):
#     def __call__(self, lexicon_size, n_outcomes, input_to_outcome_set=None):
        
#         self.n_outcomes = n_outcomes
#         self.lexicon_size = lexicon_size
        
#         model = Sequential()
#         model.add(Dense(output_dim=200, input_dim=self.lexicon_size, activation='tanh', name='hidden1'))
#         #model.add(Dense(output_dim=200, input_dim=lexicon_size, activation='tanh', name='hidden2'))
#         model.add(Dense(output_dim=n_outcomes, activation='softmax', name='output'))
#         model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])
#         return model
#     def fit(self, X, y, **kwargs):
#         #get number of input nodes
#         self.sk_params.update(lexicon_size = X.shape[1])
#         #keras doesn't handle sparse matrices
#         X = X.toarray()
#         super(MLPClassifier, self).fit(X, y, **kwargs)
#     def predict(self, X, **kwargs):
#         #keras doesn't handle sparse matrices  
#         X = X.toarray()
    
#         if "input_to_outcome_set" in self.sk_params and self.sk_params["input_to_outcome_set"] is not None:
#             #import pdb; pdb.set_trace()
#             input_to_outcome_set = self.sk_params["input_to_outcome_set"]
#             #predict from specific outcome set for each input
#             pred_y = []
#             prob_y = self.model.predict(X, **kwargs)
#             for prob_y_, outcome_choices in zip(prob_y, input_to_outcome_set):
#                 prob_y_ = prob_y_[outcome_choices]
#                 pred_y_ = outcome_choices[numpy.argmax(prob_y_)]
#                 pred_y.append(pred_y_)
#             return pred_y

#         else:
#             return super(MLPClassifier, self).predict(X, **kwargs)
    
# class Seq2SeqClassifier(KerasClassifier):
#     def __call__(self, lexicon_size, max_length, batch_size=None, stateful=False,
#                  n_encoding_layers=1, n_decoding_layers=1, 
#                  n_embedding_nodes=100, n_hidden_nodes=250, verbose=1, embedded_input=False):
        
#         self.stateful = stateful
#         self.embedded_input = embedded_input
#         self.batch_size = batch_size
#         self.max_length = max_length
#         self.lexicon_size = lexicon_size
#         self.n_embedding_nodes = n_embedding_nodes
#         self.n_encoding_layers = n_encoding_layers
#         self.n_decoding_layers = n_decoding_layers
#         self.n_hidden_nodes = n_hidden_nodes
#         self.verbose = verbose
        
#         #import pdb;pdb.set_trace()
#         model = Sequential()
        
#         if not self.embedded_input:
#             embedding = Embedding(batch_input_shape=(self.batch_size, self.max_length), 
#                                   input_dim=self.lexicon_size + 1,
#                                   output_dim=self.n_embedding_nodes, 
#                                   mask_zero=True, name='embedding')
#             model.add(embedding)

#         encoded_input = GRU(batch_input_shape=(self.batch_size, self.max_length, self.n_embedding_nodes),
#                             input_length = self.max_length,
#                             input_dim = self.n_embedding_nodes,
#                             output_dim=self.n_hidden_nodes, return_sequences=False, 
#                             name='encoded_input1', stateful=self.stateful)
#         model.add(encoded_input)

#         repeat_layer = RepeatVector(self.max_length, name="repeat_layer")
#         model.add(repeat_layer)
        
#         encoded_outcome = GRU(self.n_hidden_nodes, return_sequences=True, name='encoded_outcome1',
#                               stateful=self.stateful)#(repeat_layer)
#         model.add(encoded_outcome)

#         outcome_seq = TimeDistributed(Dense(output_dim=self.lexicon_size + 1, activation='softmax', 
#                             name='outcome_seq'))#(encoded_outcome)
#         model.add(outcome_seq)

#         optimizer = "rmsprop"
#         model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        
#         if self.verbose:
#             print "CREATED Sequence2Sequence model: embedding layer sizes:", self.n_embedding_nodes, ",",\
#             self.n_encoding_layers, "encoding layers with size:", self.n_hidden_nodes, ",",\
#             self.n_decoding_layers, "decoding layers with size:", self.n_hidden_nodes, ",",\
#             "optimizer:", optimizer, ", batch_size:", self.batch_size, ", stateful =", self.stateful
#         return model
    
#     def fit(self, X, y, rnn_params=None, **kwargs):
#         #import pdb;pdb.set_trace()
#         #y are sequences rather than classes here
#         self.sk_params.update(rnn_params)
#         if "embedded_input" in self.sk_params and self.sk_params["embedded_input"]:
#             max_length = X.shape[-2]
#         else:
#             max_length = X.shape[-1]
#         self.sk_params.update({"max_length": max_length})
#         #import pdb;pdb.set_trace()
#         patience = 2
#         n_lossless_iters = 0
#         if "stateful" in self.sk_params and self.sk_params["stateful"]:
#             #import pdb;pdb.set_trace()
#             #carry over state between batches
#             assert(len(X.shape) >= 3)
#             self.model = self.__call__(**self.filter_sk_params(self.__call__))
#             nb_epoch = self.sk_params["nb_epoch"]
#             n_batches = int(numpy.ceil(len(X) * 1. / self.batch_size))
#             #import pdb;pdb.set_trace()
#             if self.verbose:
#                 print "training stateful Seq2Seq on", len(X), "sequences for", nb_epoch, "epochs..."
#                 print n_batches, "batches with", self.batch_size, "sequences per batch"
#             start_time = timeit.default_timer()
#             min_loss = numpy.inf
#             for epoch in range(nb_epoch):
#                 train_losses = []
#                 for batch_index in range(0, len(X), self.batch_size):
#                     batch_num = batch_index / self.batch_size + 1
#                     for sent_index in range(X.shape[1]):
#                         batch_X = X[batch_index:batch_index + self.batch_size, sent_index]
#                         batch_y = y[batch_index:batch_index + self.batch_size, sent_index]
#                         assert(len(batch_X) == len(batch_y))
#                         if len(batch_X) < self.batch_size:
#                             #too few sequences for batch, so add extra rows
#                             batch_X = numpy.append(batch_X, numpy.zeros((self.batch_size - len(batch_X),) 
#                                                                     + batch_X.shape[1:]), axis=0)
#                             batch_y = numpy.append(batch_y, numpy.zeros((self.batch_size - len(batch_y),
#                                                                      batch_y.shape[-1])), axis=0)
#                         train_loss = self.model.train_on_batch(x=batch_X, y=batch_y[:, :, None])
#                         train_losses.append(train_loss)
#                     if batch_num % 100 == 0:
#                         print "completed", batch_num, "/", n_batches, "batches in epoch", epoch + 1, "..."
                            
#                     self.model.reset_states()
                
#                 if self.verbose:
#                     print("epoch {}/{} loss: {:.3f} ({:.3f}m)".format(epoch + 1, nb_epoch, 
#                                                                       numpy.mean(train_losses),
#                                                                       (timeit.default_timer() - start_time) / 60))
#                 if numpy.mean(train_losses) < min_loss:
#                     n_lossless_iters = 0  
#                     min_loss = numpy.mean(train_losses)
#                 else:
#                     n_lossless_iters += 1
#                     if n_lossless_iters == patience:
#                         #loss hasn't decreased after waiting number of patience iterations, so stop
#                         print "stopping early"
#                         break
                     
#         else:
#             #import pdb;pdb.set_trace()
#             #assert(len(X.shape) == 2)
#             #regular fit function works for non-stateful models
#             early_stop = EarlyStopping(monitor='train_loss', patience=patience, verbose=0, mode='auto')
#             super(Seq2SeqClassifier, self).fit(X, y=y[:, :, None], callbacks=[early_stop], **kwargs)
        
#     def predict(self, X, y_choices, **kwargs):
#         #check if y_choices is single set or if there are different choices for each input
#         y_choices = check_y_choices(X, y_choices)
        
#         if self.verbose:
#             print "predicting outputs for", len(X), "sequences..."
            
#         if self.stateful:
#             #iterate through sentences as input-output
#             #import pdb;pdb.set_trace()
#             probs_y = []
#             for batch_index in range(0, len(X), self.batch_size):
#                 for sent_index in range(X.shape[1]):
#                     batch_X = X[batch_index:batch_index + self.batch_size, sent_index]
#                     if len(batch_X) < self.batch_size:
#                         #too few sequences for batch, so add extra rows
#                         #import pdb;pdb.set_trace()
#                         batch_X = numpy.append(batch_X, numpy.zeros((self.batch_size - len(batch_X),) 
#                                                                     + batch_X.shape[1:]), axis=0)
#                         assert(len(batch_X) == self.batch_size)
#                     probs_next_sent = self.model.predict_on_batch(batch_X)
#                 #then reduce batch again if it has empty rows 
#                 if len(X) - batch_index < self.batch_size:
#                     #import pdb;pdb.set_trace()
#                     batch_X = batch_X[:len(X) - batch_index]
#                 batch_choices = y_choices[batch_index:batch_index + self.batch_size]
#                 assert(len(batch_X) == len(batch_choices))
#                 batch_probs_y = []
#                 for choice_index in range(batch_choices.shape[1]):
#                     #import pdb;pdb.set_trace()
#                     #evaluate each choice based on predicted probabilites from most recent sentence
#                     batch_choice = batch_choices[:, choice_index]
#                     probs_choice = probs_next_sent[numpy.arange(len(batch_choice))[:, None],
#                                             numpy.arange(batch_choice.shape[-1]), batch_choice]
#                     #have to iterate through instances because each is different length
#                     probs_choice = [prob_choice[choice > 0][-1] for choice, prob_choice in 
#                                                                      zip(batch_choice, probs_choice)]
#                     batch_probs_y.append(probs_choice)
#                 batch_probs_y = numpy.stack(batch_probs_y, axis=1)
#                 probs_y.append(batch_probs_y)                
#                 self.model.reset_states()
#             probs_y = numpy.concatenate(probs_y)
#             #import pdb;pdb.set_trace()
        
#         else:
#             probs_y = []
#             #import pdb;pdb.set_trace()
#             for x, choices in zip(X, y_choices):       
#                 probs = []
#                 for choice in choices:
#                     prob = self.model.predict(x[None, :], **kwargs)
#                     prob = prob[:, numpy.arange(len(choice)), choice]
#                     prob = prob[0, choice > 0][-1]
#                     #prob = numpy.sum(numpy.log(prob))
#                     #prob = numpy.sum(numpy.log(prob))
#                     probs.append(prob)
#                 probs_y.append(numpy.array(probs))
#             probs_y = numpy.array(probs_y)
        
#         assert(len(probs_y) == len(X))
#         #return prob for each choice for each input
#         return probs_y
        
        
# class MergeSeqClassifier(KerasClassifier):
#     def __call__(self, lexicon_size, outcome_set, max_length, n_encoding_layers=1, n_decoding_layers=1, 
#                  n_embedding_nodes=100, n_hidden_nodes=200, batch_size=1, verbose=1):
        
#         self.lexicon_size = lexicon_size
#         self.max_length = max_length
#         self.outcome_set = outcome_set
#         self.n_embedding_nodes = n_embedding_nodes
#         self.n_encoding_layers = n_encoding_layers
#         self.n_decoding_layers = n_decoding_layers
#         self.n_hidden_nodes = n_hidden_nodes
#         self.batch_size = batch_size
#         self.verbose = verbose

#         #create model
#         user_input = Input(shape=(self.max_length,), dtype='int32', name="user_input")
#         embedding = Embedding(input_length=self.max_length,
#                               input_dim=self.lexicon_size + 1, 
#                               output_dim=self.n_embedding_nodes, mask_zero=True, name='input_embedding')
#         embedded_input = embedding(user_input)
#         encoded_input = GRU(self.n_hidden_nodes, return_sequences=False, name='encoded_input1')(embedded_input)
        
#         outcome_seq = Input(shape=(self.max_length,), dtype='int32', name="outcome_seq")
#         embedded_outcome = embedding(outcome_seq)
#         encoded_outcome = GRU(self.n_hidden_nodes, return_sequences=False, name='encoded_outcome1')(embedded_outcome)
#         input_outcome_seq = merge([encoded_input, encoded_outcome], mode='concat', 
#                                   concat_axis=-1, name='input_outcome_seq')     
#         outcome = Dense(output_dim=len(self.outcome_set), activation='softmax', name='outcome')(input_outcome_seq)
#         model = Model(input=[user_input, outcome_seq], output=[outcome])
#         model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=["accuracy"])      
#         if self.verbose:
#             print "CREATED MergeSequence model: embedding layer sizes =", self.n_embedding_nodes, ",",\
#             self.n_encoding_layers, "encoding layers with size", self.n_hidden_nodes, ",",\
#             self.n_decoding_layers, "decoding layers with size", self.n_hidden_nodes, ",",\
#             "lexicon size = ", self.lexicon_size, ",",\
#             "batch size = ", self.batch_size
#         return model
#     def fit(self, X, y, **kwargs):
#         #import pdb;pdb.set_trace()
#         if not self.sk_params["outcome_set"]:
#             assert("outcome_set" in kwargs)
#             self.sk_params.update(kwargs)
            
#         #outcome_set[y] are outcome sequences
#         super(MergeSeqClassifier, self).fit([X, self.sk_params['outcome_set'][y]], y) #**kwargs)
#         #hidden = self.get_sequence.predict(x=[X, numpy.repeat(self.outcome_set[0][None, :], len(X), axis=0)])
        
#     def predict(self, X, **kwargs):
#         #import pdb;pdb.set_trace()
        
#         max_probs = []
#         for outcome, outcome_seq in enumerate(self.outcome_set):
#             #get probs for this outcome
#             probs = self.model.predict(x=[X, numpy.repeat(outcome_seq[None, :], len(X), axis=0)])[:, outcome]
#             max_probs.append(probs)
#         y = numpy.argmax(numpy.stack(max_probs, axis=1), axis=1)
#         return y

# class CausalEmbeddings(SavedModel):
#     def __init__(self, lexicon_size=None, n_embedding_nodes=300, batch_size=100, filepath=None):
#         self.lexicon_size = lexicon_size
#         self.n_embedding_nodes = n_embedding_nodes
#         self.batch_size = batch_size
#         self.filepath = filepath

#     def create_model(self):
#         cause_word_layer = Input(shape=(1,), name="cause_word_layer") #batch_shape=(batch_size, 
#         cause_emb_layer = Embedding(self.lexicon_size + 1, self.n_embedding_nodes, name='cause_emb_layer')(cause_word_layer)
#         effect_word_layer = Input(shape=(1,), name="effect_word_layer")#batch_shape=(batch_size,
#         effect_emb_layer = Embedding(self.lexicon_size + 1, self.n_embedding_nodes, name='effect_emb_layer')(effect_word_layer)#
#         merge_layer = merge([cause_emb_layer, effect_emb_layer], mode='dot', concat_axis=-1, name='merge_layer')
#         flatten_layer = Flatten(name='flatten_layer')(merge_layer)
#         sigmoid_layer = Activation('sigmoid')(flatten_layer)
#         model = Model(input=[cause_word_layer, effect_word_layer], output=sigmoid_layer)
#         model.compile(loss='binary_crossentropy', optimizer='adam')
#         return model

#     def fit(self, cause_words, effect_words, labels, lexicon_size=None, n_epochs=10):

#         if not hasattr(self, 'model'):
#             assert(lexicon_size is not None)
#             self.lexicon_size = lexicon_size
#             self.model = self.create_model()
#             print "Created model", self.__class__.__name__, ":", self.__dict__

#         self.model.fit(x=[cause_words, effect_words], y=labels, batch_size=self.batch_size, nb_epoch=n_epochs)

#         if self.filepath:
#             self.save()

#     def predict(self, cause_words, effect_words):

#         probs = self.model.predict(x=[cause_words, effect_words])
#         return probs


