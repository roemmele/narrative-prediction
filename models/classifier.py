import timeit, numpy, pickle, os, copy
import theano
import theano.tensor as T
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, TimeDistributed, merge, Dropout, Masking, Reshape, RepeatVector
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.optimizers import RMSprop, SGD, Adagrad, Adam
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K

theano_rng = T.shared_randomstreams.RandomStreams(123)


def load_classifier(filepath):
    #load keras model itself
    with open(filepath + '/classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
    classifier.model = load_model(filepath + '/classifier.h5')
    print "loaded classifier from", filepath + '/classifier.pkl'
    return classifier


class SavedModel():
    def save(self):
        #import pdb;pdb.set_trace()
        
        #save model
        if not os.path.isdir(self.filepath):
            os.mkdir(self.filepath)
        
        self.model.save(self.filepath + '/classifier.h5')
        with open(self.filepath + '/classifier.pkl', 'wb') as f:
            pickle.dump(self, f)
        print "Saved model to", self.filepath
        
    def __getstate__(self):
        #don't save model itself with classifier object, it'll be saved separately as .h5 file
        attrs = self.__dict__.copy()
        if 'model' in attrs:
            del attrs['model']
        if 'pred_model' in attrs:
            del attrs['pred_model']
        if 'encoder_model' in attrs:
            del attrs['encoder_model']
        if 'sample_words' in attrs:
            del attrs['sample_words']
        return attrs

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
    def __init__(self, lexicon_size=None, n_timesteps=None, n_embedding_nodes=300, n_hidden_nodes=250, n_hidden_layers=1,
                 embeddings=None, batch_size=1, verbose=1, filepath=None, optimizer='Adam', lr=0.001, clipvalue=5.0, decay=1e-6):#, separate_context=False,max_seq_length=None, ):
        
        self.lexicon_size = lexicon_size
        self.n_embedding_nodes = n_embedding_nodes
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
        #self.separate_context = separate_context
        #self.max_seq_length = max_seq_length
        self.sample_words = None

        if self.verbose:
            print "CREATED", self.__class__.__name__, ":embedding layer nodes = {}, hidden layers = {}, " \
                    "hidden layer nodes = {}, optimizer = {} with lr = {}, " \
                    "clipvalue = {}, and decay = {}".format(
                    self.n_embedding_nodes, self.n_hidden_layers, self.n_hidden_nodes,
                    self.optimizer, self.lr, self.clipvalue, self.decay)
    
    def create_model(self, n_timesteps=None, batch_size=1, include_pred_layer=True):

        # if self.separate_context:

        #     context_input_layer = Input(batch_shape=(batch_size, n_timesteps), 
        #                                 dtype='int32', name='context_input_layer')

        #     # seq_input_layer = Input(batch_shape=(self.batch_size, n_timesteps), 
        #     #                         dtype='int32', name='seq_input_layer')

        #     embedded_context_layer = Embedding(input_dim = self.lexicon_size + 1,
        #                                 output_dim=self.n_embedding_nodes,
        #                                 mask_zero=True, name='embedding_layer')(context_input_layer)

        #     # embedded_seq_layer = embedding_layer(seq_input_layer)

        #     context_hidden_layer = GRU(output_dim=self.n_hidden_nodes, return_sequences=False, stateful=False, name='context_hidden_layer')(embedded_context_layer)

        #     repeat_layer = RepeatVector(self.max_sent_length)(context_hidden_layer)

        #     # merge_layer = merge([context_hidden_layer, embedded_seq_layer], mode='concat', concat_axis=-1)

        #     seq_hidden_layer = GRU(output_dim=self.n_hidden_nodes, return_sequences=True, stateful=False, name='seq_hidden_layer')(repeat_layer)#(context_hidden_layer)#(merge_layer)

        #     pred_layer = TimeDistributed(Dense(self.lexicon_size + 1, activation="softmax", name='pred_layer'))(seq_hidden_layer)

        #     model = Model(input=context_input_layer, output=pred_layer)

        # else:
        
        model = Sequential()
        
        # if self.embeddings is None:
        model.add(Embedding(self.lexicon_size + 1, self.n_embedding_nodes,
                            batch_input_shape=(batch_size, n_timesteps), mask_zero=True))

        for layer_num in xrange(self.n_hidden_layers):
            model.add(GRU(self.n_hidden_nodes, 
                          batch_input_shape=(batch_size, n_timesteps, self.n_embedding_nodes),
                          return_sequences=True, stateful=True))

        if include_pred_layer: 
            model.add(TimeDistributed(Dense(self.lexicon_size + 1, activation="softmax")))
            
        #select optimizer and compile
        model.compile(loss="sparse_categorical_crossentropy", 
                      optimizer=eval(self.optimizer)(clipvalue=self.clipvalue, lr=self.lr, decay=self.decay))
                
        return model
    
    def sort_seqs(self, seqs):
        #sort by descending length
        lengths = [len(seq) for seq in seqs]
        sorted_idxs = numpy.argsort(lengths)#[::-1]
        seqs = [seqs[idx] for idx in sorted_idxs]
        return seqs
    
    # def prepend_eos(self, seq):
    #     seq = [[seq[sent_idx - 1][-1]] + sent if sent_idx > 0 else sent
    #                    for sent_idx, sent in enumerate(seq)]
    #     return seq

    def fit(self, seqs, lexicon_size=None):
        
        if not hasattr(self, 'model'):
            # for param, value in params.items():
            #     setattr(self, param, value) #set additional parameters not specified when model obj was initialized (e.g. lexicon size)
            self.lexicon_size = lexicon_size
            self.model = self.create_model(n_timesteps=self.n_timesteps, batch_size=self.batch_size)
            self.start_time = timeit.default_timer()

        if self.verbose:
            print "training RNNLM on {} sequences with batch size = {}".format(len(seqs), self.batch_size)
        
        train_losses = []

        # if self.separate_context:

        #     context_seqs = [seq[:-1] for seq in X]
        #     #unroll context seqs
        #     context_seqs = [[word for sent in seq for word in sent] for seq in X]
        #     sents = [seq[-1] for seq in X]
        #     #sort seqs by length
        #     # X = self.sort_seqs(seqs=X)
        #     #import pdb;pdb.set_trace()
        #     for batch_index in range(0, len(X), self.batch_size):
        #         #prep batch
        #         batch_context_seqs = get_batch(seqs=context_seqs[batch_index:batch_index + self.batch_size], batch_size=self.batch_size)
        #         batch_sents = get_batch(seqs=sents[batch_index:batch_index + self.batch_size], batch_size=self.batch_size, padding='post', max_length=self.max_sent_length)
        #         train_loss = self.model.train_on_batch(x=batch_context_seqs, y=batch_sents[:, :, None])
        #         train_losses.append(train_loss)
        #         # self.model.reset_states()                
        #         if batch_index and batch_index % 5000 == 0:
        #             print "processed {} sequences ({:.3f}m)...".format(batch_index, (timeit.default_timer() - self.start_time) / 60)


        # else:
            # if self.batch_size == 1 and not self.n_timesteps:
            #     #process sequences one at a time, one sentence at a time
            #     for seq_idx, seq in enumerate(X):
            #         if type(seq[0]) not in (list, numpy.ndarray, tuple):
            #             seq = [seq]
            #         seq = self.prepend_eos(seq)
            #         if y is not None:
            #             y_seq = y[seq_idx]
            #         for sent_idx, sent in enumerate(seq):
            #             #import pdb;pdb.set_trace()
            #             sent_x = numpy.array(sent)
            #             if y is not None:
            #                 sent_y = numpy.array(y_seq[sent_idx])
            #             else:
            #                 sent_y = sent_x
            #             sent_x = sent_x[None, :-1]
            #             sent_y = sent_y[None, 1:]
            #             sent_y = numpy.expand_dims(sent_y, axis=-1)
            #             assert(sent_x.size > 0 and sent_y.size > 0)
            #             assert(len(sent_x) == len(sent_y))
            #             train_loss = self.model.train_on_batch(x=sent_x, y=sent_y)
            #             train_losses.append(train_loss)
            #         self.model.reset_states()
            #         if (seq_idx + 1) % 1000 == 0:
            #             print("processed {}/{} sequences, loss = {:.3f} ({:.3f}m)...".format(seq_idx + 1, 
            #                 len(X), numpy.mean(train_losses), 
            #                 (timeit.default_timer() - self.start_time) / 60))
                
            # else:
        assert(type(seqs[0][0]) not in (list, tuple, numpy.ndarray))
        #sort seqs by length
        seqs = self.sort_seqs(seqs)
        for batch_index in xrange(0, len(seqs), self.batch_size):
            #prep batch
            batch = get_batch(seqs=seqs[batch_index:batch_index + self.batch_size], batch_size=self.batch_size, n_timesteps=self.n_timesteps)
            for step_index in xrange(0, batch.shape[-1] - 1, self.n_timesteps):
                batch_x = batch[:, step_index:step_index + self.n_timesteps]
                if not numpy.sum(batch_x):
                    #batch is all zeros, skip
                    continue
                batch_y = batch[:, step_index + 1:step_index + self.n_timesteps + 1, None]
                train_loss = self.model.train_on_batch(x=batch_x, y=batch_y)
            train_losses.append(train_loss)
            self.model.reset_states()                
            if batch_index and batch_index % 5000 == 0:
                print "processed {} sequences, loss: {:.3f} ({:.3f}m)...".format(batch_index, numpy.mean(train_losses),
                                                                                (timeit.default_timer() - self.start_time) / 60)

        if self.filepath:
            #save model if filepath given
            self.save()
        if self.verbose:
            print("loss: {:.3f} ({:.3f}m)".format(numpy.mean(train_losses),
                                                (timeit.default_timer() - self.start_time) / 60))

    def get_batch_p_next_words(self, words):
        p_next_words = self.pred_model.predict_on_batch(x=words[:, None])[:, -1]
        return p_next_words

    def pred_batch_next_words(self, p_next_words, mode='max', n_best=1, temp=1.0, prevent_unk=True):

        def sample_word(p_next_word):
            word = theano_rng.choice(size=(n_best,), a=T.arange(p_next_word.shape[0]), replace=True, p=p_next_word, dtype='int64')
            return word

        def init_sample_words(temp):
            #initilize theano function for random sampling
            Temp = T.scalar()
            P_Next_Words = T.matrix('p_next_words', dtype='float64')
            P_Adj_Next_Words = T.nnet.softmax(T.log(P_Next_Words) / Temp)
            Next_Words, Updates = theano.scan(fn=sample_word, sequences=P_Adj_Next_Words)
            sample_words = theano.function([P_Next_Words, Temp], Next_Words, updates=Updates)
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
    
    # def embed_sent(self, sent):
    #     embedded_sent = []
    #     for word in sent:
    #         #convert last predicted word to embedding
    #         if self.lexicon_lookup[word] in self.embeddings:
    #             #next_word = embeddings[lexicon_lookup[next_word]]
    #             embedded_sent.append(self.embeddings[self.lexicon_lookup[word]])
    #         else:
    #             embedded_sent.append(numpy.zeros((self.n_embedding_nodes)))
    #     return embedded_sent

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
        
    def predict(self, seqs, max_length=35, mode='max', batch_size=1, n_best=1, temp=1.0, prevent_unk=True):

        if not hasattr(self, 'pred_model') or batch_size != self.pred_model.input_shape[0]:
            # if self.separate_context:
            #     self.pred_model = self.create_model(batch_size=None, n_timesteps=None)
            # else:
            self.pred_model = self.create_model(batch_size=batch_size, n_timesteps=1)

            #set weights of prediction model
            if self.verbose:
                print "created predictor model"

        self.pred_model.set_weights(self.model.get_weights())


        # if type(X[0][0]) in [list, tuple, numpy.ndarray]:
        #     X = [[word for sent in seq for word in sent] for seq in X]

        # if self.separate_context:

        #     #generate a new word in a given sequence
        #     pred_sents = []
        #     p_pred_sents = []
        #     #merge sents, feed by n_timesteps instead

        #     #generate new sentences       
        #     #import pdb;pdb.set_trace()
        #     for batch_index in range(0, len(X), batch_size):
        #         #prep batch
        #         batch = get_batch(seqs=X[batch_index:batch_index + batch_size], batch_size=batch_size)
        #         #batch_sents = numpy.zeros((batch_size, max_length), dtype='int64')
        #         batch_p_sents = self.pred_model.predict_on_batch(x=batch)
        #         if mode == 'max':
        #             batch_sents = numpy.argmax(batch_p_sents, axis=-1)
        #         elif mode == 'random':
        #             batch_sents = []
        #             for p_sent in batch_p_sents:
        #                 sent = []
        #                 for p_word in p_sent:
        #                     word = self.sample_words(p_word, 1, temp)[0]
        #                     sent.append(word)
        #                 batch_sents.append(sent)
        #             batch_sents = numpy.array(batch_sents)
                        
        #         batch_p_sents = batch_p_sents[numpy.arange(len(batch_sents))[:,None], numpy.arange(batch_sents.shape[1]), batch_sents]

        #         if len(X[batch_index:batch_index + batch_size]) < batch_size:
        #             #remove padding if batch was padded
        #             batch_sents = batch_sents[:len(X[batch_index:batch_index + batch_size])]
        #             batch_p_sents = batch_p_sents[:len(X[batch_index:batch_index + batch_size])]

        #         batch_sents = batch_sents.tolist()
        #         batch_p_sents = batch_p_sents.tolist()

        #         # # import pdb;pdb.set_trace()
        #         # for sent_idx, (sent, p_sent) in enumerate(zip(batch_sents, batch_p_sents)):
        #         #     # sent_length = len(X[batch_index:batch_index + batch_size][sent_idx])
        #         #     # sent = sent[-sent_length:]
        #         #     # p_sent = p_sent[-sent_length:]
        #         #     for word_idx, word in enumerate(sent):
        #         #         if word in eos_tokens:
        #         #             batch_sents[sent_idx] = sent[:word_idx + 1]
        #         #             p_sent = p_sent[:word_idx + 1]
        #         #             break
        #         #     batch_p_sents[sent_idx] = numpy.mean(p_sent)

        #         pred_sents.extend(batch_sents)
        #         p_pred_sents.extend(batch_p_sents)

        #         if batch_index and batch_index % 1000 == 0:
        #             print "generated new sequences for {}/{} inputs...".format(batch_index, len(X))

        #     p_pred_sents = numpy.array(p_pred_sents)
        #     return pred_sents, p_pred_sents

        # else:



            # else:
        #generate a new word in a given sequence
        pred_seqs = []
        #p_pred_sents = []

        #generate new sentences       
        #import pdb;pdb.set_trace()
        for batch_index in xrange(0, len(seqs), batch_size):
            #prep batch
            batch = get_batch(seqs=seqs[batch_index:batch_index + batch_size], batch_size=batch_size)
            batch_pred_seqs = numpy.zeros((batch_size, max_length), dtype='int64')
            #batch_p_sents = numpy.zeros((batch_size, max_length))

            #read context
            #import pdb;pdb.set_trace()
            for step_index in xrange(batch.shape[-1]):
                # if batch.shape[1] - step_index <= self.n_timesteps:
                #     #import pdb;pdb.set_trace()
                #     batch = self.pad_timesteps(seqs=batch)
                p_next_words = self.get_batch_p_next_words(words=batch[:, step_index])

            #now predict
            for idx in xrange(max_length):
                next_words, p_next_words = self.pred_batch_next_words(p_next_words, mode, n_best, temp, prevent_unk)
                batch_pred_seqs[:, idx] = next_words
                #batch_p_sents[:, idx] = p_next_words
                p_next_words = self.get_batch_p_next_words(words=batch_pred_seqs[:, idx])
            self.pred_model.reset_states()

            if len(seqs[batch_index:batch_index + batch_size]) < batch_size:
                #remove padding if batch was padded
                batch_pred_seqs = batch_pred_seqs[:len(seqs[batch_index:batch_index + batch_size])]
                #batch_p_sents = batch_p_sents[:len(X[batch_index:batch_index + batch_size])]
            batch_pred_seqs = batch_pred_seqs.tolist()
            #batch_p_sents = batch_p_sents.tolist()
            # import pdb;pdb.set_trace()
            # for sent_idx, (sent, p_sent) in enumerate(zip(batch_sents, batch_p_sents)):
            #     for word_idx, word in enumerate(sent):
            #         if word in eos_tokens:
            #             batch_sents[sent_idx] = sent[:word_idx + 1]
            #             p_sent = p_sent[:word_idx + 1]
            #             break
            #     batch_p_sents[sent_idx] = numpy.mean(p_sent)

            pred_seqs.extend(batch_pred_seqs)
            #p_pred_sents.extend(batch_p_sents)

            if batch_index and batch_index % 1000 == 0:
                print "generated new sequences for {}/{} inputs...".format(batch_index, len(seqs))

        #p_pred_sents = numpy.array(p_pred_sents)
        return pred_seqs#, p_pred_sents

    def get_probs(self, seqs):
        '''compute probability of given sequences'''
        p_seqs = []

        for batch_index in xrange(0, len(X), batch_size):
            #prep batch
            batch_x = get_batch(seqs=seqs[batch_index:batch_index + batch_size], batch_size=batch_size)
            batch_y = get_batch(seqs=y_seqs[batch_index:batch_index + batch_size], batch_size=batch_size, padding='post')
            batch_p_seqs = numpy.zeros((batch_size, batch_y.shape[-1]))

            #read context
            #import pdb;pdb.set_trace()
            for step_idx in xrange(batch_x.shape[-1]):
                # if batch.shape[1] - step_index <= self.n_timesteps:
                #     #import pdb;pdb.set_trace()
                #     batch = self.pad_timesteps(seqs=batch)
                p_next_words = self.get_batch_p_next_words(words=batch_x[:, step_idx])

            #now predict
            for step_idx in xrange(batch_y.shape[-1]):
                p_next_words = p_next_words[numpy.arange(len(batch_y)), batch_y[:, step_idx]]
                batch_p_seqs[:, step_idx] = p_next_words
                p_next_words = self.get_batch_p_next_words(words=batch_y[:, step_idx])

            self.pred_model.reset_states()

            if len(X[batch_index:batch_index + batch_size]) < batch_size:
                #remove padding if batch was padded
                batch_p_seqs = batch_p_sents[:len(X[batch_index:batch_index + batch_size])]

            batch_p_seqs = batch_p_seqs.tolist()

            #import pdb;pdb.set_trace()
            batch_seq_lengths = [len(y_seqs[batch_index:batch_index + batch_size][idx]) for idx in xrange(len(batch_p_seqs))]
            batch_p_seqs = [p_seq[:seq_length] for p_seq, seq_length in zip(batch_p_sents, batch_seq_lengths)]
            batch_p_seqs = [numpy.mean(p_seq) for p_seq in batch_p_seqs]

            p_seqs.extend(batch_p_seqs)

        return p_seqs

    def evaluate(self, seqs):
        '''compute prob of given sequences'''
        losses_keras = []
        #losses_self = []
        eval_model = self.create_model(batch_size=self.batch_size)
        eval_model.set_weights(self.model.get_weights())
        for batch_index in xrange(0, len(seqs), self.batch_size):
            batch = get_batch(seqs=seqs[batch_index:batch_index + self.batch_size], batch_size=self.batch_size)
            batch_x = batch[:, :-1]
            batch_y = batch[:, 1:]
            # batch_probs = eval_model.predict_on_batch(batch_x)
            # batch_probs = batch_probs[numpy.arange(len(batch_y))[:,None], numpy.arange(batch_y.shape[-1]), batch_y]
            # batch_loss_self = [-numpy.mean(numpy.log(probs[numpy.where(x > 0)])) for x,probs in zip(batch_x, batch_probs)] 
            # #batch_probs = batch_probs[numpy.where(batch_x > 0)] # filter matrix padding
            # losses_self.append(batch_loss_self)
            # eval_model.reset_states()
            batch_loss_keras = eval_model.test_on_batch(batch_x, batch_y[:,:,None])
            losses_keras.append(batch_loss_keras)
            eval_model.reset_states()
        perplexity_keras = numpy.exp(numpy.mean(losses_keras))
        #perplexity_self = numpy.exp(numpy.mean(losses_self))
        return perplexity_keras#, perplexity_self


    def create_encoder(self):

        self.encoder_model = self.create_model(n_timesteps=None, batch_size=1, pred_layer=False)
        self.encoder_model.set_weights(self.model.get_weights()[:-2])
        if self.verbose:
            print "created encoder model"

    def get_embeddings(self):

        embeddings = self.model.get_weights()[0]
        return embeddings


class FeatureRNNLM(RNNLM):
    def __init__(self, n_feature_nodes=100, **params):
        RNNLM.__init__(self, **params)
        self.n_feature_nodes = n_feature_nodes

    def create_model(self, n_timesteps=1, batch_size=1, include_pred_layer=True):

        seq_input_layer = Input(batch_shape=(batch_size, n_timesteps), name="seq_input_layer")
        seq_embedding_layer = Embedding(input_dim=self.lexicon_size + 1, 
                                        output_dim=self.n_embedding_nodes, mask_zero=True, name='seq_embedding_layer')(seq_input_layer)
        seq_hidden_layer = GRU(output_dim=self.n_hidden_nodes, return_sequences=True, stateful=True, name='seq_hidden_layer')(seq_embedding_layer)

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
            # self.n_timesteps = 1
            #self.context_size = lexicon_size
            self.model = self.create_model(n_timesteps=self.n_timesteps, batch_size=self.batch_size)
            self.start_time = timeit.default_timer()

        if self.verbose:
            print("training FeatureRNNLM on {} sequences with batch size = {}".format(len(seqs), self.batch_size))
        
        train_losses = []

        # assert(type(seqs[0][0]) not in (list, tuple, numpy.ndarray))
        assert(len(seqs) == len(feature_vecs))
        #sort seqs by length
        seqs, feature_vecs = self.sort_seqs(seqs, feature_vecs)
        for batch_index in xrange(0, len(seqs), self.batch_size):
            #prep batch
            batch = get_batch(seqs=seqs[batch_index:batch_index + self.batch_size], batch_size=self.batch_size, n_timesteps=self.n_timesteps)
            batch_features = feature_vecs[batch_index:batch_index + self.batch_size]
            #batch_features = self.context_seqs_to_vecs(batch_context)
            #batch_context = numpy.insert(batch_context, 0, numpy.zeros(len(batch_context)), axis=1) #prepend column of zeros for processing first word of sequence (where no context exists)
            for step_index in xrange(0, batch.shape[-1] - 1, self.n_timesteps):
                batch_x = batch[:, step_index:step_index + self.n_timesteps]
                 #get context for entire sequence up to this point
                if not numpy.sum(batch_x):
                    #batch is all zeros, skip
                    continue
                batch_y = batch[:, step_index + 1:step_index + self.n_timesteps + 1, None]
                train_loss = self.model.train_on_batch(x=[batch_x, batch_features], y=batch_y)
            train_losses.append(train_loss)
            self.model.reset_states()                
            if batch_index and batch_index % 5000 == 0:
                print "processed {} sequences, loss: {:.3f} ({:.3f}m)...".format(batch_index, numpy.mean(train_losses),
                                                                                (timeit.default_timer() - self.start_time) / 60)

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
                print "created predictor model"

        self.pred_model.set_weights(self.model.get_weights())

        pred_seqs = []

        for batch_index in xrange(0, len(seqs), batch_size):
            #prep batch
            batch = get_batch(seqs=seqs[batch_index:batch_index + batch_size], batch_size=batch_size)
            batch_features = feature_vecs[batch_index:batch_index + batch_size]
            #batch_features = numpy.insert(batch_features, 0, numpy.zeros(len(batch_features)), axis=1) #prepend column of zeros for processing first word of sequence (where no context exists)
            
            batch_pred_seqs = numpy.zeros((batch_size, max_length), dtype='int64')

            for idx in xrange(batch.shape[-1]): #read in given sequence from which to predict
                p_next_words = self.get_batch_p_next_words(words=batch[:, idx], features=batch_features)

            for idx in xrange(max_length): #now predict
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
                print "generated new sequences for {}/{} inputs...".format(batch_index, len(seqs))

        return pred_seqs

    # def context_seqs_to_vecs(self, seqs):
    #     '''takes sequences of numbers as input and returns bag-of-words vectors'''
    #     count_vecs = []
    #     for seq in seqs:
    #         count_vec = numpy.bincount(numpy.array(seq), minlength=self.lexicon_size + 1)
    #         count_vecs.append(count_vec)
    #     count_vecs = numpy.array(count_vecs)
    #     count_vecs[:,0] = 0 #don't include 0s in vector (0's are words that are not part of context)
    #     return count_vecs


def max_margin(y_true, y_pred):
    return K.sum(K.maximum(0., 1. - y_pred*y_true + y_pred*(1. - y_true)))


class SeqBinaryClassifier(SavedModel, KerasClassifier):
    def __init__(self, context_size, lexicon_size=None, max_length=None, n_embedding_nodes=300, batch_size=None, 
                 n_hidden_layers=1, n_hidden_nodes=200, verbose=1, embedded_input=True, optimizer='RMSprop', clipvalue=numpy.inf,
                 filepath=None, use_dropout=False, pairs=False, save_freq=20):
        
        self.batch_size = batch_size
        self.lexicon_size = lexicon_size
        self.max_length = max_length
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_nodes = n_hidden_nodes
        self.n_embedding_nodes = n_embedding_nodes
        self.context_size = context_size
        self.verbose = verbose
        self.embedded_input = embedded_input
        self.use_dropout = use_dropout
        self.pairs = pairs
        self.clipvalue = clipvalue
        self.optimizer = optimizer
        self.update_num = 0
        self.save_freq = save_freq
        if filepath and not os.path.isdir(filepath):
            os.mkdir(filepath)
        self.filepath = filepath

        ####################CURRENT METHOD######################
        context_input_layer = Input(batch_shape=(self.batch_size, self.context_size, self.n_embedding_nodes), name="context_input_layer")

        seq_input_layer = Input(batch_shape=(self.batch_size, self.n_embedding_nodes), name="seq_input_layer")

        reshape_seq_layer = Reshape((1, self.n_embedding_nodes))(seq_input_layer)

        merge_layer = merge([context_input_layer, reshape_seq_layer], mode='concat', concat_axis=-2, name='merge_layer')

        mask_layer = Masking(mask_value=0.0, input_shape=(self.context_size + 1, self.n_embedding_nodes))(merge_layer)

        hidden_layer = GRU(output_dim=self.n_hidden_nodes, return_sequences=False, stateful=False, name='context_hidden_layer')(mask_layer)#(merge_layer)

        pred_layer = Dense(output_dim=1, activation='sigmoid', name='pred_layer')(hidden_layer)

        self.model = Model(input=[context_input_layer, seq_input_layer], output=pred_layer)
        ##########################################################

        ################NAOYA METHOD##############################

        # context_input_layer = Input(batch_shape=(self.batch_size, self.context_size, self.n_embedding_nodes), name="context_input_layer")

        # seq_input_layer = Input(batch_shape=(self.batch_size, self.n_embedding_nodes), name="seq_input_layer")

        # hidden_layer = GRU(output_dim=self.n_hidden_nodes, return_sequences=False, 
        #                             stateful=False, name='context_hidden_layer')(context_input_layer)

        # context_dense_layer = Dense(output_dim=self.n_hidden_nodes, activation='tanh', name='context_dense_layer')(hidden_layer)

        # seq_dense_layer = Dense(output_dim=self.n_hidden_nodes, activation='tanh', name='seq_dense_layer')(seq_input_layer)

        # merge_layer = merge([context_dense_layer, seq_dense_layer], mode='dot', name='merge_layer')

        # model = Model(input=[context_input_layer, seq_input_layer], output=merge_layer)

        ############################################################

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

        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])#loss='mean_squared_error', 'sparse_categorical_crossentropy'
        
        if self.verbose:
            print "CREATED SeqBinary model: embedding layer nodes = {}, " \
                    "hidden layers = {}, hidden layer nodes = {}, dropout = {}, " \
                    "optimizer = {}, batch size = {}".format(self.n_embedding_nodes, self.n_hidden_layers,
                                         self.n_hidden_nodes, self.use_dropout, optimizer, self.batch_size)
        
    
    def fit(self, X, y, y_seqs=None):

        if len(X.shape) < 3: #make sure X has 3 dimensions (n_samples, context_size, encoder_dim) (context may consist of a single sentence)
            X = X[:, None, :]

        model_input = [X, y_seqs]
    
        #sentences up to last are context, last sentence is ending to be judged as correct
        history = self.model.fit(model_input, y, nb_epoch=1)
        
        print "loss: {:.3f}, accuracy: {:.3f}".format(history.history['loss'][0], history.history['acc'][0])

        self.update_num += 1

        #save model if filepath given
        if self.filepath and (self.update_num % self.save_freq == 0):
            self.save()

    def predict(self, X, y_seqs):
        #import pdb;pdb.set_trace()

        if len(X.shape) < 2: #make sure X has 3 dimensions (n_samples, context_size, encoder_dim) (context may consist of a single sentence)
            X = X[None, None, :]
        elif len(X.shape) < 3:
            X = X[:, None, :]

        probs_y = []
            
        if type(y_seqs) is numpy.ndarray:
            if len(y_seqs.shape) < 2:
                y_seqs = y_seqs[None, None, :]
            elif len(y_seqs.shape) < 3:
                y_seqs = y_seqs[:, None, :]

            for choice_idx in xrange(y_seqs.shape[1]):
                choices = y_seqs[:, choice_idx]
                probs = self.model.predict([X, choices])[:,-1]
                probs_y.append(probs)

            probs_y = numpy.stack(probs_y, axis=1)
            pred_y = numpy.argmax(probs_y, axis=1)

        else:
            for seq, y_seq in zip(X, y_seqs):
                probs_choice = []
                for choice_idx in xrange(len(y_seq)):
                    #choices = y_seq[choice_idx]
                    probs = self.model.predict([seq[None], y_seq[None, choice_idx]])[:,-1][0]
                    probs_choice.append(probs)
                probs_y.append(numpy.array(probs_choice))

            pred_y = numpy.array([numpy.argmax(probs) for probs in probs_y])

        assert(len(pred_y) == len(probs_y) == len(X))
            
        return probs_y, pred_y

    # def analyze(self, X):

    #     y_wrt_X = T.grad(self.model.layers[-1].output, self.model.layers[0].input)
    #     get_y_wrt_X = theano.function([self.model.layers[0].input], y_wrt_X)
    #     y_wrt_X_output = get_y_wrt_X(X)
    #     return y_wrt_X_output


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
            print "CREATED MLPLM: embedding layer nodes = {}, hidden layers = {}, " \
                    "hidden layer nodes = {}, optimizer = {} with lr = {}, " \
                    "clipvalue = {}, and decay = {}".format(
                    self.n_embedding_nodes, self.n_hidden_layers, self.n_hidden_nodes, 
                    self.optimizer, self.lr, self.clipvalue, self.decay)
    
    def create_model(self, n_timesteps, batch_size=1, pred_layer=True):
        
        model = Sequential()
        
        # if self.embeddings is None:
        model.add(Embedding(self.lexicon_size + 1, self.n_embedding_nodes,
                            batch_input_shape=(batch_size, n_timesteps)))#, mask_zero=True))

        model.add(Reshape((self.n_embedding_nodes * n_timesteps,)))

        for layer_num in xrange(self.n_hidden_layers):
            model.add(Dense(self.n_hidden_nodes, batch_input_shape=(batch_size, n_timesteps, self.n_embedding_nodes), activation='tanh'))

        if pred_layer: 
            model.add(Dense(self.lexicon_size + 1, activation="softmax"))
            
        #select optimizer and compile
        model.compile(loss="sparse_categorical_crossentropy", 
                      optimizer=eval(self.optimizer)(clipvalue=self.clipvalue, lr=self.lr, decay=self.decay))
                
        return model

    def fit(self, seqs, **params):
        if not hasattr(self, 'model'):
            for param, value in params.items():
                setattr(self, param, value) #set additional parameters not specified when model obj was initialized (e.g. lexicon size)
            self.model = self.create_model(n_timesteps=self.n_timesteps, batch_size=self.batch_size)
            self.start_time = timeit.default_timer()
            #self.epoch = 0

        assert(type(X[0][0]) not in [list, tuple, numpy.ndarray])
            # X = [[word for sent in seq for word in sent] for seq in X] #merge sents, feed by n_timesteps instead

        X = [[seq[idx:idx+self.n_timesteps+1] for idx in xrange(len(seq) - self.n_timesteps)] for seq in seqs]
        X = numpy.array([ngram for seq in X for ngram in seq])
        y = X[:, -1][:, None]
        X = X[:, :-1]

        train_loss = self.model.fit(X, y, nb_epoch=params['n_epochs'] if 'n_epochs' in params else 1, batch_size=self.batch_size)

        if self.filepath:
            #save model after each epoch if filepath given
            self.save()
        #self.epoch += 1
        if self.verbose:
            print("loss: {:.3f} ({:.3f}m)".format(numpy.mean(train_loss.history['loss']),
                                       (timeit.default_timer() - self.start_time) / 60))

    def pred_next_words(self, p_next_words, mode='max', n_best=1, temp=1.0):

        def sample_word(p_next_word):
            word = theano_rng.choice(size=(n_best,), a=T.arange(p_next_word.shape[0]), replace=True, p=p_next_word, dtype='int64')
            return word

        def init_sample_words(temp):
            #initilize theano function for random sampling
            P_Next_Words = T.matrix('p_next_words', dtype='float64')
            P_Adj_Next_Words = T.nnet.softmax(T.log(P_Next_Words) / temp)
            Next_Words, Updates = theano.scan(fn=sample_word, sequences=P_Adj_Next_Words)
            sample_words = theano.function([P_Next_Words], Next_Words, updates=Updates)
            return sample_words

        if mode == 'random':
            #numpy is too slow at random sampling, so use theano
            if not hasattr(self, 'sample_words') or not self.sample_words:
                self.sample_words = init_sample_words(temp)
            next_words = self.sample_words(p_next_words)
            #next_words = numpy.array([self.sample_word(p_next_word, n_best, temp)[0] for p_next_word in p_next_words])
        else:
            #next_words = numpy.array([numpy.argmax(p_next_word) for p_next_word in p_next_words])
            next_words = numpy.argmax(p_next_words, axis=1)[:, None]

        #p_next_words = p_next_words[numpy.arange(len(p_next_words)), next_words]
        #p_next_words = p_next_words[numpy.arange(len(p_next_words))[:,None], next_words]

        #for right now samples will always be size 1
        #next_words = next_words[:, 0]
        #p_next_words = p_next_words[:, 0]
        return next_words#, p_next_words

    def predict(self, seqs, max_length=35, mode='max', batch_size=1, n_best=1, temp=1.0):

        # if not hasattr(self, 'pred_model') or batch_size != self.pred_model.input_shape[0]:
        #     # if self.batch_size > 1:
        #     #if model uses batch training, create a duplicate model with batch size 1 for prediction
        #     self.pred_model = self.create_model(batch_size=batch_size, n_timesteps=self.n_timesteps)

            #set weights of prediction model
            # if self.verbose:
            #     print "created predictor model"

       # self.pred_model.set_weights(self.model.get_weights())

        assert(type(X[0][0]) not in [list, tuple, numpy.ndarray])
            # X = [[word for sent in seq for word in sent] for seq in X] #merge sents, feed by n_timesteps instead

        X = [[seq[idx:idx+self.n_timesteps] for idx in xrange(len(seq) - self.n_timesteps + 1)] for seq in seqs]
        X = numpy.array([seq[-1] for seq in X]) #only predict from last ngram in each sequence

        for idx in xrange(max_length):
            p_next_words = self.model.predict(X[:, -self.n_timesteps:])
            next_words = self.pred_next_words(p_next_words, mode, n_best, temp)
            X = numpy.append(X, next_words, axis=1)

        X = X[:, self.n_timesteps:]
        pred_seqs = list(X)
        # for sent in X:
        #     # for word_idx, word in enumerate(sent):
        #     #     if word in eos_tokens:
        #     #         sent = sent[:word_idx + 1]
        #     #         #p_sent = p_sent[:word_idx + 1]
        #     #         break
        #     pred_sents.append(sent)

        #p_pred_sents = numpy.array(p_pred_sents)
        return pred_seqs#, None #None is placeholder for probability (not yet implemented)

    def evaluate(self, seqs):
        '''compute prob of given sequences'''
        losses = []
        #losses_self = []
        # eval_model = self.create_model(batch_size=self.batch_size)
        # eval_model.set_weights(self.model.get_weights())

        assert(type(X[0][0]) not in [list, tuple, numpy.ndarray])
            # X = [[word for sent in seq for word in sent] for seq in X] #merge sents, feed by n_timesteps instead

        X = [[seq[idx:idx+self.n_timesteps+1] for idx in xrange(len(seq) - self.n_timesteps)] for seq in seqs]
        X = numpy.array([ngram for seq in X for ngram in seq])
        y = X[:, -1][:, None]
        X = X[:, :-1]

        loss = self.model.evaluate(X, y)
        #eval_model.reset_states()
        perplexity = numpy.exp(loss)
        #perplexity_self = numpy.exp(numpy.mean(losses_self))
        return perplexity #, perplexity_self



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
            print "CREATED Sequence2Sequence model: embedding layer sizes:", self.n_embedding_nodes, ",",\
            self.n_encoding_layers, "encoding layers with size:", self.n_hidden_nodes, ",",\
            self.n_decoding_layers, "decoding layers with size:", self.n_hidden_nodes, ",",\
            "optimizer:", optimizer, ", batch_size:", self.batch_size, ", stateful =", self.stateful
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
            for epoch in xrange(nb_epoch):
                train_losses = []
                for batch_index in xrange(0, len(X), self.batch_size):
                    batch_num = batch_index / self.batch_size + 1
                    for sent_index in xrange(X.shape[1]):
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
            for batch_index in xrange(0, len(X), self.batch_size):
                for sent_index in xrange(X.shape[1]):
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
                for choice_index in xrange(batch_choices.shape[1]):
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
            print "CREATED MergeSequence model: embedding layer sizes =", self.n_embedding_nodes, ",",\
            self.n_encoding_layers, "encoding layers with size", self.n_hidden_nodes, ",",\
            self.n_decoding_layers, "decoding layers with size", self.n_hidden_nodes, ",",\
            "lexicon size = ", self.lexicon_size, ",",\
            "batch size = ", self.batch_size
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

