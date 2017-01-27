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
    #load entire classifier object
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
        del attrs['model']
        if 'pred_model' in attrs:
            del attrs['pred_model']
        if 'encoder_model' in attrs:
            del attrs['encoder_model']
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

# def sample_words(self, p_next_word, n_samples, temp=1.0):
#     #use theano for random sampling because numpy is too slow

#     def get_sample(p_next_word):
#         p_next_word = T.log(prob) / temp
#         p_next_word = T.nnet.softmax(p_next_word)
#         sample = theano_rng.choice(size=(n_samples,), a=T.arange(p_next_word.shape[0]), replace=True, p=p_next_word, dtype='int64')
#         return sample

#     # p_next_word = numpy.log(p_next_word) / temp
#     # p_next_word = numpy.exp(p_next_word) / numpy.sum(numpy.exp(p_next_word))
#     # next_words = numpy.random.choice(size=n_samples, a=p_next_word.shape[-1], p=p_next_word, replace=False)
#     return next_words


class RNNLM(KerasClassifier, SavedModel):
    def __call__(self, lexicon_size, n_timesteps=None, n_embedding_nodes=300, n_hidden_nodes=250, n_hidden_layers=1,
                 embeddings=None, batch_size=1, max_length=None, max_sent_length=None, verbose=1, filepath=None,
                 optimizer='Adam', lr=0.001, clipvalue=5.0, decay=1e-6, separate_context=False):
        
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
        self.separate_context = separate_context
        self.max_sent_length = max_sent_length
        self.sample_words = None

        model = self.create_model(n_timesteps=self.n_timesteps, batch_size=self.batch_size)

        if self.verbose:
            print "CREATED RNNLM: embedding layer nodes = {}, hidden layers = {}, " \
                    "hidden layer nodes = {}, optimizer = {} with lr = {}, " \
                    "clipvalue = {}, and decay = {}".format(
                    self.n_embedding_nodes, self.n_hidden_layers, self.n_hidden_nodes, 
                    self.optimizer, self.lr, self.clipvalue, self.decay)

        return model
    
    def create_model(self, n_timesteps=None, batch_size=1, pred_layer=True):

        if self.separate_context:

            context_input_layer = Input(batch_shape=(batch_size, n_timesteps), 
                                        dtype='int32', name='context_input_layer')

            # seq_input_layer = Input(batch_shape=(self.batch_size, n_timesteps), 
            #                         dtype='int32', name='seq_input_layer')

            embedded_context_layer = Embedding(input_dim = self.lexicon_size + 1,
                                        output_dim=self.n_embedding_nodes,
                                        mask_zero=True, name='embedding_layer')(context_input_layer)

            # embedded_seq_layer = embedding_layer(seq_input_layer)

            context_hidden_layer = GRU(output_dim=self.n_hidden_nodes, return_sequences=False, stateful=False, name='context_hidden_layer')(embedded_context_layer)

            repeat_layer = RepeatVector(self.max_sent_length)(context_hidden_layer)

            # merge_layer = merge([context_hidden_layer, embedded_seq_layer], mode='concat', concat_axis=-1)

            seq_hidden_layer = GRU(output_dim=self.n_hidden_nodes, return_sequences=True, stateful=False, name='seq_hidden_layer')(repeat_layer)#(context_hidden_layer)#(merge_layer)

            pred_layer = TimeDistributed(Dense(self.lexicon_size + 1, activation="softmax", name='pred_layer'))(seq_hidden_layer)

            model = Model(input=context_input_layer, output=pred_layer)

        else:
        
            model = Sequential()
            
            # if self.embeddings is None:
            model.add(Embedding(self.lexicon_size + 1, self.n_embedding_nodes,
                                batch_input_shape=(batch_size, n_timesteps), mask_zero=True))

            for layer_num in xrange(self.n_hidden_layers):
                model.add(GRU(self.n_hidden_nodes, 
                              batch_input_shape=(batch_size, n_timesteps, self.n_embedding_nodes),
                              return_sequences=True, stateful=True))

            if pred_layer: 
                model.add(TimeDistributed(Dense(self.lexicon_size + 1, activation="softmax")))
            
        #select optimizer and compile
        model.compile(loss="sparse_categorical_crossentropy", 
                      optimizer=eval(self.optimizer)(clipvalue=self.clipvalue, lr=self.lr, decay=self.decay))
                
        return model
    
    def sort_seqs(self, seqs):
        #sort by descending length
        # if self.embeddings is not None:
        #     sorted_idxs = numpy.argsort((y > 0).sum(axis=-1))[::-1]
        # else:
        lengths = [len(seq) for seq in seqs]
        #sorted_idxs = numpy.argsort((X > 0).sum(axis=-1))[::-1]
        sorted_idxs = numpy.argsort(lengths)#[::-1]
        #X = X[sorted_idxs]
        seqs = [seqs[idx] for idx in sorted_idxs]
        # if y is not None:
            #y = y[sorted_idxs]
        return seqs#, y
    
    def prepend_eos(self, seq):
        seq = [[seq[sent_idx - 1][-1]] + sent if sent_idx > 0 else sent
                       for sent_idx, sent in enumerate(seq)]
        return seq

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

        if self.separate_context:

            context_seqs = [seq[:-1] for seq in X]
            #unroll context seqs
            context_seqs = [[word for sent in seq for word in sent] for seq in X]
            sents = [seq[-1] for seq in X]
            #sort seqs by length
            # X = self.sort_seqs(seqs=X)
            #import pdb;pdb.set_trace()
            for batch_index in range(0, len(X), self.batch_size):
                #prep batch
                batch_context_seqs = get_batch(seqs=context_seqs[batch_index:batch_index + self.batch_size], batch_size=self.batch_size)
                batch_sents = get_batch(seqs=sents[batch_index:batch_index + self.batch_size], batch_size=self.batch_size, padding='post', max_length=self.max_sent_length)
                train_loss = self.model.train_on_batch(x=batch_context_seqs, y=batch_sents[:, :, None])
                train_losses.append(train_loss)
                # self.model.reset_states()                
                if batch_index and batch_index % 5000 == 0:
                    print "processed {} sequences in epoch {} ({:.3f}m)...".format(batch_index, self.epoch + 1, (timeit.default_timer() - self.start_time) / 60)


        else:
            if self.batch_size == 1 and not self.n_timesteps:
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
                
            else:
                #merge sents, feed by n_timesteps instead
                X = [[word for sent in seq for word in sent] for seq in X]
                #sort seqs by length
                X = self.sort_seqs(seqs=X)
                for batch_index in range(0, len(X), self.batch_size):
                    #prep batch
                    batch = get_batch(seqs=X[batch_index:batch_index + self.batch_size], batch_size=self.batch_size, n_timesteps=self.n_timesteps)
                    for step_index in xrange(0, batch.shape[-1] - 1, self.n_timesteps):
                        # if batch.shape[1] - step_index <= self.n_timesteps:
                        #     #import pdb;pdb.set_trace()
                        #     batch = self.pad_timesteps(seqs=batch)
                        batch_x = batch[:, step_index:step_index + self.n_timesteps]
                        if not numpy.sum(batch_x):
                            #batch is all zeros, skip
                            continue
                        batch_y = batch[:, step_index + 1:step_index + self.n_timesteps + 1, None]
                        train_loss = self.model.train_on_batch(x=batch_x, y=batch_y)
                    train_losses.append(train_loss)
                    self.model.reset_states()                
                    if batch_index and batch_index % 5000 == 0:
                        print "processed {} sequences in epoch {}, loss: {:.3f} ({:.3f}m)...".format(batch_index, self.epoch + 1,
                                                                                                        numpy.mean(train_losses),
                                                                                                        (timeit.default_timer() - self.start_time) / 60)

        if self.filepath:
            #save model after each epoch if filepath given
            self.save()
        self.epoch += 1
        if self.verbose:
            print("epoch {} loss: {:.3f} ({:.3f}m)".format(self.epoch, numpy.mean(train_losses),
                                       (timeit.default_timer() - self.start_time) / 60))

    def get_p_next_word(self, seq):
        if type(seq[0]) not in (list, tuple, numpy.ndarray):
            seq = [seq]
        for sent in seq:
            p_next_word = self.pred_model.predict_on_batch(x=numpy.array(sent)[None])[0][-1]
        assert(len(p_next_word.shape) == 1)
        return p_next_word

    def get_batch_p_next_words(self, words):
        p_next_words = self.pred_model.predict_on_batch(x=words[:, None])[:, -1]
        #assert(len(p_next_word.shape) == 1)
        return p_next_words

    def pred_batch_next_words(self, p_next_words, mode='max', n_best=1, temp=1.0):

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
            next_words = numpy.argmax(p_next_words, axis=1)

        #p_next_words = p_next_words[numpy.arange(len(p_next_words)), next_words]
        p_next_words = p_next_words[numpy.arange(len(p_next_words))[:,None], next_words]

        #for right now samples will always be size 1
        next_words = next_words[:, 0]
        p_next_words = p_next_words[:, 0]
        return next_words, p_next_words
    
    # def pred_next_words(self, seq, mode='max', n_best=1, temp=1.0):
    #     #use grid search to predict next word given current best predicted sequences
    #     #import pdb;pdb.set_trace()
    #     assert(mode == 'max' or mode == 'random')

    #     p_next_word = self.get_p_next_word(seq)
        
    #     if mode == 'random':
    #         #import pdb;pdb.set_trace()
    #         next_words = self.sample_words(p_next_word, n_best, temp)
    #     else:
    #         #_next_word = p_next_word[1:] #never predict zeros
    #         next_words = numpy.argsort(p_next_word)[::-1][:n_best]

    #     p_words = p_next_word[next_words]

    #     return next_words, p_words

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
    
    # def pred_sents(self, context_seq, sents, mode, n_best, temp, eos_markers):
    #     #import pdb;pdb.set_trace()
    #     new_sents = []
    #     p_new_words = []
    #     if not sents:
    #         sents = [[]]
    #     for sent in sents:
    #         if sent:
    #             if self.check_if_null(sent):
    #                 #if None generated, remove sentence from consideration
    #                 continue
    #             elif self.check_if_eos(sent, eos_markers):
    #                 #reached end of sentence marker in generated sentence, so stop generating
    #                 new_sents.append(sent)
    #                 continue
    #             context_seq = context_seq + [sent]
    #         # if self.embeddings is not None:
    #         #     embedded_sent = self.embed_sent(sent)#, embeddings, lexicon_lookup)
    #         #     next_words = self.pred_next_words(context_seq, embedded_sent, mode, n_best, temp)
    #         # else:
    #         next_words, p_words = self.pred_next_words(context_seq, mode, n_best, temp)
    #         self.pred_model.reset_states()
    #         ext_sents = self.extend_sent(sent=sent, words=next_words)
    #         new_sents.extend(ext_sents)
    #         p_new_words.extend(p_words)

    #     new_sents, p_sents = self.get_best_sents(new_sents, p_new_words, n_best)

    #     return new_sents, p_sents
    
    def check_if_eos(self, sent, eos_markers):
        #check if an end-of-sentence marker has been generated for this sentence
        if sent[-1] in eos_markers:
            return True
        return False

    def check_if_null(self, sent):
        if sent[-1] == 0:
            return True
        return False
    
    def get_best_sents(self, sents, p_sents, n_best):
        
        best_idxs = numpy.argsort(numpy.array(p_sents))[::-1][:n_best]
        best_sents = [sents[idx] for idx in best_idxs]
        #return probs of best sents as well as sents
        p_sents = numpy.array(p_sents)[best_idxs]
        
        return best_sents, p_sents
        
    def predict(self, X, y_seqs=None, n_words=20, mode='max', batch_size=1, n_best=1, temp=1.0, eos_markers=[], **kwargs):

        if not hasattr(self, 'pred_model') or batch_size != self.pred_model.input_shape[0]:
            # if self.batch_size > 1:
            #if model uses batch training, create a duplicate model with batch size 1 for prediction
            if self.separate_context:
                self.pred_model = self.create_model(batch_size=None, n_timesteps=None)
            else:
                self.pred_model = self.create_model(batch_size=batch_size, n_timesteps=1)

            #set weights of prediction model
            if self.verbose:
                print "created predictor model"

        self.pred_model.set_weights(self.model.get_weights())


        if type(X[0][0]) in [list, tuple, numpy.ndarray]:
            X = [[word for sent in seq for word in sent] for seq in X]

        if self.separate_context:

            #generate a new word in a given sequence
            pred_sents = []
            p_pred_sents = []
            #merge sents, feed by n_timesteps instead

            #generate new sentences       
            #import pdb;pdb.set_trace()
            for batch_index in range(0, len(X), batch_size):
                #prep batch
                batch = get_batch(seqs=X[batch_index:batch_index + batch_size], batch_size=batch_size)
                #batch_sents = numpy.zeros((batch_size, n_words), dtype='int64')
                batch_p_sents = self.pred_model.predict_on_batch(x=batch)
                if mode == 'max':
                    batch_sents = numpy.argmax(batch_p_sents, axis=-1)
                elif mode == 'random':
                    batch_sents = []
                    for p_sent in batch_p_sents:
                        sent = []
                        for p_word in p_sent:
                            word = self.sample_words(p_word, 1, temp)[0]
                            sent.append(word)
                        batch_sents.append(sent)
                    batch_sents = numpy.array(batch_sents)
                        
                batch_p_sents = batch_p_sents[numpy.arange(len(batch_sents))[:,None], numpy.arange(batch_sents.shape[1]), batch_sents]

                #read context
                # #import pdb;pdb.set_trace()
                # for step_index in xrange(batch.shape[-1]):
                #     # if batch.shape[1] - step_index <= self.n_timesteps:
                #     #     #import pdb;pdb.set_trace()
                #     #     batch = self.pad_timesteps(seqs=batch)
                #     p_next_words = self.get_batch_p_next_words(words=batch[:, step_index])

                # #now predict
                # for idx in range(n_words):
                #     next_words, p_next_words = self.pred_batch_next_words(p_next_words, mode, n_best, temp)
                #     batch_sents[:, idx] = next_words
                #     batch_p_sents[:, idx] = p_next_words
                #     p_next_words = self.get_batch_p_next_words(words=batch_sents[:, idx])
                # self.pred_model.reset_states()

                if len(X[batch_index:batch_index + batch_size]) < batch_size:
                    #remove padding if batch was padded
                    batch_sents = batch_sents[:len(X[batch_index:batch_index + batch_size])]
                    batch_p_sents = batch_p_sents[:len(X[batch_index:batch_index + batch_size])]

                batch_sents = batch_sents.tolist()
                batch_p_sents = batch_p_sents.tolist()

                # import pdb;pdb.set_trace()
                for sent_idx, (sent, p_sent) in enumerate(zip(batch_sents, batch_p_sents)):
                    # sent_length = len(X[batch_index:batch_index + batch_size][sent_idx])
                    # sent = sent[-sent_length:]
                    # p_sent = p_sent[-sent_length:]
                    for word_idx, word in enumerate(sent):
                        if word in eos_markers:
                            batch_sents[sent_idx] = sent[:word_idx + 1]
                            p_sent = p_sent[:word_idx + 1]
                            break
                    batch_p_sents[sent_idx] = numpy.mean(p_sent)

                pred_sents.extend(batch_sents)
                p_pred_sents.extend(batch_p_sents)

                if batch_index and batch_index % 1000 == 0:
                    print "generated new sequences for {}/{} inputs...".format(batch_index, len(X))

            p_pred_sents = numpy.array(p_pred_sents)
            return pred_sents, p_pred_sents

        else:

            if y_seqs is not None:
                #if y sequences given, return probability of these sequences
                p_sents = []

                for batch_index in range(0, len(X), batch_size):
                    #prep batch
                    batch_x = get_batch(seqs=X[batch_index:batch_index + batch_size], batch_size=batch_size)
                    batch_y = get_batch(seqs=y_seqs[batch_index:batch_index + batch_size], batch_size=batch_size, padding='post')
                    batch_p_sents = numpy.zeros((batch_size, batch_y.shape[-1]))

                    #read context
                    #import pdb;pdb.set_trace()
                    for step_idx in xrange(batch_x.shape[-1]):
                        # if batch.shape[1] - step_index <= self.n_timesteps:
                        #     #import pdb;pdb.set_trace()
                        #     batch = self.pad_timesteps(seqs=batch)
                        p_next_words = self.get_batch_p_next_words(words=batch_x[:, step_idx])

                    #now predict
                    for step_idx in range(batch_y.shape[-1]):
                        p_next_words = p_next_words[numpy.arange(len(batch_y)), batch_y[:, step_idx]]
                        batch_p_sents[:, step_idx] = p_next_words
                        p_next_words = self.get_batch_p_next_words(words=batch_y[:, step_idx])

                    self.pred_model.reset_states()

                    if len(X[batch_index:batch_index + batch_size]) < batch_size:
                        #remove padding if batch was padded
                        batch_p_sents = batch_p_sents[:len(X[batch_index:batch_index + batch_size])]

                    batch_p_sents = batch_p_sents.tolist()

                    #import pdb;pdb.set_trace()
                    batch_sent_lengths = [len(y_seqs[batch_index:batch_index + batch_size][idx]) for idx in range(len(batch_p_sents))]
                    batch_p_sents = [p_sent[:sent_length] for p_sent, sent_length in zip(batch_p_sents, batch_sent_lengths)]
                    batch_p_sents = [numpy.mean(p_sent) for p_sent in batch_p_sents]

                    p_sents.extend(batch_p_sents)

                return p_sents

            else:
                #generate a new word in a given sequence
                pred_sents = []
                p_pred_sents = []
                #merge sents, feed by n_timesteps instead

                #generate new sentences       
                #import pdb;pdb.set_trace()
                for batch_index in range(0, len(X), batch_size):
                    #prep batch
                    batch = get_batch(seqs=X[batch_index:batch_index + batch_size], batch_size=batch_size)
                    batch_sents = numpy.zeros((batch_size, n_words), dtype='int64')
                    batch_p_sents = numpy.zeros((batch_size, n_words))

                    #read context
                    #import pdb;pdb.set_trace()
                    for step_index in xrange(batch.shape[-1]):
                        # if batch.shape[1] - step_index <= self.n_timesteps:
                        #     #import pdb;pdb.set_trace()
                        #     batch = self.pad_timesteps(seqs=batch)
                        p_next_words = self.get_batch_p_next_words(words=batch[:, step_index])

                    #now predict
                    for idx in range(n_words):
                        next_words, p_next_words = self.pred_batch_next_words(p_next_words, mode, n_best, temp)
                        batch_sents[:, idx] = next_words
                        batch_p_sents[:, idx] = p_next_words
                        p_next_words = self.get_batch_p_next_words(words=batch_sents[:, idx])
                    self.pred_model.reset_states()

                    if len(X[batch_index:batch_index + batch_size]) < batch_size:
                        #remove padding if batch was padded
                        batch_sents = batch_sents[:len(X[batch_index:batch_index + batch_size])]
                        batch_p_sents = batch_p_sents[:len(X[batch_index:batch_index + batch_size])]
                    batch_sents = batch_sents.tolist()
                    batch_p_sents = batch_p_sents.tolist()
                    # import pdb;pdb.set_trace()
                    for sent_idx, (sent, p_sent) in enumerate(zip(batch_sents, batch_p_sents)):
                        # sent_length = len(X[batch_index:batch_index + batch_size][sent_idx])
                        # sent = sent[-sent_length:]
                        # p_sent = p_sent[-sent_length:]
                        for word_idx, word in enumerate(sent):
                            if word in eos_markers:
                                batch_sents[sent_idx] = sent[:word_idx + 1]
                                p_sent = p_sent[:word_idx + 1]
                                break
                        batch_p_sents[sent_idx] = numpy.mean(p_sent)

                    pred_sents.extend(batch_sents)
                    p_pred_sents.extend(batch_p_sents)

                    if batch_index and batch_index % 1000 == 0:
                        print "generated new sequences for {}/{} inputs...".format(batch_index, len(X))

                p_pred_sents = numpy.array(p_pred_sents)
                return pred_sents, p_pred_sents

    def create_encoder(self):

        self.encoder_model = self.create_model(n_timesteps=None, batch_size=1, pred_layer=False)
        self.encoder_model.set_weights(self.model.get_weights()[:-2])
        if self.verbose:
            print "created encoder model"


    # def encode(self, X, **kwargs):

    #     if not hasattr(self, 'encoder_model'):
    #         #omit prediction layer - output should be recurrent layer
    #         self.create_encoder()

    #     encoded_seqs = []
    #     for seq in X:
    #         encoded_sents = []
    #         for sent in seq:
    #             sent = numpy.array(sent)[None]
    #             sent = self.encoder_model.predict(sent)[0][-1]
    #             encoded_sents.append(sent)
    #         self.encoder_model.reset_states()
    #         encoded_sents = numpy.array(encoded_sents)
    #         encoded_seqs.append(encoded_sents)

    #     return encoded_seqs


    def get_embeddings(self):

        embeddings = self.model.get_weights()[0]
        return embeddings



def max_margin(y_true, y_pred):
    #import pdb;pdb.set_trace() 

    #y_pred=1, y_true=1: max(0, 1 - (1*1) + 1*(1-1)) = 0
    #y_pred=0, y_true=1: max(0, 1 - (0*1) + 0*(1-1)) = 1
    #y_pred=0, y_true=0:  max(0, 1 - (0*0) + 0*(1-0)) = 1
    #y_pred=1, y_true=0:  max(0, 1 - (1*0) + 1*(1-0)) = 2
    return K.sum(K.maximum(0., 1. - y_pred*y_true + y_pred*(1. - y_true)))



    #return K.mean(K.maximum(0., 1. - y_pred*y_true + y_pred*(1. - y_true)))


class SeqBinaryClassifier(KerasRegressor, SavedModel):
    def __call__(self, context_size, lexicon_size=None, max_length=None, n_embedding_nodes=300, batch_size=None, 
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
        self.filepath = filepath
        self.use_dropout = use_dropout
        self.pairs = pairs
        self.clipvalue = clipvalue
        self.optimizer = optimizer
        self.update_num = 0
        self.save_freq = save_freq
        
        #import pdb;pdb.set_trace()
        
        # if not self.embedded_input:
        #     context_input_layer = Input(batch_shape=(self.batch_size, self.max_length), 
        #                                 dtype='int32', name="context_input_layer")
        #     seq_input_layer = Input(batch_shape=(self.batch_size, self.max_length), 
        #                             dtype='int32', name="seq_input_layer")
        #     merge_layer = merge([context_input_layer, seq_input_layer], mode='concat', concat_axis=-1)
        #     embedding_layer = Embedding(input_dim = self.lexicon_size + 1,
        #                                 output_dim=self.n_embedding_nodes,
        #                                 name='embedding', mask_zero=True)(merge_layer)#mask_zero=True,
        #     #embedded_context = embedding_layer(context_input_layer)
        #     #embedded_seq = embedding_layer(seq_input_layer)

        #     hidden_layer = GRU(output_dim=self.n_hidden_nodes, return_sequences=False, 
        #                                 stateful=False, name='context_hidden_layer')(embedding_layer)

        #     #context_hidden_layer = GRU(output_dim=self.n_hidden_nodes, return_sequences=False, 
        #                                 # stateful=False, name='context_hidden_layer')(embedded_context)
        #     #seq_hidden_layer = GRU(output_dim=self.n_hidden_nodes, return_sequences=False, 
        #                                 # stateful=False, name='seq_hidden_layer')(embedded_seq)
            
        #     #merge_layer = merge([context_hidden_layer, seq_hidden_layer], mode='concat', concat_axis=-1)
        #     #reshape_layer = Reshape((2, self.n_hidden_nodes))(merge_layer)
        #     #top_hidden_layer = GRU(output_dim=self.n_hidden_nodes, return_sequences=False, stateful=False)(reshape_layer)

        #     #pred_layer = Dense(output_dim=1, activation='sigmoid', name='pred_layer')(merge_layer)
        #     pred_layer = Dense(output_dim=1, activation='sigmoid', name='pred_layer')(hidden_layer)#(merge_layer)

        #     model = Model(input=[context_input_layer, seq_input_layer], output=pred_layer)

        # else:
        if self.pairs:
            seq2_input_layer = Input(batch_shape=(self.batch_size, self.n_embedding_nodes), 
                                name="seq2_input_layer")

            seq2_dense_layer = seq_dense_layer(seq2_input_layer)

            merge2_layer = merge([context_hidden_layer, seq2_dense_layer], mode='concat', concat_axis=-1)
                        #output_shape=(self.context_size + 1, self.n_embedding_nodes))

            if self.use_dropout:
                merge2_layer = Dropout(p=0.25)(merge2_layer)

            merge_final_layer = merge([merge1_layer, merge2_layer], mode='concat', concat_axis=-1)

            pred_layer = Dense(output_dim=1, activation='sigmoid', name='pred_layer')(merge_final_layer)

            model = Model(input=[context_input_layer, seq1_input_layer, seq2_input_layer], output=pred_layer)

        ####################CURRENT METHOD######################
        context_input_layer = Input(batch_shape=(self.batch_size, self.context_size, self.n_embedding_nodes), name="context_input_layer")

        seq_input_layer = Input(batch_shape=(self.batch_size, self.n_embedding_nodes), name="seq_input_layer")

        reshape_seq_layer = Reshape((1, self.n_embedding_nodes))(seq_input_layer)

        merge_layer = merge([context_input_layer, reshape_seq_layer], mode='concat', concat_axis=-2, name='merge_layer')

        mask_layer = Masking(mask_value=0.0, input_shape=(self.context_size + 1, self.n_embedding_nodes))(merge_layer)

        hidden_layer = GRU(output_dim=self.n_hidden_nodes, return_sequences=False, stateful=False, name='context_hidden_layer')(mask_layer)#(merge_layer)

        pred_layer = Dense(output_dim=1, activation='sigmoid', name='pred_layer')(hidden_layer)

        model = Model(input=[context_input_layer, seq_input_layer], output=pred_layer)
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


        if self.use_dropout:
            merge_layer = Dropout(p=0.25)(merge_layer)

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

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])#loss='mean_squared_error', 'sparse_categorical_crossentropy'
        
        if self.verbose:
            print "CREATED SeqBinary model: embedding layer nodes = {}, " \
                    "hidden layers = {}, hidden layer nodes = {}, dropout = {}, " \
                    "optimizer = {}, batch size = {}".format(self.n_embedding_nodes, self.n_hidden_layers,
                                         self.n_hidden_nodes, self.use_dropout, optimizer, self.batch_size)
        
        return model
    
    def fit(self, X, y, y_seqs=None, rnn_params=None, **kwargs):

        if rnn_params:
            self.sk_params.update(rnn_params)
        #import pdb;pdb.set_trace()

        if 'embedded_input' in self.sk_params and not self.sk_params['embedded_input']:
            #flatten
            if type(X[0][0]) in [list, tuple, numpy.ndarray]:
                X = [[word for sent in seq for word in sent] for seq in X]
            if type(X) is not numpy.ndarray:
                X = get_batch(X)
            if type(y_seqs) is not numpy.ndarray:
                y_seqs = get_batch(y_seqs)

        if 'pairs' in self.sk_params and self.sk_params['pairs']:
            assert(y_seqs is not None)
            # if type(y_seqs) is list:
            #     y_seqs = numpy.array(y_seqs)
            model_input = [X, y_seqs[:,0], y_seqs[:,1]]
        else:
            model_input = [X, y_seqs]
    
        if hasattr(self, 'model'):
            #if model has already been created, continue training with this new data
            kwargs.update(copy.deepcopy(self.filter_sk_params(Sequential.fit)))
            #sentences up to last are context, last sentence is ending to be judged as correct
            history = self.model.fit(model_input, y, **kwargs)
        
        else:    

            #early_stop = EarlyStopping(monitor='train_loss', patience=patience, verbose=0, mode='auto')
            history = super(SeqBinaryClassifier, self).fit(model_input, y, **kwargs)
        
        print "loss: {:.3f}, accuracy: {:.3f}".format(history.history['loss'][0], history.history['acc'][0])

        self.update_num += 1

        #save model if filepath given
        if self.filepath and (self.update_num % self.save_freq == 0):
            self.save()

    def predict(self, X, y, y_seqs, **kwargs):
        #import pdb;pdb.set_trace()
        if not self.embedded_input:
            X = [[word for sent in seq for word in sent] for seq in X]
            X = get_batch(X)
            y_seqs = get_batch([seq for seqs in y_seqs for seq in seqs]).reshape((len(y_seqs), len(y_seqs[0]), -1))
        
        if self.pairs:
            probs_y = self.model.predict([X, y_seqs[:,0], y_seqs[:,1]])#[:,-1]
            pred_y = numpy.argmax(probs_y, axis=1)
            #pred_y = numpy.rint(probs_y)
            accuracy = numpy.mean(pred_y == y)
        else:
            probs_y = []
            for choice_idx in range(y_seqs.shape[1]):
                choices = y_seqs[:, choice_idx]
                probs = self.model.predict([X, choices])[:,-1]
                probs_y.append(probs)
            probs_y = numpy.stack(probs_y, axis=1)
            assert(len(probs_y) == len(X))
            pred_y = numpy.argmax(probs_y, axis=1)
        
        accuracy = numpy.mean(pred_y == y)
            
        return probs_y, pred_y, accuracy


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

