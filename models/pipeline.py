'''The Pipeline classes interface between the model transformers (e.g. strings to numbers) and classifiers (e.g. Keras networks that take numbers as input)'''

import pickle, warnings, os
from keras.models import load_model
from models.transformer import *
from models.classifier import *

warnings.filterwarnings('ignore', category=Warning)

def load_pipeline(Pipeline, filepath, word_embeddings=None):
    transformer = SequenceTransformer.load(filepath, word_embeddings=word_embeddings)
    classifier = SavedModel.load(filepath)
    pipeline = Pipeline(transformer, classifier)
    return pipeline

class RNNLMPipeline():
    def __init__(self, transformer, classifier):
        self.transformer = transformer
        self.classifier = classifier
    def fit(self, seqs):
        if not self.transformer.lexicon:
            self.transformer.make_lexicon(seqs)
        if self.transformer.generalize_ents:
            seqs = self.transformer.replace_ents_in_seqs(seqs)
        num_seqs = self.transformer.text_to_nums(seqs)
        pos_seqs = None
        feature_vecs = None
        if self.classifier.use_pos:
            pos_seqs = [get_pos_num_seq(seq) for seq in seqs]
        if self.classifier.use_features: #include additional context features in RNNLM
            feature_vecs = self.transformer.num_seqs_to_counts([self.transformer.tok_seq_to_nums(seq) for seq in self.transformer.seqs_to_feature_words(seqs)])
        self.classifier.fit(seqs=num_seqs, pos_seqs=pos_seqs, feature_vecs=feature_vecs, lexicon_size=self.transformer.lexicon_size)

    def predict(self, seqs, max_length=35, mode='random', batch_size=1, n_best=1, temp=1.0,
                prevent_unk=True, n_sents_per_seq=None, eos_tokens=[], detokenize=False, capitalize_ents=False, adapt_ents=False):
        if capitalize_ents or adapt_ents: #get named entities in seqs
            ents = [dict(number_ents(get_ents(seq))) for seq in seqs]
        else:
            ents = None
        if self.transformer.generalize_ents:
            seqs = self.transformer.replace_ents_in_seqs(seqs)
        num_seqs = self.transformer.text_to_nums(seqs)
        print("generating sequences...")
        if self.classifier.use_features: #include additional context features in RNNLM
            feature_vecs = self.transformer.num_seqs_to_counts([self.transformer.tok_seq_to_nums(seq) for seq in self.transformer.seqs_to_feature_words(seqs)])
        else:
            feature_vecs = None
        if self.classifier.use_pos:
            num_pos_seqs = [get_pos_num_seq(seq) for seq in seqs]
            gen_seqs = self.predict_with_pos(num_seqs=num_seqs, num_pos_seqs=num_pos_seqs, feature_vecs=feature_vecs, max_length=max_length, 
                                            mode=mode, batch_size=batch_size, n_best=n_best, temp=temp, prevent_unk=prevent_unk)
        else:
            gen_seqs = self.classifier.predict(seqs=num_seqs, feature_vecs=feature_vecs, max_length=max_length, mode=mode, batch_size=batch_size, n_best=n_best,
                                                temp=temp, prevent_unk=prevent_unk)
        print("decoding generated sequences...")
        gen_seqs = self.transformer.decode_num_seqs(gen_seqs, n_sents_per_seq=n_sents_per_seq, eos_tokens=eos_tokens, detokenize=detokenize, ents=ents,
                                                    capitalize_ents=capitalize_ents, adapt_ents=adapt_ents)
        return gen_seqs

    def predict_with_pos(self, num_seqs, num_pos_seqs, feature_vecs=None, max_length=35, mode='random', batch_size=1, n_best=1, 
                        temp=1.0, prevent_unk=True, ents=None):
        '''if using part-of-speech tags, generation is more complicated because of need to get part-of-speech tag for each newly generated word; 
        that's the reason for a separate function'''

        pred_seqs = []

        for batch_index in range(0, len(num_seqs), batch_size):
            if self.classifier.use_features:
                batch_features = get_batch_features(features=feature_vecs[batch_index:batch_index + batch_size], batch_size=batch_size)
            else:
                batch_features = None

            batch_seqs = get_batch(seqs=num_seqs[batch_index:batch_index + batch_size], batch_size=batch_size) #prep batch
            batch_pos = get_batch(seqs=num_pos_seqs[batch_index:batch_index + batch_size], batch_size=batch_size)

            self.classifier.read_batch(seqs=batch_seqs, pos=batch_pos, features=batch_features)

            batch_pred_seqs = numpy.zeros((batch_size, max_length), dtype='int64')

            p_next_words = self.classifier.get_batch_p_next_words(words=batch_seqs[:,-1], pos=batch_pos[:,-1], features=batch_features)

            for idx in range(max_length): #now predict
                next_words, p_next_words = self.classifier.pred_batch_next_words(p_next_words, mode, n_best, temp, prevent_unk)
                batch_pred_seqs[:, idx] = next_words
                batch_decoded_seqs = self.transformer.decode_num_seqs(batch_pred_seqs[:, :idx+1], detokenize=True, ents=ents, capitalize_ents=True, adapt_ents=True) #transform generated word indices back into string for pos tagging
                batch_pos = numpy.array([get_pos_num_seq(seq)[-1] for seq in batch_decoded_seqs]) #get POS tag of previous generated word
                p_next_words = self.classifier.get_batch_p_next_words(words=batch_pred_seqs[:, idx], pos=batch_pos, features=batch_features)

            self.classifier.pred_model.reset_states()

            batch_pred_seqs = batch_seqs_to_list(batch_pred_seqs, len_batch=len(num_seqs[batch_index:batch_index + batch_size]), batch_size=batch_size)
            pred_seqs.extend(batch_pred_seqs)

            if batch_index and batch_index % 1000 == 0:
                print("generated new sequences for {}/{} inputs...".format(batch_index, len(num_seqs)))

        return pred_seqs

    def get_probs(self, seqs, batch_size=1):
        num_pos_seqs = None
        feature_vecs = None
        num_seqs = self.transformer.text_to_nums(seqs)
        if self.classifier.use_pos:
            num_pos_seqs = [get_pos_num_seq(seq) for seq in seqs]
        if self.classifier.use_features:
            feature_vecs = self.transformer.num_seqs_to_counts([self.transformer.tok_seq_to_nums(seq) for seq in self.transformer.seqs_to_feature_words(seqs)])

        return self.classifier.get_probs(seqs=num_seqs, pos_seqs=num_pos_seqs, feature_vecs=feature_vecs, batch_size=batch_size)

    @classmethod
    def load(cls, filepath):
        return load_pipeline(cls, filepath)

class MLPLMPipeline():
    def __init__(self, transformer, classifier):
        self.transformer = transformer
        self.classifier = classifier

    def fit(self, seqs, n_epochs=5):
        if not self.transformer.lexicon:
            self.transformer.make_lexicon(seqs)
        if self.transformer.generalize_ents:
            seqs = self.transformer.replace_ents_in_seqs(seqs)
        seqs = self.transformer.text_to_nums(seqs)
        self.classifier.fit(seqs=seqs, lexicon_size=self.transformer.lexicon_size, n_epochs=n_epochs)

    def predict(self, seqs, max_length=35, mode='random', batch_size=1, n_best=1, temp=1.0,
                prevent_unk=True, n_sents_per_seq=None, eos_tokens=[], detokenize=False, capitalize_ents=False, adapt_ents=False):
        if capitalize_ents or adapt_ents: #get named entities in seqs
            ents = [dict(number_ents(get_ents(seq))) for seq in seqs]
        else:
            ents = None
        if self.transformer.generalize_ents:
            seqs = self.transformer.replace_ents_in_seqs(seqs)
        seqs = self.transformer.text_to_nums(seqs)
        gen_seqs = self.classifier.predict(seqs=seqs, max_length=max_length, mode=mode, batch_size=batch_size, n_best=n_best,
                                            temp=temp, prevent_unk=prevent_unk)
        print("decoding generated sequences...")
        gen_seqs = self.transformer.decode_num_seqs(gen_seqs, n_sents_per_seq=n_sents_per_seq, eos_tokens=eos_tokens, detokenize=detokenize, ents=ents,
                                                    capitalize_ents=capitalize_ents, adapt_ents=adapt_ents)
        return gen_seqs

    def get_probs(self, seqs, batch_size=None):
        seqs = self.transformer.text_to_nums(seqs)
        return self.classifier.get_probs(seqs=seqs)#, batch_size=batch_size)

    @classmethod
    def load(cls, filepath):
        return load_pipeline(cls, filepath)

class CausalEmbeddingsPipeline():
    def __init__(self, transformer, classifier, window_size=5):
        '''window size indicates for a given sentence, how many sentences after that will be used to get causal words'''
        self.transformer = transformer
        self.classifier = classifier
        self.window_size = window_size
    def fit(self, seqs, n_epochs=10):
        if not self.transformer.lexicon:
            self.transformer.make_lexicon(seqs)
        true_causal_pairs = []
        for seq in seqs:
            seq = segment(seq)
            seq = self.transformer.text_to_nums(seq)
            for sent_idx in range(len(seq) - 1):
                seq_window = seq[sent_idx:sent_idx+self.window_size]
                cause_seq = seq_window[0]
                effect_seq = [word for sent in seq_window[1:] for word in sent]
                causal_pairs = get_causal_pairs(cause_seq, effect_seq) #get all pairs of words in this sequence window
                true_causal_pairs.extend(causal_pairs)
        false_causal_pairs = numpy.array(reverse_pairs(true_causal_pairs))
        random_pairs = rng.permutation(numpy.array(true_causal_pairs).flatten()).reshape((-1, 2))[:len(true_causal_pairs) / 5]
        false_causal_pairs = numpy.concatenate((false_causal_pairs, random_pairs))
        causal_pairs = numpy.concatenate((numpy.array(true_causal_pairs), false_causal_pairs))
        labels = numpy.concatenate((numpy.ones(len(true_causal_pairs)), numpy.zeros(len(false_causal_pairs))))
        self.classifier.fit(cause_words=causal_pairs[:, 0, None], effect_words=causal_pairs[:, 1, None], labels=labels, lexicon_size=self.transformer.lexicon_size, n_epochs=n_epochs)
    def predict(self, seq1, seq2):
        '''return a total score for the causal relatedness between seq1 and seq2'''
        seq1, seq2 = self.transformer.text_to_nums([seq1, seq2])
        causal_pairs = numpy.array(get_causal_pairs(seq1, seq2))
        prob = numpy.mean(self.classifier.predict(cause_words=causal_pairs[:, 0, None], effect_words=causal_pairs[:, 1, None]))
        return prob
    @classmethod
    def load(cls, filepath):
        return load_pipeline(cls, filepath)

class RNNBinaryPipeline():
    def __init__(self, transformer, classifier):
        self.transformer = transformer
        self.classifier = classifier
    def fit(self, seqs, n_epochs=10):
        if not self.transformer.lexicon:
            self.transformer.make_lexicon(seqs)
        true_pairs = []
        for seq in seqs:
            seq = segment(seq)
            seq = self.transformer.text_to_nums(seq)
            for sent_idx in range(0, len(seq) - 1, 2):
                true_pairs.extend([seq[sent_idx], seq[sent_idx + 1]])
        reversed_pairs = numpy.array(reverse_pairs(true_pairs))
        random_pairs = rng.permutation(numpy.array(true_pairs).flatten()).reshape((-1, 2))[:len(true_pairs) / 5]
        false_pairs = numpy.concatenate((reversed_pairs, random_pairs))
        pairs = numpy.concatenate((numpy.array(true_pairs), false_pairs))
        labels = numpy.concatenate((numpy.ones(len(true_pairs)), numpy.zeros(len(false_pairs))))
        self.classifier.fit(cause_words=causal_pairs[:, 0, None], effect_words=causal_pairs[:, 1, None], labels=labels, lexicon_size=self.transformer.lexicon_size, n_epochs=n_epochs)
    def predict(self, seq1, seq2):
        '''return a total score for the causal relatedness between seq1 and seq2'''
        seq1, seq2 = self.transformer.text_to_nums([seq1, seq2])
        causal_pairs = numpy.array(get_causal_pairs(seq1, seq2))
        prob = numpy.mean(self.classifier.predict(cause_words=causal_pairs[:, 0, None], effect_words=causal_pairs[:, 1, None]))
        return prob
    @classmethod
    def load(cls, filepath):
        return load_pipeline(cls, filepath)


class SeqBinaryPipeline():
    def __init__(self, transformer, classifier):
        self.transformer = transformer
        self.classifier = classifier
        assert(self.classifier.filepath)
        self.filepath = self.transformer.filepath
        self.use_skipthoughts = True if self.transformer.__class__.__name__ == 'SkipthoughtsTransformer' else False
    def fit(self, input_seqs, output_seqs, neg_output_seqs, input_seqs_filepath, output_seqs_filepath, neg_output_seqs_filepath, output_word_embeddings=None, n_epochs=10, n_chunks=7):
        n_neg_per_seq = len(neg_output_seqs[0])
        if not self.use_skipthoughts:
            if not self.transformer.lexicon:
                self.transformer.make_lexicon(seqs=[" ".join(input_seq + [output_seq]) for input_seq, output_seq in\
                                                                                      zip(input_seqs, output_seqs)])
        if not os.path.exists(neg_output_seqs_filepath) or not neg_output_seqs_filepath:
            if self.use_skipthoughts:
                # encode_skipthought_seqs(neg_seqs, encoder_module, sent_encoder, 
                #                         encoder_dim, memmap=True, filepath=neg_seqs_filepath)
                neg_output_seqs = self.transformer.encode(neg_output_seqs, filepath=neg_output_seqs_filepath)
            else:
                neg_output_seqs = numpy.array([self.transformer.text_to_embs(seqs=seqs, word_embeddings=output_word_embeddings) for seqs in neg_output_seqs])
                numpy.save(neg_output_seqs_filepath, neg_output_seqs)
        if self.use_skipthoughts:
            neg_output_seqs = numpy.memmap(neg_output_seqs_filepath, dtype='float64', mode='r',
                                    shape=(len(input_seqs), n_neg_per_seq, self.transformer.encoder_dim))
        else:
            neg_output_seqs = numpy.load(neg_output_seqs_filepath, mmap_mode='r') #load neg seqs from mem-mapped file
    
        if not os.path.exists(input_seqs_filepath):
            if self.use_skipthoughts:
                # encode_skipthought_seqs(input_seqs, encoder_module, sent_encoder, 
                #                                      encoder_dim, memmap=True, filepath=input_seqs_filepath)
                # encode_skipthought_seqs(output_seqs, encoder_module, sent_encoder, 
                #                                       encoder_dim, memmap=True, filepath=output_seqs_filepath)
                input_seqs = self.transformer.encode(input_seqs, input_seqs_filepath)
            else:
                input_seqs = numpy.array([self.transformer.text_to_embs(seqs=seqs) for seqs in input_seqs])
                numpy.save(input_seqs_filepath, input_seqs)

        if not os.path.exists(output_seqs_filepath):
            if self.use_skipthoughts:
                output_seqs = self.transformer.encode(output_seqs, output_seqs_filepath)
            else:
                output_seqs = numpy.array(self.transformer.text_to_embs(seqs=output_seqs, word_embeddings=output_word_embeddings))
                numpy.save(output_seqs_filepath, output_seqs)

        if self.use_skipthoughts:
            input_seqs = numpy.memmap(input_seqs_filepath, dtype='float64', mode='r',
                                      shape=(len(input_seqs), self.classifier.context_size, self.transformer.encoder_dim))
            output_seqs = numpy.memmap(output_seqs_filepath, dtype='float64', mode='r',
                                      shape=(len(output_seqs), self.transformer.encoder_dim))
        else:
            #load seqs from mem-mapped file
            input_seqs = numpy.load(input_seqs_filepath, mmap_mode='r')
            output_seqs = numpy.load(output_seqs_filepath, mmap_mode='r')
        
        print("added", len(input_seqs), "positive examples")
        print("added", len(input_seqs) * n_neg_per_seq, "negative examples")
        print("examples divided into", n_chunks, "chunks for training")

                #import pdb;pdb.set_trace()
        seqs_per_chunk = len(input_seqs) / n_chunks
        for epoch in range(n_epochs):
            print("TRAINING EPOCH {}/{}".format(epoch + 1, n_epochs))
            for chunk_idx in range(n_chunks):
                chunk_input_seqs = input_seqs[chunk_idx * seqs_per_chunk: (chunk_idx + 1) * seqs_per_chunk]
                chunk_output_seqs = output_seqs[chunk_idx * seqs_per_chunk: (chunk_idx + 1) * seqs_per_chunk]
                chunk_neg_output_seqs = neg_output_seqs[chunk_idx * seqs_per_chunk: (chunk_idx + 1) * seqs_per_chunk]
                chunk_labels = numpy.concatenate([numpy.ones(len(chunk_input_seqs)), numpy.zeros(chunk_neg_output_seqs.shape[0] * chunk_neg_output_seqs.shape[1])])

                chunk_input_seqs = numpy.concatenate([chunk_input_seqs, chunk_input_seqs.repeat(n_neg_per_seq, axis=0)])
                chunk_output_seqs = numpy.concatenate([chunk_output_seqs, chunk_neg_output_seqs.reshape(-1, chunk_neg_output_seqs.shape[-1])]) #add neg output seqs to positive output seqs

                self.classifier.fit(input_seqs=chunk_input_seqs, output_seqs=chunk_output_seqs, labels=chunk_labels)
    def predict(self, input_seq, output_seq):

        if self.use_skipthoughts:
            input_seq = self.transformer.encode(input_seqs)
            output_seq = self.encode(output_seqs)
        else:
            input_seq = numpy.array(self.transformer.text_to_embs(seqs=input_seq))
            output_seq = numpy.array(self.transformer.text_to_embs(seqs=[output_seq]))[0]
        
        #import pdb;pdb.set_trace()
        prob = self.classifier.predict(input_seq, output_seq)
        return prob

    @classmethod
    def load(cls, filepath, word_embeddings):
        return load_pipeline(cls, filepath, word_embeddings)




class RNNPipeline():
    def __init__(self, transformer, classifier):
        self.transformer = transformer
        self.classifier = classifier
    def fit(self, seqs, y_seqs=None, y_classes=None):
        seqs = self.transformer.fit_transform(seqs)
        if y_seqs is not None:
            y_seqs = self.transformer.fit_transform(y_seqs)
        self.classifier.fit(seqs, y_seqs, y_classes, **params)
    def predict(self, seqs, y_seqs=None, **params):
        seqs = self.transformer.transform(seqs)
        if y_seqs is not None:
            y_seqs = self.transformer.transform(y_seqs)
        return self.classifier.predict(seqs, y_seqs, **params)
    def evaluate(self, seqs):
        seqs = self.transformer.transform(seqs)
        return self.classifier.evaluate(seqs)

# def load_rnnbinary_pipeline(filepath, embed_filepath='../ROC/AvMaxSim/vectors', batch_size=1, context_size=1, skipthoughts_filepath='../skip-thoughts-master', 
#                             use_skipthoughts=False, n_skipthought_nodes=4800, pretrained=True, verbose=True):

#     saved_model = load_model(filepath + '/classifier.h5')
#     classifier = RNNBinaryClassifier(batch_size=batch_size, context_size=context_size, 
#                                      n_embedding_nodes=saved_model.get_layer('context_hidden_layer').input_shape[-1], 
#                                      n_hidden_layers=1, 
#                                      n_hidden_nodes=saved_model.get_layer('context_hidden_layer').output_shape[-1])
#     classifier.model.set_weights(saved_model.get_weights())

#     word_embeddings = Word2Vec.load(embed_filepath, mmap='r')

#     if use_skipthoughts:
#         transformer = load_skipthoughts_transformer(filepath=skipthoughts_filepath, word_embeddings=word_embeddings, 
#                                                     n_nodes=n_skipthought_nodes, pretrained=pretrained, verbose=verbose)
        
#     else:
#         transformer = load_transformer(filepath, word_embeddings)
#         transformer.n_embedding_nodes = word_embeddings.vector_size
#         transformer.sent_encoder = None

#     model = RNNPipeline(transformer=transformer, classifier=classifier)
#     return model


# class AutoencoderPipeline():
#     #sklearn pipeline won't pass extra parameters other than input data between steps
#     def _pre_transform(self, X, y_seqs=None, **fit_params):
#         fit_params_steps = dict((step, {}) for step, _ in self.steps)
#         for pname, pval in six.iteritems(fit_params):
#             step, param = pname.split('__', 1)
#             fit_params_steps[step][param] = pval
#         Xt = X
#         for name, transform in self.steps[:-1]:
#             Xt, y_seqs = transform.fit(Xt, y_seqs, **fit_params_steps[name]).transform(Xt, y_seqs)
#         return Xt, y_seqs, fit_params_steps[self.steps[-1][0]]
#     def fit(self, X, y=None, y_seqs=None, **fit_params):
#         #import pdb;pdb.set_trace()
#         Xt, y, fit_params = self._pre_transform(X, y_seqs, **fit_params)
#         self.steps[-1][-1].fit(Xt, y, **fit_params)
#         return self
#     def predict(self, X, y_choices=None):
#         #check if y_choices is single set or if there are different choices for each input
        
#         #import pdb;pdb.set_trace()
#         Xt = X
#         for name, transform in self.steps[:-1]:
#             Xt, y_choices = transform.transform(Xt, y_choices)
#         if y_choices is not None:
#             return self.steps[-1][-1].predict(Xt, y_choices)
#         else:
#             return self.steps[-1][-1].predict(Xt)
