'''The Pipeline classes interface between the model transformers (e.g. strings to numbers) and classifiers (e.g. Keras networks that take numbers as input)'''

import six, pickle, warnings, os
# from sklearn.pipeline import Pipeline
from gensim.models.word2vec import Word2Vec
from keras.models import load_model
import transformer
reload(transformer)
from transformer import *
import classifier
reload(classifier)
from classifier import *

warnings.filterwarnings('ignore', category=Warning)

def load_rnnlm_pipeline(filepath):
    #import pdb;pdb.set_trace()
    transformer = load_transformer(filepath)
    # if set_unk_word:
    #     transformer.unk_word = u"<UNK>"
    #     transformer.lexicon_lookup[1] = transformer.unk_word
    classifier = load_classifier(filepath)
    #pipeline_type = classifier.__class__.__name__ + 'Pipeline'
    pipeline = RNNLMPipeline(transformer, classifier)
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
            # for idx, (num_seq, pos_seq) in enumerate(zip(num_seqs, pos_seqs)):
            #     if len(num_seq) != len(pos_seq):
            #         import pdb;pdb.set_trace()
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
        print "generating sequences..."
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
        print "decoding generated sequences..."
        gen_seqs = self.transformer.decode_num_seqs(gen_seqs, n_sents_per_seq=n_sents_per_seq, eos_tokens=eos_tokens, detokenize=detokenize, ents=ents,
                                                    capitalize_ents=capitalize_ents, adapt_ents=adapt_ents)
        return gen_seqs

    def predict_with_pos(self, num_seqs, num_pos_seqs, feature_vecs=None, max_length=35, mode='random', batch_size=1, n_best=1, 
                        temp=1.0, prevent_unk=True, ents=None):
        '''if using part-of-speech tags, generation is more complicated because of need to get part-of-speech tag for each newly generated word; 
        that's the reason for a separate function'''

        pred_seqs = []

        for batch_index in xrange(0, len(num_seqs), batch_size):
            if self.classifier.use_features:
                batch_features = get_batch_features(features=feature_vecs[batch_index:batch_index + batch_size], batch_size=batch_size)
            else:
                batch_features = None

            batch_seqs = get_batch(seqs=num_seqs[batch_index:batch_index + batch_size], batch_size=batch_size) #prep batch
            batch_pos = get_batch(seqs=num_pos_seqs[batch_index:batch_index + batch_size], batch_size=batch_size)

            self.classifier.read_batch(seqs=batch_seqs, pos=batch_pos, features=batch_features)

            batch_pred_seqs = numpy.zeros((batch_size, max_length), dtype='int64')

            p_next_words = self.classifier.get_batch_p_next_words(words=batch_seqs[:,-1], pos=batch_pos[:,-1], features=batch_features)

            for idx in xrange(max_length): #now predict
                next_words, p_next_words = self.classifier.pred_batch_next_words(p_next_words, mode, n_best, temp, prevent_unk)
                batch_pred_seqs[:, idx] = next_words
                # batch_decoded_seqs = [" ".join(segment(seq)[-1:]) for seq in self.transformer.decode_num_seqs(numpy.concatenate([batch_seqs, batch_pred_seqs[:, :idx+1]], axis=1)[:,-20:], 
                #                                                                                             detokenize=True, ents=ents, capitalize_ents=True, adapt_ents=True)]
                batch_decoded_seqs = self.transformer.decode_num_seqs(batch_pred_seqs[:, :idx+1], detokenize=True, ents=ents, capitalize_ents=True, adapt_ents=True) #transform generated word indices back into string for pos tagging
                batch_pos = numpy.array([get_pos_num_seq(seq)[-1] for seq in batch_decoded_seqs]) #get POS tag of previous generated word
                p_next_words = self.classifier.get_batch_p_next_words(words=batch_pred_seqs[:, idx], pos=batch_pos, features=batch_features)

            self.classifier.pred_model.reset_states()

            batch_pred_seqs = batch_seqs_to_list(batch_pred_seqs, len_batch=len(num_seqs[batch_index:batch_index + batch_size]), batch_size=batch_size)
            pred_seqs.extend(batch_pred_seqs)

            if batch_index and batch_index % 1000 == 0:
                print "generated new sequences for {}/{} inputs...".format(batch_index, len(num_seqs))

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
        print "decoding generated sequences..."
        gen_seqs = self.transformer.decode_num_seqs(gen_seqs, n_sents_per_seq=n_sents_per_seq, eos_tokens=eos_tokens, detokenize=detokenize, ents=ents,
                                                    capitalize_ents=capitalize_ents, adapt_ents=adapt_ents)
        return gen_seqs

    def get_probs(self, seqs):
        seqs = self.transformer.text_to_nums(seqs)
        return self.classifier.get_probs(seqs=seqs)#, batch_size=batch_size)


class RNNPipeline():
    def __init__(self, transformer, classifier):
        self.transformer = transformer
        self.classifier = classifier
    def fit(self, seqs, y_seqs=None, y_classes=None, **params):
        # if self.transformer.lexicon:
        #     X = self.transformer.transform(X)
        #     y_seqs = self.transformer.transform(y_seqs)
        # else:
        seqs = self.transformer.fit_transform(seqs)
        if y_seqs is not None:
            y_seqs = self.transformer.fit_transform(y_seqs)
        # if self.classifier.__class__.__name__ in ('RNNLM', 'MLPLM'):
        #     params['lexicon_size'] = self.transformer.lexicon_size
        #     #self.classifier.fit_epoch(X, y_seqs, **params)
        #     self.classifier.fit_epoch(seqs, y_seqs, **params)
        # else:
        self.classifier.fit(seqs, y_seqs, y_classes, **params)
    def predict(self, seqs, y_seqs=None, **params):
        seqs = self.transformer.transform(seqs)
        if y_seqs is not None:
            y_seqs = self.transformer.transform(y_seqs)
        # if self.classifier.__class__.__name__ in ('RNNLM', 'MLPLM'):
        #     gen_params = {param:value for param,value in params.items() if param not in ('cap_ents', 'adapt_ents', 'detokenize')}
        #     if 'eos_tokens' in params:
        #         gen_params['eos_tokens'] = self.transformer.lookup_eos(params['eos_tokens']) #convert end-of-sentence markers to indices
        #     gen_seqs, prob_seqs = self.classifier.predict(seqs, **gen_params)
        #     decode_params = {param:value for param, value in params.items() if param in ('eos_tokens', 'cap_ents',\
        #                                                                                 'adapt_ents', 'detokenize')}
        #     print "decoding generated sentences..."
        #     gen_seqs = [self.transformer.decode_seqs(seq, **decode_params) for seq in gen_seqs]
        #     return gen_seqs, prob_seqs
        # else:
        return self.classifier.predict(seqs, y_seqs, **params)
    def evaluate(self, seqs):
        seqs = self.transformer.transform(seqs)
        return self.classifier.evaluate(seqs)

def load_rnnbinary_pipeline(filepath, embed_filepath='../ROC/AvMaxSim/vectors', batch_size=1, context_size=1, skipthoughts_filepath='../skip-thoughts-master', 
                            use_skipthoughts=False, n_skipthought_nodes=4800, pretrained=True, verbose=True):

    saved_model = load_model(filepath + '/classifier.h5')
    classifier = RNNBinaryClassifier(batch_size=batch_size, context_size=context_size, 
                                     n_embedding_nodes=saved_model.get_layer('context_hidden_layer').input_shape[-1], 
                                     n_hidden_layers=1, 
                                     n_hidden_nodes=saved_model.get_layer('context_hidden_layer').output_shape[-1])
    classifier.model.set_weights(saved_model.get_weights())

    word_embeddings = Word2Vec.load(embed_filepath, mmap='r')

    if use_skipthoughts:
        transformer = load_skipthoughts_transformer(filepath=skipthoughts_filepath, word_embeddings=word_embeddings, 
                                                    n_nodes=n_skipthought_nodes, pretrained=pretrained, verbose=verbose)
        
    else:
        transformer = load_transformer(filepath, word_embeddings)
        transformer.n_embedding_nodes = word_embeddings.vector_size
        transformer.sent_encoder = None

    model = RNNPipeline(transformer=transformer, classifier=classifier)
    return model


class AutoencoderPipeline():
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
