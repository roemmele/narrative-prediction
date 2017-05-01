import six, pickle, warnings, os
from sklearn.pipeline import Pipeline
from gensim.models.word2vec import Word2Vec
from keras.models import load_model
import transformer
reload(transformer)
from transformer import *
import classifier
reload(classifier)
from classifier import *

warnings.filterwarnings('ignore', category=Warning)

class RNNPipeline():
    def __init__(self, transformer, classifier):
        self.transformer = transformer
        self.classifier = classifier
    def fit(self, X, y=None, y_seqs=None, **params):
        if self.transformer.lexicon:
            X, y_seqs = self.transformer.transform(X, y_seqs)
        else:
            X, y_seqs = self.transformer.fit_transform(X, y_seqs)
        if self.classifier.__class__.__name__ in ('RNNLM', 'MLPLM'):
            params['lexicon_size'] = self.transformer.lexicon_size
            self.classifier.fit_epoch(X, y_seqs, **params)
        else:
            self.classifier.fit(X, y_seqs, y, **params)
    def predict(self, X, y_seqs=None, **params):
        X, y_seqs = self.transformer.transform(X, y_seqs)
        if self.classifier.__class__.__name__ in ('RNNLM', 'MLPLM'):
            gen_params = {param:value for param,value in params.items() if param != 'cap_tokens'}
            if 'eos_tokens' in params:
                gen_params['eos_tokens'] = self.transformer.lookup_eos(params['eos_tokens']) #convert end-of-sentence markers to indices
            gen_seqs, prob_seqs = self.classifier.predict(X, **gen_params)
            decode_params = {param:value for param, value in params.items() if param in ('eos_tokens', 'cap_tokens')}
            print "decoding generated sentences..."
            gen_seqs = [self.transformer.decode_seqs(seq, **decode_params) for seq in gen_seqs]#, cap_tokens=params['cap_tokens'] if 'cap_tokens' in params else [])#convert from indices back to text
            return gen_seqs, prob_seqs
        else:
            return self.classifier.predict(X, y_seqs, **params)
    def evaluate(self, X):
        X, _ = self.transformer.transform(X)
        return self.classifier.evaluate(X)

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


def generate_sents(lm, context_seqs, batch_size=1, n_best=1, n_words=35, 
                   mode='max', temp=1.0, eos_tokens=[".", "!", "?"], cap_tokens=[]):

    gen_sents, p_sents = lm.predict(X=context_seqs, mode=mode,
                                    batch_size=batch_size, 
                                    n_best=n_best, n_words=n_words,
                                    temp=temp, eos_tokens=eos_tokens,
                                    cap_tokens=cap_tokens)
    return gen_sents, p_sents


# class RNNPipeline():
#     #sklearn pipeline won't pass extra parameters other than input data between steps
#     def _pre_transform(self, X, y_seqs=None, **fit_params):
#         fit_params_steps = dict((step, {}) for step, _ in self.steps)
#         for pname, pval in six.iteritems(fit_params):
#             step, param = pname.split('__', 1)
#             fit_params_steps[step][param] = pval
#         Xt = X
#         for name, transform in self.steps[:-1]:
#             if hasattr(transform, "fit_transform"):
#                 Xt, y_seqs, rnn_params = transform.fit_transform(Xt, y_seqs, **fit_params_steps[name])
#             else:
#                 Xt, y_seqs, rnn_params = transform.fit(Xt, y_seqs, **fit_params_steps[name]).transform(Xt, y_seqs)
#         return Xt, y_seqs, rnn_params, fit_params_steps[self.steps[-1][0]]
#     def fit(self, X, y=None, y_seqs=None, **fit_params):
#         if self.steps[-1][-1].__class__.__name__ == 'RNNLM' and\
#             self.steps[0][-1].__class__.__name__ == 'SequenceTransformer' and\
#             self.steps[0][-1].get_params()['word_embeddings'] is not None:
#             #import pdb;pdb.set_trace()
#             #if this is a language model, no explicit output; input will be copied to output before embedding,
#             #if input is embedded
#             self.steps[0][-1].set_params(copy_input_to_output = True)
#         # if self.steps[-1][-1].__class__.__name__ in ['Seq2SeqClassifier', 'MergeSeqClassifier',
#         #                                               'RNNLMClassifier', 'RNNLM', 'SeqBinaryClassifier']:
#         Xt, y_seqs, rnn_params, fit_params = self._pre_transform(X, y_seqs, **fit_params)
#         # else:
#         #     Xt, _, rnn_params, fit_params = self._pre_transform(X, y_seqs, **fit_params)
#         if self.steps[-1][-1].__class__.__name__ == 'RNNLM':
#             self.steps[-1][-1].fit_epoch(Xt, y_seqs, rnn_params, **fit_params)
#         elif self.steps[-1][-1].__class__.__name__ == 'SeqBinaryClassifier':
#             self.steps[-1][-1].fit(Xt, y, y_seqs, rnn_params, **fit_params)
#         else:
#             self.steps[-1][-1].fit(Xt, y, rnn_params, **fit_params)
#         return self
#     def predict(self, X, y=None, y_seqs=None, **kwargs):
#         #if choice sequences given, predict sequence from this list
#         #import pdb;pdb.set_trace()
#         Xt = X
#         for name, transform in self.steps[:-1]:
#             Xt, y_seqs = transform.transform(Xt, y_seqs)
#         if y_seqs is not None and y is not None:
#             return self.steps[-1][-1].predict(Xt, y, y_seqs, **kwargs)
#         elif y_seqs is not None:
#             return self.steps[-1][-1].predict(Xt, y_seqs, **kwargs)
#         else:
#             return self.steps[-1][-1].predict(Xt, **kwargs)
#     def encode(self, X, **kwargs):
#         Xt = X
#         for name, transform in self.steps[:-1]:
#             Xt, _ = transform.transform(Xt)
#         return self.steps[-1][-1].encode(Xt, **kwargs)


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
