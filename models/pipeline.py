'''The Pipeline classes interface between the model transformers (e.g. strings to numbers) and classifiers (e.g. Keras networks that take numbers as input)'''

from __future__ import print_function
import pickle, warnings, os
from keras.models import load_model
from models.transformer import *
from models.classifier import *

warnings.filterwarnings('ignore', category=Warning)

class Pipeline(object):
    def __init__(self, transformer, classifier, skip_vectorizer=None):
        self.transformer = transformer
        self.classifier = classifier
        self.skip_vectorizer = skip_vectorizer
    # @classmethod
    # def load(cls, filepath):
    #     transformer = SequenceTransformer.load(filepath)
    #     classifier = SavedModel.load(filepath)
    #     pipeline = cls(transformer, classifier)
    #     return pipeline
    @classmethod
    def load(cls, filepath, word_embs=None, has_skip_vectorizer=False, skip_filepath=None):
        transformer = SequenceTransformer.load(filepath, word_embs=word_embs)
        classifier = SavedModel.load(filepath)
        if has_skip_vectorizer:
            if skip_filepath:
                skip_vectorizer = SkipthoughtsTransformer.load(skip_filepath)
            else:
                skip_vectorizer = SkipthoughtsTransformer(verbose=False)
            pipeline = cls(transformer, classifier, skip_vectorizer)
        else:
            pipeline = cls(transformer, classifier)
        return pipeline

class RNNLMPipeline(Pipeline):
    def fit(self, seqs, n_epochs=1, verbose=True):
        if not self.transformer.lexicon:
            self.transformer.make_lexicon(seqs)
        if self.transformer.generalize_ents:
            seqs = [self.transformer.replace_ents_in_seq(seq) for seq in seqs]
        num_seqs = self.transformer.text_to_nums(seqs)
        pos_seqs = None
        feature_vecs = None
        if self.classifier.use_pos:
            pos_seqs = [get_pos_num_seq(seq) for seq in seqs]
        if self.classifier.use_features: #include additional context features in RNNLM
            feature_vecs = self.transformer.num_seqs_to_bow([self.transformer.tok_seq_to_nums(seq) for seq in self.transformer.seqs_to_feature_words(seqs)])
        for epoch in range(n_epochs):
            if verbose:
                print('EPOCH', epoch + 1)
            self.classifier.fit(seqs=num_seqs, pos_seqs=pos_seqs, feature_vecs=feature_vecs, lexicon_size=self.transformer.lexicon_size)

    def predict(self, seqs, max_length=35, mode='random', batch_size=1, n_best=1, temp=1.0, prevent_unk=True, 
                n_context_sents=-1, n_sents_per_seq=None, eos_tokens=[], detokenize=False, capitalize_ents=False, adapt_ents=False):
        seqs = [seq if seq.strip() else u"." for seq in seqs] #if seq is empty, generate from end-of-sentence marker "."
        if capitalize_ents or adapt_ents: #get named entities in seqs
            ents = [number_ents(*get_ents(seq)) for seq in seqs]
        else:
            ents = None
        if self.transformer.generalize_ents:
            seqs = [self.transformer.replace_ents_in_seq(seq) for seq in seqs]
        print("generating sequences...")
        if self.classifier.use_features: #include additional context features in RNNLM
            feature_vecs = self.transformer.num_seqs_to_bow([self.transformer.tok_seq_to_nums(seq) for seq in self.transformer.seqs_to_feature_words(seqs)])
        else:
            feature_vecs = None
        if n_context_sents > -1:
            '''include only most recent n_context_sents in context sequence given to recurrent layer; if -1, all sentences in context will be included;
            regardless of this setting, the whole context sequence is still taken into account in the feature vectors, if using'''
            seqs = [" ".join(segment(seq)[-n_context_sents:]) for seq in seqs]
        num_seqs = self.transformer.text_to_nums(seqs)
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
            feature_vecs = self.transformer.num_seqs_to_bow([self.transformer.tok_seq_to_nums(seq) for seq in self.transformer.seqs_to_feature_words(seqs)])

        return self.classifier.get_probs(seqs=num_seqs, pos_seqs=num_pos_seqs, feature_vecs=feature_vecs, batch_size=batch_size)


class MLPLMPipeline(Pipeline):

    def fit(self, seqs, n_epochs=5):
        if not self.transformer.lexicon:
            self.transformer.make_lexicon(seqs)
        if self.transformer.generalize_ents:
            seqs = [self.transformer.replace_ents_in_seq(seq) for seq in seqs]
        seqs = self.transformer.text_to_nums(seqs)
        self.classifier.fit(seqs=seqs, lexicon_size=self.transformer.lexicon_size, n_epochs=n_epochs)

    def predict(self, seqs, max_length=35, mode='random', batch_size=1, n_best=1, temp=1.0,
                prevent_unk=True, n_sents_per_seq=None, eos_tokens=[], detokenize=False, capitalize_ents=False, adapt_ents=False):
        if capitalize_ents or adapt_ents: #get named entities in seqs
            ents = [dict(number_ents(get_ents(seq))) for seq in seqs]
        else:
            ents = None
        if self.transformer.generalize_ents:
            seqs = [self.transformer.replace_ents_in_seq(seq) for seq in seqs]
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

class CausalEmbeddingsPipeline(Pipeline):
    def get_true_pairs(self, seqs, window_size=1):
        '''window size indicates for a given sentence, how many sentences after that will be used to get causal words'''
        pairs = []
        for seq in seqs:
            seq = segment(seq)
            if self.transformer.use_spacy_embs or self.transformer.word_embeddings:
                seq = self.transformer.text_to_embs(seq)
            else:
                seq = self.transformer.text_to_nums(seq)
            for sent_idx in range(len(seq) - 1):
                window = seq[sent_idx:sent_idx+window_size+1]
                seq1 = window[0]
                seq2 = [word for sent in window[1:] for word in sent]
                seq_pairs = numpy.array(get_word_pairs(seq1, seq2)) #get all pairs of words in this sequence window
                pairs.extend(seq_pairs)
        return pairs
    def fit(self, seqs, n_epochs=1):
        embedded_input = True if (self.transformer.word_embeddings or self.transformer.use_spacy_embs) else False
        if not embedded_input and not self.transformer.lexicon:
            self.transformer.make_lexicon(seqs) # Use true pairs to build lexicon
        true_pairs = self.get_true_pairs(seqs)
        reversed_pairs = reverse_pairs(true_pairs)
        random_pairs = randomize_pairs(true_pairs)
        false_pairs = reversed_pairs + random_pairs
        pairs = numpy.array(true_pairs + false_pairs)
        labels = numpy.concatenate((numpy.ones(len(true_pairs)), numpy.zeros(len(false_pairs))))
        self.classifier.fit(cause_words=pairs[:, 0], effect_words=pairs[:, 1], 
                            labels=labels, lexicon_size=self.transformer.lexicon_size, embedded_input=embedded_input, n_epochs=n_epochs)
    def predict(self, seq1, seq2):
        '''return a total score for the causal relatedness between seq1 and seq2'''
        embedded_input = True if (self.transformer.word_embeddings or self.transformer.use_spacy_embs) else False
        if embedded_input:
            seq1, seq2 = self.transformer.text_to_embs([seq1, seq2])
        else:
            seq1, seq2 = self.transformer.text_to_nums([seq1, seq2])
        pairs = numpy.array(get_word_pairs(seq1, seq2))
        prob = numpy.mean(self.classifier.predict(cause_words=pairs[:, 0], effect_words=pairs[:, 1]))
        return prob

class MLPBinaryPipeline(Pipeline):
    def get_true_pairs(self, seqs, segment_clauses=False, max_clause_length=15):
        if segment_clauses:
            seqs = [segment_into_clauses(seq) for seq in seqs]
        else:
            seqs = [segment(seq) for seq in seqs] #segment by sentence instead of clause
        pairs = get_adj_clause_pairs(seqs, max_length=max_clause_length) #add 10 to max length to account for grammatical words
        pairs = [self.transformer.text_to_nums(pair) for pair in pairs]
        #pairs = [self.transformer.num_seqs_to_bow(pair) for pair in pairs if pair[0] and pair[1]]
        pairs = [pair for pair in pairs if pair[0] and pair[1]]
        return pairs
    def fit(self, seqs, clause_window=1, max_clause_length=15, n_epochs=1, verbose=True):
        #embedded_input = True if (self.transformer.word_embeddings or self.transformer.use_spacy_embs) else False
        if not self.transformer.lexicon:
            self.transformer.make_lexicon(seqs)
        # if embedded_input:
        #     '''TO-DO: NEED TO ENSURE SEQUENCES WITH ALL 0 VECTORS AREN'T GETTING THROUGH'''
        #     true_pairs = [self.transformer.text_to_embs(pair) for pair in true_pairs]
        #     false_pairs = [self.transformer.text_to_embs(pair) for pair in false_pairs]
        # else:
        true_pairs = self.get_true_pairs(seqs, max_clause_length=max_clause_length)
        #false_pairs = [self.transformer.text_to_nums(pair) for pair in false_pairs]
        reversed_pairs = reverse_pairs(true_pairs)
        random_pairs = randomize_pairs(true_pairs)# + randomize_pairs(true_pairs)
        false_pairs = random_pairs + reversed_pairs
        pairs = true_pairs + false_pairs
        labels = numpy.concatenate((numpy.ones(len(true_pairs)), numpy.zeros(len(false_pairs))))
        if verbose:
            print("training on", len(true_pairs), "true pairs, and", len(false_pairs), "false pairs, (total=", len(true_pairs + false_pairs), ")")
        self.classifier.fit(seqs1=[pair[0] for pair in pairs], seqs2=[pair[1] for pair in pairs], labels=labels,
                            lexicon_size=self.transformer.lexicon_size, n_epochs=n_epochs) #embedded_input=embedded_input,
    def predict(self, seq1, seq2):
        '''return a total score for the causal relatedness between seq1 and seq2'''
        # if self.classifier.embedded_input:
        #     seq1, seq2 = self.transformer.pad_embs(self.transformer.text_to_embs([seq1, seq2]), max_length=self.classifier.n_timesteps)
        # else:
        seq1, seq2 = self.transformer.num_seqs_to_bow(self.transformer.text_to_nums([seq1, seq2]))
        prob = self.classifier.predict(seq1=seq1[None], seq2=seq2[None])
        return prob

class RNNBinaryPipeline(Pipeline):
    def get_bkwrd_samples(self, seqs1, seqs2, n_samples=1):#, max_clause_length=20):
        samples = [(seq1, seq1[rng.choice(len(seqs1))]) for seq1, seq2 in zip(seqs1 * n_samples, seqs2 * n_samples)]
        seqs1 = [sample[0] for sample in samples]
        # seqs2 = [sample[1] for sample in samples]
        bkwrd_seqs2 = [sample[1] for sample in samples]
        return seqs1, bkwrd_seqs2

    #     '''given seqs, return instances that pair sentences in sequence with all those that appear before it in the sequence'''
    #     #import pdb;pdb.set_trace()
    #     # seqs1 = []
    #     # pos_seqs2 = []
    #     # neg_seqs2 = []
    #     # for seq in seqs1:
    #     #     seq = segment(seq)
    #     #     for sent_idx, sent in enumerate(seq[:-1]):
    #     #         for bkwrd_idx in range(sent_idx - 1, -1, -1):
    #     #             next_sent = seq[sent_idx + 1]
    #     #             bkwrd_sent = seq[bkwrd_idx]
    #     #             if len(sent) <= max_clause_length and len(next_sent) <= max_clause_length and len(bkwrd_sent) <= max_clause_length: #only add instances with sentences shorter than maximum length
    #     #                 seqs1.append(sent) #for every negative sentence, need to match it to a positive
    #     #                 pos_seqs2.append(next_sent) # positive example is immediate next sentence
    #     #                 neg_seqs2.append(bkwrd_sent)
    #     bkwrd_seqs = []
    #     for seq1 in seqs1:
    #         assert(type(seq1) in (tuple, list))
    #         for sent_idx, sent in enumerate(seq1):
    #             for bkwrd_idx in range(sent_idx, -1, -1):
    #                 # next_sent = seq[sent_idx + 1]
    #                 bkwrd_sent = seq1[bkwrd_idx]
    #                 bkwrd_seqs.append(bkwrd_sent)
    #                 # if len(bkwrd_sent) <= max_clause_length and len(bwrd_sent) <= max_clause_length and len(bkwrd_sent) <= max_clause_length: #only add instances with sentences shorter than maximum length
    #                 #     seqs1.append(sent) #for every negative sentence, need to match it to a positive
    #                 #     pos_seqs2.append(next_sent) # positive example is immediate next sentence
    #                 #     neg_seqs2.append(bkwrd_sent)
    #     return bkwrd_seqs

    def get_random_samples(self, seqs1, seqs2, n_samples=1):
        '''for each sentence in a seq, pair it with its next sentence (the positive instance)
        and a random sentence from the pool of all sequences (the negative instance)'''
        # seqs1 = []
        # pos_seqs2 = []
        # neg_seqs2 = []
        samples = [(seq1, seqs2[rng.choice(len(seqs2))]) for seq1, seq2 in zip(seqs1 * n_samples, seqs2 * n_samples)]
        seqs1 = [sample[0] for sample in samples]
        # seqs2 = [sample[1] for sample in samples]
        random_seqs2 = [sample[1] for sample in samples]
        # for seq in seqs2:
        #     for sent_idx, sent in enumerate(seq[:-1]):
        #         for idx in range(n_random): #add n_random random instances
        #             next_sent = seq[sent_idx + 1]
        #             random_seq_idx = rng.choice(len(seqs))
        #             random_sent_idx = rng.choice(len(seqs[random_seq_idx]))
        #             random_sent = seqs[random_seq_idx][random_sent_idx]
        #             if len(sent) <= max_clause_length and len(next_sent) <= max_clause_length and len(random_sent) <= max_clause_length: #only add instances with sentences shorter than maximum length
        #                 seqs1.append(sent) #for every negative sentence, need to match it to a positive
        #                 pos_seqs2.append(next_sent) # positive example is immediate next sentence
        #                 neg_seqs2.append(random_sent)
        return seqs1, random_seqs2

    def fit(self, seqs1, seqs2, labels=[], generate_neg=True, n_bkwrd=0, n_random=1, n_epochs=1, eval_fn=None, eval_freq=1, verbose=True):
        # embedded_input = True if (self.transformer.word_embs or self.transformer.use_spacy_embs) else False
        # if not self.classifier.embedded_input and not self.transformer.lexicon:
        #     self.transformer.make_lexicon(input_seqs)
        # if pos_output_seqs:
        #     assert(neg_output_seqs is not None)
        if not self.transformer.lexicon:
            self.transformer.make_lexicon(seqs1 + seqs2)

        if self.classifier.embedded_input:
            seqs1 = self.transformer.text_to_embs(seqs1, reduce_emb_mode='sum').tolist()
            seqs2 = self.transformer.text_to_embs(seqs2, reduce_emb_mode='sum').tolist()
            # if neg_seqs2:
            #     neg_seqs2 = self.transformer.text_to_embs(seqs2)
            # neg_output_seqs = self.transformer.text_to_embs(neg_output_seqs)
        else:
            seqs1 = self.transformer.text_to_nums(seqs1)
            seqs2 = self.transformer.text_to_nums(seqs2)
            # if neg_seqs2:
            #     neg_seqs2 = self.transformer.text_to_embs(seqs2)
            # neg_output_seqs = self.transformer.text_to_nums(neg_output_seqs)
            # if verbose:
            #     print("training on", len(input_seqs), "positive-negative pairs:")
        #else: #if pos and neg output seqs not given, generate them from input seqs
                # if embedded_input:
            #     '''TO-DO: NEED TO ENSURE SEQUENCES WITH ALL 0 VECTORS AREN'T GETTING THROUGH'''
            #input_seqs = [self.transformer.text_to_nums(segment(seq)) for seq in input_seqs]
        #import pdb;pdb.set_trace()
        if not labels:
            labels = numpy.ones((len(seqs1),))
        assert(len(seqs1) == len(labels))

        if generate_neg:
            if n_bkwrd:
                assert(type(seqs1[0]) in (tuple, list) and n_input_sents > 1)
                sample_seqs1, sample_neg_seqs2 = self.get_bkwrd_samples(seqs1, seqs2, n_samples=n_bkwrd)
                # if self.classifier.embedded_input:
                #     seqs1 = numpy.append(seqs1, sample_seqs1, axis=0)
                #     seqs2 = numpy.append(seqs2, sample_neg_seqs2, axis=0)
                # else:
                seqs1.extend(sample_seqs1)
                seqs2.extend(sample_neg_seqs2)
                labels = numpy.append(labels, numpy.zeros((len(sample_neg_seqs2,))))
            #bkwrd_input_seqs, bkwrd_pos_output_seqs, bkwrd_neg_output_seqs = self.get_bkwrd_seqs(input_seqs, max_clause_length=max_clause_length)
            if n_random:
                sample_seqs1, sample_neg_seqs2 = self.get_random_samples(seqs1, seqs2, n_samples=n_random)
                # if self.classifier.embedded_input:
                #     seqs1 = numpy.append(seqs1, sample_seqs1, axis=0)
                #     seqs2 = numpy.append(seqs2, sample_neg_seqs2, axis=0)
                # else:
                seqs1.extend(sample_seqs1)
                seqs2.extend(sample_neg_seqs2)
                labels = numpy.append(labels, numpy.zeros((len(sample_neg_seqs2,))))
                #random_input_seqs, random_pos_output_seqs, random_neg_output_seqs = self.get_random_seqs(input_seqs, n_random=5, max_clause_length=max_clause_length)
            # seqs1 = bkwrd_input_seqs + random_input_seqs
            # pos_output_seqs = bkwrd_pos_output_seqs + random_pos_output_seqs
            # neg_output_seqs = bkwrd_neg_output_seqs + random_neg_output_seqs

            if verbose:
                print("training on", len(seqs1), "positive-negative pairs:")#, len(random_pos_output_seqs), "random,", len(bkwrd_pos_output_seqs), "backward")

        # shuffle instances
        random_idxs = rng.permutation(len(seqs1))
        seqs1 = [seqs1[idx] for idx in random_idxs]
        seqs2 = [seqs2[idx] for idx in random_idxs]
        labels = labels[random_idxs]

        if not hasattr(self, 'best_accuracy'):
            self.best_accuracy = -numpy.inf
        #import pdb;pdb.set_trace()
        for epoch in range(n_epochs):
            if n_epochs > 1:
                print("EPOCH:", epoch + 1)
            chunk_size = int(numpy.ceil(len(seqs1) * 1. / eval_freq))
            for chunk_idx in range(0, len(seqs1), chunk_size):
                self.classifier.fit(seqs1=seqs1[chunk_idx:chunk_idx + chunk_size], seqs2=seqs2[chunk_idx:chunk_idx + chunk_size], labels=labels[chunk_idx:chunk_idx + chunk_size], 
                                    lexicon_size=self.transformer.lexicon_size, n_epochs=1, save_to_filepath=False) #embedded_input=embedded_input, 
                                #n_timesteps=max_clause_length, lexicon_size=self.transformer.lexicon_size, embedded_input=embedded_input, n_epochs=n_epochs)
                if eval_fn:
                    if not hasattr(self, 'best_accuracy'):
                        self.best_accuracy = -numpy.inf
                    #import pdb;pdb.set_trace()
                    accuracy = eval_fn(self)
                    if accuracy >= self.best_accuracy:
                        self.best_accuracy = accuracy
                        if self.classifier.filepath:
                            self.classifier.save()
                elif self.classifier.filepath:
                    self.classifier.save()
            # self.classifier.fit(seqs1=seqs1, seqs2=seqs2, neg_seqs2=neg_seqs2, lexicon_size=self.transformer.lexicon_size, n_epochs=1)
            # if eval_fn:
            #     #import pdb;pdb.set_trace()
            #     accuracy = eval_fn(self)
            #     if accuracy >= self.best_accuracy:
            #         self.best_accuracy = accuracy
            #         self.classifier.save()
            # else:
            #     self.classifier.save()

    def predict(self, seqs1, seqs2, pred_method=None):
        '''return a total score for the causal relatedness between seq1 and seq2'''
        # import pdb;pdb.set_trace()
        if self.classifier.embedded_input:
            seqs1 = self.transformer.text_to_embs(seqs1, reduce_emb_mode='sum')
            seqs2 = self.transformer.text_to_embs(seqs2, reduce_emb_mode='sum')

        probs = []
        for seq1, seq2 in zip(seqs1, seqs2):
            prob = self.classifier.predict(seq1=seq1, seq2=seq2, pred_method=pred_method)
            probs.append(prob)

        probs = numpy.array(probs)
        return probs

    # @classmethod
    # def load(cls, filepath, word_embs=None, has_skip_vectorizer=False, skip_filepath=None):
    #     transformer = SequenceTransformer.load(filepath, word_embs=word_embs)
    #     classifier = SavedModel.load(filepath)
    #     if has_skip_vectorizer:
    #         if skip_filepath:
    #             skip_vectorizer = SkipthoughtsTransformer.load(skip_filepath)
    #         else:
    #             skip_vectorizer = SkipthoughtsTransformer(verbose=False)
    #         pipeline = cls(transformer, classifier, skip_vectorizer)
    #     else:
    #         pipeline = cls(transformer, classifier)
    #     return pipeline

class Seq2SeqPipeline(Pipeline):

    # def __init__(self, transformer, classifier, skip_vectorizer=None):
    #     self.transformer = transformer
    #     self.skip_vectorizer = skip_vectorizer
    #     self.classifier = classifier

    # def get_sim_pairs(self, seq_pairs, sim_model, n_sim=5, reverse=False):
    #     #import pdb;pdb.set_trace()

    #     sim_pairs = []
    #     if reverse:
    #         first_clauses = [pair[1] for pair in seq_pairs]
    #     else:
    #         first_clauses = [pair[0] for pair in seq_pairs]
    #     for chunk_idx in range(0, len(first_clauses), 5000):
    #         chunk_first_clauses = first_clauses[chunk_idx:chunk_idx+5000]
    #         sim_second_clauses = sim_model.get_sim_next_seqs(chunk_first_clauses, n_best=n_sim + 1)
    #         if reverse:
    #             sim_pairs.extend([[second_clause, first_clause] for first_clause, second_clauses in zip(chunk_first_clauses, sim_second_clauses) 
    #                                                                     for second_clause in second_clauses[1:] if second_clause])
    #         else:
    #             sim_pairs.extend([[first_clause, second_clause] for first_clause, second_clauses in zip(chunk_first_clauses, sim_second_clauses) 
    #                                                                                     for second_clause in second_clauses[1:] if second_clause])
    #         print("retrieved similar pairs for", chunk_idx + 5000, "/", len(first_clauses), "pairs...")

    #     return sim_pairs

    def fit(self, seqs1, seqs2, max_length=25, n_epochs=1, eval_fn=None, eval_freq=1, verbose=True): #segment_clauses=False, max_clause_distance=1, max_clause_length=15, flat_input=False, reverse=False, 
            #drop_words=False, n_samples_per_seq=1, sim_model=None, n_sim=5,

        #embedded_input = True if (self.transformer.word_embeddings or self.transformer.use_spacy_embs or flat_input) else False
        if not self.transformer.lexicon:
            self.transformer.make_lexicon(seqs1 + seqs2)
        # if sim_model:
        #     pairs = get_adj_clause_pairs(seqs, reverse=reverse)
        #     sim_pairs = self.get_sim_pairs(pairs, sim_model, n_sim=n_sim, reverse=reverse)
        #     pairs = pairs + sim_pairs
        #     pairs = [self.transformer.text_to_nums(pair) for pair in pairs]
        #     pairs = [pair for pair in pairs if len(pair[0]) and len(pair[0]) <= max_clause_length and len(pair[1]) and len(pair[1]) <= max_clause_length]
        # else:
        #seqs = [self.transformer.text_to_nums(segment(seq, clauses=segment_clauses)) for seq in seqs]
        if self.classifier.flat_input and self.classifier.embedded_input:
            if self.skip_vectorizer is not None: #input sequences will be transformer into flat vectors
                seqs1 = self.skip_vectorizer.text_to_embs(seqs1)[:,0,:]
            elif self.classifier.embedded_input:
                seqs1 = self.transformer.text_to_embs(seqs1, reduce_emb_mode='sum')
                #seqs1 = self.transformer.tok_seqs_to_embs(seqs1, reduce_emb_mode='sum')
            #assert(type(seqs1) == numpy.ndarray)
        else:
            seqs1 = self.transformer.text_to_nums(seqs1)
            #seqs1 = self.transformer.tok_seqs_to_nums(seqs1)

        seqs2 = self.transformer.text_to_nums(seqs2)
        #seqs2 = self.transformer.tok_seqs_to_nums(seqs2)

        assert(len(seqs1) == len(seqs2))

        for epoch in range(n_epochs):
            if n_epochs > 1:
                if verbose:
                    print("EPOCH:", epoch + 1)
            if verbose:
                print("training on", len(seqs1), "sequence pairs")
            # print("training on", len(seq_pairs), "sequence pairs (sim model =", True if sim_model else False, ", segment clauses =", segment_clauses, ", max clause distance =", max_clause_distance,
            #     ", max clause length =", max_clause_length, ", reverse =", reverse, ", drop words =", drop_words, ", n samples per seq =", n_samples_per_seq, ")")
            chunk_size = int(numpy.ceil(len(seqs1) * 1. / eval_freq))
            for chunk_idx in range(0, len(seqs1), chunk_size):
                self.classifier.fit(seqs1[chunk_idx:chunk_idx + chunk_size], seqs2[chunk_idx:chunk_idx + chunk_size], 
                                    n_timesteps=max_length, lexicon_size=self.transformer.lexicon_size,  
                                    n_epochs=1, save_to_filepath=False) #embedded_input=embedded_input, 
                                #n_timesteps=max_clause_length, lexicon_size=self.transformer.lexicon_size, embedded_input=embedded_input, n_epochs=n_epochs)

                if eval_fn:
                    if not hasattr(self, 'best_accuracy'):
                        self.best_accuracy = -numpy.inf
                    #import pdb;pdb.set_trace()
                    accuracy = eval_fn(self)
                    if accuracy >= self.best_accuracy:
                        self.best_accuracy = accuracy
                        if self.classifier.filepath:
                            self.classifier.save()
                elif self.classifier.filepath:
                    self.classifier.save()

    def predict(self, seqs1, seqs2, pred_method='multiply', unigram_probs=None):
        '''return a total score for the prob that seq2 follows seq1'''
        if self.classifier.flat_input and self.classifier.embedded_input:
            if self.skip_vectorizer is not None: #input sequences will be transformer into flat vectors
                seqs1 = self.skip_vectorizer.text_to_embs(seqs1)[:,0,:]
            else:
                seqs1 = self.transformer.text_to_embs(seqs1, reduce_emb_mode='sum')
            #assert(type(seqs1) == numpy.ndarray)
        else:
            seqs1 = self.transformer.text_to_nums(seqs1)

        seqs2 = self.transformer.text_to_nums(seqs2)

        probs = []
        for seq1, seq2 in zip(seqs1, seqs2):
            prob = self.classifier.predict(seq1=seq1, seq2=seq2, pred_method=pred_method, unigram_probs=unigram_probs)
            probs.append(prob)

        probs = numpy.array(probs)
        return probs

    def get_most_probable_words(self, seqs1, top_n_words=10, unigram_probs=None):
        '''return the most probable output words for input sequences'''
        if self.classifier.flat_input and self.classifier.embedded_input:
            if self.skip_vectorizer is not None: #input sequences will be transformer into flat vectors
                seqs1 = self.skip_vectorizer.text_to_embs(seqs1)[:,0,:]
            else:
                seqs1 = self.transformer.text_to_embs(seqs1, reduce_emb_mode='sum')
        else:
            seqs1 = self.transformer.text_to_nums(seqs1)

        most_probable_words = []
        probs = []
        for seq1 in seqs1:
            seq_probable_words, seq_probs = self.classifier.get_most_probable_words(seq1, top_n_words=top_n_words, unigram_probs=unigram_probs)
            probs.append(seq_probs)
            seq_probable_words = [self.transformer.lexicon_lookup[word] for word in seq_probable_words] #transform word indices to strings
            most_probable_words.append(seq_probable_words)

        probs = numpy.array(probs)

        return most_probable_words, probs


    # @classmethod
    # def load(cls, filepath, word_embs=None, has_skip_vectorizer=False, skip_filepath=None):
    #     transformer = SequenceTransformer.load(filepath, word_embs=word_embs)
    #     classifier = SavedModel.load(filepath)
    #     if has_skip_vectorizer:
    #         if skip_filepath:
    #             skip_vectorizer = SkipthoughtsTransformer.load(skip_filepath)
    #         else:
    #             skip_vectorizer = SkipthoughtsTransformer(verbose=False)
    #         pipeline = cls(transformer, classifier, skip_vectorizer)
    #     else:
    #         pipeline = cls(transformer, classifier)
    #     return pipeline

class ClassifierPipeline(Pipeline):
    def fit(self, seqs, labels, n_epochs=1):
        
        if self.transformer.use_spacy_embs or self.transformer.word_embeddings is not None:
            seqs = self.transformer.text_to_embs(seqs)
            n_input_nodes = seqs.shape[-1]
        else:
            if not self.transformer.lexicon:
                self.transformer.make_lexicon(seqs)
            seqs = self.transformer.text_to_nums(seqs)
            n_input_nodes = self.transformer.lexicon_size + 1
        self.classifier.fit(seqs, labels, n_input_nodes=n_input_nodes, n_epochs=n_epochs)

    def predict(self, seqs):
        if self.transformer.use_spacy_embs or self.transformer.word_embeddings is not None:
            seqs = self.transformer.text_to_embs(seqs)
        else:
            seqs = self.transformer.text_to_nums(seqs)
        return self.classifier.predict(seqs)

# class RNNClassifierPipeline():
#     def fit(self, seqs, labels, n_epochs=1):
        
#         if not self.transformer.lexicon:
#             self.transformer.make_lexicon(seqs)
        
#         seqs = self.transformer.text_to_nums(seqs)
#         self.classifier.fit(seqs, labels, lexicon_size=self.transformer.lexicon_size, n_epochs=n_epochs)
#     def predict(self, seqs):
#         seqs = self.transformer.text_to_nums(seqs)
#         return self.classifier.predict(seqs)


# class SeqBinaryPipeline(Pipeline):
#     # def __init__(self, transformer, classifier):
#     #     Pipeline.__init__(self, transformer, classifier)
#     #     self.use_skipthoughts = True if self.transformer.__class__.__name__ == 'SkipthoughtsTransformer' else False
#     def fit(self, input_seqs, output_seqs, neg_output_seqs, input_seqs_filepath, output_seqs_filepath, neg_output_seqs_filepath, output_word_embeddings=None, n_epochs=10, n_chunks=7):
#         n_neg_per_seq = len(neg_output_seqs[0])
#         if not self.use_skipthoughts:
#             if not self.transformer.lexicon:
#                 self.transformer.make_lexicon(seqs=[" ".join(input_seq + [output_seq]) for input_seq, output_seq in\
#                                                                                       zip(input_seqs, output_seqs)])
#         if not os.path.exists(neg_output_seqs_filepath) or not neg_output_seqs_filepath:
#             if self.use_skipthoughts:
#                 # encode_skipthought_seqs(neg_seqs, encoder_module, sent_encoder, 
#                 #                         encoder_dim, memmap=True, filepath=neg_seqs_filepath)
#                 neg_output_seqs = self.transformer.text_to_embs(neg_output_seqs, filepath=neg_output_seqs_filepath)
#             else:
#                 neg_output_seqs = numpy.array([self.transformer.text_to_embs(seqs=seqs, word_embeddings=output_word_embeddings) for seqs in neg_output_seqs])
#                 numpy.save(neg_output_seqs_filepath, neg_output_seqs)
#         if self.use_skipthoughts:
#             neg_output_seqs = numpy.memmap(neg_output_seqs_filepath, dtype='float64', mode='r',
#                                     shape=(len(input_seqs), n_neg_per_seq, self.transformer.encoder_dim))
#         else:
#             neg_output_seqs = numpy.load(neg_output_seqs_filepath, mmap_mode='r') #load neg seqs from mem-mapped file
    
#         if not os.path.exists(input_seqs_filepath):
#             if self.use_skipthoughts:
#                 # encode_skipthought_seqs(input_seqs, encoder_module, sent_encoder, 
#                 #                                      encoder_dim, memmap=True, filepath=input_seqs_filepath)
#                 # encode_skipthought_seqs(output_seqs, encoder_module, sent_encoder, 
#                 #                                       encoder_dim, memmap=True, filepath=output_seqs_filepath)
#                 input_seqs = self.transformer.text_to_embs(input_seqs, input_seqs_filepath)
#             else:
#                 input_seqs = numpy.array([self.transformer.text_to_embs(seqs=seqs) for seqs in input_seqs])
#                 numpy.save(input_seqs_filepath, input_seqs)

#         if not os.path.exists(output_seqs_filepath):
#             if self.use_skipthoughts:
#                 output_seqs = self.transformer.text_to_embs(output_seqs, output_seqs_filepath)
#             else:
#                 output_seqs = numpy.array(self.transformer.text_to_embs(seqs=output_seqs, word_embeddings=output_word_embeddings))
#                 numpy.save(output_seqs_filepath, output_seqs)

#         if self.use_skipthoughts:
#             input_seqs = numpy.memmap(input_seqs_filepath, dtype='float64', mode='r',
#                                       shape=(len(input_seqs), self.classifier.context_size, self.transformer.encoder_dim))
#             output_seqs = numpy.memmap(output_seqs_filepath, dtype='float64', mode='r',
#                                       shape=(len(output_seqs), self.transformer.encoder_dim))
#         else:
#             #load seqs from mem-mapped file
#             input_seqs = numpy.load(input_seqs_filepath, mmap_mode='r')
#             output_seqs = numpy.load(output_seqs_filepath, mmap_mode='r')
        
#         print("added", len(input_seqs), "positive examples")
#         print("added", len(input_seqs) * n_neg_per_seq, "negative examples")
#         print("examples divided into", n_chunks, "chunks for training")

#                 #import pdb;pdb.set_trace()
#         seqs_per_chunk = len(input_seqs) / n_chunks
#         for epoch in range(n_epochs):
#             print("TRAINING EPOCH {}/{}".format(epoch + 1, n_epochs))
#             for chunk_idx in range(n_chunks):
#                 chunk_input_seqs = input_seqs[chunk_idx * seqs_per_chunk: (chunk_idx + 1) * seqs_per_chunk]
#                 chunk_output_seqs = output_seqs[chunk_idx * seqs_per_chunk: (chunk_idx + 1) * seqs_per_chunk]
#                 chunk_neg_output_seqs = neg_output_seqs[chunk_idx * seqs_per_chunk: (chunk_idx + 1) * seqs_per_chunk]
#                 chunk_labels = numpy.concatenate([numpy.ones(len(chunk_input_seqs)), numpy.zeros(chunk_neg_output_seqs.shape[0] * chunk_neg_output_seqs.shape[1])])

#                 chunk_input_seqs = numpy.concatenate([chunk_input_seqs, chunk_input_seqs.repeat(n_neg_per_seq, axis=0)])
#                 chunk_output_seqs = numpy.concatenate([chunk_output_seqs, chunk_neg_output_seqs.reshape(-1, chunk_neg_output_seqs.shape[-1])]) #add neg output seqs to positive output seqs

#                 self.classifier.fit(input_seqs=chunk_input_seqs, output_seqs=chunk_output_seqs, labels=chunk_labels)
#     def predict(self, input_seq, output_seq):

#         if self.use_skipthoughts:
#             input_seq = self.transformer.text_to_embs(input_seqs)
#             output_seq = self.transformer.text_to_embs(output_seqs)
#         else:
#             input_seq = numpy.array(self.transformer.text_to_embs(seqs=input_seq))
#             output_seq = numpy.array(self.transformer.text_to_embs(seqs=[output_seq]))[0]
        
#         #import pdb;pdb.set_trace()
#         prob = self.classifier.predict(input_seq, output_seq)
#         return prob

#     @classmethod
#     def load(cls, filepath, word_embeddings):
#         pipeline = Pipeline.load(filepath)
#         pipeline.transformer.word_embeddings = word_embeddings
#         return pipeline

class EmbeddingSimilarityPipeline(Pipeline):
    def predict(self, seqs1, seqs2, use_max_word=False):
        scores = []
        if use_max_word: #avemax
            seqs1 = self.transformer.text_to_embs(seqs1)
            seqs2 = self.transformer.text_to_embs(seqs2)
            for seq1, seq2 in zip(seqs1, seqs2):
                word1_scores = []
                for word1 in seq1:
                    word2_scores = []
                    for word2 in seq2:
                        word2_score = self.classifier.predict(word1, word2)
                        word2_scores.append(word2_score)
                    word1_scores.append(numpy.max(word2_scores))
                scores.append(numpy.mean(word1_scores))
        else:
            if self.transformer.__class__.__name__ == 'SkipthoughtsTransformer':
                seqs1 = self.transformer.text_to_embs(seqs1)[:,0,:]
                seqs2 = self.transformer.text_to_embs(seqs2)[:,0,:]
            else:
                seqs1 = self.transformer.text_to_embs(seqs1, reduce_emb_mode='sum')
                seqs2 = self.transformer.text_to_embs(seqs2, reduce_emb_mode='sum')
            for seq1, seq2 in zip(seqs1, seqs2):
                score = self.classifier.predict(seq1, seq2)
                scores.append(score)
            scores = numpy.array(scores)
        return scores

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
