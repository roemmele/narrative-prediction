'''The Pipeline classes interface between the model transformers (e.g. strings to numbers) and classifiers (e.g. Keras networks that take numbers as input)'''

from __future__ import print_function
import pickle
import warnings
import os
from keras.models import load_model
from models.transformer import *
from models.classifier import *

warnings.filterwarnings('ignore', category=Warning)


class Pipeline(object):

    def __init__(self, transformer, classifier, skip_vectorizer=None):
        self.transformer = transformer
        self.classifier = classifier
        self.skip_vectorizer = skip_vectorizer

    @classmethod
    def load(cls, filepath, word_embs=None, transformer_is_skip=False, has_skip_vectorizer=False, skip_filepath=None):
        if transformer_is_skip:
            if skip_filepath:
                transformer = SkipthoughtsTransformer(filepath=skip_filepath, verbose=False)
            else:
                transformer = SkipthoughtsTransformer(verbose=False)
        else:
            transformer = SequenceTransformer.load(filepath, word_embs=word_embs)
        classifier = SavedModel.load(filepath)
        classifier.filepath = filepath
        if has_skip_vectorizer:
            if skip_filepath:
                skip_vectorizer = SkipthoughtsTransformer(filepath=skip_filepath, verbose=False)
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
        if self.classifier.use_features:  # include additional context features in RNNLM
            feature_vecs = self.transformer.num_seqs_to_bow(
                [self.transformer.tok_seq_to_nums(seq) for seq in self.transformer.seqs_to_feature_words(seqs)])
        for epoch in range(n_epochs):
            if verbose:
                print('EPOCH', epoch + 1)
            self.classifier.fit(seqs=num_seqs, pos_seqs=pos_seqs,
                                feature_vecs=feature_vecs, lexicon_size=self.transformer.lexicon_size)

    def predict(self, seqs, max_length=35, mode='random', batch_size=1, n_best=1, temp=1.0, prevent_unk=True,
                n_context_sents=-1, n_sents_per_seq=None, eos_tokens=[], detokenize=False, capitalize_ents=False, adapt_ents=False):
        # if seq is empty, generate from end-of-sentence marker "."
        seqs = [seq if seq.strip() else u"." for seq in seqs]
        if capitalize_ents or adapt_ents:  # get named entities in seqs
            ents = [number_ents(*get_ents(seq)) for seq in seqs]
        else:
            ents = None
        if self.transformer.generalize_ents:
            seqs = [self.transformer.replace_ents_in_seq(seq) for seq in seqs]
        print("generating sequences...")
        if self.classifier.use_features:  # include additional context features in RNNLM
            feature_vecs = self.transformer.num_seqs_to_bow(
                [self.transformer.tok_seq_to_nums(seq) for seq in self.transformer.seqs_to_feature_words(seqs)])
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
                batch_features = get_batch_features(features=feature_vecs[batch_index:batch_index + batch_size],
                                                    batch_size=batch_size)
            else:
                batch_features = None

            batch_seqs = get_batch(seqs=num_seqs[batch_index:batch_index + batch_size],
                                   batch_size=batch_size)  # prep batch
            batch_pos = get_batch(seqs=num_pos_seqs[batch_index:batch_index + batch_size],
                                  batch_size=batch_size)

            self.classifier.read_batch(seqs=batch_seqs, pos=batch_pos, features=batch_features)

            batch_pred_seqs = numpy.zeros((batch_size, max_length), dtype='int64')

            p_next_words = self.classifier.get_batch_p_next_words(words=batch_seqs[:, -1],
                                                                  pos=batch_pos[:, -1],
                                                                  features=batch_features)

            for idx in range(max_length):  # now predict
                next_words, p_next_words = self.classifier.pred_batch_next_words(
                    p_next_words, mode, n_best, temp, prevent_unk)
                batch_pred_seqs[:, idx] = next_words
                # transform generated word indices back into string for pos tagging
                batch_decoded_seqs = self.transformer.decode_num_seqs(batch_pred_seqs[:, :idx + 1],
                                                                      detokenize=True,
                                                                      ents=ents,
                                                                      capitalize_ents=True,
                                                                      adapt_ents=True)
                # get POS tag of previous generated word
                batch_pos = numpy.array([get_pos_num_seq(seq)[-1] for seq in batch_decoded_seqs])
                p_next_words = self.classifier.get_batch_p_next_words(
                    words=batch_pred_seqs[:, idx], pos=batch_pos, features=batch_features)

            self.classifier.pred_model.reset_states()

            batch_pred_seqs = batch_seqs_to_list(batch_pred_seqs,
                                                 len_batch=len(
                                                     num_seqs[batch_index:batch_index + batch_size]),
                                                 batch_size=batch_size)
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
            feature_vecs = self.transformer.num_seqs_to_bow([self.transformer.tok_seq_to_nums(seq)
                                                             for seq in self.transformer.seqs_to_feature_words(seqs)])

        return self.classifier.get_probs(seqs=num_seqs, pos_seqs=num_pos_seqs, feature_vecs=feature_vecs, batch_size=batch_size)


class MLPLMPipeline(Pipeline):

    def fit(self, seqs, n_epochs=5):
        if not self.transformer.lexicon:
            self.transformer.make_lexicon(seqs)
        if self.transformer.generalize_ents:
            seqs = [self.transformer.replace_ents_in_seq(seq) for seq in seqs]
        seqs = self.transformer.text_to_nums(seqs)
        self.classifier.fit(seqs=seqs,
                            lexicon_size=self.transformer.lexicon_size, n_epochs=n_epochs)

    def predict(self, seqs, max_length=35, mode='random', batch_size=1, n_best=1, temp=1.0,
                prevent_unk=True, n_sents_per_seq=None, eos_tokens=[], detokenize=False, capitalize_ents=False, adapt_ents=False):
        if capitalize_ents or adapt_ents:  # get named entities in seqs
            ents = [dict(number_ents(get_ents(seq))) for seq in seqs]
        else:
            ents = None
        if self.transformer.generalize_ents:
            seqs = [self.transformer.replace_ents_in_seq(seq) for seq in seqs]
        seqs = self.transformer.text_to_nums(seqs)
        gen_seqs = self.classifier.predict(seqs=seqs, max_length=max_length, mode=mode, batch_size=batch_size, n_best=n_best,
                                           temp=temp, prevent_unk=prevent_unk)
        print("decoding generated sequences...")
        gen_seqs = self.transformer.decode_num_seqs(gen_seqs, n_sents_per_seq=n_sents_per_seq, eos_tokens=eos_tokens,
                                                    detokenize=detokenize, ents=ents,
                                                    capitalize_ents=capitalize_ents, adapt_ents=adapt_ents)
        return gen_seqs

    def get_probs(self, seqs, batch_size=None):
        seqs = self.transformer.text_to_nums(seqs)
        return self.classifier.get_probs(seqs=seqs)  # , batch_size=batch_size)


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
                window = seq[sent_idx:sent_idx + window_size + 1]
                seq1 = window[0]
                seq2 = [word for sent in window[1:] for word in sent]
                # get all pairs of words in this sequence window
                seq_pairs = numpy.array(get_word_pairs(seq1, seq2))
                pairs.extend(seq_pairs)
        return pairs

    def fit(self, seqs, n_epochs=1):
        embedded_input = True if (
            self.transformer.word_embeddings or self.transformer.use_spacy_embs) else False
        if not embedded_input and not self.transformer.lexicon:
            self.transformer.make_lexicon(seqs)  # Use true pairs to build lexicon
        true_pairs = self.get_true_pairs(seqs)
        reversed_pairs = reverse_pairs(true_pairs)
        random_pairs = randomize_pairs(true_pairs)
        false_pairs = reversed_pairs + random_pairs
        pairs = numpy.array(true_pairs + false_pairs)
        labels = numpy.concatenate((numpy.ones(len(true_pairs)), numpy.zeros(len(false_pairs))))
        self.classifier.fit(cause_words=pairs[:, 0], effect_words=pairs[:, 1],
                            labels=labels, lexicon_size=self.transformer.lexicon_size,
                            embedded_input=embedded_input, n_epochs=n_epochs)

    def predict(self, seq1, seq2):
        '''return a total score for the causal relatedness between seq1 and seq2'''
        embedded_input = True if (
            self.transformer.word_embeddings or self.transformer.use_spacy_embs) else False
        if embedded_input:
            seq1, seq2 = self.transformer.text_to_embs([seq1, seq2])
        else:
            seq1, seq2 = self.transformer.text_to_nums([seq1, seq2])
        pairs = numpy.array(get_word_pairs(seq1, seq2))
        prob = numpy.mean(self.classifier.predict(cause_words=pairs[:, 0],
                                                  effect_words=pairs[:, 1]))
        return prob


class MLPBinaryPipeline(Pipeline):

    def get_true_pairs(self, seqs, segment_clauses=False, max_clause_length=15):
        if segment_clauses:
            seqs = [segment_into_clauses(seq) for seq in seqs]
        else:
            seqs = [segment(seq) for seq in seqs]  # segment by sentence instead of clause
        # add 10 to max length to account for grammatical words
        pairs = get_adj_clause_pairs(seqs, max_length=max_clause_length)
        pairs = [self.transformer.text_to_nums(pair) for pair in pairs]
        #pairs = [self.transformer.num_seqs_to_bow(pair) for pair in pairs if pair[0] and pair[1]]
        pairs = [pair for pair in pairs if pair[0] and pair[1]]
        return pairs

    def fit(self, seqs, clause_window=1, max_clause_length=15, n_epochs=1, verbose=True):
        #embedded_input = True if (self.transformer.word_embeddings or self.transformer.use_spacy_embs) else False
        if not self.transformer.lexicon:
            self.transformer.make_lexicon(seqs)
        true_pairs = self.get_true_pairs(seqs, max_clause_length=max_clause_length)
        reversed_pairs = reverse_pairs(true_pairs)
        random_pairs = randomize_pairs(true_pairs)  # + randomize_pairs(true_pairs)
        false_pairs = random_pairs + reversed_pairs
        pairs = true_pairs + false_pairs
        labels = numpy.concatenate((numpy.ones(len(true_pairs)), numpy.zeros(len(false_pairs))))
        if verbose:
            print("training on", len(true_pairs), "true pairs, and", len(false_pairs),
                  "false pairs, (total=", len(true_pairs + false_pairs), ")")
        self.classifier.fit(seqs1=[pair[0] for pair in pairs], seqs2=[pair[1] for pair in pairs], labels=labels,
                            lexicon_size=self.transformer.lexicon_size, n_epochs=n_epochs)  # embedded_input=embedded_input,

    def predict(self, seq1, seq2):
        '''return a total score for the causal relatedness between seq1 and seq2'''
        seq1, seq2 = self.transformer.num_seqs_to_bow(self.transformer.text_to_nums([seq1, seq2]))
        prob = self.classifier.predict(seq1=seq1[None], seq2=seq2[None])
        return prob


class RNNBinaryPipeline(Pipeline):

    def get_bkwrd_sample_idxs(self, n_seqs, n_bkwrd_sents, n_samples=1):
        '''get indices of randomly selected sentences in the input seqs (seqs1)'''
        bkwrd_idxs = numpy.array([rng.choice(n_bkwrd_sents, size=n_samples, replace=False)
                                  for seq in range(n_seqs)])
        return bkwrd_idxs

    def get_random_sample_idxs(self, n_seqs, n_idxs, n_samples=1):
        '''get indices of randomly selected sentences in the output sentences (seqs2)'''
        random_idxs = numpy.array([rng.choice(n_idxs, size=n_samples, replace=False)
                                   for seq in range(n_seqs)])
        return random_idxs

    def fit(self, seqs1, seqs2, n_bkwrd=0, n_random=1, n_epochs=1, eval_fn=None, chunk_size=2000):

        if self.classifier.filepath and not os.path.isdir(self.classifier.filepath):
            os.mkdir(self.classifier.filepath)

        if not self.transformer.__class__.__name__ == 'SkipthoughtsTransformer' and not self.transformer.lexicon:
            self.transformer.make_lexicon([sent for seq in seqs1 for sent in seq] + seqs2)

        if self.transformer.__class__.__name__ == 'SkipthoughtsTransformer':
            seqs1 = self.transformer.text_to_embs(seqs1,
                                                  seqs_filepath=self.classifier.filepath + '/seqs1.npy' if self.classifier.filepath else None)
            seqs2 = self.transformer.text_to_embs(seqs2,
                                                  seqs_filepath=self.classifier.filepath + '/seqs2.npy' if self.classifier.filepath else None)
        else:
            if self.classifier.embedded_input:
                seqs1 = [self.transformer.text_to_embs(seq, reduce_emb_mode='sum')
                         for seq in seqs1]
                seqs2 = self.transformer.text_to_embs(seqs2, reduce_emb_mode='sum')[:, None]
            else:
                seqs1 = [self.transformer.text_to_nums(seq, reduce_emb_mode='sum')
                         for seq in seqs1]
                seqs2 = self.transformer.text_to_nums(seqs2)[:, None]

        print("training model for", n_epochs, "epochs on", len(seqs1), "positive instances,", len(seqs1) * n_random,
              "random negative instances, and", len(seqs1) * n_bkwrd, "backward negative instances")

        if not hasattr(self, 'best_accuracy'):
            self.best_accuracy = -numpy.inf
        for epoch in range(n_epochs):
            if n_epochs > 1:
                print("EPOCH:", epoch + 1)
            for chunk_idx in range(0, len(seqs1), chunk_size):

                seqs1_chunk = seqs1[chunk_idx:chunk_idx + chunk_size]
                seqs2_chunk = seqs2[chunk_idx:chunk_idx + chunk_size]
                labels_chunk = numpy.ones((len(seqs1_chunk),))

                if n_bkwrd:
                    bkwrd_sample_idxs = self.get_bkwrd_sample_idxs(n_seqs=chunk_size,
                                                                   n_bkwrd_sents=self.classifier.n_input_sents,
                                                                   n_samples=n_bkwrd)
                    bkwrd_seqs1_chunk, bkwrd_seqs2_chunk = zip(*[[seq1, seq1[idx][None]] for seq1, idxs
                                                                 in zip(seqs1[chunk_idx:chunk_idx + chunk_size], bkwrd_sample_idxs)
                                                                 for idx in idxs])
                    seqs1_chunk = numpy.concatenate([seqs1_chunk, numpy.array(bkwrd_seqs1_chunk)])
                    seqs2_chunk = numpy.concatenate([seqs2_chunk, numpy.array(bkwrd_seqs2_chunk)])
                    labels_chunk = numpy.concatenate([labels_chunk,
                                                      numpy.zeros((len(bkwrd_seqs1_chunk),))])
                if n_random:
                    random_sample_idxs = self.get_random_sample_idxs(n_seqs=chunk_size,
                                                                     n_idxs=len(seqs1),
                                                                     n_samples=n_random)
                    random_seqs1_chunk, random_seqs2_chunk = zip(*[[seq1, seqs2[idx]] for seq1, idxs
                                                                   in zip(seqs1[chunk_idx:chunk_idx + chunk_size], random_sample_idxs)
                                                                   for idx in idxs])
                    seqs1_chunk = numpy.concatenate([seqs1_chunk, numpy.array(random_seqs1_chunk)])
                    seqs2_chunk = numpy.concatenate([seqs2_chunk, numpy.array(random_seqs2_chunk)])
                    labels_chunk = numpy.concatenate([labels_chunk,
                                                      numpy.zeros((len(random_seqs1_chunk),))])

                assert(len(seqs1_chunk) == len(seqs2_chunk) == len(labels_chunk))

                # shuffle instances
                shuffle_idxs = rng.permutation(len(seqs1_chunk))
                seqs1_chunk = numpy.array([seqs1_chunk[idx] for idx in shuffle_idxs])
                seqs2_chunk = numpy.array([seqs2_chunk[idx] for idx in shuffle_idxs])
                labels_chunk = numpy.array(labels_chunk[shuffle_idxs])

                self.classifier.fit(seqs1=seqs1_chunk, seqs2=seqs2_chunk,
                                    labels=labels_chunk, n_epochs=1, save_to_filepath=False)

                if eval_fn:
                    if not hasattr(self, 'best_accuracy'):
                        self.best_accuracy = -numpy.inf
                    accuracy = eval_fn(self)
                    print("validation accuracy:", accuracy)
                    if accuracy >= self.best_accuracy:
                        self.best_accuracy = accuracy
                        if self.classifier.filepath:
                            self.classifier.save()
                elif self.classifier.filepath:
                    self.classifier.save()

    def predict(self, seqs1, seqs2):

        if self.transformer.__class__.__name__ == 'SkipthoughtsTransformer':
            seqs1 = self.transformer.text_to_embs(seqs1)
            seqs2 = self.transformer.text_to_embs(seqs2)

        else:
            if self.classifier.embedded_input:
                seqs1 = [self.transformer.text_to_embs(seq, reduce_emb_mode='sum')
                         for seq in seqs1]
                seqs2 = self.transformer.text_to_embs(seqs2, reduce_emb_mode='sum')[:, None]
            else:
                seqs1 = [self.transformer.text_to_nums(seq, reduce_emb_mode='sum')
                         for seq in seqs1]
                seqs2 = self.transformer.text_to_nums(seqs2)[:, None]

        probs = []
        for seq1, seq2 in zip(seqs1, seqs2):
            prob = self.classifier.predict(seq1=seq1, seq2=seq2)
            probs.append(prob)

        probs = numpy.array(probs)
        return probs


class EncoderDecoderPipeline(Pipeline):

    def fit(self, seqs1, seqs2, max_length=25, n_epochs=1, eval_fn=None, chunk_size=200000, verbose=True):

        if not self.transformer.lexicon:
            self.transformer.make_lexicon(seqs1 + seqs2)

        seqs1 = self.transformer.text_to_nums(seqs1)
        seqs2 = self.transformer.text_to_nums(seqs2)

        assert(len(seqs1) == len(seqs2))

        for epoch in range(n_epochs):
            if n_epochs > 1:
                if verbose:
                    print("EPOCH:", epoch + 1)
            if verbose:
                print("training on", len(seqs1), "sequence pairs")
            for chunk_idx in range(0, len(seqs1), chunk_size):
                self.classifier.fit(seqs1[chunk_idx:chunk_idx + chunk_size], seqs2[chunk_idx:chunk_idx + chunk_size],
                                    n_timesteps=max_length, lexicon_size=self.transformer.lexicon_size,
                                    n_epochs=1, save_to_filepath=False)

                if eval_fn:
                    if not hasattr(self, 'best_accuracy'):
                        self.best_accuracy = -numpy.inf
                    accuracy = eval_fn(self)
                    if accuracy >= self.best_accuracy:
                        self.best_accuracy = accuracy
                        if self.classifier.filepath:
                            self.classifier.save()
                elif self.classifier.filepath:
                    self.classifier.save()

    def predict(self, seqs1, seqs2):
        '''return a total score for the prob that seq2 follows seq1'''

        seqs1 = self.transformer.text_to_nums(seqs1)
        seqs2 = self.transformer.text_to_nums(seqs2)

        probs = []
        for seq1, seq2 in zip(seqs1, seqs2):
            prob = self.classifier.predict(seq1=seq1, seq2=seq2)
            probs.append(prob)

        probs = numpy.array(probs)
        return probs

    def get_most_probable_words(self, seqs1, top_n_words=10, unigram_probs=None):
        '''return the most probable output words for input sequences'''
        if self.classifier.flat_input and self.classifier.embedded_input:
            if self.skip_vectorizer is not None:  # input sequences will be transformer into flat vectors
                seqs1 = self.skip_vectorizer.text_to_embs(seqs1)[:, 0, :]
            else:
                seqs1 = self.transformer.text_to_embs(seqs1, reduce_emb_mode='sum')
        else:
            seqs1 = self.transformer.text_to_nums(seqs1)

        most_probable_words = []
        probs = []
        for seq1 in seqs1:
            seq_probable_words, seq_probs = self.classifier.get_most_probable_words(seq1,
                                                                                    top_n_words=top_n_words,
                                                                                    unigram_probs=unigram_probs)
            probs.append(seq_probs)
            seq_probable_words = [self.transformer.lexicon_lookup[word]
                                  for word in seq_probable_words]  # transform word indices to strings
            most_probable_words.append(seq_probable_words)

        probs = numpy.array(probs)

        return most_probable_words, probs


class ClassifierPipeline(Pipeline):

    def fit(self, seqs, labels, n_epochs=1):

        if self.transformer.use_spacy_embs or self.transformer.word_embeddings is not None:
            seqs = self.transformer.text_to_embs(seqs, reduce_emb_mode='sum')
            n_input_nodes = seqs.shape[-1]
        else:
            if not self.transformer.lexicon:
                self.transformer.make_lexicon(seqs)
            seqs = self.transformer.text_to_nums(seqs)
            n_input_nodes = self.transformer.lexicon_size + 1
        self.classifier.fit(seqs, labels, n_input_nodes=n_input_nodes, n_epochs=n_epochs)

    def predict(self, seqs):
        if self.transformer.use_spacy_embs or self.transformer.word_embeddings is not None:
            seqs = self.transformer.text_to_embs(seqs, reduce_emb_mode='sum')
        else:
            seqs = self.transformer.text_to_nums(seqs)
        return self.classifier.predict(seqs)


class EmbeddingSimilarityPipeline(Pipeline):

    def predict(self, seqs1, seqs2, use_max_word=False):
        scores = []
        if use_max_word:  # avemax
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
                seqs1 = self.transformer.text_to_embs(seqs1)[:, 0, :]
                seqs2 = self.transformer.text_to_embs(seqs2)[:, 0, :]
            else:
                seqs1 = self.transformer.text_to_embs(seqs1, reduce_emb_mode='sum')
                seqs2 = self.transformer.text_to_embs(seqs2, reduce_emb_mode='sum')
            for seq1, seq2 in zip(seqs1, seqs2):
                score = self.classifier.predict(seq1, seq2)
                scores.append(score)
            scores = numpy.array(scores)
        return scores
