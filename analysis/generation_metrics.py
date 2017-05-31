import sys, numpy, pandas

from scipy.spatial.distance import cosine

from textacy import *

import models.transformer
reload(models.transformer)
from models.transformer import *

import models.ngram
reload(models.ngram)
from models.ngram import *

#sys.path.append('../Bleu-master')
#sys.path.append('Bleu-master')
#from calculatebleu import *

from nltk.translate.bleu_score import *

from pycorenlp import StanfordCoreNLP
corenlp = StanfordCoreNLP('http://localhost:9000')

sys.path.append('../grammaticality-metrics-master/codalab/scoring_program')
sys.path.append('grammaticality-metrics-master/codalab/scoring_program')
import evaluate
reload(evaluate)
from evaluate import call_lt

skipthoughts_transformer = None

def check_seqs_format(seqs):
    '''functions below expect generated sequences to be a list of lists, i.e. multiple sequences for each context sequence;
    transform to this format if seqs are a flat list'''
    if type(seqs[0]) not in (list, tuple):
        seqs = [[seq] for seq in seqs]
    return seqs

def get_seq_lengths(seq):
    lengths = [len(segment_and_tokenize(seq)) for seq in seqs]
    return lengths

def get_bleu_scores(gen_seqs, gold_seqs):
    '''compute bleu scores of generated sequences relative to gold'''

    gen_seqs = check_seqs_format(gen_seqs)
    gold_seqs = check_seqs_format(gold_seqs)
    bleu_scores = []
    for seq_idx, (gen_seqs_, gold_seqs_) in enumerate(zip(gen_seqs, gold_seqs)):
        gold_seqs_ = [segment_and_tokenize(seq) for seq in gold_seqs_]
        seq_bleu_scores = [sentence_bleu(gold_seqs_, segment_and_tokenize(seq)) for seq in gen_seqs_]
        if seq_idx % 500 == 0:
            print "computed blue scores for", seq_idx, "gold sequences..."
        bleu_scores.append(seq_bleu_scores)
    bleu_scores = numpy.array(bleu_scores)
    return bleu_scores


def get_verified_ngrams(transformer, gen_seqs, n, db_filepath):
    '''return the ngrams in gen_seqs that also occur in another corpus of n-grams (in db_filepath)'''

    gen_seqs = check_seqs_format(gen_seqs)
    verified_ngrams = {}
    for seqs in gen_seqs:
        ngrams = extract_ngrams(seqs, n)
        counts = get_ngram_counts(transformer, ngrams, db_filepath)
        for ngram,count in zip(ngrams, counts):
            if count:
                if ngram in verified_ngrams:
                    verified_ngrams[ngram] += count
                else:
                    verified_ngrams[ngram] = count
    return verified_ngrams

def get_n_ngrams(gen_seqs, n):
    '''return the total number of ngrams and total number of unique ngrams'''
    gen_seqs = check_seqs_format(gen_seqs)
    unique_ngrams = set()
    n_ngrams = 0
    for seqs in gen_seqs:
        seq_ngrams = extract_ngrams(seqs, n)
        n_ngrams += len(seq_ngrams)
        unique_ngrams.update(seq_ngrams)
    return {"total":n_ngrams, "unique":len(unique_ngrams)}

def get_phrases(seqs, min_count=5, threshold=10):
    #import pdb;pdb.set_trace()
    bigram_phraser = Phrases((segment_and_tokenize(seq) for seq in seqs), delimiter=' ', min_count=min_count, threshold=threshold)
    bigram_phrases = set(list(bigram_phraser.export_phrases((segment_and_tokenize(seq) for seq in seqs))))
    trigram_phraser = Phrases(bigram_phraser[(segment_and_tokenize(seq) for seq in seqs)], delimiter=' ', min_count=min_count, threshold=threshold)
    trigram_phrases = set(list(trigram_phraser.export_phrases(bigram_phraser[(segment_and_tokenize(seq) for seq in seqs)])))
    phrases = bigram_phrases.union(trigram_phrases)
    phrases = [(phrase, score) for score, phrase in 
               sorted([(score, phrase) for phrase, score in phrases], reverse=True)]
    return phrases

def lookup_phrases(ref_phrases, cand_phrases):
    '''check whether an ngram (cand phrase) exists as phrase in corpus of known phrases'''
    known_phrases = [phrase for phrase in cand_phrases if phrase in set(ref_phrases)]
    return known_phrases

def compare_phrases(phrases1, phrases2):
    phrases1 = set([phrase for phrase, score in phrases1])
    print "# of phrases1:", len(phrases1)
    phrases2 = set([phrase for phrase, score in phrases2])
    print "# of phrases2:", len(phrases2)
    common_phrases = phrases1.intersection(phrases2)
    print "# of common phrases:", len(common_phrases)
    print "common phrases:", common_phrases, "\n"
    unique_phrases1 = phrases1.difference(phrases2)
    print "# of unique phrases in phrases1:", len(unique_phrases1)
    print "phrases1 unique phrases:", unique_phrases1, "\n"
    unique_phrases2 = phrases2.difference(phrases1)
    print "# of unique phrases in phrases2:", len(unique_phrases2)
    print "phrases2 unique phrases:", unique_phrases2, "\n"

def get_candidate_phrases(gen_sents):
    if type(gen_sents[0]) not in (list, tuple):
        gen_sents = [[sent] for sent in gen_sents]
    cand_phrases = []
    for sents in gen_sents:
        for sent in sents:
            sent = encoder(sent)
            for n in (2,3): #get phrases of both 2 and 3 words
                cand_phrases.extend([ngram.text.lower() for ngram in 
                                     extract.ngrams(sent, n=n, filter_punct=False, filter_stops=False)])
    return cand_phrases

def get_word_pairs(context_seq, gen_seq, include_pos=['ADJ', 'ADV', 'INTJ', 'NOUN', 'PRON', 'PROPN', 'VERB']):
    '''get all word pairs between context sequence and generated sequence'''
    context_seq = encoder(context_seq)
    gen_seq = encoder(gen_seq)
    context_seq = extract.words(context_seq, include_pos=include_pos)
    gen_seq = list(extract.words(gen_seq, include_pos=include_pos))
    pairs = []
    for context_token in context_seq:
        for gen_token in gen_seq:
            pairs.append((context_token.string.lower().strip(), gen_token.string.lower().strip()))
    return pairs

# def get_word_matches(word_pairs):
#     '''find all pairs of same words'''
#     word_matches = [(word1, word2) for word1, word2 in word_pairs if word1.string == word2.string]
#     return word_matches

def get_jaccard_sim(context_seq, gen_seq, include_pos=['ADJ', 'ADV', 'INTJ', 'NOUN', 'PRON', 'PROPN', 'VERB']):
    context_words = set([word.string.lower().strip() for word in extract.words(encoder(context_seq), include_pos=include_pos)])
    gen_seq_words = set([word.string.lower().strip() for word in extract.words(encoder(gen_seq), include_pos=include_pos)])
    if not len(context_words) and not len(gen_seq_words):
        jaccard_sim = 0
    else:
        common_words = context_words.intersection(gen_seq_words)
        jaccard_sim = len(common_words) * 1. / (len(context_words) + len(gen_seq_words) - len(common_words))
    return jaccard_sim

def get_word2vec_sim(context_seq, gen_seq):
    word_pairs = get_word_pairs(context_seq, gen_seq)
    if word_pairs:
        pair_scores = [similarity.word2vec(encoder(word1),encoder(word2)) for word1,word2 in word_pairs]
    else: #no word pairs between context and generated sequences (e.g. generated sequence might be punctuation only)
        pair_scores = [0]
    # assert(len(word_pairs) == len(pair_scores))
    word2vec_sim = numpy.mean(pair_scores)
    return word2vec_sim

# def get_word_pair_dist(transformer, context_seqs, fin_sents):
#     word_pairs = {}
#     context_seqs, fin_sents = transformer.transform(context_seqs, fin_sents)
#     for context_seq, fin_sent in zip(context_seqs, fin_sents):
#         for context_word in context_seq:
#             for fin_word in fin_sent:
#                 word_pair = (transformer.lexicon_lookup[context_word], 
#                              transformer.lexicon_lookup[fin_word])
#                 if word_pair not in word_pairs:
#                     word_pairs[word_pair] = 0
#                 word_pairs[word_pair] += 1
#     return word_pairs

# def get_sim_scores(word_pairs):
#     sim_scores = {}
#     for word_pair in word_pairs:
#         word1, word2 = word_pair
#         #sim_score = encoder(unicode(word1)).similarity(encoder(unicode(word2)))
#         if word1 in model.vocab and word2 in model.vocab:
#             sim_score = sim_model.similarity(word1, word2)
#         else:
#             sim_score = 0.0
#         sim_scores[word_pair] = sim_score
#     return sim_scores

# def get_mean_sim(word_pairs, sim_scores):
#     '''compute mean similarity between contexts and final sentences'''
#     weighted_sim_scores = []
#     for word_pair, count in word_pairs.items():
#         weighted_score = sim_scores[word_pair] * count #scale by frequency of this pair
#         weighted_sim_scores.append(weighted_score)
#     total_pair_count = sum(word_pairs.values())
#     mean_sim_score = sum(weighted_sim_scores) / total_pair_count
#     return mean_sim_score

def get_lexical_sim(context_seqs, gen_seqs, mode='word2vec'):
    '''compute average word2vec or jaccard similarity between all pairs of words between context and generated sequences'''

    gen_seqs = check_seqs_format(gen_seqs)
    sim_scores = []
    for context_seq, seqs in zip(context_seqs, gen_seqs):
        scores = []
        for gen_seq in seqs:
            if mode == 'word2vec':
                score = get_word2vec_sim(context_seq, gen_seq)
            elif mode == 'jaccard':
                score = get_jaccard_sim(context_seq, gen_seq)
            scores.append(score)
        sim_scores.append(scores)
    sim_scores = numpy.array(sim_scores)
    return sim_scores

def get_skipthought_similarity(context_seqs, gen_seqs, verbose=True):
    global skipthoughts_transformer
    if not skipthoughts_transformer:
        skipthoughts_transformer = SkipthoughtsTransformer(verbose=verbose)
    sim_scores = []
    context_seqs = [segment(seq) for seq in context_seqs]
    #take mean of vectors for all sentences in context
    context_seqs = numpy.array([numpy.mean(encoded_context_seqs, axis=0) for encoded_context_seqs\
                                        in skipthoughts_transformer.encode(context_seqs)])
    gen_seqs = skipthoughts_transformer.encode(gen_seqs)
    for context_seq, gen_seqs_ in zip(context_seqs, gen_seqs):
        scores = [(1 - cosine(context_seq, gen_seq)) for gen_seq in gen_seqs_]
        sim_scores.append(scores)
    sim_scores = numpy.array(sim_scores)

    return sim_scores

def get_corefs(context_seqs, gen_seqs):

    assert(type(gen_seqs) in (list,tuple) and type(context_seqs) in (list,tuple))

    gen_seqs = check_seqs_format(gen_seqs)

    corefs = {'ents':[], 'corefs':[]}
    for context_seq_idx, (context_seq, gen_seqs_) in enumerate(zip(context_seqs, gen_seqs)):
        gen_ents = []
        gen_corefs = []
        for gen_seq in gen_seqs_:
            try:
                parse = corenlp.annotate((context_seq + " " + gen_seq).encode('utf-8',errors='replace'),\
                                         properties={'annotators': 'coref', 'outputFormat': 'json'})
            except:
                print "error:", context_seq + " " + gen_seq
                parse = {}
        #         except RPCInternalError:
        #             parse = {}
            #get all potential coferring entities (i.e. nouns) in generated sequence
            sents = parse['sentences']
            seq_ents = [(token['index'], token['word']) for token in sents[-1]['tokens']\
                                                    if token['pos'] in ('NN','NNP','NNPS','NNS','PRP','PRP$')]
            seq_corefs = []
            for coref_ent_idx, coref_ent in parse['corefs'].items():
                mentions = {'rep_mention':None, 'context_mentions':[], 'gen_mentions':[]}
                for mention in coref_ent:
                    if mention['isRepresentativeMention']:
                        mentions['rep_mention'] = (mention['sentNum'], mention['text'])
                    if mention['sentNum'] == len(sents): #mention is in generated sentence
                        mentions['gen_mentions'].append((mention['sentNum'], mention['text']))
                    else:
                        mentions['context_mentions'].append((mention['sentNum'], mention['text']))
                seq_corefs.append(mentions)
            gen_ents.append(seq_ents)
            gen_corefs.append(seq_corefs)
        if context_seq_idx % 500 == 0:
            print "processed coreferences in", context_seq_idx, "sequences..."

        # if single_gen_seq:
        #     gen_ents = gen_ents[0]
        #     gen_corefs = gen_corefs[0]

        corefs['ents'].append(gen_ents)
        corefs['corefs'].append(gen_corefs)

    # if single_seq:
    #     corefs['ents'] = corefs['ents'][0]
    #     corefs['corefs'] = corefs['corefs'][0]

    return corefs

def get_coref_counts(context_seqs, gen_seqs):

    counts = {'ents':[], 'corefs':[], 'prev_mention_sents':[]}
    corefs = get_corefs(context_seqs, gen_seqs)

    for gen_ents, gen_corefs in zip(corefs['ents'], corefs['corefs']):
        gen_coref_counts = []
        gen_ent_counts = []
        gen_prev_mention_sents = []
        for seq_ents, seq_corefs in zip(gen_ents, gen_corefs):
            #print "ents:", seq_ents
            ent_count = len(seq_ents)
            #print "ent count:", ent_count
            gen_ent_counts.append(ent_count)
            #print "corefs:", seq_corefs
            coref_counts = sum([len(coref['gen_mentions']) for coref in seq_corefs])
            #print "coref_counts:", coref_counts
            gen_coref_counts.append(coref_counts)
            prev_mentions = []
            for coref in seq_corefs:
                #find the sentence position (number) of the most recent previous mention of each coreferring entity; 
                #if an entity is the first mention in the generated sequence, look for a coreference in the preceding context sequence;
                #if none found or the entity is not the first mention, the previous mention position is the number of the generated sentence itself
                # coref_prev_mentions = []
                for mention_idx, mention in enumerate(coref['gen_mentions']):
                    if mention_idx > 0:
                        prev_mentions.append(coref['gen_mentions'][mention_idx-1][0])
                    elif not coref['context_mentions']:
                        prev_mentions.append(mention[0])
                    else:
                        prev_mentions.append(coref['context_mentions'][-1][0])
                #prev_mentions.append(coref_prev_mentions)
            #print "prev mentions:", prev_mentions, "\n"
            gen_prev_mention_sents.append(prev_mentions)
        counts['ents'].append(gen_ent_counts)
        counts['corefs'].append(gen_coref_counts)
        counts['prev_mention_sents'].append(gen_prev_mention_sents)

    counts['ents'] = numpy.array(counts['ents'])
    counts['mean_ents'] = numpy.mean(counts['ents'])
    counts['corefs'] = numpy.array(counts['corefs'])
    counts['mean_corefs'] = numpy.mean(counts['corefs'])
    counts['resolution_rates'] = numpy.nan_to_num(counts['corefs'] * 1. / counts['ents'])
    counts['mean_resolution_rates'] = numpy.mean(counts['resolution_rates'])

    return counts

def get_grammaticality_scores(gen_seqs):
    gen_seqs = check_seqs_format(gen_seqs)
    n_gen_per_seq = numpy.array([len(seqs) for seqs in gen_seqs])
    if numpy.all(n_gen_per_seq == n_gen_per_seq[0]):
        #every sequence has the same length
        n_gen_per_seq = n_gen_per_seq[0]
    grammaticality_scores = call_lt([seq for seqs in gen_seqs for seq in seqs])
    if type(n_gen_per_seq) == list:
        #different lengths per sequence
        idxs = [numpy.sum(n_gen_per_seq[:idx]) for idx in range(len(n_gen_per_seq))] + [None] #add -1 for last entry
        grammaticality_scores = [grammaticality_scores[idxs[start]:idxs[start+1]] for start in range(len(idxs) - 1)]
    else:
        grammaticality_scores = numpy.array(grammaticality_scores).reshape(len(gen_seqs), n_gen_per_seq)
    #grammaticality_scores = numpy.array(grammaticality_scores)
    return grammaticality_scores

def get_type_token_ratios(gen_seqs):
    gen_seqs = check_seqs_format(gen_seqs)
    token_counts = {}
    for seqs in gen_seqs:
        for seq in seqs:
            seq_tokens = segment_and_tokenize(seq)
            for token in seq_tokens:
                if token in token_counts:
                    token_counts[token] += 1
                else:
                    token_counts[token] = 1
    n_types = len(token_counts)
    n_tokens = sum(token_counts.values())
    return {'n_types':n_types, 'n_tokens':n_tokens}

def get_pos_ngram_similarity(context_seqs, gen_seqs, n=3):

    gen_seqs = check_seqs_format(gen_seqs)
    pos_sim_scores = []
    for context_seq, gen_seqs_ in zip(context_seqs, gen_seqs):
        scores = []
        context_pos_ngrams = [tuple([word.tag for word in ngram])
                            for ngram in extract.ngrams(encoder(context_seq), n=n, filter_stops=False, filter_punct=False)]
        for gen_seq in gen_seqs_:
            gen_pos_ngrams = [tuple([word.tag for word in ngram])
                                for ngram in extract.ngrams(encoder(gen_seq), n=n, filter_stops=False, filter_punct=False)]
            if gen_pos_ngrams:
                score = len([ngram for ngram in gen_pos_ngrams if ngram in set(context_pos_ngrams)]) * 1. / len(gen_pos_ngrams)
            else:
                score = 0.0
            scores.append(score)
        pos_sim_scores.append(scores)
    pos_sim_scores = numpy.array(pos_sim_scores)
    return pos_sim_scores

def get_noun_chunk_complexity(gen_seqs):
    '''return number and length of noun chunks in each generated sequence'''
    gen_seqs = check_seqs_format(gen_seqs)
    chunk_lengths = []
    n_chunks = []
    for gen_seqs_ in gen_seqs:
        mean_lengths = []
        ns = []
        for gen_seq in gen_seqs_:
            gen_seq = encoder(gen_seq)
            seq_chunks = [chunk for chunk in gen_seq.noun_chunks]
            n = len(seq_chunks)
            if n:
                mean_length = numpy.mean([len(chunk) for chunk in seq_chunks])
            else:
                mean_length = 0 #if no chunks in this sequence, set mean length to 0
            mean_lengths.append(mean_length)
            ns.append(n)
        n_chunks.append(ns)
        chunk_lengths.append(mean_lengths)
    n_chunks = numpy.array(n_chunks)
    chunk_lengths = numpy.array(chunk_lengths)
    return pandas.Series({'n_chunks':n_chunks, 'chunk_lengths':chunk_lengths, 'mean_n_chunks':numpy.mean(n_chunks),\
                        'mean_chunk_lengths':numpy.mean(chunk_lengths[~numpy.isnan(chunk_lengths)])})


def get_svo_complexity(gen_seqs):
    '''return number of subject-verb-object structures in generated sequences'''
    gen_seqs = check_seqs_format(gen_seqs)
    n_svos = []
    for gen_seqs_ in gen_seqs:
        ns = []
        for gen_seq in gen_seqs_:
            gen_seq = encoder(gen_seq)
            n = len([svo for svo in extract.subject_verb_object_triples(gen_seq)])
            ns.append(n)
        n_svos.append(ns)
    n_svos = numpy.array(n_svos)
    return pandas.Series({'n_svos':n_svos, 'mean_n_svos':numpy.mean(n_svos)})

def get_frequency_scores(gen_seqs, word_freqs):

    gen_seqs = check_seqs_format(gen_seqs)
    freq_scores = []
    for gen_seqs_ in gen_seqs:
        scores = []
        for gen_seq in gen_seqs_:
            score = numpy.mean([numpy.log(word_freqs[word]) if word in word_freqs else numpy.log(1) for word in segment_and_tokenize(gen_seq)])
            scores.append(score)
        freq_scores.append(scores)
    freq_scores = numpy.array(freq_scores)
    return pandas.Series({'freq_scores':freq_scores, 'mean_freq_scores':numpy.mean(freq_scores)})

#def get_lsm_scores
#categories: pronouns, determiners, auxillary verbs, adverbs, prepositions, conjuctions, negations, quantifiers




            
