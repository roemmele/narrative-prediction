import sys, numpy, pandas

from scipy.spatial.distance import cosine

from textacy import *

from gensim.models import Word2Vec

import models.transformer
reload(models.transformer)
from models.transformer import *

import models.ngram
reload(models.ngram)
from models.ngram import *

# sys.path.append('../Bleu-master')
# sys.path.append('Bleu-master')
# from calculatebleu import *

sys.path.append('skip-thoughts-master/')
sys.path.append('../skip-thoughts-master/')
import skipthoughts

from nltk.translate.bleu_score import *

from pycorenlp import StanfordCoreNLP
corenlp = StanfordCoreNLP('http://localhost:9000')

sys.path.append('../grammaticality-metrics-master/codalab/scoring_program')
sys.path.append('grammaticality-metrics-master/codalab/scoring_program')
import evaluate
reload(evaluate)
from evaluate import call_lt

skipthoughts_transformer = None
google_word2vec = None

def check_seqs_format(seqs):
    '''functions below expect generated sequences to be a list of lists, i.e. multiple sequences for each context sequence;
    transform to this format if seqs are a flat list'''
    assert(type(seqs) in (list,tuple))
    if type(seqs[0]) not in (list, tuple):
        seqs = [[seq] for seq in seqs]
    return seqs

def get_seq_lengths(gen_seqs):
    gen_seqs = check_seqs_format(gen_seqs)
    lengths = []
    for gen_seqs_ in gen_seqs:
        seq_lengths = numpy.array([len(tokenize(seq)) for seq in gen_seqs_])
        lengths.append(seq_lengths)
    lengths = numpy.array(lengths)
    return {'lengths':lengths, 'mean_length': numpy.mean(lengths)}

def get_bleu_scores(gen_seqs, gold_seqs, verbose=False):
    '''compute bleu scores of generated sequences relative to gold'''

    gen_seqs = check_seqs_format(gen_seqs)
    gold_seqs = check_seqs_format(gold_seqs)
    bleu_scores = []
    for seq_idx, (gen_seqs_, gold_seqs_) in enumerate(zip(gen_seqs, gold_seqs)):
        gold_seqs_ = [tokenize(seq) for seq in gold_seqs_]
        seq_bleu_scores = [sentence_bleu(gold_seqs_, tokenize(seq)) for seq in gen_seqs_]
        if verbose and seq_idx % 500 == 0:
            print "computed blue scores for", seq_idx, "gold sequences..."
        bleu_scores.append(seq_bleu_scores)
    bleu_scores = numpy.array(bleu_scores)
    return {'bleu_scores': bleu_scores, 'mean_bleu_scores': numpy.mean(bleu_scores)}


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

def get_phrases(gen_seq):
    '''given a generated sequence, return all bigrams (phrases) that have entries in google word2vec'''
    global google_word2vec
    if not google_word2vec:
        print "\nloading google word2vec model..."
        google_word2vec = Word2Vec.load("data/google.vectors", mmap='r')
    phrases = []
    gen_seq = tokenize(gen_seq, lowercase=False)
    for idx in xrange(0, len(gen_seq) - 1, 2):
        bigram = gen_seq[idx:idx+2] #phrases in this model are represented as Word1_Word2 (case-sensitive)
        if "_".join(bigram) in google_word2vec:
            phrases.append(" ".join(bigram))
    return phrases

def get_phrase_counts(gen_seqs):
    '''return number of phrases per sequence'''
    gen_seqs = check_seqs_format(gen_seqs)
    n_phrases = []
    for gen_seqs_ in gen_seqs:
        n_phrases_ = [len(get_phrases(gen_seq)) for gen_seq in gen_seqs_]
        n_phrases.append(n_phrases_)
    n_phrases = numpy.array(n_phrases)
    return {'n_phrases':n_phrases, 'mean_n_phrases':numpy.mean(n_phrases)}


# def get_phrases(seqs, min_count=5, threshold=10):
#     #import pdb;pdb.set_trace()
#     bigram_phraser = Phrases((tokenize(seq) for seq in seqs), delimiter=' ', min_count=min_count, threshold=threshold)
#     bigram_phrases = set(list(bigram_phraser.export_phrases((tokenize(seq) for seq in seqs))))
#     trigram_phraser = Phrases(bigram_phraser[(tokenize(seq) for seq in seqs)], delimiter=' ', min_count=min_count, threshold=threshold)
#     trigram_phrases = set(list(trigram_phraser.export_phrases(bigram_phraser[(tokenize(seq) for seq in seqs)])))
#     phrases = bigram_phrases.union(trigram_phrases)
#     phrases = [(phrase, score) for score, phrase in 
#                sorted([(score, phrase) for phrase, score in phrases], reverse=True)]
#     return phrases

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

def get_lexical_sim(context_seqs, gen_seqs, verbose=False):
    '''compute average word2vec and jaccard similarity between all pairs of words between context and generated sequences'''
    assert(len(context_seqs) == len(gen_seqs))
    gen_seqs = check_seqs_format(gen_seqs)
    sim_word2vec_scores = []
    sim_jaccard_scores = []
    for context_seq_idx, (context_seq, seqs) in enumerate(zip(context_seqs, gen_seqs)):
        word2vec_scores = []
        jaccard_scores = []
        for gen_seq in seqs:
            word2vec_score = get_word2vec_sim(context_seq, gen_seq)
            jaccard_score = get_jaccard_sim(context_seq, gen_seq)
            word2vec_scores.append(word2vec_score)
            jaccard_scores.append(jaccard_score)
        sim_word2vec_scores.append(word2vec_scores)
        sim_jaccard_scores.append(jaccard_scores)
        if verbose and context_seq_idx % 500 == 0:
            print "computed lexical similarity for", context_seq_idx, "sequences..."
    sim_word2vec_scores = numpy.array(sim_word2vec_scores)
    sim_jaccard_scores = numpy.array(sim_jaccard_scores)
    return {'word2vec':sim_word2vec_scores, 'jaccard': sim_jaccard_scores,\
            'mean_word2vec':numpy.mean(sim_word2vec_scores), 'mean_jaccard':numpy.mean(sim_jaccard_scores)}

def get_skipthought_similarity(context_seqs, gen_seqs, verbose=False):
    assert(len(context_seqs) == len(gen_seqs))
    gen_seqs = check_seqs_format(gen_seqs)
    global skipthoughts_transformer
    if not skipthoughts_transformer:
        skipthoughts_transformer = SkipthoughtsTransformer(verbose=verbose)
    context_seqs = [segment(seq) for seq in context_seqs]
    #take mean of vectors for all sentences in context
    encoded_context_seqs = numpy.array([numpy.mean(encoded_sents, axis=0) for encoded_sents\
                                        in skipthoughts_transformer.encode(context_seqs)])
    #need to flatten gen_seqs into list of sequences
    n_gen_per_seq = len(gen_seqs[0])
    encoded_gen_seqs = [segment(gen_seq) for gen_seqs_ in gen_seqs for gen_seq in gen_seqs_]
    #take mean of vectors for all sentences in generated sequence (possible that generated sequence could contain more than one sentence)
    encoded_gen_seqs = numpy.array([numpy.mean(gen_seqs_, axis=0) for gen_seqs_\
                                        in skipthoughts_transformer.encode(encoded_gen_seqs)])
    #restructure gen_seqs into list of lists
    encoded_gen_seqs = encoded_gen_seqs.reshape(len(gen_seqs), n_gen_per_seq, -1)

    sim_scores = []
    #calcuate cosine distance between mean context and generated vectors
    for context_seq, gen_seqs_ in zip(encoded_context_seqs, encoded_gen_seqs):
        scores = [(1 - cosine(context_seq, gen_seq)) for gen_seq in gen_seqs_]
        sim_scores.append(scores)
    sim_scores = numpy.array(sim_scores)

    return {'skipthought_scores': sim_scores, 'mean_skipthought_scores': numpy.mean(sim_scores)}

def get_corefs(context_seqs, gen_seqs, verbose=False): # gen_ents=None, 
    assert(len(context_seqs) == len(gen_seqs))
    assert(type(gen_seqs) in (list,tuple) and type(context_seqs) in (list,tuple))

    gen_seqs = check_seqs_format(gen_seqs)

    corefs = []

    for context_seq_idx, (context_seq, gen_seqs_) in enumerate(zip(context_seqs, gen_seqs)):
        n_sents_in_context = len(segment(context_seq))
        gen_corefs = []
        for gen_seq in gen_seqs_:
            try:
                parse = corenlp.annotate((context_seq + " " + gen_seq).encode('utf-8',errors='replace'),\
                                         properties={'annotators': 'coref', 'outputFormat': 'json'})
            except:
                print "error:", context_seq + " " + gen_seq
                #parse = {}
            sents = parse['sentences']
            seq_corefs = []
            for coref_ent_idx, coref_ent in parse['corefs'].items():
                mentions = {'rep_mention':None, 'context_mentions':[], 'gen_mentions':[]}
                for mention in coref_ent:
                    if mention['isRepresentativeMention']:
                        mentions['rep_mention'] = (mention['sentNum'], mention['text'])
                    if mention['sentNum'] > n_sents_in_context: #mention is in generated sequence
                        mentions['gen_mentions'].append((mention['sentNum'], mention['text']))
                    elif mention['sentNum'] <= n_sents_in_context:
                        mentions['context_mentions'].append((mention['sentNum'], mention['text']))
                if mentions['context_mentions']: #only count corefs between context and generated sequence, not corefs only within generated sequence
                    seq_corefs.append(mentions)
            gen_corefs.append(seq_corefs)
        if verbose and context_seq_idx % 500 == 0:
            print "processed coreferences in", context_seq_idx, "sequences..."
        corefs.append(gen_corefs)

    return corefs

def get_coref_counts(context_seqs, gen_seqs):
    assert(len(context_seqs) == len(gen_seqs))
    counts = {'corefs':[], 'prev_mention_sents':[]}

    corefs = get_corefs(context_seqs, gen_seqs)

    for gen_corefs in corefs:
        gen_coref_counts = []
        #gen_ent_counts = []
        gen_prev_mention_sents = []
        for seq_corefs in gen_corefs:
            coref_counts = sum([len(coref['gen_mentions']) for coref in seq_corefs])
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
            gen_prev_mention_sents.append(prev_mentions)
        counts['corefs'].append(gen_coref_counts)
        counts['prev_mention_sents'].append(gen_prev_mention_sents)

    counts['ents'] = get_noun_chunk_complexity(gen_seqs)['n_chunks']
    #counts['ents'] = numpy.array(counts['ents'])
    counts['mean_ents'] = numpy.mean(counts['ents'])
    counts['corefs'] = numpy.array(counts['corefs'])
    counts['mean_corefs'] = numpy.mean(counts['corefs'])
    counts['res_rates'] = numpy.nan_to_num(counts['corefs'] * 1. / counts['ents'])
    counts['mean_res_rates'] = numpy.mean(counts['res_rates'])

    return counts

def get_grammaticality_scores(gen_seqs):
    gen_seqs = check_seqs_format(gen_seqs)
    n_gen_seqs = len(gen_seqs)
    n_gen_per_seq = len(gen_seqs[0])
    gen_seqs = [segment(gen_seq) for gen_seqs_ in gen_seqs for gen_seq in gen_seqs_] #flatten into list of sentences, because grammaticality tool assigns score per single entence
    n_sents_per_seq = numpy.array([len(seq) for seq in gen_seqs]) #keep track of number of sentences
    grammaticality_scores = call_lt([sent.replace("\n", " ") for gen_seq in gen_seqs for sent in gen_seq]) #this tool can't handle line breaks in sentences, so replace them
    #restructure sents into sequences, so score per sequence will be mean of sentence scores
    idxs = [numpy.sum(n_sents_per_seq[:idx]) for idx in xrange(len(n_sents_per_seq))] + [None] #add -1 for last entry
    grammaticality_scores = numpy.array([numpy.mean(grammaticality_scores[idxs[start]:idxs[start+1]]) for start in xrange(len(idxs) - 1)])
    grammaticality_scores = grammaticality_scores.reshape(n_gen_seqs, n_gen_per_seq)
    return {'gram_scores':grammaticality_scores, 'mean_gram_scores':numpy.mean(grammaticality_scores)}

def get_type_token_ratio(gen_seqs, lexicon=None):
    '''if lexicon given, only consider words in lexicon'''
    gen_seqs = check_seqs_format(gen_seqs)
    token_counts = {}
    for seqs in gen_seqs:
        for seq in seqs:
            seq_tokens = tokenize(seq)
            for token in seq_tokens:
                if lexicon is None or token in lexicon:
                    if token in token_counts:
                        token_counts[token] += 1
                    else:
                        token_counts[token] = 1
    n_types = len(token_counts)
    n_tokens = sum(token_counts.values())
    ratio = n_types * 1. / n_tokens
    return {'n_types':n_types, 'n_tokens':n_tokens, 'ratio':ratio}

def get_unique_ngram_ratio(gen_seqs, n=3, lexicon=None):
    '''same as type-token ratio, but for ngrams'''
    gen_seqs = check_seqs_format(gen_seqs)
    ngram_counts = {}
    for seqs in gen_seqs:
        for seq in seqs:
            seq_ngrams = [tuple([word.lower_ for word in ngram])
                            for ngram in extract.ngrams(encoder(seq), n=n, filter_stops=False, filter_punct=False)]
            if lexicon is not None:
                seq_ngrams = [ngram for ngram in seq_ngrams if numpy.all([token in lexicon for token in ngram])] #filter ngrams where at least one word is not in lexicon
            for ngram in seq_ngrams:
                if ngram in ngram_counts:
                    ngram_counts[ngram] += 1
                else:
                    ngram_counts[ngram] = 1
    n_unique = len(ngram_counts)
    n_total = sum(ngram_counts.values())
    ratio = n_unique * 1. / n_total
    return {'n_unique':n_unique, 'n_total':n_total, 'ratio':ratio}

def get_pos_ngram_similarity(context_seqs, gen_seqs, n=3):
    assert(len(context_seqs) == len(gen_seqs))
    gen_seqs = check_seqs_format(gen_seqs)
    pos_sim_scores = []
    for context_seq, gen_seqs_ in zip(context_seqs, gen_seqs):
        scores = []
        context_pos_ngrams = set([tuple([word.pos for word in ngram])
                            for ngram in extract.ngrams(encoder(context_seq), n=n, filter_stops=False, filter_punct=False)])
        for gen_seq in gen_seqs_:
            gen_pos_ngrams = set([tuple([word.pos for word in ngram])
                                for ngram in extract.ngrams(encoder(gen_seq), n=n, filter_stops=False, filter_punct=False)])
            if gen_pos_ngrams:
                common_pos_ngrams = context_pos_ngrams.intersection(gen_pos_ngrams)
                score = len(common_pos_ngrams) * 1. / (len(context_pos_ngrams) + len(gen_pos_ngrams) - len(common_pos_ngrams))
                #score = len([ngram for ngram in gen_pos_ngrams if ngram in set(context_pos_ngrams)]) * 1. / len(gen_pos_ngrams)
            else:
                score = 0.0
            scores.append(score)
        pos_sim_scores.append(scores)
    pos_sim_scores = numpy.array(pos_sim_scores)
    return {'pos_sim_scores': pos_sim_scores, 'mean_pos_sim_scores': numpy.mean(pos_sim_scores)}

def get_noun_chunk_complexity(gen_seqs):
    '''return number and length of noun chunks in each generated sequence'''
    gen_seqs = check_seqs_format(gen_seqs)
    chunk_lengths = []
    n_chunks = []
    seq_lengths = [] #also track sequence length for normalized scores
    for gen_seqs_ in gen_seqs:
        chunk_lengths_ = []
        n_chunks_ = []
        seq_lengths_ = []
        for gen_seq in gen_seqs_:
            gen_seq = encoder(gen_seq)
            seq_lengths_.append(len(gen_seq))
            seq_chunks = [chunk for chunk in gen_seq.noun_chunks]
            n = len(seq_chunks)
            if n:
                mean_chunk_length = numpy.mean([len(chunk) for chunk in seq_chunks])
            else:
                mean_chunk_length = 0 #if no chunks in this sequence, set mean length to 0
            chunk_lengths_.append(mean_chunk_length)
            n_chunks_.append(n)
        n_chunks.append(n_chunks_)
        chunk_lengths.append(chunk_lengths_)
        seq_lengths.append(seq_lengths_)
    n_chunks = numpy.array(n_chunks)
    chunk_lengths = numpy.array(chunk_lengths)
    seq_lengths = numpy.array(seq_lengths)
    norm_n_chunks = n_chunks * 1. / seq_lengths
    norm_chunk_lengths = chunk_lengths * 1. / seq_lengths
    return {'n_chunks':n_chunks, 'chunk_lengths':chunk_lengths, 'norm_n_chunks':norm_n_chunks, 'norm_chunk_lengths':norm_chunk_lengths,\
            'mean_n_chunks':numpy.mean(n_chunks), 'mean_chunk_lengths':numpy.mean(chunk_lengths),\
            'norm_mean_n_chunks':numpy.mean(norm_n_chunks), 'norm_mean_chunk_lengths':numpy.mean(norm_chunk_lengths)}#[~numpy.isnan(chunk_lengths)])}

def get_verb_phrase_complexity(gen_seqs):
    '''return number and length of verb phrases in each generated sequence'''
    gen_seqs = check_seqs_format(gen_seqs)
    phrase_lengths = []
    n_phrases = []
    gen_seq_lengths = [] #also track sequence length for normalized scores
    for gen_seqs_ in gen_seqs:
        mean_lengths = []
        ns = []
        seq_lengths = []
        for gen_seq in gen_seqs_:
            gen_seq = encoder(gen_seq)
            seq_lengths.append(len(gen_seq))
            seq_phrases = [list(word.children) for word in gen_seq if word.pos_ == 'VERB']
            n = len(seq_phrases)
            if n:
                mean_length = numpy.mean([len(phrase) + 1 for phrase in seq_phrases]) #add one for verb itself
            else:
                mean_length = 0 #if no chunks in this sequence, set mean length to 0
            mean_lengths.append(mean_length)
            ns.append(n)
        n_phrases.append(ns)
        phrase_lengths.append(mean_lengths)
        gen_seq_lengths.append(seq_lengths)
    n_phrases = numpy.array(n_phrases)
    phrase_lengths = numpy.array(phrase_lengths)
    seq_lengths = numpy.array(seq_lengths)
    #also compute numbers normalized by sequence length
    norm_n_phrases = n_phrases * 1. / seq_lengths
    norm_phrase_lengths = phrase_lengths * 1. / seq_lengths
    return {'n_phrases': n_phrases, 'phrase_lengths': phrase_lengths, 'norm_n_phrases': norm_n_phrases, 'norm_phrase_lengths': norm_phrase_lengths,\
            'mean_n_phrases': numpy.mean(n_phrases), 'mean_phrase_lengths': numpy.mean(phrase_lengths),\
            'mean_norm_n_phrases': numpy.mean(norm_n_phrases), 'mean_norm_phrase_lengths': numpy.mean(norm_phrase_lengths)}


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
    return {'n_svos':n_svos, 'mean_n_svos':numpy.mean(n_svos)}

def get_frequency_scores(gen_seqs):
    '''use spacy's word frequency stats to get average unigram frequencies across words in a sequence'''
    gen_seqs = check_seqs_format(gen_seqs)
    freq_scores = []
    for gen_seqs_ in gen_seqs:
        scores = []
        for gen_seq in gen_seqs_:
            gen_seq = encoder(gen_seq)
            score = numpy.mean([word.prob for word in gen_seq])
            scores.append(score)
        freq_scores.append(scores)
    freq_scores = numpy.array(freq_scores)
    return {'freq_scores':freq_scores, 'mean_freq_scores':numpy.mean(freq_scores)}

def get_lsm_scores(context_seqs, gen_seqs):
    '''get similarity between context and generated sequences in terms of part-of-speech category distribution (i.e. similarity in frequency distributions of each tag in context and generated sequence)'''
    assert(len(context_seqs) == len(gen_seqs))
    gen_seqs = check_seqs_format(gen_seqs)
    categories = {'nouns':('NOUN','PROPN'), 'adjectives': ('ADJ',), 'adverbs':('ADV',), 'conjunctions':('CONJ','SCONJ'), 'determiners': ('DET',),\
                'prepositions': ('ADP',), 'pronouns':('PRON',), 'punctuation':('PUNCT',), 'verbs': ('VERB',)}#, 'auxillary_verbs':('AUX',)}
    tags_to_cats = {tag:category for category,tags in categories.items() for tag in tags}

    lsm_scores = {cat:[] for cat in categories}
    for context_seq, gen_seqs_ in zip(context_seqs, gen_seqs):
        context_cat_counts = {cat:0 for cat in categories}
        cat_scores = {cat:[] for cat in categories}
        context_seq = encoder(context_seq)
        for word in context_seq:
            if word.pos_ in tags_to_cats:
                context_cat_counts[tags_to_cats[word.pos_]] += 1

        for gen_seq in gen_seqs_:
            gen_seq_cat_counts = {cat:0 for cat in categories}
            gen_seq = encoder(gen_seq)
            for word in gen_seq:
                if word.pos_ in tags_to_cats:
                    gen_seq_cat_counts[tags_to_cats[word.pos_]] += 1
            for category in gen_seq_cat_counts:
                context_prop = context_cat_counts[category] * 1. / len(context_seq)
                gen_seq_prop =  gen_seq_cat_counts[category] * 1. / len(gen_seq)
                cat_score = 1 - (numpy.abs(context_prop - gen_seq_prop) / (context_prop + gen_seq_prop + 1e-6))
                cat_scores[category].append(cat_score)
        for category in lsm_scores:
            lsm_scores[category].append(cat_scores[category])
    lsm_scores = {category:numpy.array(scores) for category,scores in lsm_scores.items()}
    lsm_scores['all'] = numpy.array(lsm_scores.values()).mean(axis=0)
    lsm_means = {category + '_mean':numpy.mean(scores) for category,scores in lsm_scores.items()}
    lsm_scores.update(lsm_means)
    return lsm_scores




            