import sys, os, numpy, pandas

from scipy.spatial.distance import cosine

from textacy import *

from gensim.models import Word2Vec

import models.transformer
reload(models.transformer)
from models.transformer import *

import models.ngram
reload(models.ngram)
from models.ngram import *

from models.narrative_pmi.narrative_dataset import Narrative_Dataset
from models.narrative_pmi.pmi import PMI_Model

# sys.path.append('skip-thoughts-master/')
# sys.path.append('../skip-thoughts-master/')
# import skipthoughts

from nltk.translate.bleu_score import *

from pycorenlp import StanfordCoreNLP
corenlp = StanfordCoreNLP('http://localhost:9000')

from subprocess import Popen, PIPE


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
    '''return the length of each generated sequence in terms of number of words'''
    gen_seqs = check_seqs_format(gen_seqs)
    lengths = []
    for gen_seqs_ in gen_seqs:
        seq_lengths = numpy.array([len(tokenize(seq)) for seq in gen_seqs_])
        lengths.append(seq_lengths)
    lengths = numpy.array(lengths)
    return {'lengths':lengths, 'mean_length': numpy.mean(lengths)}

def get_perplexity(seqs):
    '''return the perplexity of these sequences'''
    return


def get_bleu_scores(gen_seqs, gold_seqs, verbose=False):
    '''compute bleu scores of generated sequences relative to their gold counterparts'''
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


def get_ngrams(gen_seqs, n, lexicon_filepath='data/blog_lexicon.pkl', db_filepath='data/blog_ngrams.db'):
    '''return the ngrams in gen_seqs with counts for often each occurs in another corpus of n-grams (the sqlite database in db_filepath);
    since n-grams as stored as word indices in this db, lexicon_filepath is the pickled dictionary that converts words to indices'''
    gen_seqs = check_seqs_format(gen_seqs)
    gen_ngrams = {}
    #n_total_unique_ngrams = 0
    for seqs in gen_seqs:
        ngrams = extract_ngrams(seqs, n)
        #n_total_unique_ngrams += len(ngrams)
        counts = get_ngram_counts_from_db(ngrams, lexicon_filepath, db_filepath)
        for ngram,count in zip(ngrams, counts):
            if ngram in gen_ngrams:
                gen_ngrams[ngram] += count
            else:
                gen_ngrams[ngram] = count
    return gen_ngrams#, n_total_unique_ngrams

def get_n_ngrams(gen_seqs, n, lexicon_filepath='data/blog_lexicon.pkl', db_filepath='data/blog_ngrams.db'):
    '''return the number of unique ngrams (of length n) in the generated sequences that also appear in another corpus (see above); 
    result includes total number of unique ngrams in generated sequences, total number verified (i.e. have at least one occurence in outside corpus), 
    and the proportion of verified to total'''
    gen_ngrams = get_ngrams(gen_seqs, n, lexicon_filepath, db_filepath)
    n_ngrams = len(gen_ngrams)
    n_verified_ngrams = numpy.sum(numpy.array(gen_ngrams.values()) > 0)
    return {"n_verified": n_verified_ngrams, "n_total": n_ngrams, "ratio": n_verified_ngrams * 1. / n_ngrams}

# def get_n_ngrams(gen_seqs, n):
#     '''return the total number of ngrams and total number of unique ngrams in the generated sequences'''
#     gen_seqs = check_seqs_format(gen_seqs)
#     unique_ngrams = set()
#     n_ngrams = 0
#     for seqs in gen_seqs:
#         seq_ngrams = extract_ngrams(seqs, n)
#         n_ngrams += len(seq_ngrams)
#         unique_ngrams.update(seq_ngrams)
#     return {"total":n_ngrams, "unique":len(unique_ngrams)}

def get_phrases(gen_seq):
    '''given a generated sequence, return all bigrams (phrases) that have entries in google word2vec'''
    global google_word2vec
    if not google_word2vec:
        print "\nloading google word2vec model..."
        google_word2vec = Word2Vec.load("data/google.vectors", mmap='r')
    phrases = []
    bigrams = []
    gen_seq = tokenize(gen_seq, lowercase=False)
    for idx in xrange(0, len(gen_seq) - 1, 2):
        bigram = gen_seq[idx:idx+2] #phrases in this model are represented as Word1_Word2 (case-sensitive)
        bigrams.append(" ".join(bigram))
        if "_".join(bigram) in google_word2vec:
            phrases.append(" ".join(bigram))
    return bigrams, phrases

def get_phrase_counts(gen_seqs):
    '''return number of total unique two-word phrases and phrases per generated sequence (phrase rate, where phrases are bigrams that have word2vec representations; see above)'''
    gen_seqs = check_seqs_format(gen_seqs)
    gen_bigrams = {}
    gen_phrases = {}
    gen_n_phrases = []
    for gen_seqs_ in gen_seqs:
        n_phrases = []
        for gen_seq in gen_seqs_:
            bigrams, phrases = get_phrases(gen_seq)
            for bigram in bigrams:
                if bigram in gen_bigrams:
                    gen_bigrams[bigram] += 1
                else:
                    gen_bigrams[bigram] = 1
            for phrase in phrases:
                if phrase in gen_phrases:
                    gen_phrases[phrase] += 1
                else:
                    gen_phrases[phrase] = 1
            n_phrases.append(len(phrases))
        gen_n_phrases.append(n_phrases)
    gen_n_phrases = numpy.array(gen_n_phrases)
    return {'phrase_rates':gen_n_phrases, 'mean_phrase_rates':numpy.mean(gen_n_phrases), 'n_bigrams': len(gen_bigrams), 
            'n_phrases': len(gen_phrases), 'phrase_bigram_ratio': len(gen_phrases) * 1. / len(gen_bigrams)}


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
    '''get all word pairs between each context sequence and corresponding generated sequence'''
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
    '''return the jaccard similarity between a context and generated sequence, optionally filtering by words with specific part-of-speech tags'''
    context_words = set([word.string.lower().strip() for word in extract.words(encoder(context_seq), include_pos=include_pos)])
    gen_seq_words = set([word.string.lower().strip() for word in extract.words(encoder(gen_seq), include_pos=include_pos)])
    if not len(context_words) and not len(gen_seq_words):
        jaccard_sim = 0
    else:
        common_words = context_words.intersection(gen_seq_words)
        jaccard_sim = len(common_words) * 1. / (len(context_words) + len(gen_seq_words) - len(common_words))
    return jaccard_sim

def get_word2vec_sim(context_seq, gen_seq):
    '''return the word2vec cosine similarity between the context and each generated sequence 
    (where the word2vec representation for a sequence is just the average of its word vectors)'''
    word_pairs = get_word_pairs(context_seq, gen_seq)
    if word_pairs:
        pair_scores = [similarity.word2vec(encoder(word1),encoder(word2)) for word1,word2 in word_pairs]
    else: #no word pairs between context and generated sequences (e.g. generated sequence might be punctuation only)
        pair_scores = [0]
    # assert(len(word_pairs) == len(pair_scores))
    word2vec_sim = numpy.mean(pair_scores)
    return word2vec_sim

def get_lexical_sim(context_seqs, gen_seqs, verbose=False):
    '''compute average word2vec and jaccard similarity between all pairs of words between each context and corresponding generated sequence'''
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
    '''return the cosine similarity between the mean of the skipthought (sentence) vectors for each context and corresponding generated sequence'''
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

def get_corefs(context_seqs, gen_seqs, verbose=False):
    '''return all the entities in each generated sequence that co-ref to an entity in the corresponding context'''
    assert(len(context_seqs) == len(gen_seqs))
    assert(type(gen_seqs) in (list,tuple) and type(context_seqs) in (list,tuple))

    gen_seqs = check_seqs_format(gen_seqs)

    corefs = []

    for context_seq_idx, (context_seq, gen_seqs_) in enumerate(zip(context_seqs, gen_seqs)):
        n_sents_in_context = len(segment(context_seq))
        gen_corefs = []
        for gen_seq in gen_seqs_:
            seq_corefs = []
            try:
                parse = corenlp.annotate((context_seq + " " + gen_seq).encode('utf-8',errors='replace'),\
                                         properties={'annotators': 'coref', 'outputFormat': 'json'})
            except:
                print "error:", context_seq + " " + gen_seq
                parse = None
            if type(parse) is dict:
                #sents = parse['sentences']
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
    '''return 1) the number of entities (noun chunks) in each generated sequence, 2) the number of entities in each generated sequence that co-refer to entities in its context,
    and 3) the proportion of entities in each generated sequence that co-refer to entities in the corresponding context'''
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
    counts['ents'] = numpy.maximum(counts['ents'], counts['corefs']) #don't let number of entities exceed the number of coreferences
    counts['mean_corefs'] = numpy.mean(counts['corefs'])
    counts['res_rates'] = numpy.nan_to_num(counts['corefs'] * 1. / counts['ents'])
    counts['mean_res_rates'] = numpy.mean(counts['res_rates'])

    return counts

def get_grammaticality_scores(gen_seqs):
    '''return grammaticality scores for generated sequences, using Language Tool to compute scores'''
    gen_seqs = check_seqs_format(gen_seqs)
    n_gen_seqs = len(gen_seqs)
    n_gen_per_seq = len(gen_seqs[0])
    gen_seqs = [segment(gen_seq) for gen_seqs_ in gen_seqs for gen_seq in gen_seqs_] #flatten into list of sentences, because grammaticality tool assigns score per single entence
    n_sents_per_seq = numpy.array([len(seq) for seq in gen_seqs]) #keep track of number of sentences
    #scorer = grammar_check.LanguageTool('en-US')
    gram_scores = _get_LT_scores([sent.replace("\n", " ") for gen_seq in gen_seqs for sent in gen_seq]) #this tool can't handle line breaks in sentences, so replace them
    #get total number of errors in each sentence, then divide errors by sentence length; subtract from 1, thus score of 1 indicates no errors
    #gram_scores = numpy.array([1 - len(scorer.check(sent.replace("\n", " "))) * 1. / len(tokenize(sent)) for gen_seq in gen_seqs for sent in gen_seq])
    #restructure sents into sequences, so score per sequence will be mean of sentence scores
    idxs = [numpy.sum(n_sents_per_seq[:idx]) for idx in xrange(len(n_sents_per_seq))] + [None] #add -1 for last entry
    gram_scores = numpy.array([numpy.mean(gram_scores[idxs[start]:idxs[start+1]]) for start in xrange(len(idxs) - 1)])
    gram_scores = gram_scores.reshape(n_gen_seqs, n_gen_per_seq)
    return {'gram_scores':gram_scores, 'mean_gram_scores':numpy.mean(gram_scores)}


def _get_LT_scores(sentences, debug=False):
    '''counts errors with an external call to LanguageTool; this code was borrowed from
    https://github.com/cnap/grammaticality-metrics/blob/master/codalab/scoring_program/evaluate.py'''
    # sys.stderr.write('Running LanguageTool...\n')
    if debug:
        sys.stderr.write('Java info: %s %s\n' %
                         (os.system('which java'), os.system('java -version')))
    process = Popen(['java', '-Dfile.encoding=utf-8',
                     '-jar', os.path.join(os.getcwd(), 'LanguageTool-3.1/languagetool-commandline.jar'),
                     '-d', 'COMMA_PARENTHESIS_WHITESPACE,WHITESPACE_RULE,' +
                     'EN_UNPAIRED_BRACKETS,EN_QUOTES',
                     '-b', '-l', 'en-US', '-c', 'utf-8'],
                    stdin=PIPE, stdout=PIPE, stderr=PIPE)
    ret = process.communicate(input=('\n'.join(sentences)).encode('utf-8'))
    if debug:
        sys.stderr.write('LT out: %s\n' % str(ret))
    error_counts = [0] * len(sentences)
    for l in ret[0].split('\n'):
        if 'Rule ID' in l:
            ll = l.split()
            ind = (int(ll[2][:-1]) - 1)
            error_counts[ind] += 1
    _token_counts = [len(s.split()) for s in sentences]
    token_counts = numpy.array([num_toks if num_toks > 0 else 1 for num_toks in _token_counts],
                            dtype=float)
    error_counts = numpy.array(error_counts, dtype=float)

    return 1 - numpy.divide(error_counts, token_counts)

def get_type_token_ratio(gen_seqs, lexicon=None):
    '''return proportion of unique words to total word occurences across generated sequences; if lexicon given, only consider words in lexicon'''
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
    '''return jaccard similarity between n-grams in each context sequence and corresponding generated sequence'''
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
    '''use spacy's word frequency stats to get average unigram frequencies across words in each sequence'''
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
    '''get similarity between context sequences and generated sequences in terms of part-of-speech category distribution
    (i.e. similarity in frequency distributions of each tag in context and generated sequence)'''
    assert(len(context_seqs) == len(gen_seqs))
    gen_seqs = check_seqs_format(gen_seqs)
    categories = {'nouns':('NOUN','PROPN'), 'adjectives': ('ADJ',), 'adverbs':('ADV',), 'conjunctions':('CONJ','SCONJ'), 'determiners': ('DET',),\
                'prepositions': ('ADP',), 'pronouns':('PRON',), 'punctuation':('PUNCT',), 'verbs': ('VERB',)}
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

def get_sentiment_sim(context_seqs, gen_seqs):
    '''return the cosine similarity between the sentiment scores of each context and corresponding generated sequence;
    the sentiment scores are given in spacy'''
    gen_seqs = check_seqs_format(gen_seqs)
    emotion_types = ['AFRAID', 'AMUSED', 'ANGRY', 'ANNOYED', 'DONT_CARE', 'HAPPY', 'INSPIRED', 'SAD']
    gen_sentiment_sim_scores = []
    for context_seq, gen_seqs_ in zip(context_seqs, gen_seqs):
        context_sentiment = lexicon_methods.emotional_valence(encoder(context_seq))
        context_sentiment = numpy.array([context_sentiment[emotion_type] for emotion_type in emotion_types]) + 1e-8 #add tiny number to avoid NaN when all scores are 0
        sentiment_sim_scores = []
        for gen_seq in gen_seqs_:
            gen_sentiment = lexicon_methods.emotional_valence(encoder(gen_seq))
            gen_sentiment = numpy.array([gen_sentiment[emotion_type] for emotion_type in emotion_types]) + 1e-8 #add tiny number to avoid NaN when all scores are 0
            sentiment_sim = 1 - cosine(context_sentiment, gen_sentiment)
            sentiment_sim_scores.append(sentiment_sim)
        gen_sentiment_sim_scores.append(sentiment_sim_scores)

    gen_sentiment_sim_scores = numpy.array(gen_sentiment_sim_scores)
    return {'sentiment_sim_scores': gen_sentiment_sim_scores, 'mean_sentiment_sim_scores': numpy.mean(gen_sentiment_sim_scores)}

def get_pmi_scores(context_seqs, gen_seqs, model_filepath='data/narrative_dataset_1million'):
    dataset = Narrative_Dataset(model_filepath)
    model = PMI_Model(dataset)
    pmi_scores = []
    for context_seq, gen_seqs_ in zip(context_seqs, gen_seqs):
        encoded_context_seq = dataset.encode_sequence(context_seq)
        scores = []
        for gen_seq in gen_seqs_:
            encoded_gen_seq = dataset.encode_sequence(gen_seq)
            score = model.score(sequence1=encoded_context_seq, sequence2=encoded_gen_seq)
            scores.append(score)
        pmi_scores.append(scores)
    pmi_scores = numpy.array(pmi_scores)
    return {'pmi_scores': pmi_scores, 'mean_pmi_scores': numpy.mean(pmi_scores)}




            