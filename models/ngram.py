import sqlite3, numpy, pickle

from transformer import *

rng = numpy.random.RandomState(123)

def save_ngrams(ngram_counts, n, filepath):

    db = sqlite3.connect(filepath)
    cursor = db.cursor()

    #need to create bigram counts db if it hasn't been created
    cursor.execute("CREATE TABLE IF NOT EXISTS ngram(\
                    word1 INTEGER,\
                    word2 INTEGER,\
                    word3 INTEGER,\
                    word4 INTEGER,\
                    word5 INTEGER,\
                    count INTEGER DEFAULT 0,\
                    PRIMARY KEY (word1, word2, word3, word4, word5))")

    #create an index on count and words
    cursor.execute("CREATE INDEX IF NOT EXISTS count_index ON ngram(count)")
    cursor.execute("CREATE INDEX IF NOT EXISTS word1_index ON ngram(word1)")
    cursor.execute("CREATE INDEX IF NOT EXISTS word2_index ON ngram(word2)")
    cursor.execute("CREATE INDEX IF NOT EXISTS word3_index ON ngram(word3)")
    cursor.execute("CREATE INDEX IF NOT EXISTS word4_index ON ngram(word4)")
    cursor.execute("CREATE INDEX IF NOT EXISTS word5_index ON ngram(word5)")
    
    #insert current counts into db
    ngrams, counts = zip(*ngram_counts.items())
    ngrams = [(ngram + ((-1,) * (5 - n))) for ngram in ngrams]
    #insert words if they don't already exist
    cursor.executemany("INSERT OR IGNORE INTO ngram(word1, word2, word3, word4, word5)\
                        VALUES (?, ?, ?, ?, ?)", ngrams)
    #now update counts
    cursor.executemany("UPDATE ngram\
                        SET count = (count + ?)\
                        WHERE word1 = ? AND word2 = ? AND word3 = ? AND word4 = ? AND word5 = ?", 
                       [(count,) + (ngram) for ngram, count in zip(ngrams, counts)])

    #commit insert
    db.commit()

    #close connection
    db.close()

def lookup_ngram_count(filepath, ngram):
    '''specify an ngram prefix to get all counts of n+1 grams that include that prefix'''

    db = sqlite3.connect(filepath)
    cursor = db.cursor()

    #queries will match all ngrams that start with the given n-gram (if length of ngrams in db is greater than length of given ngram), so multiple rows may be returned
    if len(ngram) == 1:
        cursor.execute("SELECT * FROM ngram WHERE word1 = ?", ngram)
    elif len(ngram) == 2:
        cursor.execute("SELECT * FROM ngram WHERE word1 = ? AND word2 = ?", ngram)
    elif len(ngram) == 3:
        cursor.execute("SELECT * FROM ngram WHERE word1 = ? AND word2 = ? AND word3 = ?", ngram)
    elif len(ngram) == 4:
        cursor.execute("SELECT * FROM ngram WHERE word1 = ? AND word2 = ? AND word3 = ? AND word4 = ?", ngram)
    elif len(ngram) == 5:
        cursor.execute("SELECT * FROM ngram WHERE word1 = ? AND word2 = ? AND word3 = ? AND word4 = ? AND word5 = ?", ngram)

    count = numpy.sum(numpy.array([ngram[-1] for ngram in cursor.fetchall()]))
    
    cursor.close()
    db.close()

    return count

def lookup_counts_for_n(filepath, n):
    '''get all ngrams of length n'''

    db = sqlite3.connect(filepath)
    cursor = db.cursor()

    if n == 1:
        cursor.execute("SELECT * FROM ngram WHERE word1 > -1 AND word2 = -1 AND word3 = -1\
                        AND word4 = -1 AND word5 = -1")
    elif n == 2:
        cursor.execute("SELECT * FROM ngram WHERE word1 > -1 AND word2 > -1 AND word3 = -1\
                        AND word4 = -1 AND word5 = -1")
    elif n == 3:
        cursor.execute("SELECT * FROM ngram WHERE word1 > -1 AND word2 > -1 AND word3 > -1\
                        AND word4 = -1 AND word5 = -1")
    elif n == 4:
        cursor.execute("SELECT * FROM ngram WHERE word1 > -1 AND word2 > -1 AND word3 > -1\
                        AND word4 > -1 AND word5 = -1")
    elif n == 5:
        cursor.execute("SELECT * FROM ngram WHERE word1 > -1 AND word2 > -1 AND word3 > -1\
                        AND word4 > -1 AND word5 > -1")
        
    ngram_counts = cursor.fetchall()
    cursor.close()
    db.close()
    
    ngrams = [ngram[:n] for ngram in ngram_counts]
    counts = numpy.array([ngram[-1] for ngram in ngram_counts])
    assert(len(ngrams) == len(counts))
    return ngrams, counts

def add_ngrams_to_model(transformer, seqs, n_min, n_max, filepath):

    for n_idx in range(n_min, n_max+1):
        ngram_counts = {}
        seqs, _ = transformer.transform(seqs)
        for seq in seqs:
            seq_ngrams = [tuple(seq[idx:idx+n_idx]) for idx in range(len(seq) - (n_idx - 1))]
            for ngram in seq_ngrams:
                if ngram not in ngram_counts:
                    ngram_counts[ngram] = 0
                ngram_counts[ngram] += 1
        
        save_ngrams(ngram_counts, n=n_idx, filepath=filepath)

def extract_ngrams(seqs, n):
    '''return all ngrams of length n in these sequences'''
    ngrams = {}
    if type(seqs[0]) in (list,tuple):
        seqs = [seq for seqs_ in seqs for seq in seqs_]
    #seqs, _ = transformer.transform(seqs)
    for seq in seqs:
        seq = tokenize(seq)
        seq_ngrams = [tuple(seq[idx:idx+n]) for idx in range(len(seq) - (n - 1))]
        for ngram in seq_ngrams:
            if ngram not in ngrams:
                ngrams[ngram] = 1
            else:
                ngrams[ngram] += 1
    return ngrams

def get_ngram_counts_from_db(ngrams, lexicon_filepath, db_filepath):
    '''this function takes text ngrams as input, converts them to word indices according to the given lexicon, and then looks up their count in the given db'''
    with open(lexicon_filepath, 'rb') as f:
        lexicon = pickle.load(f)
    counts = []
    for ngram in ngrams:
        ngram = tuple([lexicon[word] if word in lexicon else 1 for word in ngram])
        count = lookup_ngram_count(db_filepath, ngram=ngram)
        counts.append(count)
    counts = numpy.array(counts)
    return counts
            

def gen_ngram_sents(transformer, seqs, n, filepath, eos_tokens=[".", "!", "?"], cap_tokens=[]):
    
    unigram_counts, unigrams = get_ngram_counts(filepath + '.db', n=1)
    unigram_count_sum = unigram_counts.sum() #backoff count will always be the same for unigrams
    unigram_probs = unigram_counts.astype('float32') / unigram_count_sum

    pred_sents = []
    
    seqs, _ = transformer.transform(seqs)
    
    for seq_idx, seq in enumerate(seqs):
        if seq_idx % 1000 == 0:
            print seq_idx
        if n == 1:
            pred_sent = []
        else: 
            pred_sent = seq[-(n-1):] #predict based on last ngram from previous sentence
        pred_token = 0
        while transformer.lexicon_lookup[pred_token] not in eos_tokens\
                                                and len(pred_sent) < 25:
            for n_idx in range(n, 0, -1):
                if n_idx == 1:
                    ngrams = unigrams
                    ngram_probs = unigram_probs
                else:
                    counts, ngrams = get_ngram_counts(filepath + '.db', ngram_prefix=tuple(pred_sent[-n_idx+1:]))
                    if counts is not None:
                        ngram_probs = counts.astype('float32') / counts.sum()
                        break
                
            pred_idx = rng.choice(a=len(ngram_probs), p=ngram_probs)
            pred_token = ngrams[pred_idx][-1]
            #decode token
#             if pred_token > 0:
            pred_sent.append(pred_token)
        #decode indices into strings, don't include last token from context sequence in generated sentence
        #print "decoding generated sentences..."
        #gen_seqs = [self.transformer.decode_seqs(seq, **decode_params) for seq in gen_seqs]
        pred_sent = transformer.decode_seqs(pred_sent, eos_tokens, cap_tokens)
        # pred_sent = " ".join([transformer.lexicon_lookup[pred_token] 
        #                       for pred_token in pred_sent[n-1:] if transformer.lexicon_lookup[pred_token]])
        pred_sents.append(pred_sent)
    
    print "generated", len(pred_sents), "sentences with n-gram model ( n =", n, ")"
    
    return pred_sents

def get_top_ngrams(transformer, n, filepath, top=50):
    '''show top most frequent ngrams'''
#     import pdb;pdb.set_trace()
    counts, ngrams = get_ngram_counts(filepath, n=n)
    sorted_idxs = numpy.argsort(counts)
    ngrams = [ngrams[idx] for idx in sorted_idxs]
    counts = counts[sorted_idxs]
        
    #decode ngrams into strings
    # if len(ngram_probs.shape) == 1: #unigrams
    #     top_ngrams = [transformer.lexicon_lookup[unigram] for unigram in top_ngrams]
    # else:
    ngrams = [tuple([transformer.lexicon_lookup[token] for token in ngram]) for ngram in ngrams]

    top_ngrams = zip(ngrams, counts)[-top:]
    
    return top_ngrams

def get_perplexity(transformer, seqs, n, filepath):
    '''compute perplecity of ngram of model on give dataset'''

    seqs, _ = transformer.transform(seqs)

    ngram_probs = []
    for n_idx in range(n): #compute probs to have on hand
        counts, ngrams = get_ngram_counts(filepath + '.db', n=n_idx+1)
        count_sum = counts.sum()
        probs = counts.astype('float32') / count_sum
        probs = dict(zip(ngrams, probs))
        ngram_probs.append(probs)
    
    #for seq_idx, seq in enumerate(seqs):
    seq_probs = []
    #seq_probs2 = []
    for seq in seqs:
        #probs_by_seq = []
        ngrams = [tuple(seq[idx:idx+n]) for idx in range(len(seq) - (n - 1))]
        for ngram in ngrams:
            for n_idx in range(n, 0, -1):
                if ngram[:n_idx] in ngram_probs[n_idx-1]:
                    ngram_prob = ngram_probs[n_idx-1][ngram[:n_idx]]
                    seq_probs.append(ngram_prob)
                    #probs_by_seq.append(ngram_prob)
                    break
        #seq_probs2.append(numpy.exp(-numpy.mean(numpy.log(probs_by_seq))))

    #perplexity = numpy.exp(-numpy.mean(numpy.log(seq_probs)))
    seq_probs = numpy.array(seq_probs)
    perplexity = numpy.exp(-numpy.mean(numpy.log(seq_probs)))
    #perplexity2 = numpy.mean(seq_probs2)

    return perplexity#, perplexity2



