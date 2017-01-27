
# coding: utf-8

# In[1]:

import sys, os, warnings
sys.path.append('../')

import models.transformer
reload(models.transformer)
from models.transformer import load_transformer

import models.classifier
reload(models.classifier)
from models.classifier import *

import roc
reload(roc)
from roc import *

import roc_lm
reload(roc_lm)
from roc_lm import *

import models.similarity
reload(models.similarity)
from models.similarity import SimilarityIndex

sys.path.append('skip-thoughts-master')
import skipthoughts
reload(skipthoughts)
from training import vocab
from training import train as encoder_train
from training import tools as encoder_tools
reload(encoder_tools)

warnings.filterwarnings('ignore', category=Warning)


# In[35]:

def generate_sim_sents(seq_index, input_seqs, fin_sents, n_neg_per_seq):
    #flatten input seqs
    input_seqs = [" ".join(seq) for seq in input_seqs]
    batch_size = 1000
    sim_sents = []
    for batch_idx in range(0, len(input_seqs), batch_size):
        sim_idxs, _ = seq_index.get_similar_seqs(input_seqs[batch_idx: batch_idx + batch_size], 
                                                    n_best=n_neg_per_seq + 1)
        #exclude most similar seq, which will be same seq itself
        sim_idxs = sim_idxs[:, :-1]
        sents = [[fin_sents[idx] for idx in idxs] for idxs in sim_idxs]
        sim_sents.extend(sents)
        if batch_idx > 0 and batch_idx % 5000 == 0:
            print "paired negative sentences with {}/{} sequences...".format(batch_idx, len(input_seqs))
    print "generated", n_neg_per_seq, "similarity negative examples for each of", len(input_seqs), "positive examples"
    return sim_sents

def generate_lm_sents(lm, input_seqs, temp, n_neg_per_seq):
#     assert(n_neg_per_seq % len(gen_temps) == 0)
#     n_neg_per_temp = n_neg_per_seq / len(gen_temps)
#     neg_seqs = []
    input_idxs = numpy.arange(len(input_seqs)).repeat(n_neg_per_seq)
    gen_seqs, p_gen_seqs = generate_sents(lm, [input_seqs[idx] for idx in input_idxs], batch_size=1000, 
                                          n_best=1, mode='random', temp=temp)#, decode=decode)
    gen_seqs = [gen_seqs[idx:idx + n_neg_per_seq] 
                for idx in range(0, len(input_seqs) * n_neg_per_seq, n_neg_per_seq)]
#     p_gen_seqs = numpy.array([p_gen_seqs[idx:idx + n_neg_per_seq] 
#                 for idx in range(0, len(input_seqs) * n_neg_per_seq, n_neg_per_seq)])
#     p_gen_seqs = p_gen_seqs * (1.0 - temp)
#     for temp in gen_temps:
# #         seqs = generate_lm_sents(lm, input_seqs, temp, n_neg_per_temp, verbose=False)
# #         neg_seqs.extend(seqs)
#         input_idxs = numpy.arange(len(input_seqs)).repeat(n_neg_per_temp)
#         gen_seqs, p_gen_seqs = generate_sents(lm, [input_seqs[idx] for idx in input_idxs], batch_size=1000, 
#                                               n_best=1, mode='random', temp=temp)#, decode=decode)
#         gen_seqs = [gen_seqs[idx:idx + n_neg_per_seq] 
#                     for idx in range(0, len(input_seqs) * n_neg_per_temp, n_neg_per_temp)]
#         p_gen_seqs = numpy.array([p_gen_seqs[idx:idx + n_neg_per_seq] 
#                     for idx in range(0, len(input_seqs) * n_neg_per_temp, n_neg_per_temp)])
#         p_gen_seqs = p_gen_seqs * (1.0 - temp)
    print "generated", n_neg_per_seq, "lm negative examples for each of", len(input_seqs), "positive examples"
    return gen_seqs

def generate_rand_sents(output_seqs, n_neg_per_seq):#, n_rand_seqs):
    #import pdb;pdb.set_trace()
    n_rand_seqs = len(output_seqs) * n_neg_per_seq
    #pair input sequences with randomly picked final sentences from other input sequences
    rand_output_idxs = rng.randint(low=0, high=len(output_seqs), 
                                   size=n_rand_seqs)
    input_idxs = numpy.arange(len(output_seqs)).repeat(n_neg_per_seq)
    while numpy.any(input_idxs == rand_output_idxs):
        #ensure that outputs are not paired with their correct inputs
        matched_idxs = numpy.where(input_idxs == rand_output_idxs)
        rand_output_idxs[matched_idxs] = rng.randint(low=0, high=len(output_seqs), size=len(matched_idxs))
    rand_output_idxs = rand_output_idxs.reshape((len(output_seqs), n_neg_per_seq))
    output_seqs = [[output_seqs[idx] for idx in idxs] for idxs in rand_output_idxs]
    print "generated", n_neg_per_seq, "random negative examples for each of", len(output_seqs), "positive examples"
    return output_seqs

def generate_bkwrd_sents(input_seqs, output_seqs, n_neg_per_seq): #specify # of instances to generate
    #import pdb;pdb.set_trace()
    n_rand_seqs = len(output_seqs) * n_neg_per_seq
    rand_sent_idxs = numpy.array([rng.choice(len(input_seqs[0]), size=n_neg_per_seq, replace=False)
                      for idx in range(len(input_seqs))])
    output_seqs =  [[seqs[idx] for idx in sent_idxs] for seqs, sent_idxs in zip(input_seqs, rand_sent_idxs)]
    assert(len(input_seqs) == len(output_seqs))
    print "generated", n_neg_per_seq, "backward examples for each of", len(input_seqs), "positive examples"
    return output_seqs

# def get_rand_neg_seqs(input_seqs, output_seqs, n_neg_per_seq):#, n_bkwrd_per_seq):
#     '''two different ways to generate negative instances: either pair an input (ie first four sentences)
#     with a random final sentence from another story (rand seqs), or select one of first four sentences as 
#     ending (repeated seqs)'''
#     #import pdb;pdb.set_trace()
# #     n_neg_per_seq = n_rand_per_seq + n_bkwrd_per_seq
#     #rand_y_seqs = [[]] * len(input_seqs)
# #     bkwrd_y_seqs = [[]] * len(input_seqs)
# #     if n_rand_per_seq:
#     rand_y_seqs = generate_rand_sents(output_seqs, n_rand_per_seq)
# #     if n_bkwrd_per_seq:
# #         bkwrd_y_seqs = generate_bkwrd_sents(input_seqs, output_seqs, n_bkwrd_per_seq)
#     #neg_y_seqs = [rand_seqs + bkwrd_seqs for rand_seqs, bkwrd_seqs in zip(rand_y_seqs, bkwrd_y_seqs)]
#     print "generated", n_neg_per_seq, " negative examples for each of", len(input_seqs), "positive examples"
#     return neg_y_seqs

# def get_lm_neg_seqs(lm, input_seqs, n_neg_per_seq, gen_temps):
#     #import pdb;pdb.set_trace()
#     #negative sequences generated by language model

#     return neg_seqs

# def get_sim_neg_seqs(seq_index, input_seqs, fin_sents, n_neg_per_seq, seqs_per_chunk):
#     #import pdb;pdb.set_trace()
#     neg_seqs = generate_sim_sents(seq_index, input_seqs, fin_sents, n_neg_per_seq)
#     print "generated", n_neg_per_seq, "negative examples for each of", len(input_seqs), "positive examples"
#     #save neg seqs to file to avoid loading all in memory
#     #num chunks is equal to number of negative examples per positive
#     return neg_seqs

def get_trunc_pos_seqs(input_seqs, output_seqs, model_filepath):
    #generate additional positive sequences by truncating stories (e.g. use first X<4 sentences to predict next)
    input_len = len(input_seqs[0])
    input_filepath = model_filepath + '/input_trunc_pos_seqs.npy'
    output_filepath = model_filepath + '/output_trunc_pos_seqs.npy'
    if os.path.exists(input_filepath) and os.path.exists(output_filepath):
        #import pdb;pdb.set_trace()
        trunc_input_seqs = numpy.memmap(input_filepath, mode='r',
                                        shape=((len(input_seqs) * (input_len - 1), input_len,
                                                input_seqs[0].shape[-1])))
        trunc_output_seqs = numpy.memmap(output_filepath, mode='r',
                                         shape=((len(input_seqs) * (input_len - 1), input_seqs[0].shape[-1])))
    else:
        trunc_input_seqs = numpy.memmap(input_filepath, dtype='float64', mode='w+',
                                shape=((len(input_seqs) * (input_len - 1), input_len,
                                        input_seqs[0].shape[-1])))
        trunc_output_seqs = numpy.memmap(output_filepath, dtype='float64', mode='w+',
                                         shape=((len(input_seqs) * (input_len - 1), input_seqs[0].shape[-1])))
        for trunc_len in range(1, input_len):
            #import pdb;pdb.set_trace()
            trunc_input_seqs[len(input_seqs) * (trunc_len - 1):len(input_seqs) * trunc_len, 
                                -trunc_len:] = numpy.array([seq[:trunc_len] for seq in input_seqs])
            trunc_output_seqs[len(input_seqs) * (trunc_len - 1)                                 :len(input_seqs) * trunc_len] = numpy.array([seq[trunc_len] for seq in input_seqs])
    #import pdb;pdb.set_trace()
    return trunc_input_seqs, trunc_output_seqs

def get_neg_seqs(input_seqs, output_seqs, modes, n_neg_per_mode, gen_temp=0.4, max_neg_per_mode=6):
    #max of 6 negative samples per mode (or 4 for backward)
    #import pdb;pdb.set_trace()
    neg_seqs = []
    for mode, n_neg_per_seq in zip(modes, n_neg_per_mode):
        csv_filepath = mode + str(len(input_seqs)) + '_neg' + (str(max_neg_per_mode) if mode != 'backward' else '4') + '.csv'
        assert(n_neg_per_seq <= max_neg_per_mode)
        if os.path.exists(csv_filepath):
            seqs = pandas.read_csv(csv_filepath, encoding='utf-8', header=None).values.tolist()
        else:
            if mode == 'similarity':
                seq_index = SimilarityIndex(filepath='roc_97027.index')
                #there's an assumption here that all output seqs are in the similarity index
                seqs = generate_sim_sents(seq_index, input_seqs, output_seqs, n_neg_per_seq=max_neg_per_mode)
            elif mode == 'lm':
                lm = load_lm('roc_lm97027_batchtrained_clio_updated')
                seqs = generate_lm_sents(lm, input_seqs, gen_temp, n_neg_per_seq=max_neg_per_mode)
            elif mode == 'random':
                seqs = generate_rand_sents(output_seqs, n_neg_per_seq=max_neg_per_mode)
            elif mode == 'backward':
                seqs = generate_bkwrd_sents(input_seqs, output_seqs, n_neg_per_seq=4)
            pandas.DataFrame(seqs).to_csv(csv_filepath, header=False, index=False, encoding='utf-8')
        seqs = [sents[:n_neg_per_seq] for sents in seqs]
        neg_seqs.append(seqs)
        
    if len(neg_seqs) == 1:
        neg_seqs = neg_seqs[0]
    else:
        neg_seqs = [[sent for mode in seqs for sent in mode] for seqs in zip(*neg_seqs)]
    return neg_seqs
    


# In[3]:

if __name__ == "__main__":
    train_input_seqs, train_output_seqs,     val_input_seqs, val_output_choices, val_output_gold = load_train_val_data()


# In[120]:

# if __name__ == "__main__":
#     test_input_seqs,  test_output_choices, test_output_gold = get_cloze_data(filepath='cloze_test_ALL_test.tsv', 
#                                                                          mode='concat')
#     use_skipthoughts = False
#     filepath = 'roc_seqbinary_baseline97027_neg6_roc97027_embeddings'
#     if use_skipthoughts:
#         transformer = SequenceTransformer(min_freq=1, verbose=1, word_embeddings=None)
#     else:
#         embeddings = similarity_score.load_model('roc97027_embeddings') #'AvMaxSim/vectors'
#         transformer = load_transformer(filepath, embeddings)
#         transformer.n_embedding_nodes = embeddings.vector_size
#         transformer.sent_encoder = None
#     classifier = load_classifier(filepath)
#     seq_binary = RNNPipeline(steps=[("transformer", transformer), ("classifier", classifier)])
#     #import pdb;pdb.set_trace()
#     #prob_y, pred_y, accuracy = evaluate(val_input_seqs, val_output_choices, val_output_gold)
#     prob_y, pred_y, accuracy = evaluate(test_input_seqs, test_output_choices, test_output_gold) #encoder_dim=
# #     show_predictions(X=test_input_seqs[-20:], y=test_output_gold[-20:],
# #                  prob_y=prob_y[-20:], y_choices=test_output_choices[-20:])


# In[36]:

def train(input_seqs, output_seqs, modes, n_neg_per_mode, model_filepath, encoder_dim=4800, context_len=4,
          gen_temp=0.4, n_chunks=7, use_trunc=True, nb_epoch=10):
        
    #import pdb;pdb.set_trace()
    n_neg_per_seq = sum(n_neg_per_mode)
    
    if not use_skipthoughts:
        #import pdb;pdb.set_trace()
        seq_binary.named_steps['transformer'].fit(X=input_seqs, y_seqs=output_seqs)

    if not os.path.isdir(model_filepath):
        os.mkdir(model_filepath)
    neg_seqs_filepath = model_filepath + '/neg_seqs.npy'
    
    if not os.path.exists(neg_seqs_filepath):
        #import pdb;pdb.set_trace()
        neg_seqs = get_neg_seqs(input_seqs, output_seqs, modes, n_neg_per_mode, gen_temp=0.4)
        
        if use_skipthoughts:
            encode_skipthought_seqs(neg_seqs, encoder_module, sent_encoder, 
                                    encoder_dim, memmap=True, filepath=neg_seqs_filepath)
        else:
            neg_seqs, _ = seq_binary.named_steps['transformer'].transform(X=neg_seqs)
            numpy.save(neg_seqs_filepath, neg_seqs)
    
    if use_skipthoughts:
        neg_seqs = numpy.memmap(neg_seqs_filepath, dtype='float64', mode='r',
                                shape=(len(input_seqs), n_neg_per_seq, encoder_dim))
    else:
        #load neg seqs from mem-mapped file
        neg_seqs = numpy.load(neg_seqs_filepath, mmap_mode='r')
            
    print "loaded negative examples from filepath", neg_seqs_filepath
    
    input_seqs_filepath = model_filepath + '/input_seqs.npy'
    output_seqs_filepath = model_filepath + '/output_seqs.npy'
    #import pdb;pdb.set_trace()
    if not (os.path.exists(input_seqs_filepath) and os.path.exists(output_seqs_filepath)):
        if use_skipthoughts:
            encode_skipthought_seqs(input_seqs, encoder_module, sent_encoder, 
                                                 encoder_dim, memmap=True, filepath=input_seqs_filepath)
            encode_skipthought_seqs(output_seqs, encoder_module, sent_encoder, 
                                                  encoder_dim, memmap=True, filepath=output_seqs_filepath)
        else:
            input_seqs, output_seqs = seq_binary.named_steps['transformer'].transform(X=input_seqs,
                                                                                      y_seqs=output_seqs)
            #import pdb;pdb.set_trace()
            numpy.save(input_seqs_filepath, input_seqs)
            numpy.save(output_seqs_filepath, output_seqs)
    
    if use_skipthoughts:
        input_seqs = numpy.memmap(input_seqs_filepath, dtype='float64', mode='r',
                                  shape=(len(input_seqs), context_len, encoder_dim))
        output_seqs = numpy.memmap(output_seqs_filepath, dtype='float64', mode='r',
                                  shape=(len(input_seqs), encoder_dim))
    else:
        #load seqs from mem-mapped file
        input_seqs = numpy.load(input_seqs_filepath, mmap_mode='r')
        output_seqs = numpy.load(output_seqs_filepath, mmap_mode='r')
        
    print "training seq_binary classifier on corpus of", len(input_seqs), "stories"
    
    if use_trunc:
        trunc_input_seqs, trunc_output_seqs = get_trunc_pos_seqs(input_seqs, output_seqs, model_filepath)
        print "added", len(trunc_input_seqs), "truncated positive examples"
    
    print "added", len(input_seqs) * n_neg_per_seq, "negative examples"
    print "instances divided into", n_chunks, "chunks for training"
        
    #import pdb;pdb.set_trace()
    seqs_per_chunk = len(input_seqs) / n_chunks
    for epoch in range(nb_epoch):
        print "TRAINING EPOCH {}/{}".format(epoch + 1, nb_epoch)
        for chunk_idx in range(n_chunks):
            #import pdb;pdb.set_trace()
            train_seqs, train_y_seqs, train_y = prep_train_seqs(input_seqs[chunk_idx * seqs_per_chunk: 
                                                                           (chunk_idx + 1) * seqs_per_chunk], 
                                                                output_seqs[chunk_idx * seqs_per_chunk: 
                                                                            (chunk_idx + 1) * seqs_per_chunk], 
                                                                neg_seqs[chunk_idx * seqs_per_chunk: 
                                                                           (chunk_idx + 1) * seqs_per_chunk],
                                                                trunc_input_seqs[chunk_idx * seqs_per_chunk: 
                                                                           (chunk_idx + 1) * seqs_per_chunk]\
                                                                if use_trunc else None, 
                                                                trunc_output_seqs[chunk_idx * seqs_per_chunk: 
                                                                            (chunk_idx + 1) * seqs_per_chunk]\
                                                                if use_trunc else None)
                                                                         #, chunk_shape))
            #import pdb;pdb.set_trace()
            seq_binary.named_steps['classifier'].fit(X=train_seqs, y_seqs=train_y_seqs, 
                                                     y=train_y, nb_epoch=1)
            #seq_binary.fit(X=train_seqs, y_seqs=train_y_seqs, y=train_y, classifier__nb_epoch=1)
    seq_binary.named_steps['classifier'].save()

def evaluate(input_seqs, output_choices, output_gold, encoder_dim=4800):
    if use_skipthoughts:
        input_seqs = encode_skipthought_seqs(input_seqs, encoder_module, sent_encoder, encoder_dim=encoder_dim)
        output_choices = encode_skipthought_seqs(output_choices, encoder_module, sent_encoder, encoder_dim)
    else:
        input_seqs, output_choices = seq_binary.named_steps['transformer'].transform(X=input_seqs,
                                                                                  y_seqs=output_choices)
        
    #import pdb;pdb.set_trace()
    prob_y, pred_y, accuracy = seq_binary.named_steps['classifier'].predict(X=input_seqs,
                                                                            y_seqs=output_choices,
                                                                            y=output_gold)
    print "accuracy:", accuracy
    return prob_y, pred_y, accuracy


if __name__ == "__main__":
    use_skipthoughts = False
    skipthoughts_model = 'roc' #'bookcorpus' #'roc' 
    modes = ('backward',)#,'lm') #('random','backward','similarity','lm')#('random', 'backward', 'lm')
    n_neg_per_mode = (4,)#(2, 2, 2)
        
    n_train_seqs = len(train_input_seqs)
    gen_temp = 0.4
    
    #import pdb;pdb.set_trace()
    filepath = 'roc_seqbinary_' + str(n_train_seqs)
    filepath += '_neg_' + "_".join([mode + str(n_neg) for mode, n_neg in zip(modes, n_neg_per_mode)])
    #import pdb;pdb.set_trace()
    if use_skipthoughts:
        if skipthoughts_model == 'roc':
            encoder_module = encoder_tools
            embed_filepath = 'roc97027_embeddings'
            embeddings = similarity_score.load_model(embed_filepath)
            model_dir = 'skipthoughts_roc' + str(n_train_seqs)
            lexicon_filepath = model_dir + '/lexicon'
            encoder_filepath = model_dir + '/encoder'
            sent_encoder = encoder_module.load_model(embed_map=embeddings, 
                                       path_to_model=encoder_filepath, path_to_dictionary=lexicon_filepath)
            n_embedding_nodes = 2400
        elif skipthoughts_model == 'bookcorpus':
            encoder_module = skipthoughts
            sent_encoder = encoder_module.load_model()
            n_embedding_nodes = 4800
        filepath += '_skipthoughts_' + skipthoughts_model
        print 'loaded skipthoughts encoder'
        #just create a dummy transformer object since skipthoughts model doesn't use it but pipeline requires it
        transformer = SequenceTransformer(min_freq=1, verbose=1, word_embeddings=None)
    
    else:
        #import pdb;pdb.set_trace()
        embed_filepath = 'roc97027_embeddings'
        embed_name = 'roc_emb'
        #embed_filepath = 'AvMaxSim/vectors'
        #embed_name = 'google_emb'
        embeddings = similarity_score.load_model(embed_filepath)
        n_embedding_nodes = embeddings.vector_size
        print "using word embeddings from", embed_filepath
        filepath += "_" + embed_name
    
        if os.path.exists(filepath + '/transformer.pkl'):
            #load existing transformer
            transformer = load_transformer(filepath, embeddings=embeddings)
        else:
            transformer = SequenceTransformer(min_freq=1, verbose=1,
                                              word_embeddings=embeddings,
                                              embed_y=True, reduce_emb_mode='mean',
                                              filepath=filepath)
        
    seq_binary = RNNPipeline(steps=[("transformer", transformer),
                                    ("classifier", SeqBinaryClassifier(batch_size=100, nb_epoch=1, 
                                                                   verbose=0, n_hidden_layers=1,
                                                                   n_embedding_nodes=n_embedding_nodes,
                                                                   n_hidden_nodes=1000, context_size=4,
                                                                   filepath=filepath, use_dropout=False))])

    #import pdb;pdb.set_trace()
    train(train_input_seqs[:n_train_seqs], train_output_seqs[:n_train_seqs], modes=modes, 
          n_neg_per_mode=n_neg_per_mode, model_filepath=filepath, encoder_dim=n_embedding_nodes, 
          gen_temp=gen_temp, n_chunks=21, use_trunc=False)
    import pdb;pdb.set_trace()
    prob_y, pred_y, accuracy = evaluate(val_input_seqs, val_output_choices, val_output_gold, 
                                        encoder_dim=n_embedding_nodes)
    show_predictions(X=val_input_seqs[-20:], y=val_output_gold[-20:],
                 prob_y=prob_y[-20:], y_choices=val_output_choices[-20:])



# In[10]:

#seq_index = SimilarityIndex(filepath='roc_97027.index')
'''seqs = [" ".join(seq) for seq in train_input_seqs[100:150]]
sim_idxs, _ = seq_index.get_similar_seqs(seqs, n_best=2)
for idx, (seq, sim_idx) in enumerate(zip(seqs, sim_idxs)):
    print seq
    print idx, [train_output_seqs[idx] for idx in sim_idx], "\n"'''


# In[6]:

# if __name__ == '__main__':
#     train_stories = get_train_stories(filepath='ROC-Stories.tsv') + \
#                     get_train_stories(filepath='ROCStories_winter2017.csv')
#     #train_stories = train_stories[:1000]
#     context_seqs = [" ".join(story[:-1]) for story in train_stories]
#     fin_sents = [story[-1] for story in train_stories]
#     #import pdb;pdb.set_trace()
#     story_index = SimilarityIndex(filepath='roc_' + str(len(train_stories)) + '.index', 
#                                   stories=context_seqs, min_freq=1)

