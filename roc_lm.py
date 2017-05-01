
# coding: utf-8

# In[3]:

import sys, os, random, pandas, numpy, re

# if os.getcwd() != '/Users/roemmele/Documents/Interactive_Narrative/ROC':
#     os.chdir('/Users/roemmele/Documents/Interactive_Narrative/ROC')

from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec, Phrases, LsiModel
from simplejson import loads

import roc
reload(roc)
from roc import *

sys.path.append('../')
import models.transformer
reload(models.transformer)
from models.transformer import *

import models.pipeline
reload(models.pipeline)
from models.pipeline import *

import models.classifier
reload(models.classifier)
from models.classifier import MLPLM
from models.classifier import *

import models.ngram
reload(models.ngram)
from models.ngram import *

import models.stats
from models.stats import *

from textacy import *

# sys.path.append('../google-ngrams-master')
# from getngrams import *

# sys.path.append('../stanford-corenlp-python/')
# from corenlp import *

# sys.path.append('../grammaticality-metrics-master/codalab/scoring_program')
# import evaluate
# reload(evaluate)
# from evaluate import call_lt

# sys.path.append('../coheoka-master/coheoka')
# from coherence_probability import *


# In[332]:

def load_lm(filepath):
    #import pdb;pdb.set_trace()
    transformer = load_transformer(filepath)
    transformer.sent_encoder = None
    transformer.unk_word = u"<UNK>"
    transformer.lexicon_lookup[1] = transformer.unk_word
    classifier = load_classifier(filepath)
    lm = RNNPipeline(transformer, classifier)
    return lm

def eval_pmi_sents(transformer, model, context_seq, gen_sents, eval_mode='pmi'):
    sent_scores = []
    if eval_mode == 'pmi':
        context_seq = transformer.transform(context_seq)[0][0]
        gen_sents = transformer.transform(gen_sents)[0]
    for gen_sent in gen_sents:
        if eval_mode == 'pmi':
            sent_score = model.score(sequences=[context_seq, gen_sent])
        elif eval_mode == 'avemax':
            sent_score = similarity_score.score(context_seq, gen_sent, model, tail=0, head=-1)
        sent_scores.append(sent_score)
    return sent_scores
    

def show_gen_sents(context_seqs, gen_sents=None, p_gen_sents=None, gold_sents=None, p_gold_sents=None):
    for seq_idx in range(len(context_seqs)):
        print "STORY:", context_seqs[seq_idx]
        if gold_sents:
            print "GOLD:", gold_sents[seq_idx], "({:.3f})".format(p_gold_sents[seq_idx])
        if gen_sents:
    #         for sent, p_sent in zip(gen_sents[seq_idx], p_gen_sents[seq_idx]):
            print "PRED:", gen_sents[seq_idx], "({:.3f})".format(p_gen_sents[seq_idx])
        print "\n"
    
def get_binary_values(n_pos, n_samples):
    '''create an array of zeros of size n_samples, and set n_pos of the values to 1;
    used for permutation tests that need total number of samples'''
    
    binary_values = numpy.concatenate((numpy.ones((n_pos)),
                                       numpy.zeros((n_samples - n_pos))))
    return binary_values
        


# In[160]:

if __name__ == '__main__': 
    '''load cloze val and test stories'''
#     #import pdb;pdb.set_trace()
    val_input_seqs, val_output_choices, val_output_gold = get_cloze_data('cloze_test_ALL_val.tsv', flatten=True)
    test_input_seqs, test_output_choices, test_output_gold = get_cloze_data('cloze_test_ALL_test.tsv', flatten=True)
    context_seqs = val_input_seqs + test_input_seqs
    context_sents = get_cloze_data('cloze_test_ALL_val.tsv')[0] + get_cloze_data('cloze_test_ALL_test.tsv')[0]
    gold_fin_sents = [choices[gold] for choices, gold in zip(val_output_choices, val_output_gold)]                + [choices[gold] for choices, gold in zip(test_output_choices, test_output_gold)]
    lm_transformer = load_lm('../roc_lm97027_batchtrained').transformer
    train_fin_sents = [story[-1] for story in get_train_stories(filepath='ROC-Stories.tsv')                                            + get_train_stories(filepath='ROCStories_winter2017.csv')]
    


# In[6]:

if __name__ == '__main__':
    '''load generated sentences from file'''
    unigram_gen_sents = pandas.read_csv('unigram_sents3742_5.csv', encoding='utf-8', header=None).values.tolist()
    ngram_gen_sents = pandas.read_csv('ngram_sents3742_5.csv', encoding='utf-8', header=None).values.tolist()
    mlp_gen_sents = pandas.read_csv('mlp_sents3742_5.csv', encoding='utf-8', header=None).values.tolist()
    rnn_gen_sents = pandas.read_csv('rnn_sents3742_5.csv', encoding='utf-8', header=None).values.tolist()
    model_gen_sents = {"unigram": unigram_gen_sents, "ngram": ngram_gen_sents, 
                       "mlp": mlp_gen_sents, "rnn": rnn_gen_sents, 
                       "gold":[[fin_sent] for fin_sent in gold_fin_sents]}
    


# # In[161]:

# def get_sent_lengths(sents):
#     lengths = [len(tokenize(sent)) for sent in sents]
#     return lengths

# if __name__ == '__main__':
#     import pdb;pdb.set_trace()
#     len_gold_sents = get_sent_lengths(gold_fin_sents)
#     print "mean gold sentence length:", numpy.mean(len_gold_sents)
#     len_unigram_sents = [sent_lengths for sents in unigram_gen_sents 
#                                         for sent_lengths in get_sent_lengths(sents)]
#     print "mean unigram sentence length:", numpy.mean(len_unigram_sents)
#     len_ngram_sents = [sent_lengths for sents in ngram_gen_sents
#                                     for sent_lengths in get_sent_lengths(sents)]
#     print "mean ngram sentence length:", numpy.mean(len_ngram_sents)
#     len_mlp_sents = [sent_lengths for sents in mlp_gen_sents
#                                     for sent_lengths in get_sent_lengths(sents)]
#     print "mean mlp sentence length:", numpy.mean(len_mlp_sents)
#     len_rnn_sents = [sent_lengths for sents in rnn_gen_sents 
#                                     for sent_lengths in get_sent_lengths(sents)]
#     print "mean rnn sentence length:", numpy.mean(len_rnn_sents)
#     len_train_sents = get_sent_lengths(train_fin_sents)
#     print "mean train sentence length:", numpy.mean(len_train_sents)


# # In[276]:

# if __name__ == '__main__':
#     '''write file with list of lexicon words from model'''
    
#     with open('../fiction_lm_lexicon.txt', 'w') as f:
#     #     f.write("\t".join(word) for word in fiction_lm_transformer.lexicon.items())
#         f.write("\n".join([word.encode('utf-8', errors='replace') for word in fiction_lm_transformer.lexicon.keys()]))


# # In[ ]:

# if __name__ == '__main__':
#     '''read in list of named entities for capitalization'''
    
#     with open("roc_named_ents.txt", 'r') as f:
#         named_ents = [unicode(ent.strip(), errors='replace') for ent in f.readlines()]


# # In[ ]:

# if __name__ == '__main__':
#     '''get list of all named entities from training stories'''
#     train_stories = get_train_stories(filepath='ROC-Stories.tsv', flatten=True)                    + get_train_stories(filepath='ROCStories_winter2017.csv', flatten=True)
#     named_ents = []
#     for story in train_stories:
#         #print story
#         ents = [ent.lower() for ent, ent_type in get_entities(story)]#.items() 
# #                 if (ent[0].isupper() and ent_type not in ('DATE', 'TIME', 'ORDINAL', 'CARDINAL'))
# #                    or (ent[0].isupper() and ent_type in ('DATE', 'TIME', 'ORDINAL', 'CARDINAL') 
# #                        and sum([not sent.strip().startswith(ent) #uppercase DATE/TIME entity occurs not at start
# #                                 for sent in re.split('[' + "".join(eos_markers) + ']', story)
# #                                    if ent in sent]))]
#         named_ents.extend(set(ents))
#         #print "\n".join(ents), "\n"
#     named_ents = set(named_ents)


# # # Model creation and training

# # In[295]:

# sys.path.append("../")
# import story_corpus
# reload(story_corpus)
# from story_corpus import *

# if __name__ == "__main__":
#     '''create unigram model from train stories'''
# #     lm_transformer = load_lm('roc_lm97027_batchtrained_clio_updated').transformer
# #     stories = get_train_stories(filepath='ROC-Stories.tsv', flatten=True)\
# #                 + get_train_stories(filepath='ROCStories_winter2017.csv', flatten=True)
# #     stories = stories[:100]
#     import pdb;pdb.set_trace()
#     story_ids = get_story_ids(db_type='sqlite', n_stories=1000000)
# #     with open('../wordpress_blog_ids_filtered.txt', 'r') as id_file:
# #         story_ids = [int(story_id) for story_id in id_file.readlines()][:1000]
#     n_stories_per_chunk = len(story_ids) / 1000
#     for chunk_idx in range(0, len(story_ids), n_stories_per_chunk):
#         stories = get_stories(story_ids[chunk_idx:chunk_idx + n_stories_per_chunk])
#         update_ngram_model(lm_transformer, stories, n_min=4, n_max=4, filepath='blog_ngrams')
#         print "computed ngrams for", chunk_idx+n_stories_per_chunk, "stories..."
# #     get_ngram_dist(lm_transformer, stories, n=1)
# #     get_ngram_dist(lm_transformer, stories, n=2)
# #     get_ngram_dist(lm_transformer, stories, n=3)
# #     get_ngram_dist(lm_transformer, stories, n=4)
# #     get_ngram_dist(lm_transformer, stories, n=5)


# # In[247]:

# def train_batched_mlplm(stories):
#     n_epochs = 10
#     n_chunks = 1
#     seqs_per_chunk = len(stories) / n_chunks
#     #fit the transformer first
#     if not mlp_lm.transformer.lexicon:
#         mlp_lm.transformer.fit(stories)
#     #import pdb;pdb.set_trace()
#     for epoch in range(n_epochs):
#         print "training epoch {}/{}...".format(epoch + 1, n_epochs)
#         for chunk_idx in range(n_chunks):
#             mlp_lm.fit(stories[chunk_idx:chunk_idx + seqs_per_chunk])
#         samp_stories = random.sample(stories, 10)
#         context_seqs = [story[:-1] for story in samp_stories]
#         gen_sents = gen_mlp_sents(context_seqs)
#         for story, sent in zip(context_seqs, gen_sents):
#             print story
#             print sent
#             print "\n"

# def gen_mlp_sents(seqs):
#     import pdb;pdb.set_trace()
# #     for seq_idx, seq in enumerate(seqs):
#     gen_sents, _ = mlp_lm.predict(seqs, mode='random', eos_tokens=[".", "!", "?"])
#     return gen_sents
    

# if __name__ == '__main__':
#     '''train an MLP language model'''
# #     import pdb;pdb.set_trace()
# #     train_stories = get_train_stories(filepath='ROC-Stories.tsv') + \
# #                     get_train_stories(filepath='ROCStories_winter2017.csv')
# #     train_stories = train_stories[:100]
# #     #train_context_seqs = [" ".join(story[:-1]) for story in train_stories]
# #     #train_stories = [" ".join(story) for story in train_stories]
# #     filepath = 'roc_mlplm' + str(len(train_stories))
# #     if os.path.exists(filepath + '/transformer.pkl'):
# #         #load existing transformer
# #         transformer = load_transformer(filepath)
# #     else:
# #         transformer = SequenceTransformer(min_freq=1, verbose=1, filepath=filepath)
# #     if os.path.exists(filepath + '/classifier.pkl') and os.path.exists(filepath + '/classifier.h5'):
# #         #load existing transformer
# #         classifier = load_classifier(filepath)
# #     else:
# #         n_timesteps = 4
# #         classifier = MLPLM(verbose=1, batch_size=100, n_timesteps=n_timesteps,
# #                                n_hidden_layers=2, n_embedding_nodes=300, 
# #                                n_hidden_nodes=500, filepath=filepath)
# #     mlp_lm = RNNPipeline(transformer, classifier)                   
# #     train_batched_mlplm(train_stories)
#     #import pdb;pdb.set_trace()
#     #mlplm_perp = mlp_lm.evaluate(train_stories[:100])


# # In[ ]:

# def train_batched_rnnlm(stories):
#     n_epochs = 50
#     import pdb;pdb.set_trace()
#     for epoch in range(n_epochs):
#         print "training epoch {}/{}...".format(epoch + 1, n_epochs)
#         lm.fit(X=stories)
#         #generate samples to show progress
#         samp_size = 10
#         temp = 0.6
#         samp_stories = random.sample(stories, samp_size)
#         context_seqs = [story[:-1] for story in samp_stories]
#         gold_sents = [story[-1] for story in samp_stories]
#         p_gold_sents = lm.predict(X=context_seqs, y_seqs=gold_sents, batch_size=samp_size)
#         gen_sents, p_sents = generate_sents(lm, context_seqs, n_best=1, mode='max', batch_size=samp_size)
#         print "MAX PROB SENTENCES:"
#         show_gen_sents(context_seqs, gen_sents, p_sents, gold_sents, p_gold_sents)
#         import pdb;pdb.set_trace()
#         gen_sents, p_sents = generate_sents(lm, context_seqs, n_best=1, mode='random', 
#                                             temp=temp, batch_size=samp_size)
#         print "SENTENCES WITH TEMP =", temp
#         show_gen_sents(context_seqs, gen_sents, p_sents)

# if __name__ == '__main__':
#     '''train an RNN language model'''
#     import pdb;pdb.set_trace()
#     train_stories = get_train_stories(filepath='ROC-Stories.tsv') +                     get_train_stories(filepath='ROCStories_winter2017.csv')
#     train_stories = train_stories[:10]
#     filepath = 'roc_lm' + str(len(train_stories)) + '_batch1trained'
#     lm = RNNPipeline(steps=[('transformer', SequenceTransformer(min_freq=1, verbose=1, filepath=filepath)),
#                             ('classifier', RNNLM(verbose=1, batch_size=1, n_timesteps=None,
#                                              n_hidden_layers=2,
#                                              n_embedding_nodes=300, n_hidden_nodes=500,
#                                              filepath=filepath))])
#     train_batched_lm(train_stories)



# # # Sentence generation

# # In[235]:

# if __name__ == '__main__': 
#     '''generate sents from unigrams'''
#     import pdb;pdb.set_trace()
#     eos_tokens = [".", "!", "?"]
#     n_gen_sents = 5 #num of sents to generate per context
#     unigram_gen_sents = gen_ngram_sents(lm_transformer, 
#                                         [context_seqs[idx] for idx in\
#                                         numpy.arange(len(context_seqs)).repeat(n_gen_sents)], 
#                                         n=1, filepath='ngrams', eos_tokens=eos_tokens,
#                                         cap_tokens=named_ents)
#     unigram_gen_sents = [unigram_gen_sents[idx:idx + n_gen_sents] 
#                          for idx in range(0, len(context_seqs) * n_gen_sents, n_gen_sents)]
#     pandas.DataFrame(unigram_gen_sents).to_csv('unigram_sents' + str(len(context_seqs))
#                                                + '_' + str(n_gen_sents) + '.csv', header=False, 
#                                                index=False, encoding='utf-8') #save generated sents to file


# # In[288]:

# if __name__ == '__main__':
#     '''generate sents from MLPLM'''
    
#     mlplm = load_lm('roc_mlplm97027')
#     import pdb;pdb.set_trace()
#     eos_tokens = [".", "!", "?"]
#     context_seqs = val_input_seqs + test_input_seqs 
#     n_gen_sents = 5 #num of sents to generate per context
#     context_seqs = context_seqs[-100:]
#     import pdb;pdb.set_trace()
# #     heldout_stories = [context_seq + " " + fin_sent for context_seq, fin_sent in zip(context_seqs, gold_fin_sents)]
# #     mlplm_perp = mlplm.evaluate(heldout_stories)
# #     print mlplm_perp
#     mlp_gen_sents, _ = generate_sents(mlplm, [context_seqs[idx] for idx in                                            numpy.arange(len(context_seqs)).repeat(n_gen_sents)],
#                                       batch_size=1000, n_best=1, n_words=25, 
#                                       mode='random', temp=1.0, eos_tokens=eos_tokens, cap_tokens=named_ents)
#     mlp_gen_sents = [mlp_gen_sents[idx:idx + n_gen_sents] 
#                      for idx in range(0, len(context_seqs) * n_gen_sents, n_gen_sents)]
# #     pandas.DataFrame(mlp_gen_sents).to_csv('mlp_sents' + str(len(context_seqs))
# #                                            + '_' + str(n_gen_sents) + '.csv', header=False, 
# #                                            index=False, encoding='utf-8') #save generated sents to file
#     #rnn_gen_sents = pandas.read_csv('rnn_sents3742_5.csv', encoding='utf-8', header=None).values.tolist()
# #     for idx, sents in enumerate(rnn_gen_sents):
# #         print context_seqs[idx]
# #         for sent in sents:
# #             print sent
# #         print "\n"


# # In[9]:

# if __name__ == '__main__': 
#     '''generate sents from rnn'''

#     #import pdb;pdb.set_trace()
#     context_seqs = val_input_seqs + test_input_seqs 
#     lm = load_lm('roc_lm97027_batchtrained_clio_updated')
#     #context_seqs = context_seqs[:50]
#     n_gen_sents = 5 #num of sents to generate per context
#     rnn_gen_sents, _ = generate_sents(lm, [context_seqs[idx] for idx in                                        numpy.arange(len(context_seqs)).repeat(n_gen_sents)], 
#                                    batch_size=1000, n_best=1, n_words=25, 
#                                    mode='random', temp=1.0, eos_tokens=[".", "!", "?"])
#     rnn_gen_sents = [rnn_gen_sents[idx:idx + n_gen_sents] 
#                      for idx in range(0, len(context_seqs) * n_gen_sents, n_gen_sents)]
#     pandas.DataFrame(rnn_gen_sents).to_csv('rnn_sents' + str(len(context_seqs))
#                                                + '_' + str(n_gen_sents) + '.csv', header=False, 
#                                                index=False, encoding='utf-8') #save generated sents to file

    


# # In[206]:

# if __name__ == "__main__":
#     '''read in Reid's ngram-generated sentences and save them as pandas-friendly csv'''
#     ngram_gen_sents1 = pandas.read_csv('reid_ngram_sents1.tsv', encoding='utf-8', 
#                                        header=None, sep='\t').values.tolist()
#     ngram_gen_sents2 = pandas.read_csv('reid_ngram_sents2.tsv', encoding='utf-8', 
#                                    header=None, sep='\t').values.tolist()
#     ngram_gen_sents = ngram_gen_sents1 + ngram_gen_sents2
#     ngram_gen_sents = [[lm_transformer.detokenize_sent(sent.split()[:25], #limit sentences to 25 words
#                                                        eos_tokens=eos_tokens, 
#                                                        cap_tokens=named_ents)
#                         for sent in sents] for sents in ngram_gen_sents]
#     pandas.DataFrame(ngram_gen_sents).to_csv('ngram_gen_sents' + str(len(ngram_gen_sents))
#                                            + '_5.csv', header=False, 
#                                            index=False, encoding='utf-8') #save generated sents to file


# # In[ ]:

# if __name__ == '__main__': 
#     '''generate sents from n-gram models (unused)'''
# #     import pdb;pdb.set_trace()
# #     bigram_gen_sents = gen_ngram_sents(lm_transformer, context_seqs[:25], n=2, filepath='ngrams_test')
# #     for context, fin_sent, gen_sent in zip(context_seqs, fin_sents, bigram_gen_sents)[:50]:
# #         print context
# #         print "GOLD:", fin_sent
# #         print "PRED:", gen_sent
# #         print "\n"
#     #bigram_gen_sents = gen_ngram_sents(lm_transformer, n=2, prev_tokens=story_last_tokens)
#     #trigram_gen_sents = gen_ngram_sents(lm_transformer, n=3, prev_tokens=story_last_tokens)
    
#     #quingram_gen_sents = gen_ngram_sents(lm_transformer, context_seqs, n=5)
# #     for context, fin_sent, gen_sent in zip(context_seqs, fin_sents,  quingram_gen_sents)[:50]:
# #         print context
# #         print "GOLD:", fin_sent
# #         print "PRED:", gen_sent
# #         print "\n"


# # # Perplexity

# # In[6]:

# if __name__ == '__main__': 
#     '''calcuate average perplexity of ngram models from Reid's files'''
    
#     import pdb;pdb.set_trace()
#     with open('cloze_test_ALL_val.ppl', 'r') as f:
#         perps = [perp.strip().split() for perp in f.readlines()]
#         correct_perps = [float(perp[choice]) for perp, choice in zip(perps, val_output_gold)]
#         incorrect_perps = [float(perp[numpy.logical_not(choice)]) for perp, choice in zip(perps, val_output_gold)]
#         print "val correct perplexity:", numpy.mean(correct_perps)
#         print "val incorrect perplexity:", numpy.mean(incorrect_perps)
            
#     import pdb;pdb.set_trace()
#     with open('cloze_test_ALL_test.ppl', 'r') as f:
#         perps = [perp.strip().split() for perp in f.readlines()]
#         correct_perps = [float(perp[choice]) for perp, choice in zip(perps, val_output_gold)]
#         incorrect_perps = [float(perp[numpy.logical_not(choice)]) for perp, choice in zip(perps, val_output_gold)]
#         print "test correct perplexity:", numpy.mean(correct_perps)
#         print "test incorrect perplexity:", numpy.mean(incorrect_perps)
        


# # In[172]:

# if __name__ == '__main__': 
#     '''compute perplexity of unigram model'''
    
#     heldout_stories = [context_seq + " " + fin_sent for context_seq, fin_sent in zip(context_seqs, fin_sents)]
#     import pdb;pdb.set_trace()
#     unigram_perp = get_perplexity(lm_transformer, heldout_stories, 1, 'ngrams')
#     print unigram_perp
# #     bigram_perp = get_perplexity(lm_transformer, heldout_stories, 2, 'ngrams')
# #     print bigram_perp


# # In[167]:

# if __name__ == '__main__': 
#     '''compute perplexity of RNNLM'''
    
#     lm = load_lm('roc_lm97027_batchtrained_clio_updated')
# # #     stories = get_train_stories(filepath='ROC-Stories.tsv', flatten=True)\
# # #             + get_train_stories(filepath='ROCStories_winter2017.csv', flatten=True)
#     #import pdb;pdb.set_trace()
#     rnnlm_perp = lm.evaluate(heldout_stories)
#     print rnnlm_perp


# # # Bleu score

# # In[401]:

# sys.path.append('../Bleu-master')
# from calculatebleu import *

# from nltk.translate.bleu_score import *

# if __name__ == '__main__': 
#     unigram_bleu_scores = []
#     '''compute bleu scores of generated sentences relative to gold'''
#     import pdb;pdb.set_trace() 
#     for story_idx, (gen_sents, gold_sent) in enumerate(zip(unigram_gen_sents, gold_fin_sents)):
#         for sent in gen_sents: 
#             #bleu_score = BLEU([sent], [[gold_sent]])
#             bleu_score = sentence_bleu([tokenize(gold_sent)], tokenize(sent))
#             unigram_bleu_scores.append(bleu_score)
# #             if bleu_score:
# #                 import pdb;pdb.set_trace() 
#             #print sent, gold_sent, bleu_score
#         if story_idx % 100 == 0:
#             print "processed sentences for", story_idx, "stories"
#     print "mean unigram bleu scores:", numpy.mean(unigram_bleu_scores)
    



# # In[402]:


# if __name__ == '__main__': 
#     ngram_bleu_scores = []
#     '''compute bleu scores of generated sentences relative to gold'''
#     #import pdb;pdb.set_trace() 
#     for story_idx, (gen_sents, gold_sent) in enumerate(zip(ngram_gen_sents, gold_fin_sents)):
#         for sent in gen_sents: 
#             bleu_score = sentence_bleu([tokenize(gold_sent)], tokenize(sent))
#             ngram_bleu_scores.append(bleu_score)
#     print "mean ngram bleu scores:", numpy.mean(ngram_bleu_scores)
    



# # In[403]:

# if __name__ == '__main__': 
#     mlp_bleu_scores = []
#     '''compute bleu scores of generated sentences relative to gold'''
#     #import pdb;pdb.set_trace() 
#     for story_idx, (gen_sents, gold_sent) in enumerate(zip(mlp_gen_sents, gold_fin_sents)):
#         for sent in gen_sents: 
#             bleu_score = sentence_bleu([tokenize(gold_sent)], tokenize(sent))
#             mlp_bleu_scores.append(bleu_score)
#     print "mean mlp bleu scores:", numpy.mean(mlp_bleu_scores)
    


# # In[404]:

# if __name__ == '__main__': 
#     rnn_bleu_scores = []
#     '''compute bleu scores of generated sentences relative to gold'''
#     #import pdb;pdb.set_trace() 
#     for story_idx, (gen_sents, gold_sent) in enumerate(zip(rnn_gen_sents, gold_fin_sents)):
#         for sent in gen_sents: 
#             bleu_score = sentence_bleu([tokenize(gold_sent)], tokenize(sent))
#             rnn_bleu_scores.append(bleu_score)
#     print "mean rnn bleu scores:", numpy.mean(rnn_bleu_scores)
    



# # In[406]:

# if __name__ == '__main__':    
#     p_unigram_ngram_bleu = evaluate_difference(unigram_bleu_scores, ngram_bleu_scores,
#                                                num_trials=10000, verbose=False)
#     print "p-value for bleu score difference between unigram and ngram sents:", p_unigram_ngram_bleu
#     p_ngram_mlp_bleu = evaluate_difference(ngram_bleu_scores, mlp_bleu_scores,
#                                             num_trials=10000, verbose=False)
#     print "p-value for bleu score difference between ngram and mlp sents:", p_ngram_mlp_bleu
#     p_mlp_rnn_bleu = evaluate_difference(mlp_bleu_scores, rnn_bleu_scores,
#                                          num_trials=10000, verbose=False)
#     print "p-value for bleu score difference between mlp and rnn sents:", p_mlp_rnn_bleu


# # N-gram Analysis

# In[427]:

def analyze_ngrams(transformer, gen_sents, n, filepath):
    if type(gen_sents[0]) not in (list, tuple):
        gen_sents = [[sent] for sent in gen_sents]
    gen_ngrams = []
    known_ngrams = []
    for sents in gen_sents:
        ngrams, counts = get_ngrams_from_seqs(transformer, sents, n, filepath)
        gen_ngrams.extend(ngrams)
        for ngram, count in zip(ngrams, counts):
            if count:
                #import pdb;pdb.set_trace()
                known_ngrams.append(ngram)
                #print ngram, count
    return gen_ngrams, known_ngrams


# ## Count generated ngrams that also occur in blog corpus

# In[430]:

import models.ngram
reload(models.ngram)
from models.ngram import *

#gen_ngrams = {"unigram":unigram_gen_ngrams,"ngram":ngram_gen_ngrams, "mlp":mlp_gen_ngrams, 
#              "rnn":rnn_gen_ngrams, "gold":gold_gen_ngrams}
gen_ngrams = {}
gen_ngrams_in_blog = {}

if __name__ == '__main__':
    #unigram_gen_sents
    import pdb;pdb.set_trace()
    for model in model_gen_sents:
        ngrams, ngrams_in_blog = analyze_ngrams(lm_transformer, model_gen_sents[model], n=4, filepath="blog_ngrams.db")
        gen_ngrams[model] = ngrams
        gen_ngrams_in_blog[model] = ngrams_in_blog
        print "blog results for model:", model
        print "\ttotal # of 4-grams:", len(gen_ngrams[model])
        print "\t# unique:",  len(set(gen_ngrams[model]))
        print "\t# of unique 4-grams also in blog stories:", len(set(gen_ngrams_in_blog[model]))
        print "\t%:", len(set(gen_ngrams_in_blog[model])) * 1. / len(set(gen_ngrams[model]))


# In[432]:

if __name__ == '__main__':
    blog_ngram_values = {}
    for model in gen_ngrams:
        blog_ngram_values[model] = get_binary_values(len(set(gen_ngrams_in_blog[model])), 
                                                         len(set(gen_ngrams[model])))
    p_blog_ngrams = {}
    p_blog_ngrams[("unigram", "ngram")] = evaluate_difference(blog_ngram_values["unigram"], 
                                                              blog_ngram_values["ngram"])
    print "p-value for blog 4-gram difference between unigram and ngram sents:", p_blog_ngrams[("unigram", "ngram")]

    p_blog_ngrams[("ngram", "mlp")] = evaluate_difference(blog_ngram_values["ngram"], 
                                                          blog_ngram_values["mlp"])
    print "p-value for blog 4-gram difference between ngram and mlp sents:", p_blog_ngrams[("ngram", "mlp")]
    
    p_blog_ngrams[("mlp", "rnn")] = evaluate_difference(blog_ngram_values["mlp"],
                                                        blog_ngram_values["rnn"])
    print "p-value for blog 4-gram difference between mlp and rnn sents:", p_blog_ngrams[("mlp", "rnn")]
    
    p_blog_ngrams[("rnn", "gold")] = evaluate_difference(blog_ngram_values["rnn"], 
                                                         blog_ngram_values["gold"])
    print "p-value for blog 4-gram difference between rnn and gold sents:", p_blog_ngrams[("rnn", "gold")]


# # ## Count generated ngrams that also occur ROC training stories

# # In[326]:

# if __name__ == '__main__':
#     #unigram_gen_sents
#     import pdb;pdb.set_trace()
#     unigram_gen_ngrams, unigram_known_ngrams = analyze_ngrams(lm_transformer, unigram_gen_sents, n=4)
#     #n_gen_ngrams = len(gen_ngrams)
#     print "total # of 4-grams in unigram sents:", len(unigram_gen_ngrams)
#     print "# unique:",  len(set(unigram_gen_ngrams))
#     print "# of unique 4-grams also in training stories:", len(set(unigram_known_ngrams))
#     print "%:", len(set(unigram_known_ngrams)) * 1. / len(set(unigram_gen_ngrams))


# # In[327]:

# if __name__ == '__main__':

#     ngram_gen_ngrams, ngram_known_ngrams = analyze_ngrams(lm_transformer, ngram_gen_sents, n=4)
#     print "total # of 4-grams in ngram sents:", len(ngram_gen_ngrams)
#     print "# unique:",  len(set(ngram_gen_ngrams))
#     print "# of unique 4-grams also in training stories:", len(set(ngram_known_ngrams))
#     print "%:", len(set(ngram_known_ngrams)) * 1. / len(set(ngram_gen_ngrams))


# # In[328]:

# if __name__ == '__main__':

#     mlp_gen_ngrams, mlp_known_ngrams = analyze_ngrams(lm_transformer, mlp_gen_sents, n=4)
#     print "total # of 4-grams in mlp sents:", len(mlp_gen_ngrams)
#     print "# unique:",  len(set(mlp_gen_ngrams))
#     print "# of unique 4-grams also in training stories:", len(set(mlp_known_ngrams))
#     print "%:", len(set(mlp_known_ngrams)) * 1. / len(set(mlp_gen_ngrams))


# # In[329]:

# if __name__ == '__main__':

#     rnn_gen_ngrams, rnn_known_ngrams = analyze_ngrams(lm_transformer, rnn_gen_sents, n=4)
#     print "total # of 4-grams in rnn sents:", len(rnn_gen_ngrams)
#     print "# unique:",  len(set(rnn_gen_ngrams))
#     print "# of unique 4-grams also in training stories:", len(set(rnn_known_ngrams))
#     print "%:", len(set(rnn_known_ngrams)) * 1. / len(set(rnn_gen_ngrams))


# # In[331]:

# if __name__ == '__main__':
#     gold_gen_ngrams, gold_known_ngrams = analyze_ngrams(lm_transformer, gold_fin_sents, n=4)
#     print "total # of 4-grams in gold sents:", len(gold_gen_ngrams)
#     print "# unique:",  len(set(gold_gen_ngrams))
#     print "# of unique 4-grams also in training stories:", len(set(gold_known_ngrams))
#     print "%:", len(set(gold_known_ngrams)) * 1. / len(set(gold_gen_ngrams))


# # In[334]:

# if __name__ == '__main__':
#     unigram_ngram_values = get_binary_values(len(set(unigram_known_ngrams)), len(set(unigram_gen_ngrams)))
#     ngram_ngram_values = get_binary_values(len(set(ngram_known_ngrams)), len(set(ngram_gen_ngrams)))
#     mlp_ngram_values = get_binary_values(len(set(mlp_known_ngrams)), len(set(mlp_gen_ngrams)))
#     rnn_ngram_values = get_binary_values(len(set(rnn_known_ngrams)), len(set(rnn_gen_ngrams)))
#     gold_ngram_values = get_binary_values(len(set(gold_known_ngrams)), len(set(gold_gen_ngrams)))
#     p_unigram_ngram_ngrams = evaluate_difference(unigram_ngram_values, ngram_ngram_values, 
#                                                  num_trials=10000, verbose=False)
#     print "p-value for training 4-gram difference between unigram and ngram sents:", p_unigram_ngram_ngrams
#     p_ngram_mlp_ngrams = evaluate_difference(ngram_ngram_values, mlp_ngram_values, 
#                                              num_trials=10000, verbose=False)
#     print "p-value for training 4-gram difference between ngram and mlp sents:", p_ngram_mlp_ngrams
#     p_mlp_rnn_ngrams = evaluate_difference(mlp_ngram_values, rnn_ngram_values, 
#                                              num_trials=10000, verbose=False)
#     print "p-value for training 4-gram difference between mlp and rnn sents:", p_mlp_rnn_ngrams
#     p_mlp_gold_ngrams = evaluate_difference(rnn_ngram_values, gold_ngram_values,
#                                             num_trials=10000, verbose=False)
#     print "p-value for training 4-gram difference between mlp and gold sents:", p_mlp_gold_ngrams


# # # Phrases

# # In[359]:

# def get_phrases(sents, min_count=5, threshold=10):
#     #import pdb;pdb.set_trace()
#     tok_sents = [tokenize(sent) for sent in sents]
#     bigram_phraser = Phrases(tok_sents, delimiter=' ', min_count=min_count, threshold=threshold)
#     bigram_phrases = set(list(bigram_phraser.export_phrases(tok_sents)))
#     #print "bigram phrases:", bigram_phrases, "\n"
#     trigram_phraser = Phrases(bigram_phraser[tok_sents], delimiter=' ', min_count=min_count, threshold=threshold)
#     trigram_phrases = set(list(trigram_phraser.export_phrases(bigram_phraser[tok_sents])))
#     #print "trigram phrases:", trigram_phrases
#     phrases = bigram_phrases.union(trigram_phrases)
#     phrases = [(phrase, score) for score, phrase in 
#                sorted([(score, phrase) for phrase, score in phrases], reverse=True)]
#     print phrases
#     return phrases

# def lookup_phrases(ref_phrases, cand_phrases):
#     '''check whether an ngram (cand phrase) exists as phrase in corpus of known phrases'''
#     known_phrases = ref_phrases.intersection(set(cand_phrases))
#     return known_phrases

# def compare_phrases(phrases1, phrases2):
#     phrases1 = set([phrase for phrase, score in phrases1])
#     print "# of phrases1:", len(phrases1)
#     phrases2 = set([phrase for phrase, score in phrases2])
#     print "# of phrases2:", len(phrases2)
#     common_phrases = phrases1.intersection(phrases2)
#     print "# of common phrases:", len(common_phrases)
#     print "common phrases:", common_phrases, "\n"
#     unique_phrases1 = phrases1.difference(phrases2)
#     print "# of unique phrases in phrases1:", len(unique_phrases1)
#     print "phrases1 unique phrases:", unique_phrases1, "\n"
#     unique_phrases2 = phrases2.difference(phrases1)
#     print "# of unique phrases in phrases2:", len(unique_phrases2)
#     print "phrases2 unique phrases:", unique_phrases2, "\n"

# def get_candidate_phrases(gen_sents):
#     if type(gen_sents[0]) not in (list, tuple):
#         gen_sents = [[sent] for sent in gen_sents]
#     cand_phrases = []
#     for sents in gen_sents:
#         for sent in Corpus(lang=u'en', texts=sents):
#             for n in (2,3): #get phrases of both 2 and 3 words
#                 cand_phrases.extend([ngram.text.lower() for ngram in 
#                                      extract.ngrams(sent, n=n, filter_punct=False, filter_stops=False)])
#     return cand_phrases
    


# # In[344]:

# # if __name__ == '__main__':
# #     '''get phrases in training stories'''
# #     import pdb;pdb.set_trace()
    
# #     train_sents = [sent for story in (get_train_stories(filepath='ROC-Stories.tsv')\
# #                                     + get_train_stories(filepath='ROCStories_winter2017.csv'))
# #                   for sent in story]
# #     train_phrases = get_phrases(train_sents)
# #     print "# of phrases in training stories:", len(train_phrases)
# #     with open('roc_train_phrases.pkl', 'wb') as f:
# #         pickle.dump(train_phrases, f)
        
# #     train_phrases = set([phrase for phrase, score in train_phrases]) #stores phrases as set


# # In[377]:

# if __name__ == '__main__':
#     '''get phrases in unigram sentences'''
#     unigram_cand_phrases = get_candidate_phrases(unigram_gen_sents)
#     unigram_phrases = lookup_phrases(train_phrases, unigram_cand_phrases)
#     unigram_phrase_values = get_binary_values(len(unigram_phrases), len(set(unigram_cand_phrases)))
#     print "# of unigram candidate phrases:", len(unigram_cand_phrases)
#     print "# of unique unigram candidate phrases:", len(set(unigram_cand_phrases))
#     print "# of recognized phrases (from training):", len(unigram_phrases)
#     print "%:", len(unigram_phrases) * 1. / len(set(unigram_cand_phrases))


# # In[378]:

# if __name__ == '__main__':
#     '''get phrases in unigram sentences'''
#     ngram_cand_phrases = get_candidate_phrases(ngram_gen_sents)
#     ngram_phrases = lookup_phrases(train_phrases, ngram_cand_phrases)
#     ngram_phrase_values = get_binary_values(len(ngram_phrases), len(set(ngram_cand_phrases)))
#     print "# of ngram candidate phrases:", len(ngram_cand_phrases)
#     print "# of unique ngram candidate phrases:", len(set(ngram_cand_phrases))
#     print "# of recognized phrases (from training):", len(ngram_phrases)
#     print "%:", len(ngram_phrases) * 1. / len(set(ngram_cand_phrases))


# # In[379]:

# if __name__ == '__main__':
#     '''get phrases in unigram sentences'''
#     mlp_cand_phrases = get_candidate_phrases(mlp_gen_sents)
#     mlp_phrases = lookup_phrases(train_phrases, mlp_cand_phrases)
#     mlp_phrase_values = get_binary_values(len(mlp_phrases), len(set(mlp_cand_phrases)))
#     print "# of mlp candidate phrases:", len(mlp_cand_phrases)
#     print "# of unique mlp candidate phrases:", len(set(mlp_cand_phrases))
#     print "# of recognized phrases (from training):", len(mlp_phrases)
#     print "%:", len(mlp_phrases) * 1. / len(set(mlp_cand_phrases))


# # In[380]:

# if __name__ == '__main__':
#     '''get phrases in unigram sentences'''
#     rnn_cand_phrases = get_candidate_phrases(rnn_gen_sents)
#     rnn_phrases = lookup_phrases(train_phrases, rnn_cand_phrases)
#     rnn_phrase_values = get_binary_values(len(rnn_phrases), len(set(rnn_cand_phrases)))
#     print "# of rnn candidate phrases:", len(rnn_cand_phrases)
#     print "# of unique rnn candidate phrases:", len(set(rnn_cand_phrases))
#     print "# of recognized phrases (from training):", len(rnn_phrases)
#     print "%:", len(rnn_phrases) * 1. / len(set(rnn_cand_phrases))


# # In[381]:

# if __name__ == '__main__':
#     '''get phrases in unigram sentences'''
#     gold_cand_phrases = get_candidate_phrases(gold_fin_sents)
#     gold_phrases = lookup_phrases(train_phrases, gold_cand_phrases)
#     gold_phrase_values = get_binary_values(len(gold_phrases), len(set(gold_cand_phrases)))
#     print "# of gold candidate phrases:", len(gold_cand_phrases)
#     print "# of unique gold candidate phrases:", len(set(gold_cand_phrases))
#     print "# of recognized phrases (from training):", len(gold_phrases)
#     print "%:", len(gold_phrases) * 1. / len(set(gold_cand_phrases))


# # In[382]:

# if __name__ == '__main__':    
#     p_unigram_ngram_phrases = evaluate_difference(unigram_phrase_values, ngram_phrase_values,
#                                                   num_trials=10000, verbose=False)
#     print "p-value for phrase differences between unigram and ngram sents:", p_unigram_ngram_phrases
    
#     p_ngram_mlp_phrases = evaluate_difference(ngram_phrase_values, mlp_phrase_values,
#                                               num_trials=10000, verbose=False)
#     print "p-value for phrase differences between ngram and mlp sents:", p_ngram_mlp_phrases
    
#     p_ngram_rnn_phrases = evaluate_difference(ngram_phrase_values, rnn_phrase_values,
#                                               num_trials=10000, verbose=False)
#     print "p-value for phrase differences between ngram and rnn sents:", p_ngram_rnn_phrases
    
#     p_mlp_rnn_phrases = evaluate_difference(mlp_phrase_values, rnn_phrase_values,
#                                             num_trials=10000, verbose=False)
#     print "p-value for phrase differences between mlp and rnn sents:", p_mlp_rnn_phrases
    
#     p_ngram_gold_phrases = evaluate_difference(ngram_phrase_values, gold_phrase_values,
#                                                num_trials=10000, verbose=False)
#     print "p-value for phrase differences between ngram and gold sents:", p_ngram_gold_phrases


# # In[84]:

# # if __name__ == '__main__':
# #     print "TEST GOLD SENTENCES VERSUS NGRAM SENTENCES:"
# #     compare_phrases(test_gold_phrases, ngram_gen_phrases)


# # In[88]:

# # if __name__ == '__main__':
# #     '''get phrases in 5-gram sentences'''
    
# #     import pdb;pdb.set_trace()
# #     #ngram_gen_sents = pandas.read_csv('ngram_sents3742_5.csv', encoding='utf-8', header=None).values.tolist()
# #     ngram_gen_phrases = get_phrases([sent for sents in ngram_gen_sents for sent in sents], threshold=1)
# #     #print phrases


# # # Word similarity across story

# # In[416]:

def get_word_pairs(context_seq, fin_sent):
    '''get all word pairs between context sequence and final sentence in story'''
    context_seq = Doc(context_seq, lang=u'en')
    fin_sent = Doc(fin_sent, lang=u'en')
    include_pos = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PRON', 'PROPN', 'VERB']
    context_seq = extract.words(context_seq, include_pos=include_pos)
    fin_sent = list(extract.words(fin_sent, include_pos=include_pos))
    pairs = []
    for context_token in context_seq:
        for fin_token in fin_sent:
            pairs.append((context_token, fin_token))
    return pairs

def get_word_matches(word_pairs):
    '''find all pairs of same words'''
    word_matches = [(word1, word2) for word1, word2 in word_pairs if word1.string == word2.string]
    return word_matches
    

def get_word_pair_dist(transformer, context_seqs, fin_sents):
    word_pairs = {}
    context_seqs, fin_sents = transformer.transform(context_seqs, fin_sents)
    for context_seq, fin_sent in zip(context_seqs, fin_sents):
        for context_word in context_seq:
            for fin_word in fin_sent:
                word_pair = (transformer.lexicon_lookup[context_word], 
                             transformer.lexicon_lookup[fin_word])
                if word_pair not in word_pairs:
                    word_pairs[word_pair] = 0
                word_pairs[word_pair] += 1
    return word_pairs

def get_sim_scores(word_pairs):
    sim_scores = {}
    for word_pair in word_pairs:
        word1, word2 = word_pair
        #sim_score = encoder(unicode(word1)).similarity(encoder(unicode(word2)))
        if word1 in model.vocab and word2 in model.vocab:
            sim_score = sim_model.similarity(word1, word2)
        else:
            sim_score = 0.0
        sim_scores[word_pair] = sim_score
    return sim_scores

def get_mean_sim(word_pairs, sim_scores):
    '''compute mean similarity between contexts and final sentences'''
    weighted_sim_scores = []
    for word_pair, count in word_pairs.items():
        weighted_score = sim_scores[word_pair] * count #scale by frequency of this pair
        weighted_sim_scores.append(weighted_score)
    total_pair_count = sum(word_pairs.values())
    mean_sim_score = sum(weighted_sim_scores) / total_pair_count
    return mean_sim_score

def get_sentence_similarity(gen_sent, gold_sent, use_skipthought=False):
    gen_sent = Doc(gen_sent, lang=u'en')
    gold_sent = Doc(gold_sent, lang=u'en')
    sim = similarity.word2vec(gen_sent, gold_sent)
    return sim

# def get_skipthought_similarity(context_sents, gen_sents):
#     encode_skipthought_seqs(seqs, encoder_module, encoder, encoder_dim=4800, 
#                             memmap=False, filepath='roc_context_sents')
#     encode_skipthought_seqs(seqs, encoder_module, encoder, encoder_dim=4800, memmap=False, filepath='context_sents')
            


# # In[420]:

# if __name__ == "__main__":
#     import models.transformer
#     reload(models.transformer)
#     from models.transformer import *
#     import pdb;pdb.set_trace()
#     skip_transformer = SkipthoughtsTransformer()


# # In[418]:

# if __name__ == "__main__":
#     '''compute sentence embedding similarity between generated sentences and final sentence'''

#     import pdb;pdb.set_trace()
    
#     sentence_sims = {model:{sent_idx:[] for sent_idx in range(len(context_sents[0]))} for model in model_gen_sents}
    
#     for model, gen_sents in model_gen_sents.items():
#         for story_idx, (story_context_sents, story_gen_sents) in enumerate(zip(context_sents, gen_sents)):
#             for context_sent_idx, context_sent in enumerate(story_context_sents):
#                 for gen_sent in story_gen_sents:
#                     sentence_sim = get_sentence_similarity(context_sent, gen_sent)
#                     sentence_sims[model][context_sent_idx].append(sentence_sim)
# #             if story_idx % 1000 == 0:
# #                 print "processed sentences up to story", story_idx
                
            
#         print "mean sentence similarity for", model, "model:"
#         mean_sim = []
#         for sent_idx, sent_sims in sentence_sims[model].items():
#             mean_sent_sim = numpy.mean(sent_sims[sent_idx])
#             print "\tsentence", sent_idx + 1, ":", mean_sent_sim
#             mean_sim.append(mean_sent_sim)
#         print "\ttotal:", numpy.mean(mean_sim)
        


# # In[250]:

# if __name__ == "__main__":
#     '''compute word embedding similarity between words in context and generated sentences in gold cloze set'''
    
#     #lm_transformer = load_lm('roc_lm97027_batchtrained_clio_updated').transformer
#     #import pdb;pdb.set_trace()
#     gold_word_pairs = []
#     gold_pair_scores = []
#     for context_seq, fin_sent in zip(context_seqs, gold_fin_sents):
#         word_pairs = get_word_pairs(context_seq, fin_sent)
#         pair_scores = [similarity.word2vec(word1,word2) for word1,word2 in word_pairs]
#         gold_word_pairs.extend(word_pairs)
#         gold_pair_scores.extend(pair_scores)
#     print "# of gold word pairs:", len(gold_word_pairs)
#     print "gold average similarity:", numpy.mean(gold_pair_scores)
    


# # In[252]:

# if __name__ == "__main__":
#     '''compute word similarity between words in context and generated sentences in unigram sentences'''
    
#     unigram_word_pairs = []
#     unigram_pair_scores = []
#     for context_seq, fin_sents in zip(context_seqs, unigram_gen_sents):
#         for fin_sent in fin_sents:
#             word_pairs = get_word_pairs(context_seq, fin_sent)
#             pair_scores = [similarity.word2vec(word1,word2) for word1,word2 in word_pairs]
#             unigram_word_pairs.extend(word_pairs)
#             unigram_pair_scores.extend(pair_scores)
#     print "# of unigram word pairs:", len(unigram_word_pairs)
#     print "unigram average similarity:", numpy.mean(unigram_pair_scores)
    


# # In[253]:

# if __name__ == "__main__":
#     '''compute word similarity between words in context and generated sentences in ngram sentences'''
    
#     #lm_transformer = load_lm('roc_lm97027_batchtrained_clio_updated').transformer
#     #import pdb;pdb.set_trace()
#     ngram_word_pairs = []
#     ngram_pair_scores = []
#     for context_seq, fin_sents in zip(context_seqs, ngram_gen_sents):
#         for fin_sent in fin_sents:
#             word_pairs = get_word_pairs(context_seq, fin_sent)
#             pair_scores = [similarity.word2vec(word1,word2) for word1,word2 in word_pairs]
#             ngram_word_pairs.extend(word_pairs)
#             ngram_pair_scores.extend(pair_scores)
#     print "# of ngram word pairs:", len(ngram_word_pairs)
#     print "ngram average similarity:", numpy.mean(ngram_pair_scores)
    


# # In[254]:

# if __name__ == "__main__":
#     '''compute word similarity between words in context and generated sentences in mlp sentences'''
    
#     mlp_word_pairs = []
#     mlp_pair_scores = []
#     for context_seq, fin_sents in zip(context_seqs, mlp_gen_sents):
#         for fin_sent in fin_sents:
#             word_pairs = get_word_pairs(context_seq, fin_sent)
#             pair_scores = [similarity.word2vec(word1,word2) for word1,word2 in word_pairs]
#             mlp_word_pairs.extend(word_pairs)
#             mlp_pair_scores.extend(pair_scores)
#     print "# of mlp word pairs:", len(mlp_word_pairs)
#     print "mlp average similarity:", numpy.mean(mlp_pair_scores)
    


# # In[265]:

# if __name__ == "__main__":
#     '''compute word embedding similarity between words in context and generated sentences in rnn sents'''
    
#     #lm_transformer = load_lm('roc_lm97027_batchtrained_clio_updated').transformer
#     import pdb;pdb.set_trace()
#     rnn_word_pairs = []
#     rnn_pair_scores = []
#     for context_seq, fin_sents in zip(context_seqs, rnn_gen_sents):
#         for fin_sent in fin_sents:
#             word_pairs = get_word_pairs(context_seq, fin_sent)
#             pair_scores = [similarity.word2vec(word1,word2) for word1,word2 in word_pairs]
#             rnn_word_pairs.extend(word_pairs)
#             rnn_pair_scores.extend(pair_scores)
#     print "# of rnn word pairs:", len(rnn_word_pairs)
#     print "rnn average similarity:", numpy.mean(rnn_pair_scores)


# # In[278]:

# if __name__ == "__main__":
#     '''stats tests for word embedding similarity'''
    
#     #import pdb;pdb.set_trace()
#     p_unigram_ngram_sim = evaluate_difference(unigram_pair_scores, ngram_pair_scores, num_trials=10000, verbose=True)
#     print "p-value for embedding similarity between unigram and ngram sents:", p_unigram_ngram_sim
    


# # In[279]:

# if __name__ == "__main__":
#     '''stats tests for word embedding similarity'''
    
#     #import pdb;pdb.set_trace
#     p_ngram_mlp_sim = evaluate_difference(ngram_pair_scores, mlp_pair_scores,  num_trials=10000, verbose=True)
#     print "p-value for embedding similarity between ngram and mlp sents:", p_ngram_mlp_sim
    


# # In[280]:

# if __name__ == "__main__":
#     '''stats tests for word embedding similarity'''
    
#     #import pdb;pdb.set_trace()
#     p_mlp_rnn_sim = evaluate_difference(mlp_pair_scores, rnn_pair_scores,  num_trials=10000, verbose=True)
#     print "p-value for embedding similarity between mlp and rnn sents:", p_mlp_rnn_sim
    


# # In[281]:

# if __name__ == "__main__":
#     '''stats tests for word embedding similarity'''
    
#     #import pdb;pdb.set_trace()
#     p_rnn_gold_sim = evaluate_difference(rnn_pair_scores, gold_pair_scores,  num_trials=10000, verbose=True)
#     print "p-value for embedding similarity between rnn and gold sents:", p_rnn_gold_sim
    


# # In[276]:

# if __name__ == "__main__":
#     '''stats tests for word embedding similarity'''
    
#     #import pdb;pdb.set_trace()
#     p_rnn_gold_sim = evaluate_difference(rnn_pair_scores, gold_pair_scores,  num_trials=10000, verbose=True)
#     print "p-value for embedding similarity between rnn and gold sents:", p_rnn_gold_sim
# #     p_rnn_gold_sim = evaluate_difference(unigram_pair_scores, gold_pair_scores,  num_trials=10000, verbose=True)
# #     print "p-value for embedding similarity between rnn and gold sents:", p_rnn_gold_sim
    


# # In[298]:

# if __name__ == "__main__":
#     '''Get number of shared words between context and ending'''
    
#     unigram_word_matches = get_word_matches(unigram_word_pairs)
#     print "# of matching unigram words:", len(unigram_word_matches)
#     print "# of total word pairs:", len(unigram_word_pairs)
#     print "% of matching unigram words:", len(unigram_word_matches) * 1. / len(unigram_word_pairs), "\n"
    
#     #import pdb;pdb.set_trace()
#     ngram_word_matches = get_word_matches(ngram_word_pairs)
#     print "# of matching ngram words:", len(ngram_word_matches)
#     print "# of total word pairs:", len(ngram_word_pairs)
#     print "% of matching ngram words:", len(ngram_word_matches) * 1. / len(ngram_word_pairs), "\n"
    
#     #import pdb;pdb.set_trace()
#     mlp_word_matches = get_word_matches(mlp_word_pairs)
#     print "# of matching mlp words:", len(mlp_word_matches)
#     print "# of total word pairs:", len(mlp_word_pairs)
#     print "% of matching mlp words:", len(mlp_word_matches) * 1. / len(mlp_word_pairs), "\n"
    
#     #import pdb;pdb.set_trace()
#     rnn_word_matches = get_word_matches(rnn_word_pairs)
#     print "# of matching rnn words:", len(rnn_word_matches)
#     print "# of total word pairs:", len(rnn_word_pairs)
#     print "% of matching rnn words:", len(rnn_word_matches) * 1. / len(rnn_word_pairs), "\n"
    
#     #import pdb;pdb.set_trace()
#     gold_word_matches = get_word_matches(gold_word_pairs)
#     print "# of matching gold words:", len(gold_word_matches)
#     print "# of total word pairs:", len(gold_word_pairs)
#     print "% of matching gold words:", len(gold_word_matches) * 1. / len(gold_word_pairs), "\n"


# # In[299]:

# def get_word_match_values(word_pairs, word_matches):
#     '''create a array where each matching word pair has a value of 1;
#     supply a value of 0 for each pair that is not a match'''
    
#     word_match_values = numpy.concatenate((numpy.ones((len(word_matches))), 
#                                            numpy.zeros((len(word_pairs) - len(word_matches)))))
#     return word_match_values

# if __name__ == "__main__":
#     '''stats tests for word embedding similarity'''
    
#     import pdb;pdb.set_trace()
#     unigram_match_values = get_word_match_values(unigram_word_pairs, unigram_word_matches)
#     ngram_match_values = get_word_match_values(ngram_word_pairs, ngram_word_matches)
#     mlp_match_values = get_word_match_values(mlp_word_pairs, mlp_word_matches)
#     rnn_match_values = get_word_match_values(rnn_word_pairs, rnn_word_matches)
#     gold_match_values = get_word_match_values(rnn_word_pairs, gold_word_matches)
#     p_unigram_ngram_matches = evaluate_difference(unigram_match_values, ngram_match_values, 
#                                                   num_trials=10000, verbose=True)
#     print "p-value for word match frequency between unigram and ngram sents:", p_unigram_ngram_matches
#     p_ngram_mlp_matches = evaluate_difference(ngram_match_values, mlp_match_values, 
#                                               num_trials=10000, verbose=True)
#     print "p-value for word match frequency between ngram and mlp sents:", p_ngram_mlp_matches
#     p_mlp_rnn_matches = evaluate_difference(mlp_match_values, rnn_match_values, 
#                                             num_trials=10000, verbose=True)
#     print "p-value for word match frequency between mlp and rnn sents:", p_mlp_rnn_matches
#     p_rnn_gold_matches = evaluate_difference(rnn_match_values, gold_match_values,
#                                              num_trials=10000, verbose=True)
#     print "p-value for word match frequency between rnn and gold sents:", p_rnn_gold_matches
    


# # In[53]:

# if __name__ == "__main__":
#     '''word pair similarity'''
#     # import pdb;pdb.set_trace()
#     # uni_mean_sim = get_mean_sim(uni_word_pairs, uni_sim_scores)
#     print uni_mean_sim
#     quin_mean_sim = get_mean_sim(quin_word_pairs, quin_sim_scores)
#     print quin_mean_sim
#     gold_mean_sim = get_mean_sim(gold_word_pairs, gold_sim_scores)
#     print gold_mean_sim


# # In[301]:

# if __name__ == "__main__":
#     '''find n-grams in generated sentences that also exist in google n-grams corpus'''
#     import pdb;pdb.set_trace()
# #     quadgrams = []
# #     for sents in unigram_gen_sents:
# #         for sent in sents:
# #             sent = Doc(sent, lang=u'en')
# #             sent_quadgrams = extract.ngrams(sent, n=1, filter_stops=False)
# #             print list(sent_quadgrams)
# #             quadgrams.extend([quadgram.string.lower().strip() for quadgram in sent_quadgrams])
# #     quadgrams = list(set(quadgrams))
# #     known_quadgrams = []
# #     chunk_size = 10
# #     for idx in range(0, len(quadgrams), chunk_size):
# #         #ngram_res = getNgrams(quadgram, 'eng_2012', 2007, 2008, 3, True)[-1]
# #         ngram_res = getNgrams(", ".join(quadgrams[idx:idx+chunk_size]), 'eng_2012', 2007, 2008, 3, False)[-1]
# #         print "given:", sorted(quadgrams[idx:idx+chunk_size])
# #         ngrams = list(ngram_res.keys())
# #         if len(ngrams) > 1:
# #             known_quadgrams.extend(ngrams[1:])
# #             print "recognized:", sorted(ngrams[1:]), "\n"
# #         if idx % 100 == 0 :
# #             print "checked", idx, "ngrams"
    
#     for gen_sents in unigram_gen_sents:
#         gen_sents = lm_transformer
            
    
    


# # In[176]:

# if __name__ == "__main__":
#     '''extract noun chunks from sentences'''
    
#     for fin_sent in fin_sents[:100]:
#         fin_sent = Doc(fin_sent, lang=u'en')
        
#         print list(noun_chunks(fin_sent))
#     print "\n\n\n"
#     for uni_sent in unigram_gen_sents[:100]:
#         uni_sent = Doc(uni_sent, lang=u'en')
#         print list(noun_chunks(uni_sent))


# # In[72]:

# if __name__ == "__main__":
#     '''analyze distributions of n-grams in generated sentences'''
# #     import pdb;pdb.set_trace()
# #     #get_ngram_dist(lm_transformer, quingram_gen_sents, n=2, filepath='quin_gen_ngrams.db')
#     top_quin_bigrams = get_top_ngrams(lm_transformer, n=2, filepath='quin_gen_ngrams.db')
#     print "top bigrams from quingram-generated sentences:"
#     for ngram, count in top_quin_bigrams:
#         print "{}\t{:.4f}".format(ngram, count)
#     print "\n"
        
#     #get_ngram_dist(lm_transformer, fin_sents, n=2, filepath='gold_ngrams.db')
#     top_gold_bigrams = get_top_ngrams(lm_transformer, n=2, filepath='gold_ngrams.db')
#     print "top bigrams from gold sentences:"
#     for ngram, count in top_gold_bigrams:
#         print "{}\t{:.4f}".format(ngram, count)


# # # Coreference resolution

# # In[138]:

def get_coreferences(stories):
    fin_sent_corefs = []
    prev_mention_idxs = []
    #get references in final sent
    for idx, story in enumerate(stories):
        sent_corefs = []
        sent_idxs = []
        #story = context + " " + fin_sent
        #print story
        try:
            parse = loads(corenlp_server.parse(story))
        except:
            print "error:", story
            parse = {}
#         except RPCInternalError:
#             parse = {}
        if 'coref' in parse:
            corefs = parse['coref']
            for ent in corefs:
                for mention_idx in range(len(ent)-1, -1, -1): 
                    last_mention = ent[mention_idx]
                    last_mention_idx = last_mention[0][1]
                    if last_mention_idx < 4: #coreferring entity not in final generated sentence
                        break
                    if mention_idx == 0:
                        prev_mention = last_mention
                        prev_mention_idx = prev_mention[1][1]
                        sent_corefs.append((prev_mention[1][0], last_mention[0][0]))
                    else:
                        prev_mention = ent[mention_idx-1]
                        prev_mention_idx = prev_mention[0][1]   
                        sent_corefs.append((prev_mention[0][0], last_mention[0][0]))
                    sent_idxs.append(prev_mention_idx)
        fin_sent_corefs.append(sent_corefs)
        prev_mention_idxs.append(sent_idxs)
        if idx % 500 == 0:
            print "processed", idx, "stories"
    assert(len(fin_sent_corefs) == len(prev_mention_idxs) == len(stories))
    return fin_sent_corefs, prev_mention_idxs

def count_coreferences(corefs, mention_idxs):
    mention_idxs_counts = {idx:0 for idx in range(5)}
    coref_count = 0
    for sent_corefs, sent_mention_idxs in zip(corefs, mention_idxs):
        coref_count += len(sent_corefs)
        for mention_idx in sent_mention_idxs:
            if mention_idx > 4:
                mention_idx = 4
            mention_idxs_counts[mention_idx] += 1
    return coref_count, mention_idxs_counts


# # In[141]:

# if __name__ == '__main__':
#     '''find coreferring entities between story context and gold final sentences'''
#     #import pdb;pdb.set_trace()
#     #corenlp = StanfordCoreNLP()
# #     corenlp_server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),
# #                              jsonrpc.TransportTcpIp(addr=("127.0.0.1", 8080)))
#     gold_stories = [context_seq + " " + fin_sent for context_seq, fin_sent in zip(context_seqs, gold_fin_sents)]
#     gold_corefs, gold_mention_idxs = get_coreferences(gold_stories)
#     coref_counts, mention_idxs_counts = count_coreferences(gold_corefs, gold_mention_idxs)
#     print "# of corefs in gold sentences:", coref_counts
#     for idx, count in mention_idxs_counts.items():
#         print "# of corefs that resolve in sentence", idx+1, ":", count


# # In[142]:

# if __name__ == '__main__':
#     '''find coreferring entities between story context and unigram final sentences'''
#     #import pdb;pdb.set_trace()
#     #corenlp = StanfordCoreNLP()
# #     corenlp_server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),
# #                              jsonrpc.TransportTcpIp(addr=("127.0.0.1", 8080)))
#     unigram_stories = [context_seq + " " + sent for context_seq, fin_sents in zip(context_seqs, unigram_gen_sents)
#                                                       for sent in fin_sents]
#     unigram_corefs, unigram_mention_idxs = get_coreferences(unigram_stories)
#     unigram_coref_counts, unigram_mention_idxs_counts = count_coreferences(unigram_corefs, unigram_mention_idxs)
#     print "# of corefs in unigram sentences:", unigram_coref_counts
#     for idx, count in unigram_mention_idxs_counts.items():
#         print "# of corefs that resolve in sentence", idx+1, ":", count


# # In[143]:

# if __name__ == '__main__':
#     '''find coreferring entities between story context and unigram final sentences'''
#     #import pdb;pdb.set_trace()
#     #corenlp = StanfordCoreNLP()
# #     corenlp_server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),
# #                              jsonrpc.TransportTcpIp(addr=("127.0.0.1", 8080)))
#     ngram_stories = [context_seq + " " + sent for context_seq, fin_sents in zip(context_seqs, ngram_gen_sents)
#                                                       for sent in fin_sents]
#     ngram_corefs, ngram_mention_idxs = get_coreferences(ngram_stories)
#     ngram_coref_counts, ngram_mention_idxs_counts = count_coreferences(ngram_corefs, ngram_mention_idxs)
#     print "# of corefs in ngram sentences:", ngram_coref_counts
#     for idx, count in ngram_mention_idxs_counts.items():
#         print "# of corefs that resolve in sentence", idx+1, ":", count


# # In[144]:

# if __name__ == '__main__':
#     '''find coreferring entities between story context and unigram final sentences'''
#     #import pdb;pdb.set_trace()
#     #corenlp = StanfordCoreNLP()
# #     corenlp_server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),
# #                              jsonrpc.TransportTcpIp(addr=("127.0.0.1", 8080)))
#     mlp_stories = [context_seq + " " + sent for context_seq, fin_sents in zip(context_seqs, mlp_gen_sents)
#                                                       for sent in fin_sents]
#     mlp_corefs, mlp_mention_idxs = get_coreferences(mlp_stories)
#     mlp_coref_counts, mlp_mention_idxs_counts = count_coreferences(mlp_corefs, mlp_mention_idxs)
#     print "# of corefs in mlp sentences:", mlp_coref_counts
#     for idx, count in mlp_mention_idxs_counts.items():
#         print "# of corefs that resolve in sentence", idx+1, ":", count


# # In[145]:

# if __name__ == '__main__':
#     '''find coreferring entities between story context and unigram final sentences'''
#     #import pdb;pdb.set_trace()
#     #corenlp = StanfordCoreNLP()
# #     corenlp_server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),
# #                              jsonrpc.TransportTcpIp(addr=("127.0.0.1", 8080)))
#     rnn_stories = [context_seq + " " + sent for context_seq, fin_sents in zip(context_seqs, rnn_gen_sents)
#                                                       for sent in fin_sents]
#     rnn_corefs, rnn_mention_idxs = get_coreferences(rnn_stories)
#     rnn_coref_counts, rnn_mention_idxs_counts = count_coreferences(rnn_corefs, rnn_mention_idxs)
#     print "# of corefs in rnn sentences:", rnn_coref_counts
#     for idx, count in rnn_mention_idxs_counts.items():
#         print "# of corefs that resolve in sentence", idx+1, ":", count


# # In[215]:

# def coreference_counts_to_freqs(corefs, mention_idxs, len_sents):
#     '''normalize coreference rates by dividing by sentence length'''
    
#     freq_corefs = []
#     freq_mention_idxs = {idx:[] for idx in range(5)}
#     for corefs, mention_idxs, len_sent in zip(corefs, mention_idxs, len_sents):
#         freq_corefs.append(len(corefs) * 1. / len_sent)
#         #print "corefs:", n_sent_corefs * 1. / len_sents
#         #print "counts per token:"
#         for idx in range(5):
#             freq_mention_idxs[idx].append(mention_idxs.count(idx) * 1. / len_sent)
#     return freq_corefs, freq_mention_idxs
    


# # In[216]:

# if __name__ == '__main__':
#     print "unigram results:"
#     #get standardized count of coreferences (divide by number of tokens)
#     unigram_freq_corefs, unigram_freq_mention_idxs = coreference_counts_to_freqs(unigram_corefs, 
#                                                                                  unigram_mention_idxs,
#                                                                                  len_unigram_sents)
#     import pdb;pdb.set_trace()
#     print "mean unigram coreference rate:", numpy.mean(unigram_freq_corefs)
#     for idx in unigram_freq_mention_idxs:
#         print "mean unigram coreference resolution rate in sentence", idx+1, ":",                                        numpy.mean(unigram_freq_mention_idxs[idx])


# # In[217]:

# if __name__ == '__main__':
#     print "ngram results:"
#     #get standardized count of coreferences (divide by number of tokens)
#     ngram_freq_corefs, ngram_freq_mention_idxs = coreference_counts_to_freqs(ngram_corefs,
#                                                                              ngram_mention_idxs,
#                                                                              len_ngram_sents)
#     #import pdb;pdb.set_trace()
#     print "mean ngram coreference rate:", numpy.mean(ngram_freq_corefs)
#     for idx in ngram_freq_mention_idxs:
#         print "mean ngram coreference resolution rate in sentence", idx+1, ":",                                        numpy.mean(ngram_freq_mention_idxs[idx])
    


# # In[218]:

# if __name__ == '__main__':
#     print "mlp results:"
#     #get standardized count of coreferences (divide by number of tokens)
#     mlp_freq_corefs, mlp_freq_mention_idxs = coreference_counts_to_freqs(mlp_corefs,
#                                                                              mlp_mention_idxs,
#                                                                              len_mlp_sents)
#     #import pdb;pdb.set_trace()
#     print "mean mlp coreference rate:", numpy.mean(mlp_freq_corefs)
#     for idx in mlp_freq_mention_idxs:
#         print "mean mlp coreference resolution rate in sentence", idx+1, ":",                                        numpy.mean(mlp_freq_mention_idxs[idx])
    


# # In[232]:

# if __name__ == '__main__':
#     print "rnn results:"
#     #get standardized count of coreferences (divide by number of tokens)
#     rnn_freq_corefs, rnn_freq_mention_idxs = coreference_counts_to_freqs(rnn_corefs,
#                                                                              rnn_mention_idxs,
#                                                                              len_rnn_sents)
#     #import pdb;pdb.set_trace()
#     print "mean rnn coreference rate:", numpy.mean(rnn_freq_corefs)
#     for idx in rnn_freq_mention_idxs:
#         print "mean rnn coreference resolution rate in sentence", idx+1, ":",                                        numpy.mean(rnn_freq_mention_idxs[idx])
    


# # In[233]:

# if __name__ == '__main__':
#     print "gold results:"
#     #get standardized count of coreferences (divide by number of tokens)
#     import pdb;pdb.set_trace()
#     gold_freq_corefs, gold_freq_mention_idxs = coreference_counts_to_freqs(gold_corefs,
#                                                                              gold_mention_idxs,
#                                                                              len_gold_sents)
#     print "mean gold coreference rate:", numpy.mean(gold_freq_corefs)
#     for idx in gold_freq_mention_idxs:
#         print "mean gold coreference resolution rate in sentence", idx+1, ":",                                        numpy.mean(gold_freq_mention_idxs[idx])
    


# # In[234]:

# if __name__ == '__main__':
#     '''stats tests for coreference frequencies'''
#     unigram_ngram_coref_p = evaluate_difference(unigram_freq_corefs, ngram_freq_corefs)
#     print "p value between unigram and ngram coreference rate:", unigram_ngram_coref_p
#     ngram_mlp_coref_p = evaluate_difference(ngram_freq_corefs, mlp_freq_corefs)
#     print "p value between ngram and mlp coreference rate:", ngram_mlp_coref_p
#     mlp_rnn_coref_p = evaluate_difference(mlp_freq_corefs, rnn_freq_corefs)
#     print "p value between mlp and rnn coreference rate:", mlp_rnn_coref_p
#     rnn_gold_coref_p = evaluate_difference(rnn_freq_corefs, gold_freq_corefs)
#     print "p value between rnn and gold coreference rate:", rnn_gold_coref_p


# # In[236]:

# if __name__ == '__main__':
#     '''stats tests for coreference frequencies by sentence position'''
#     for sent_idx in range(5):
#         print "stats tests for sentence position", sent_idx + 1
#         unigram_ngram_mention_p = evaluate_difference(unigram_freq_mention_idxs[sent_idx], 
#                                                       ngram_freq_mention_idxs[sent_idx])
#         print "\tp value between unigram and ngram coreference rate:", unigram_ngram_mention_p
        
#         ngram_mlp_mention_p = evaluate_difference(ngram_freq_mention_idxs[sent_idx], 
#                                                   mlp_freq_mention_idxs[sent_idx])
#         print "\tp value between ngram and mlp coreference rate:", ngram_mlp_mention_p
        
#         mlp_rnn_mention_p = evaluate_difference(mlp_freq_mention_idxs[sent_idx],
#                                                 rnn_freq_mention_idxs[sent_idx])
#         print "\tp value between mlp and rnn coreference rate:", mlp_rnn_mention_p
        
#         rnn_gold_mention_p = evaluate_difference(rnn_freq_mention_idxs[sent_idx],
#                                                  gold_freq_mention_idxs[sent_idx])
#         print "\tp value between rnn and gold coreference rate:", rnn_gold_mention_p
        


# # # Coherence scoring

# # In[247]:

# from gensim.models import TfidfModel, LsiModel, HdpModel
# from gensim import corpora
# from textacy.doc import Doc
# from textacy.corpus import Corpus
# from textacy import extract
# if __name__ == "__main__":
#     import pdb;pdb.set_trace()
#     train_stories = get_train_stories(filepath='ROC-Stories.tsv', flatten=True)                    + get_train_stories(filepath='ROCStories_winter2017.csv', flatten=True)
#     train_stories = [[word.text.lower() for word in extract.words(story, include_pos=['ADJ', 'ADV', 'NOUN', 'VERB'])]
#                              for story in Corpus(lang=u'en', texts=train_stories)]
#     train_dict = corpora.Dictionary(train_stories)
#     train_bow_stories = [train_dict.doc2bow(story) for story in train_stories]
#     tfidf = TfidfModel(train_bow_stories)
#     train_tfidf_stories = tfidf[train_bow_stories]
#     topic_model = HdpModel(train_tfidf_stories, id2word=train_dict)#, num_topics=250)
#     topic_model.save('topic_model.hdp')
#     stories_by_topic = {topic_idx:[] for topic_idx in range(lda_topic_model.num_topics)}
#     for story_idx in range(500):
#         topic_idxs, scores = zip(*lda_topic_model[train_bow_stories[story_idx]])
#         pred_topic_idx = topic_idxs[numpy.argmax(scores)]
#         stories_by_topic[pred_topic_idx].append(train_stories[story_idx])
# #         print train_stories[story_idx]
# #         print pred_topic_idx, lda_topic_model.show_topics(-1)[pred_topic_idx][1], "\n"
    


# # # Grammar scoring

# # In[164]:

# if __name__ == "__main__":
#     '''compute grammaticaly scores of generated sentences'''
    
# #     unigram_grammar_scores = call_lt([sent for sents in unigram_gen_sents for sent in sents])
# #     print "unigram grammar scores:", unigram_grammar_scores, "mean:", numpy.mean(unigram_grammar_scores)
    
# #     ngram_grammar_scores = call_lt([sent for sents in ngram_gen_sents for sent in sents])
# #     print "ngram grammar scores:", ngram_grammar_scores, "mean:", numpy.mean(ngram_grammar_scores)
    
# #     mlp_grammar_scores = call_lt([sent for sents in mlp_gen_sents for sent in sents])
# #     print "mlp grammar scores:", mlp_grammar_scores, "mean:", numpy.mean(mlp_grammar_scores)
    
#     rnn_grammar_scores = call_lt([sent for sents in rnn_gen_sents for sent in sents])
#     print "rnn grammar scores:", rnn_grammar_scores, "mean:", numpy.mean(rnn_grammar_scores)
    
# #     gold_grammar_scores = call_lt(gold_fin_sents)
# #     print "gold grammar scores:", gold_grammar_scores, "mean:", numpy.mean(gold_grammar_scores)


# # In[167]:

# import models.stats
# reload(models.stats)
# from models.stats import *

# if __name__ == "__main__":
#     '''statistically compare mean grammaticality between models'''
# #     unigram_ngram_p = evaluate_difference(unigram_grammar_scores, ngram_grammar_scores)
# #     print "p-value difference between unigram and ngram sentences:", unigram_ngram_p
    
# #     ngram_mlp_p = evaluate_difference(ngram_grammar_scores, mlp_grammar_scores)
# #     print "p-value difference between ngram and mlp sentences:", ngram_mlp_p
    
# #     mlp_rnn_p = evaluate_difference(mlp_grammar_scores, rnn_grammar_scores)
# #     print "p-value difference between mlp and rnn sentences:", mlp_rnn_p
#     import pdb;pdb.set_trace()
#     rnn_gold_p = evaluate_difference(rnn_grammar_scores, gold_grammar_scores)
#     print "p-value difference between rnn and gold sentences:", rnn_gold_p


# # In[ ]:

# if __name__ == "__main__":
#     '''analyze distributions of bigrams in generated sentences'''
# #     uni_gen_sent_bigrams = get_ngram_dist(lm.transformer, unigram_gen_sents, n=2)
# #     top_bi_unigrams = get_top_ngrams(lm.transformer, uni_gen_sent_bigrams)
# #     print "top bigrams from unigram-generated sentences:"
# #     for ngram, prob in top_bi_unigrams:
# #         print "{}\t{}".format(ngram, prob)
        
# #     bi_gen_sent_bigrams = get_ngram_dist(lm.transformer, bigram_gen_sents, n=2)
# #     top_bi_bigrams = get_top_ngrams(lm.transformer, bi_gen_sent_bigrams)
# #     print "top bigrams from bigram-generated sentences:"
# #     for ngram, prob in top_bi_unigrams:
# #         print "{}\t{}".format(ngram, prob)


# # In[ ]:

# def train_decoder():
    
#     '''train an encoder-decoder RNN'''
    
#     train_stories = get_train_stories(filepath='ROC-Stories.tsv')[:10000]
#     filepath = 'roc_lm' + str(len(train_stories)) + '_decoder'
#     max_sent_length = max([len(tokenize(seq[-1])) for seq in train_stories])
#     lm = RNNPipeline(steps=[('transformer', SequenceTransformer(min_freq=1, verbose=1, filepath=filepath)),
#                             ('classifier', RNNLM(verbose=1, batch_size=50, max_sent_length=max_sent_length, 
#                                                  n_hidden_layers=1,
#                                                  n_embedding_nodes=100, n_hidden_nodes=200, separate_context=True,
#                                                  filepath=filepath))])
#     n_epochs = 50
#     for epoch in range(n_epochs):
#         print "training epoch {}/{}...".format(epoch + 1, n_epochs)
#         #import pdb;pdb.set_trace()
#         lm.fit(X=train_stories)
#         #generate samples to show progress
#         samp_size = 10
#         temp = 0.75
#         samp_stories = random.sample(train_stories, samp_size)
#         context_seqs = [story[:-1] for story in samp_stories]
#         gold_sents = [story[-1] for story in samp_stories]
#         #         p_gold_sents = lm.predict(X=context_seqs, y_seqs=gold_sents, batch_size=samp_size)
#         import pdb;pdb.set_trace()
#         gen_sents, p_sents = generate_sents(lm, context_seqs, n_best=1, mode='random', batch_size=samp_size)
#         print "MAX PROB SENTENCES:"
#         show_gen_sents(context_seqs, gen_sents, p_sents)#, gold_sents, p_gold_sents)
# #         gen_sents, p_sents = generate_sents(lm, context_seqs, n_best=1, mode='random', 
# #                                             temp=temp, batch_size=samp_size)
# #         print "SENTENCES WITH TEMP =", temp
# #         show_gen_sents(context_seqs, gen_sents, p_sents)



