
# coding: utf-8

# In[94]:

import sys, pandas
sys.path.append('skip-thoughts-master/')
import skipthoughts

#import pdb;pdb.set_trace()
import models.pipeline
reload(models.pipeline)
from models.pipeline import *


import models.transformer
reload(models.transformer)
from models.transformer import *

import models.classifier
reload(models.classifier)
from models.classifier import *

import story_corpus
reload(story_corpus)
from story_corpus import *

import models.similarity
reload(models.similarity)
from models.similarity import *

import models.ngram
reload(models.ngram)
from models.ngram import *

sys.path.append('DINE/')
from dine import show_predictions


# In[74]:

rng = numpy.random.RandomState(0)

def get_cbt_seqs(filepaths):
    if type(filepaths) not in (list,tuple):
        filepaths = [filepaths] #single file
    context_seqs = []
    gold_sents = []
    for filepath in filepaths:
        with open(filepath, 'r') as fp:
            context_seq = []
            for sent in fp:
                sent = unicode(sent.strip(), errors='replace')
                if not sent:
                    continue
                num_split_idx = sent.index(" ")
                sent_num, sent = sent[:num_split_idx], sent[num_split_idx + 1:]
                if sent_num == '21':
                    #prediction (gold) sentence
                    sent, missing_word, _, _ = sent.split("\t")
                    sent = sent.replace("XXXXX", missing_word)
                    gold_sents.append(sent)
                    context_seqs.append(context_seq)
                    context_seq = []
                    continue
                else:
                    context_seq.append(sent)
    return context_seqs, gold_sents
            
                
            


# # In[22]:

if __name__ == '__main__':
    '''generate sentences for CBT stories'''
    context_seqs, gold_sents = get_cbt_seqs(["CBTest/data/cbtest_CN_test_2500ex.txt",                                             "CBTest/data/cbtest_CN_valid_2000ex.txt",                                             "CBTest/data/cbtest_NE_test_2500ex.txt",                                             "CBTest/data/cbtest_NE_valid_2000ex.txt",                                             "CBTest/data/cbtest_P_test_2500ex.txt",                                             "CBTest/data/cbtest_P_valid_2000ex.txt",                                             "CBTest/data/cbtest_V_test_2500ex.txt",                                             "CBTest/data/cbtest_V_valid_2000ex.txt"])
    #context_seqs = [seq[:-1] for seq in seqs[-25:]]
    n_gen_sents = 1
    with open('cbt_cap_ents.pkl', 'rb') as f:
        cap_ents = pickle.load(f)
    # print context_seqs[-1]
    # print gold_sents[-1]
    
    #     context_sents = [u"Last night I had a crazy dream.",
#                  u"A strange thing happened on my way home yesterday.",
#                  u"I had the most awkward dinner of my life last night.",
#                  u"I received a surprising phone call yesterday.",
#                  u"Last week some old friends came in town for a visit."]


# # In[95]:

# def gen_unigram_seqs(n_seqs, word_counts, eos_tokens=[".", "!", "?"], max_length=35):
#     '''use the unigram counts collected by the RNN transformer to generate sents according to unigram probs'''
    
#     #unigram_counts, unigrams = get_ngram_counts(filepath + '.db', n=1)
#     words, counts = zip(*word_counts.items())
#     counts = numpy.array(counts)
#     count_sum = counts.sum() #backoff count will always be the same for unigrams
#     probs = counts.astype('float32') / count_sum

#     pred_seqs = []
    
#     #seqs, _ = transformer.transform(seqs)
    
#     for seq_idx in range(n_seqs):
#         pred_seq = []
#         pred_token = None
#         while pred_token not in eos_tokens and len(pred_seq) < max_length:
#             pred_idx = rng.choice(a=len(probs), p=probs)
#             pred_token = words[pred_idx]
#             pred_seq.append(pred_token)
#         pred_seq = detokenize_sent(pred_seq, eos_tokens)
#         pred_seqs.append(pred_seq)
#         if seq_idx % 1000 == 0:
#             print "generated", seq_idx, "sequences..."
    
#     return pred_seqs

# def get_rand_seqs(n_seqs, db_filepath):
#     '''pick a random sentence from the db as each generated sentence'''
#     n_choices = get_n_seqs(db_filepath)
#     rand_idxs = rng.randint(1, n_choices + 1, size=n_seqs) #db indices start at 1
#     rand_seqs = []
#     chunk_size = 10000
#     for chunk_idx in range(0, len(rand_idxs), chunk_size):
#         rand_seqs.extend(get_seqs(rand_idxs[chunk_idx:chunk_idx+chunk_size], db_filepath))
#     assert(len(rand_seqs) == n_seqs)
#     return rand_seqs


# In[33]:

#word_counts = {word:count for word, count in rnnlm.transformer.word_counts.items() if word in rnnlm.transformer.lexicon}


# In[84]:

# for i in range(50):
#     import pdb;pdb.set_trace()
#     print len(get_rand_seqs(n_seqs=len(context_seqs), db_filepath='fiction_lm607627_3epoch/sents.db'))


# In[85]:

# if __name__ == "__main__":
#     '''generate sents with unigram probs'''
#     #import pdb;pdb.set_trace()
#     cbrrand_gen_sents = get_rand_seqs(n_seqs=len(context_seqs), db_filepath='fiction_lm607627_3epoch/sents.db')
#     cbrrand_gen_sents = [cbrrand_gen_sents[idx:idx + n_gen_sents] 
#                          for idx in range(0, len(context_seqs) * n_gen_sents, n_gen_sents)]
#     pandas.DataFrame(cbrrand_gen_sents).to_csv('cbt_cbrrand_sents' + str(len(context_seqs))
#                                                + '_' + str(n_gen_sents) + '.csv', header=False, 
#                                                index=False, encoding='utf-8') #save generated sents to file


# # In[96]:

# if __name__ == "__main__":
#     '''generate sents with unigram probs'''
#     import pdb;pdb.set_trace()
#     unigram_gen_sents = gen_unigram_seqs(n_seqs=len(context_seqs), word_counts=word_counts, max_length=60)
#     unigram_gen_sents = [unigram_gen_sents[idx:idx + n_gen_sents] 
#                          for idx in range(0, len(context_seqs) * n_gen_sents, n_gen_sents)]
#     pandas.DataFrame(unigram_gen_sents).to_csv('cbt_unigram_sents' + str(len(context_seqs))
#                                                + '_' + str(n_gen_sents) + '.csv', header=False, 
#                                                index=False, encoding='utf-8') #save generated sents to file


# # In[12]:

# if __name__ == '__main__':
#     with open('fiction_ids_filtered.txt', 'r') as id_file:
#         story_ids = [int(story_id) for story_id in id_file.readlines()]
#     #story_ids = filter_nonstory_ids(story_ids, min_length=125) #remove ids with null, spam, or duplicate entries
#     n_stories = len(story_ids)
#     story_ids = story_ids[-n_stories:]
#     filepath = 'fiction_lm' + str(len(story_ids)) + "_3epoch"
#     import pdb;pdb.set_trace()
#     #split stories into sentences, store in table
#     index_prefix = "sents"
#     if not os.path.exists(filepath + "/" + index_prefix + ".db"):
#         if not os.path.isdir(filepath):
#             os.mkdir(filepath)
#         n_sents = make_seqs_table(story_ids, n_sents=1, db_filepath=filepath + "/" + index_prefix + ".db")
#     else:
#         #get # of sents
#         n_sents = get_n_seqs(db_filepath=filepath + "/" + index_prefix + ".db")
#     retrieval_index = RetrievalIndex(dirname=filepath, fileprefix=index_prefix, 
#                                      seq_ids=numpy.arange(n_sents) + 1)
#     context_sents = [seq[-1] for seq in context_seqs][:10]
#     gen_sents = []
#     batch_size = 5
#     sim_ids = []
#     for batch_idx in range(0, len(context_sents), batch_size):
#         batch_context_sents = context_sents[batch_idx:batch_idx+batch_size]
#         batch_sim_ids = retrieval_index.get_similar_ids(seqs=[batch_context_sents[idx] for idx in                                                        numpy.arange(len(batch_context_sents)).repeat(n_gen_sents)], 
#                                                         n_best=5)
#         print "generated", batch_idx + batch_size, "sequences..."
#         sim_ids.extend(batch_sim_ids)
#     for (context_sent, sent_sim_ids) in zip(context_sents, sim_ids):
#         for sim_id in sent_sim_ids:
#             sim_sent = get_seqs(sim_id, db_filepath=filepath + "/" + index_prefix + ".db")
#             next_sent = get_next_seq(seq_id=sim_id, db_filepath=filepath + "/" + index_prefix + ".db")
#             if next_sent:
#                 break
#         assert(next_sent is not None)
# #         print "sent:", context_sent
# #         print "most similar sent:", sim_sent
# #         print "retrieved sent:", next_sent, "\n"
#         gen_sents.append(next_sent)
        
#     gen_sents = [gen_sents[idx:idx + n_gen_sents] 
#                  for idx in range(0, len(context_seqs) * n_gen_sents, n_gen_sents)]
        
#     pandas.DataFrame(gen_sents).to_csv('cbt_cbr_sents' + str(len(context_seqs))                                       + '_' + str(n_gen_sents) + '.csv', header=False,
#                                        index=False, encoding='utf-8') #save generated sents to file
        
        


# In[53]:

if __name__ == "__main__":
    #import pdb;pdb.set_trace()
# #     story_ids = sort_ids_by_len(story_ids)
# #     filepath = 'fiction_lm' + str(len(story_ids))
#     if os.path.exists(filepath + '/transformer.pkl'):
#         #load existing transformer
#         transformer = load_transformer(filepath)
#         update_transformer = False
#     else:
#         transformer = SequenceTransformer(min_freq=1, verbose=1, filepath=filepath)
#         update_transformer = True
#     if os.path.exists(filepath + '/classifier.pkl') and os.path.exists(filepath + '/classifier.h5'):
#         #load existing transformer
#         classifier = load_classifier(filepath)
#     else:
#         classifier = RNNLM(verbose=1, batch_size=25, n_timesteps=10, n_hidden_layers=2,
#                            n_embedding_nodes=300, n_hidden_nodes=500, filepath=filepath)
#     lm = RNNPipeline(transformer, classifier)
    rnnlm = load_rnn_pipeline('fiction_lm607627')


# In[33]:

# context_seqs = [u"I lived on Henry Avenue. The neighborhood began in the late 1800s. The first house in the area was a large, yellow farm house with a white wrap-around porch. A boy named Henry lived in the farmhouse that overlooked the Mississippi River. Henry loved to go to the river, but his mother forbade him from ever playing there alone. It was dangerous. The currents were strong and the undertow caused even the strongest swimmers to drown. One day, Henry left the house. His mother called out \"Don't go to the river Henry!\", but he ignored her. He never returned. His body was never found.", 
#                u"My friend Mallory was going to spend the night at my house. Her dad, Officer Barrett, was the Fort Wade Chief of Police. Mallory loved hanging out at our house because my parents weren\'t that strict, so it was an awesome break from all her dad\'s rules. We were watching TV in our family room and my parents were already asleep. My older sister Jenny and our neighbor Mary Frances quietly came down the wooden staircase from Jenny\'s bedroom and started walking out the door.\
#                \"Wait! Where are y\'all going?\" I asked. It was getting late.\
#                 \"We\'re just going to the Dairy Bar,\" Jenny\'s keys jingled as she grabbed them off the hook."]
# context_seqs = [segment(seq) for seq in context_seqs]
# print context_seqs[-1][-1]


# In[20]:

if __name__ == '__main__':
    '''generate sentences with RNN'''
    import pdb;pdb.set_trace()
    #context_seqs = context_seqs[:100]
    gen_sents, p_sents = generate_sents(rnnlm, [context_seqs[idx] for idx in 
                                                numpy.arange(len(context_seqs)).repeat(n_gen_sents)], 
                                        n_best=1, max_length=60, mode='random', temp=0.6, batch_size=1000,
                                        detokenize=True, cap_ents=[cap_ents[idx] for idx in numpy.arange(len(context_seqs)).repeat(n_gen_sents)])
    gen_sents = [gen_sents[idx:idx + n_gen_sents] 
                 for idx in range(0, len(context_seqs) * n_gen_sents, n_gen_sents)]
    
    pandas.DataFrame(gen_sents).to_csv('cbt_rnn_sents' + str(len(context_seqs))
                                        + '_' + str(n_gen_sents) + '.csv', header=False, 
                                        index=False, encoding='utf-8') #save generated sents to file
#     print "\n"
#     for context_seq, sents in zip(context_seqs, gen_sents):
#         print "CONTEXT:", " ".join(context_seq)
#         for sent in sents:
#             print "GENERATED:", sent
#         print "\n"


# # In[2]:

# def train_lm(lm, story_ids, n_epochs=5, n_chunks=3, update_transformer=True):
#     n_stories_per_chunk = len(story_ids) / n_chunks
#     if update_transformer: #transformer should be fit further; otherwise it is already fit
#         for chunk_idx in range(0, len(story_ids), n_stories_per_chunk):
#             stories = get_stories(story_ids[chunk_idx:chunk_idx + n_stories_per_chunk])
#             lm.transformer.fit(X=stories) #make lexicon from all stories
#     #import pdb;pdb.set_trace()
#     for epoch in range(n_epochs):
#         print "training epoch {}/{}...".format(epoch + 1, n_epochs)
#         for chunk_idx in range(0, len(story_ids), n_stories_per_chunk):
#             stories = get_stories(story_ids[chunk_idx:chunk_idx + n_stories_per_chunk])
#             lm.fit(X=stories)
#             #generate samples to show progress
#             context_sents = [u"Last night I had a crazy dream.",
#                              u"A strange thing happened on my way home yesterday.",
#                              u"I had the most awkward dinner of my life last night.",
#                              u"I received a surprising phone call yesterday.",
#                              u"Last week some old friends came in town for a visit."]
#             #import pdb;pdb.set_trace()
#             gen_sents, p_sents = generate_sents(lm, context_sents, n_best=1, mode='random', 
#                                                 temp=0.4, batch_size=1)
#             print "\n"
#             for context_sent, sent in zip(context_sents, gen_sents):
#                 print context_sent + " " + sent
#             print "\n"


# # In[ ]:

# context_sents = [u"Last night I had a crazy dream. My brother was in it. ",
#                  u"A strange thing happened on my way home yesterday. A girl approached me. ",
#                  u"I had the most awkward dinner of my life last night. I was with Grey.",
#                  u"Harry received a surprising phone call yesterday from Edward.",
#                  u"Last week Bella came in town for a visit."]
# shuffle(context_sents)
# context_sents = context_sents * 5
# #     context_sents = [u"My friend Mallory was going to spend the night at my house. Her dad, Officer Barrett, was the Fort Wade Chief of Police. Mallory loved hanging out at our house because my parents weren't that strict, so it was an awesome break from all her dad’s rules. We were watching TV in our family room and my parents were already asleep.\
# #                         My older sister Jenny and our neighbor Mary Frances quietly came down the wooden staircase from Jenny’s bedroom and started walking out the door.",
# #                     u"\"Will you recount what happened on the afternoon of July 17th?\" The Judge shifted her gaze towards me from her high place behind the metal bench in the dimly lit courtroom.",
# #                     u"The hallway stretches out long and winding in front of you. It's lined with screens on either side. All of them are blank but as you begin walking down the hall, one after the other: they become illuminated.",
# #                     u"I liked to think that I was a great parent. I taught my kids necessary life skills like swimming and how to ride a bike. I sat in the front row of all their recitals and concerts.",
# #                     u"I lived on Henry Avenue. The neighborhood began in the late 1800s. The first house in the area was a large, yellow farm house with a white wrap-around porch. A boy named Henry lived in the farmhouse that overlooked the Mississippi River.",
# #                     u"Throughout my entire life, I had been advised to abandon my dream and to find a career industry more reliable than fashion design. For years, I had felt as if no one was on my side and that no one believed in me. Yet the lack of support didn't hinder my pursuits, but became a chip on my shoulder that, in fact, spurred me into the daunting field of fashion.",
# #                     u"It was a bad business. There weren't many of us left. Saskia had a little shop now on Grozny Street selling hats and pinafores. Frivolities.",
# #                     u"I had to get my rig to Albuquerque before the shop opened at 8AM the next morning. Mendleton was already riding me hard for late deliveries. One more bad turnover and I was out of a job.",
# #                     u"I faced the control panel. To my right was the button. Hanging above me was the lever. In front of me was the speaker. In the balance: the lives of eleven people.",
# #                     u"It all got a little out of hand. I can admit that now. I was at the conservatory on scholarship to play piano.",
# #                     u"\Hi, I'm Jordan,\" a girl with a long blonde ponytail greeted me in the lobby of the fitness research center.",
# #                     ]


# # In[19]:

# if __name__ == "__main__":
#     '''count total number of words in corpus'''
#     import pdb;pdb.set_trace()
# #     with open('fiction_ids_filtered.txt', 'r') as id_file:
# #         story_ids = [int(story_id) for story_id in id_file.readlines()]
#     print len(story_ids), "stories"
#     n_words = 0
#     chunk_size=50000
#     for story_idx in range(0, len(story_ids), chunk_size):
#         for idx, story in enumerate(get_stories(story_ids[story_idx:story_idx+chunk_size])):
#             #n_words += len(segment_and_tokenize(story))
#             n_words += len(story.split())
#         print "processed", story_idx+chunk_size, "stories:", n_words, "words"
    
#     print "total words:", n_words


# # In[24]:

# '''count total number of words in corpus'''

# if __name__ == "__main__":
#     import pdb;pdb.set_trace()
    
# #     with open('wordpress_blog_ids.txt', 'r') as id_file:
# #         story_ids = [int(story_id) for story_id in id_file.readlines()]
# #     story_ids = filter_nonstory_ids(story_ids) #remove ids with null, spam, or duplicate entries
# #     with open('wordpress_blog_ids_filtered.txt', 'w') as id_file:
# #         id_file.write("\n".join(map(str, story_ids)))
        
#     with open('wordpress_blog_ids_filtered.txt', 'r') as id_file:
#         story_ids = [int(story_id) for story_id in id_file.readlines()]
#     print len(story_ids), "stories"
#     n_words = 0
#     chunk_size=50000
#     for story_idx in range(0, len(story_ids), chunk_size):
#         for idx, story in enumerate(get_stories(story_ids[story_idx:story_idx+chunk_size])):
#             #n_words += len(segment_and_tokenize(story))
#             n_words += len(story.split())
#         print "processed", story_idx+chunk_size, "stories:", n_words, "words"
    
#     print "total words:", n_words


# # In[85]:

# if __name__ == '__main__':
#     import pdb;pdb.set_trace()
#     sent = u"I walked to school yesterday."
#     gen_sents, p_sents = generate_sents(lm, sent, n_best=1, mode='random', 
#                                                 temp=0.5, batch_size=1)


# # In[85]:

# if __name__ == '__main__':
#     import pdb;pdb.set_trace()
#     sent = u"I walked to school yesterday."
#     gen_sents, p_sents = generate_sents(lm, sent, n_best=1, mode='random', 
#                                                 temp=0.5, batch_size=1)


# # In[63]:

# if __name__ == '__main__':
#     with open('fiction_ids_filtered.txt', 'w') as id_file:
#         id_file.write("\n".join(map(str, story_ids)))


# # In[ ]:



