
# coding: utf-8

# In[1]:

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


# In[2]:

def train_lm(lm, story_ids, n_epochs=5, n_chunks=3, update_transformer=True):
    n_stories_per_chunk = len(story_ids) / n_chunks
    if update_transformer: #transformer should be fit further; otherwise it is already fit
        for chunk_idx in range(0, len(story_ids), n_stories_per_chunk):
            stories = get_stories(story_ids[chunk_idx:chunk_idx + n_stories_per_chunk])
            lm.transformer.fit(X=stories) #make lexicon from all stories
    #import pdb;pdb.set_trace()
    for epoch in range(n_epochs):
        print "training epoch {}/{}...".format(epoch + 1, n_epochs)
        for chunk_idx in range(0, len(story_ids), n_stories_per_chunk):
            stories = get_stories(story_ids[chunk_idx:chunk_idx + n_stories_per_chunk])
            lm.fit(X=stories)
            #generate samples to show progress
            context_sents = [u"Last night I had a crazy dream.",
                             u"A strange thing happened on my way home yesterday.",
                             u"I had the most awkward dinner of my life last night.",
                             u"I received a surprising phone call yesterday.",
                             u"Last week some old friends came in town for a visit."]
            #import pdb;pdb.set_trace()
            gen_sents, p_sents = generate_sents(lm, context_sents, n_best=1, mode='random', 
                                                temp=0.4, batch_size=1)
            print "\n"
            for context_sent, sent in zip(context_sents, gen_sents):
                print context_sent + " " + sent
            print "\n"


# In[3]:

if __name__ == "__main__":
    with open('fiction_ids_filtered.txt', 'r') as id_file:
        story_ids = [int(story_id) for story_id in id_file.readlines()]
#     story_ids = filter_nonstory_ids(story_ids, min_length=125) #remove ids with null, spam, or duplicate entries
    n_stories = len(story_ids)
    story_ids = story_ids[-n_stories:]
    story_ids = sort_ids_by_len(story_ids)
    filepath = 'fiction_lm' + str(len(story_ids))
    import pdb;pdb.set_trace()
    if os.path.exists(filepath + '/transformer.pkl'):
        #load existing transformer
        transformer = load_transformer(filepath)
        update_transformer = False
    else:
        transformer = SequenceTransformer(min_freq=25, verbose=1, filepath=filepath)
        update_transformer = True
    if os.path.exists(filepath + '/classifier.pkl') and os.path.exists(filepath + '/classifier.h5'):
        #load existing transformer
        classifier = load_classifier(filepath)
    else:
        classifier = RNNLM(verbose=1, batch_size=50, n_timesteps=25, n_hidden_layers=2,
                           n_embedding_nodes=300, n_hidden_nodes=500, filepath=filepath)
    lm = RNNPipeline(transformer, classifier)
    train_lm(lm, story_ids, n_epochs=15, n_chunks=100, update_transformer=False)


# In[19]:

# '''count total number of words in corpus'''

# if __name__ == "__main__":
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



