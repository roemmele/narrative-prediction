
# coding: utf-8

# In[16]:

import sys, random
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

import roc
reload(roc)
from roc import get_train_stories

from models.pmi import PMI_Model as PMI_Model

sys.path.append('../')
import models.transformer
reload(models.transformer)
from models.transformer import *
import models.pipeline
reload(models.pipeline)
from models.pipeline import RNNPipeline
import models.classifier
reload(models.classifier)
from models.classifier import *

sys.path.append("../AvMaxSim")
import similarity_score


# In[5]:

def load_lm(filepath):
    #import pdb;pdb.set_trace()
    #transformer, classifier = load_pipeline(filepath)
    transformer = load_transformer(filepath)
    transformer.sent_encoder = None
    classifier = load_classifier(filepath)
    lm = RNNPipeline(steps=[('transformer', transformer), ('classifier', classifier)])
    return lm

def generate_sents(lm, context_seqs, batch_size=1, n_best=1, n_words=25, 
                   mode='max', temp=1.0, eos_markers=[".", "!", "?", "\""], decode=True):
    #import pdb;pdb.set_trace()
    eos_idxs = lm.named_steps['transformer'].lookup_eos(eos_markers)
    if decode:
        gen_sents, p_sents = lm.predict(X=context_seqs, mode=mode, batch_size=batch_size, 
                                        n_best=n_best, n_words=n_words, temp=temp, eos_markers=eos_idxs)
        #convert from indices to text
        gen_sents = [lm.named_steps["transformer"].decode_seqs(sents) for sents in gen_sents]
    else:
        gen_sents, p_sents = lm.named_steps['classifier'].predict(X=context_seqs, mode=mode, 
                                                                  batch_size=batch_size, 
                                                                  n_best=n_best, n_words=n_words, 
                                                                  temp=temp, eos_markers=eos_idxs)
    return gen_sents, p_sents

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
        #context = context_seqs[seq_idx]
#         if type(context) in [tuple, list]:
#             context = " ".join(context_seqs[seq_idx])
        print "STORY:", context_seqs[seq_idx]
        if gold_sents:
            print "GOLD:", gold_sents[seq_idx], "({:.3f})".format(p_gold_sents[seq_idx])
        if gen_sents:
    #         for sent, p_sent in zip(gen_sents[seq_idx], p_gen_sents[seq_idx]):
            print "PRED:", gen_sents[seq_idx], "({:.3f})".format(p_gen_sents[seq_idx])
        print "\n"


# In[29]:

if __name__ == "__main__":
    #lm = load_lm('roc_lm97027_batchtrained_clio_updated')
#     train_stories = get_train_stories(filepath='ROC-Stories.tsv')[:100]
#     context_seqs = [" ".join(story[:-1]) for story in train_stories]
#     filepath = 'roc_pmi97027minfreq1'
#     transformer = load_transformer(filepath)
#     pmi_model = PMI_Model(filepath)
    #embeddings = similarity_score.load_model('../AvMaxSim/vectors')
    for context_seq in context_seqs[30:45]:
        gen_sents, _ = generate_sents(lm, [context_seq] * 5, n_best=1,
                                      batch_size=5, mode='random', temp=0.2, n_words=35)
#         pmi_scores = eval_pmi_sents(transformer, pmi_model, context_seqs[0], gen_sents)
#         show_gen_sents([context_seq] * 5, [[sent] for sent in gen_sents], pmi_scores)
        #import pdb;pdb.set_trace()
        avemax_scores = eval_pmi_sents(transformer, embeddings, context_seqs[0], gen_sents, eval_mode='avemax')
        show_gen_sents([context_seq] * 5, [[sent] for sent in gen_sents], avemax_scores)


# In[8]:

if __name__ == "__main__":
    lm = load_lm('roc_lm97027_batchtrained_clio')
    train_stories = get_train_stories(filepath='ROCStories_winter2017.csv') #get_train_stories(filepath='ROC-Stories.tsv')[:100]
    samp_stories = train_stories[-20:]
    context_seqs = [story[:-1] for story in samp_stories]
    #import pdb;pdb.set_trace()
    gen_sents, p_sents = generate_sents(lm, context_seqs, n_best=1, batch_size=1, mode='random', temp=0.5, n_words=35)
    show_gen_sents(context_seqs, gen_sents, p_sents)


# In[15]:

if __name__ == "__main__":
    lm = load_lm('roc_lm97027_batchtrained_clio_updated')
    train_stories = get_train_stories(filepath='ROCStories_winter2017.csv') #get_train_stories(filepath='ROC-Stories.tsv')[:100]
    samp_stories = train_stories[-20:]
    context_seqs = [story[:-1] for story in samp_stories]
    #import pdb;pdb.set_trace()
    gen_sents, p_sents = generate_sents(lm, context_seqs, n_best=1, batch_size=1, mode='max', temp=0.1, n_words=35)
    show_gen_sents(context_seqs, gen_sents, p_sents)


# In[6]:

def train_batched_lm(stories):
    n_epochs = 50
    import pdb;pdb.set_trace()
    for epoch in range(n_epochs):
        print "training epoch {}/{}...".format(epoch + 1, n_epochs)
        lm.fit(X=stories)
        #generate samples to show progress
        samp_size = 10
        temp = 0.6
        samp_stories = random.sample(stories, samp_size)
        context_seqs = [story[:-1] for story in samp_stories]
        gold_sents = [story[-1] for story in samp_stories]
        p_gold_sents = lm.predict(X=context_seqs, y_seqs=gold_sents, batch_size=samp_size)
        gen_sents, p_sents = generate_sents(lm, context_seqs, n_best=1, mode='max', batch_size=samp_size)
        print "MAX PROB SENTENCES:"
        show_gen_sents(context_seqs, gen_sents, p_sents, gold_sents, p_gold_sents)
        import pdb;pdb.set_trace()
        gen_sents, p_sents = generate_sents(lm, context_seqs, n_best=1, mode='random', 
                                            temp=temp, batch_size=samp_size)
        print "SENTENCES WITH TEMP =", temp
        show_gen_sents(context_seqs, gen_sents, p_sents)

if __name__ == '__main__':
    import pdb;pdb.set_trace()
    train_stories = get_train_stories(filepath='ROC-Stories.tsv') +                     get_train_stories(filepath='ROCStories_winter2017.csv')
    train_stories = train_stories[:10]
    filepath = 'roc_lm' + str(len(train_stories)) + '_batch1trained'
    lm = RNNPipeline(steps=[('transformer', SequenceTransformer(min_freq=1, verbose=1, filepath=filepath)),
                            ('classifier', RNNLM(verbose=1, batch_size=1, n_timesteps=None,
                                             n_hidden_layers=2,
                                             n_embedding_nodes=300, n_hidden_nodes=500,
                                             filepath=filepath))])
    train_batched_lm(train_stories)



# In[ ]:

def train_decoder():
    train_stories = get_train_stories(filepath='ROC-Stories.tsv')[:10000]
    filepath = 'roc_lm' + str(len(train_stories)) + '_decoder'
    max_sent_length = max([len(tokenize(seq[-1])) for seq in train_stories])
    lm = RNNPipeline(steps=[('transformer', SequenceTransformer(min_freq=1, verbose=1, filepath=filepath)),
                            ('classifier', RNNLM(verbose=1, batch_size=50, max_sent_length=max_sent_length, 
                                                 n_hidden_layers=1,
                                                 n_embedding_nodes=100, n_hidden_nodes=200, separate_context=True,
                                                 filepath=filepath))])
    n_epochs = 50
    for epoch in range(n_epochs):
        print "training epoch {}/{}...".format(epoch + 1, n_epochs)
        #import pdb;pdb.set_trace()
        lm.fit(X=train_stories)
        #generate samples to show progress
        samp_size = 10
        temp = 0.75
        samp_stories = random.sample(train_stories, samp_size)
        context_seqs = [story[:-1] for story in samp_stories]
        gold_sents = [story[-1] for story in samp_stories]
        #         p_gold_sents = lm.predict(X=context_seqs, y_seqs=gold_sents, batch_size=samp_size)
        import pdb;pdb.set_trace()
        gen_sents, p_sents = generate_sents(lm, context_seqs, n_best=1, mode='random', batch_size=samp_size)
        print "MAX PROB SENTENCES:"
        show_gen_sents(context_seqs, gen_sents, p_sents)#, gold_sents, p_gold_sents)
#         gen_sents, p_sents = generate_sents(lm, context_seqs, n_best=1, mode='random', 
#                                             temp=temp, batch_size=samp_size)
#         print "SENTENCES WITH TEMP =", temp
#         show_gen_sents(context_seqs, gen_sents, p_sents)




# In[9]:

if __name__ == "__main__":
    lm = load_lm('roc_lm44362_batchtrained_clio')
#     train_stories = get_train_stories(filepath='ROC-Stories.tsv')[:100]
    samp_stories = train_stories[-20:]
    context_seqs = [story[:-1] for story in samp_stories]
    #import pdb;pdb.set_trace()
    gen_sents, p_sents = generate_sents(lm, context_seqs, n_best=1, batch_size=1, mode='max', temp=0.1, n_words=35)
    show_gen_sents(context_seqs, gen_sents, p_sents)


# In[9]:

if __name__ == "__main__":
    lm = load_lm('roc_lm97027_batchtrained_clio_updated')
#     train_stories = get_train_stories(filepath='ROC-Stories.tsv')[:100]
    samp_stories = train_stories[-20:]
    context_seqs = [story[:-1] for story in samp_stories]
    #import pdb;pdb.set_trace()
    gen_sents, p_sents = generate_sents(lm, context_seqs, n_best=1, batch_size=1, mode='max', temp=0.1, n_words=35)
    show_gen_sents(context_seqs, gen_sents, p_sents)


# In[8]:

if __name__ == "__main__":
    #get cloze items in validation set
    val_input_seqs,    val_output_choices, val_output_gold = get_cloze_data(filepath='cloze_test_ALL_val.tsv', mode='concat')


# In[39]:

if __name__ == "__main__":
    stories = get_train_stories(filepath='ROC-Stories.tsv')[-100:]
    context_seqs = [story[:-1] for story in stories]
    gold_sents = [story[-1] for story in stories]
    #import pdb;pdb.set_trace()
    p_gold_sents = lm_pipeline.predict(X=context_seqs, y=gold_sents, batch_size=10)
#     gen_sents, p_sents = generate_sents(lm_pipeline, context_seqs, batch_size=50, 
#                                           n_best=1, mode='random', temp=0.15)

    #import pdb;pdb.set_trace()
    show_gen_sents(context_seqs=context_seqs, gold_sents=gold_sents, p_gold_sents=p_gold_sents)
#     generate_endings(lm_pipeline, train_stories[40:45], n_best=1, mode='random', temp=0.1)
#     generate_endings(lm_pipeline, train_stories[40:45], n_best=1, mode='random', temp=0.1)


# In[119]:

# if __name__ == '__main__':
#     my_embeddings = lm_classifier.get_embeddings()
#     n_embedding_nodes = my_embeddings.shape[-1]
#     my_embeddings = {lm_transformer.lexicon_lookup[idx]: values for idx, values in enumerate(my_embeddings)}
#     my_embeddings = Word2Vec(sentences=sents, min_count=1, size=300, window=10, negative=50, iter=20)
#     n_embedding_nodes = embeddings.vector_size
#     google_embeddings = similarity_score.load_model("../AvMaxSim/vectors")
# #     #embeddings = Word2Vec(sentences=sents, min_count=2, size=300, sg=1)#negative=20
#     n_embedding_nodes = google_embeddings.vector_size


# In[188]:

# if __name__ == "__main__":
#     import pdb;pdb.set_trace()
#     encoder = SequenceTransformer(min_freq=1, verbose=1,
#                                 word_embeddings=my_embeddings, reduce_emb_mode='mean')
#     encoder.lexicon = lm_transformer.lexicon
#     encoder.lexicon_size = lm_transformer.lexicon_size
#     encoder.lexicon_lookup = lm_transformer.lexicon_lookup
#     encoded_gold_sents = encoder.transform(gold_sents)[0]
#     #encoded_gold_sents = numpy.mean(encoded_gold_sents, axis=1)
#     encoded_context_seqs = encoder.transform([" ".join(seq) for seq in context_seqs])[0]
#     #encoded_context_seqs = numpy.mean(context_seqs, axis=1)
#     encoded_gen_sents = encoder.transform(gen_sents)[0]


# In[171]:

#context_seq = [[segment_and_tokenize(sent) for sent in seq] for seq in context_seqs]
#context_seqs = [[word for word in segment_and_tokenize(" ".join(seq))] for  seq in context_seqs]


# In[195]:

# if __name__ == "__main__":
#     for context_seq, gold_sent, gen_sent, enc_gold_sent, enc_gen_sent, enc_context_seq\
#         in zip(context_seqs, gold_sents, gen_sents, encoded_gold_sents, encoded_gen_sents, encoded_context_seqs):
#         sim = cosine_similarity(enc_context_seq[None], enc_gold_sent[None])[0][0]
#         #sim = similarity_score.sim_score(tokenize(gold_sent), tokenize(" ".join(context_seq)), my_embeddings)
#         print (context_seq, gold_sent), "{:.3f}".format(sim)


# In[46]:

if __name__ == '__main__':
    sents = [segment_and_tokenize(sent) for seq in get_train_stories('ROC-Stories.tsv') for sent in seq]


# In[91]:

if __name__ == '__main__':
    word1 = "computer"
    word2 = "phone"
    print cosine_similarity(my_embeddings[word1], my_embeddings[word2])
    print cosine_similarity(google_embeddings[word1], google_embeddings[word2])

