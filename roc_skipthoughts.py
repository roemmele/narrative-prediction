
# coding: utf-8

# In[1]:

import sys, warnings, os
from gensim.models import Word2Vec
import roc
reload(roc)
from roc import get_train_stories, encode_skipthought_seqs
from models.transformer import segment_and_tokenize, tokenize

sys.path.append('skip-thoughts-master')
from training import vocab
from training import train as encoder_train
from training import tools as encoder_tools
reload(encoder_tools)
from decoding import train as decoder_train
from decoding import tools as decoder_tools

warnings.filterwarnings('ignore', category=Warning)


# In[2]:

def prep_skipthought_seqs(seqs):
    #sents should be already tokenized
    seqs = [[" ".join(segment_and_tokenize(sent)) for sent in seq] 
            for seq in seqs]
    #mark beginning and end of stories with special tokens
    seqs = [["<BEGIN>"] + seq + ["<END>"] for seq in seqs]
    #unravel seqs so each row is one sentence
    seqs = [sent for seq in seqs for sent in seq]
    return seqs
    
def build_skipthought_lexicon(sents, filepath):
    lexicon, word_counts = vocab.build_dictionary(sents)
    filedir = os.path.dirname(filepath)
    if not os.path.isdir(filedir):
        os.mkdir(filedir)
    vocab.save_dictionary(lexicon, word_counts, filepath)
    return lexicon, word_counts


# In[6]:

if __name__ == "__main__":
    #train skipthoughts encoder model
    #import pdb;pdb.set_trace()
    train_stories = get_train_stories(filepath='ROC-Stories.tsv') +                     get_train_stories(filepath='ROCStories_winter2017.csv')
    #train_stories = train_stories[:10000]
    model_dir = 'skipthoughts4800_roc' + str(len(train_stories))
    lexicon_filepath = model_dir + '/lexicon'
    encoder_filepath = model_dir + '/encoder'
    train_sents = prep_skipthought_seqs(train_stories)
    lexicon, word_counts = build_skipthought_lexicon(train_sents, lexicon_filepath)
    encoder_dim = 4800
    encoder_train.trainer(train_sents, dictionary=lexicon_filepath, n_words=max(lexicon.values()), batch_size=64,
                          dim=encoder_dim, saveto=encoder_filepath, saveFreq=50, max_epochs=5)

    #load skipthoughts encoder with vectors for extending lexicon
    #embeddings = similarity_score.load_model('../AvMaxSim/vectors')
    embeddings = Word2Vec.load('roc97027_embeddings300')
    import pdb;pdb.set_trace()
    encoder = encoder_tools.load_model(embed_map=embeddings, 
                                       path_to_model=encoder_filepath, path_to_dictionary=lexicon_filepath)


# In[5]:

# encode_skipthought_seqs(train_stories[:5], encoder_tools, encoder, encoder_dim=1200)


# # In[ ]:

# #train decoder to generate final story sentences
# if __name__ == "__main__":
#     decoder_filepath = model_dir + '/decoder'
# #     input_sents = [" ".join(segment_and_tokenize(story[-2])) for story in train_stories]
# #     output_sents = [" ".join(segment_and_tokenize(story[-1])) for story in train_stories]
    
#     input_sents = [" ".join(segment_and_tokenize(seq)) for seq in input_seqs]
#     output_sents = [" ".join(segment_and_tokenize(seq)) for seq in output_seqs]
#     decoder_train.trainer(X=output_sents[:(500 * 4)], C=input_sents[:(500 * 4)], 
#                           stmodel=encoder, dimctx=encoder_dim,
#                           dim_word=embeddings.vector_size, dictionary=lexicon_filepath, 
#                           saveto=decoder_filepath, saveFreq=50, max_epochs=5)
#     #load decoder model and generate text
#     decoder = decoder_tools.load_model(path_to_model=decoder_filepath, path_to_dictionary=lexicon_filepath)
#     #encode sents


# # In[ ]:

# if __name__ == '__main__':
#     import pdb;pdb.set_trace()
#     input_sents = [" ".join(segment_and_tokenize(story[-2])) for story in train_stories[:100]]
#     encoded_sents = encoder_tools.encode(encoder, input_sents)
#     for story, sent, encoded_sent in zip(train_stories[:100], input_sents, encoded_sents):
#         print "CONTEXT:", " ".join(story[:-2])
#         print "GIVEN:", sent
#         pred_sent = decoder_tools.run_sampler(decoder, encoded_sent, beam_width=1, stochastic=False, use_unk=False)
#         print "PRED:"," ".join(pred_sent), "\n"

