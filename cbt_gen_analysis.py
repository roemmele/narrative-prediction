
# coding: utf-8

# In[26]:

import sys, pandas, pprint, pickle
# sys.path.append('skip-thoughts-master/')
# import skipthoughts

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

import analysis.generation_metrics
reload(analysis.generation_metrics)
from analysis.generation_metrics import *

import analysis.stats
reload(analysis.stats)
from analysis.stats import *

pandas.set_option('precision', 3)


# In[15]:

def get_cbt_seqs(filepaths, flatten=True):
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
                    if flatten:
                        context_seq = " ".join(context_seq)
                    context_seqs.append(context_seq)
                    context_seq = []
                    continue
                else:
                    context_seq.append(sent)
    return context_seqs, gold_sents

if __name__ == '__main__':
    '''load CBT stories and gold sentences'''
    context_seqs, gold_sents = get_cbt_seqs(["CBTest/data/cbtest_CN_test_2500ex.txt",                                             "CBTest/data/cbtest_CN_valid_2000ex.txt",                                             "CBTest/data/cbtest_NE_test_2500ex.txt",                                             "CBTest/data/cbtest_NE_valid_2000ex.txt",                                             "CBTest/data/cbtest_P_test_2500ex.txt",                                             "CBTest/data/cbtest_P_valid_2000ex.txt",                                             "CBTest/data/cbtest_V_test_2500ex.txt",                                             "CBTest/data/cbtest_V_valid_2000ex.txt"])


# In[10]:

if __name__ == '__main__':
    '''load generated sentences from file'''
    #import pdb;pdb.set_trace()
    gen_sents = {}
    gen_sents['cbr'] = pandas.read_csv('cbt_cbr_sents18000_1.csv', encoding='utf-8', header=None).values.tolist()
    gen_sents['cbrrand'] = pandas.read_csv('cbt_cbrrand_sents18000_1.csv', encoding='utf-8', header=None).values.tolist()
    gen_sents['unigram'] = pandas.read_csv('cbt_unigram_sents18000_1.csv', encoding='utf-8', header=None).values.tolist()
    gen_sents['rnn'] = pandas.read_csv('cbt_rnn_sents18000_1.csv', encoding='utf-8', header=None).values.tolist()
    gen_sents['gold'] = gold_sents


# # Coreference

# In[47]:

if __name__ == '__main__':
    '''find unresolved entities in generated sentences'''

    import pdb;pdb.set_trace()
    coref_counts = {'models':{}, 'p-values':{}}
    for model in gen_sents.keys():
        coref_counts['models'][model] = get_coref_counts(context_seqs, gen_sents[model])
        with open('cbt_' + model + '_coref_counts.pkl', 'wb') as f:
            pickle.dump(coref_counts['models'][model], f)
    pprint.pprint(pandas.DataFrame.from_dict(coref_counts['models'], orient='index'))


# In[48]:

if __name__ == "__main__":
    '''stats tests for coref counts'''
    
    coref_counts['p-values']['ents'] = eval_all_diffs({model:analysis['ents']                                                        for model,analysis                                                        in coref_counts['models'].items()})
    coref_counts['p-values']['corefs'] = eval_all_diffs({model:analysis['corefs']                                                        for model,analysis                                                        in coref_counts['models'].items()})
    coref_counts['p-values']['resolution_rates'] = eval_all_diffs({model:analysis['resolution_rates']                                                                    for model,analysis                                                                    in coref_counts['models'].items()})
    pprint.pprint(coref_counts['p-values'])

