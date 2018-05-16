from __future__ import print_function
import sys, pandas

sys.path.append('../')

from models.pipeline import *
from models.classifier import *
from models.transformer import *

from copa import *

def get_stories(filepath, header=None, chunk_size=10000):
    stories = (chunk.iloc[:,0].values.tolist() for chunk in pandas.read_csv(filepath, encoding='utf-8', header=header, chunksize=chunk_size))
    return stories

def get_causal_pairs(filepath, column_idx=1, chunk_size=1000000):
    pairs = (chunk.iloc[:,column_idx:].values.tolist() for chunk in pandas.read_csv(filepath, encoding='utf-8', header=None, chunksize=chunk_size))
    return pairs

def get_causal_pairs_from_stories(stories, fwd_causal_markers, bkwd_causal_markers):
    causal_pairs = []
    for story_idx, story in enumerate(stories):                                                                                                                     
        for sent in segment(story):
            for marker in fwd_causal_markers:
                fwd_match = re.search(marker, sent)
                if fwd_match:
                    causal_pairs.append(fwd_match.groups()[:2])
                    break
            for marker in bkwd_causal_markers:
                bkwd_match = re.search(marker, sent)
                if bkwd_match:
                    causal_pairs.append(bkwd_match.groups()[:2][::-1])
                    break
        if story_idx % 20000 == 0:
            print("processed", story_idx, "stories...")
    return causal_pairs

def filter_pairs(pairs, max_sent_length=None):
    if not max_sent_length:
        pairs = [pair for pair in pairs if pair[0] and pair[1]]
    else:
        pairs = [pair for pair in pairs if pair[0] and len(pair[0].split()) <= max_sent_length\
                and pair[1] and len(pair[1].split()) <= max_sent_length]
    return pairs


def eval_fn(model_scheme, data_filepath):
    def eval_fn_(model):

        premises, alts, answers, modes = load(filepath=data_filepath)
        alt1_scores, alt2_scores, pred_alts = get_copa_scores(model, premises, alts, modes,
                                                                model_scheme=model_scheme)
        accuracy = get_copa_accuracy(pred_alts, answers)
        print("COPA accuracy: {:.3f}".format(accuracy))
        return accuracy
    return eval_fn_


if __name__ == '__main__':
    stories_filepath = '../ROC/dataset/roc_train_stories97027.csv'
    save_filepath = 'roc_test'

    if os.path.exists(save_filepath + '/transformer.pkl'): #if transformer already exists, load it
        transformer = SequenceTransformer.load(save_filepath)
    else:
        transformer = SequenceTransformer(min_freq=5, lemmatize=True, filepath=save_filepath, 
                                        include_tags=['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'RB', 'RBR', 'RBS', 'RP',
                                                        'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])

    segment_clauses = True
    max_sent_distance = 4
    max_segment_length = 20
    recurrent = False

    classifier = EncoderDecoder(filepath=save_filepath, recurrent=False, batch_size=100)
    model = EncoderDecoderPipeline(transformer, classifier)

    if not transformer.lexicon:
        for stories in get_stories(stories_filepath, chunk_size=2000000):
            transformer.make_lexicon(stories)

    #import pdb;pdb.set_trace()
    n_epochs = 50
    #for epoch in range(n_epochs):
        #print "training epoch {}/{}...".format(epoch + 1, n_epochs)
    for stories in get_stories(stories_filepath, chunk_size=2000000):

        seq_pairs = get_adj_sent_pairs(stories, segment_clauses=segment_clauses, max_distance=max_sent_distance, max_sent_length=max_segment_length)
        model.fit(seqs1=[pair[0] for pair in seq_pairs], seqs2=[pair[1] for pair in seq_pairs], max_length=max_segment_length,
                    eval_fn=eval_fn(model_scheme='forward', data_filepath=copa_dev_filepath), 
                    n_epochs=n_epochs, eval_freq=5)




