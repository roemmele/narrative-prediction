from __future__ import print_function
import sys, os, warnings, pandas
sys.path.append('../')

from models.pipeline import *
from models.classifier import *
from models.transformer import *

# sys.path.append('../skip-thoughts-master/')
# import skipthoughts
# from training import tools as encoder_tools

warnings.filterwarnings('ignore', category=Warning)

def check_punct(stories, eos_markers=["\"", "\'", ".", "?", "!"]):
    #make sure all stories end with an end-of-sentence marker; if not, insert period
    for story_idx, story in enumerate(stories):
        for sent_idx, sent in enumerate(story):
            if sent[-1] not in eos_markers:
                sent += '.' #add period
                stories[story_idx][sent_idx] = sent
    return stories

def get_train_stories(filepath, flatten=False):
    fileext = filepath.split('.')[1] #determine if file is csv or tsv
    if fileext[0] == 't':
        sep = '\t'
    elif fileext[0] == 'c':
        sep = ','
    stories = pandas.DataFrame.from_csv(filepath, sep=sep, encoding='utf-8')
    #ensure readable encoding
    stories = stories[stories[['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']].apply(
            lambda sentences: numpy.all([type(sentence) is unicode for sentence in sentences]), axis=1)]
    stories = stories[['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']].values.tolist()
    #a few stories are missing end-of-sentence punctuation, so fix it
    stories = check_punct(stories)
    if flatten:
        #combine sents for each story into single string
        stories = [" ".join(sents) for sents in stories]
    return stories

def prep_input_outputs(seqs, mode='adjacent'):
    assert(mode == "adjacent" or mode == "concat" or mode == "pairs")
    if mode == "adjacent":
        #input is sentences 1-4 (first four) as a list, output is sentences 2-5 (last four) as list
        input_seqs = [seq[:-1] for seq in seqs]
        output_seqs = [seq[1:] for seq in seqs]
        assert(len(seqs) == len(input_seqs) == len(output_seqs))
    elif mode == "concat":
        #input is first four sentences, output is last sentence
        input_seqs = [seq[:-1] for seq in seqs]
        output_seqs = [seq[-1] for seq in seqs]
        assert(len(seqs) == len(input_seqs) == len(output_seqs))
    elif mode == "pairs":
        #all pairs of adjacent sentences as independent samples
        input_seqs = [sent for seq in seqs for sent in seq[:-1]]
        output_seqs = [sent for seq in seqs for sent in seq[1:]]       
        assert(len(input_seqs) == len(output_seqs)) 
    return input_seqs, output_seqs

def get_train_input_outputs(filepath, mode='adjacent', flatten=False):
    stories = get_train_stories(filepath)
    input_seqs, output_seqs = prep_input_outputs(stories, mode)
    if flatten:
        #combine sents for each story into single string
        input_seqs = [" ".join(seq) for seq in input_seqs]
    return input_seqs, output_seqs

def prep_cloze_test(cloze_test):
    input_seqs = cloze_test[['InputSentence1', 'InputSentence2',
                            'InputSentence3', 'InputSentence4']].values.tolist()
    output_choices = pandas.concat([cloze_test[['RandomFifthSentenceQuiz1',
                                               'RandomFifthSentenceQuiz2']]], axis=1).values.tolist()
    #substract 1 from ending choice index so indices start at 0
    output_gold = pandas.concat([cloze_test["AnswerRightEnding"]], axis=1).values.flatten() - 1
    assert(len(input_seqs) == len(output_choices) == len(output_gold))
    return input_seqs, output_choices, output_gold

def get_cloze_data(filepath, flatten=False):
    cloze_test = pandas.DataFrame.from_csv(filepath, sep='\t', encoding='utf-8')
    #ensure readable encoding
    cloze_test = cloze_test[cloze_test[['InputSentence1', 'InputSentence2', 'InputSentence3', 
                            'InputSentence4', 'RandomFifthSentenceQuiz1', 'RandomFifthSentenceQuiz2']].apply(
                lambda sentences: numpy.all([type(sentence) is unicode for sentence in sentences]), axis=1)]
    input_seqs, output_choices, output_gold = prep_cloze_test(cloze_test)
    #a few stories are missing end-of-sentence punctuation, so fix it
    # input_seqs = check_punct(input_seqs)
    if flatten:
        #combine sents for each story into single string
        input_seqs = [" ".join(seq) for seq in input_seqs]
    return input_seqs, output_choices, output_gold


def evaluate_roc_cloze(model, input_seqs, output_choices, output_gold):
    choices1 = [choices[0] for choices in output_choices]
    choices2 = [choices[1] for choices in output_choices]
    probs_choice1 = model.predict(input_seqs, choices1)
    probs_choice2 = model.predict(input_seqs, choices2)
    pred_choices = numpy.argmax(numpy.stack([probs_choice1, probs_choice2], axis=1), axis=1)
    accuracy = numpy.mean(pred_choices == numpy.array(output_gold))
    return accuracy

if __name__ == '__main__':
 	train_input_seqs1, train_output_seqs1 = get_train_input_outputs('dataset/ROC-Stories.tsv', mode='concat')
 	train_input_seqs2, train_output_seqs2 = get_train_input_outputs('dataset/ROCStories_winter2017.csv', mode='concat')
 	train_input_seqs = train_input_seqs1 + train_input_seqs2
 	train_output_seqs = train_output_seqs1 + train_output_seqs2
 	#train_input_seqs = train_input_seqs[:25]
 	#train_output_seqs = train_output_seqs[:25]

 	save_filepath = 'roc_rnnbinary'
 	#If using skipthoughts to represent data, must provide path to skipthoughts model
 	transformer = SkipthoughtsTransformer(filepath='../skip-thoughts-master/', verbose=False)
 	classifier = RNNBinaryClassifier(filepath=save_filepath, n_input_sents=4, 
                                     n_embedding_nodes=transformer.encoder_dim, n_hidden_nodes=1000)

 	model = RNNBinaryPipeline(transformer, classifier)

 	model.fit(train_input_seqs, train_output_seqs, n_bkwrd=2, n_random=4, n_epochs=10, n_chunks=50)
 	
 	#model = RNNBinaryPipeline.load(filepath=save_filepath, transformer_is_skip=True, skip_filepath='../skip-thoughts-master/')
 	input_seqs, output_choices, output_gold = get_cloze_data('dataset/cloze_test_ALL_val.tsv')
 	val_accuracy = evaluate_roc_cloze(model, input_seqs, output_choices, output_gold)
 	print("ROC cloze val accuracy:", val_accuracy)

 	input_seqs, output_choices, output_gold = get_cloze_data('dataset/cloze_test_ALL_test.tsv')
 	test_accuracy = evaluate_roc_cloze(model, input_seqs, output_choices, output_gold)
 	print("ROC cloze test accuracy:", test_accuracy)

