from __future__ import print_function
import sys, os, warnings, pandas, argparse
sys.path.append('../')

from models.pipeline import *
from models.classifier import *
from models.transformer import *

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

def load_model(filepath, skip_filepath):
    model = RNNBinaryPipeline.load(filepath=filepath, transformer_is_skip=True, skip_filepath=skip_filepath)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an RNN-based binary classifier to perform the Story Cloze Test")
    parser.add_argument("--train_seqs", "-train", help="Specify filename (.tsv) containing ROCStories used as training data.", type=str, required=True)
    parser.add_argument("--val_items", "-val", help="Specify filename (.tsv) containing cloze items in validation set.", type=str, required=True)
    parser.add_argument("--test_items", "-test", help="Specify filename (.tsv) containing cloze items in test set.", type=str, required=True)
    parser.add_argument("--save_filepath", "-save", help="Specify the directory filepath where the trained model should be stored.", type=str, required=True)
    parser.add_argument("--skip_filepath", "-skip", help="Specify the directory filepath where the model for the skipthought vectors is located.", type=str, required=True)
    parser.add_argument("--n_backward", "-bkwrd", help="Specify number of \"backward\" generated endings (i.e. sentences selected from initial story) to include in incorrect training samples. Default is 2.", 
                        required=False, type=int, default=2)
    parser.add_argument("--n_random", "-rand", help="Specify number of \"random\" generated endings (i.e. endings randomly selected from other stories) to include in incorrect training samples. Default is 4.", 
                        required=False, type=int, default=4)
    parser.add_argument("--batch_size", "-batch", help="Specify number of sequences in batch during training. Default is 100.", required=False, type=int, default=100)
    parser.add_argument("--n_hidden_layers", "-lay", help="Specify number of recurrent hidden layers in model. Default is 1.", required=False, type=int, default=1)
    parser.add_argument("--n_hidden_nodes", "-hid", help="Specify number of nodes in each recurrent hidden layer. Default is 1000.", required=False, type=int, default=1000)
    parser.add_argument("--n_epochs", "-epoch", help="Specify the number of epochs the model should be trained for. Default is 10.", required=False, type=int, default=10)
    args = parser.parse_args()

    train_input_seqs, train_output_seqs = get_train_input_outputs(args.train_seqs, mode='concat')

    val_input_seqs, val_output_choices, val_output_gold = get_cloze_data(args.val_items)

    transformer = SkipthoughtsTransformer(filepath=args.skip_filepath, verbose=False)
    classifier = RNNBinaryClassifier(filepath=args.save_filepath, n_input_sents=4, n_embedding_nodes=transformer.encoder_dim,
                                    n_hidden_layers=args.n_hidden_layers, n_hidden_nodes=args.n_hidden_nodes, batch_size=args.batch_size)

    model = RNNBinaryPipeline(transformer, classifier)
    #eval_fn is a function that computes accuracy of model on validation cloze items during training
    model.fit(train_input_seqs, train_output_seqs, n_bkwrd=args.n_backward, n_random=args.n_random, n_epochs=args.n_epochs,
            eval_fn=lambda model: evaluate_roc_cloze(model, val_input_seqs, val_output_choices, val_output_gold))

    #After training, compute accuracy on test set
    test_input_seqs, test_output_choices, test_output_gold = get_cloze_data(args.test_items)
    test_accuracy = evaluate_roc_cloze(model, test_input_seqs, test_output_choices, test_output_gold)
    print("ROC cloze test accuracy:", test_accuracy)

