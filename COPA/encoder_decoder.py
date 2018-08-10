from __future__ import print_function
import sys, pandas, argparse
import xml.etree.cElementTree as et

sys.path.append('../')

from models.pipeline import *
from models.classifier import *
from models.transformer import *

def load_copa(filepath=None):
    xml_tree = et.parse(filepath)
    corpus = xml_tree.getroot()
    premises = []
    alts = []
    answers = []
    modes = []
    for item in corpus:
        mode = item.attrib["asks-for"]
        modes.append(mode)
        answer = int(item.attrib["most-plausible-alternative"]) - 1 #answers are 1 and 2, convert to 0 and 1 
        answers.append(answer)
        premise = item.find("p").text
        premises.append(premise)
        alt1 = item.find("a1").text
        alt2 = item.find("a2").text
        alts.append([alt1, alt2])
    answers = numpy.array(answers)
    return premises, alts, answers, modes

def get_copa_scores(model, premises, alts, modes):
    alt1_pairs = []
    alt2_pairs = []
    for premise, (alt1, alt2), mode in zip(premises, alts, modes):
        if mode == 'cause':
            alt1_pairs.append([alt1, premise])
            alt2_pairs.append([alt2, premise])
        else:
            alt1_pairs.append([premise, alt1])
            alt2_pairs.append([premise, alt2])

    alt1_scores = model.predict(seqs1=[pair[0] for pair in alt1_pairs], seqs2=[pair[1] for pair in alt1_pairs])
    alt2_scores = model.predict(seqs1=[pair[0] for pair in alt2_pairs], seqs2=[pair[1] for pair in alt2_pairs])

    pred_alts = numpy.argmax(numpy.array(zip(alt1_scores, alt2_scores)), axis=1)
    
    return alt1_scores, alt2_scores, pred_alts

def get_copa_accuracy(pred_alts, answers):
    pred_is_correct = numpy.array(pred_alts) == numpy.array(answers)
    accuracy = numpy.mean(pred_is_correct)
    return accuracy

def eval_copa(model, data_filepath):
    premises, alts, answers, modes = load_copa(filepath=data_filepath)
    alt1_scores, alt2_scores, pred_alts = get_copa_scores(model, premises, alts, modes)
    accuracy = get_copa_accuracy(pred_alts, answers)
    print("COPA accuracy: {:.3f}".format(accuracy))
    return accuracy

def get_seqs(filepath, header=None, chunk_size=None):
    if chunk_size:
        seqs = (chunk.iloc[:,0].values.tolist() for chunk in pandas.read_csv(filepath, encoding='utf-8', header=header, chunksize=chunk_size))
    else:
        seqs = pandas.read_csv(filepath, encoding='utf-8', header=header).iloc[:,0].values.tolist()
    return seqs

def load_model(filepath):
    model = EncoderDecoderPipeline.load(filepath=filepath)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an encoder-decoder model that predicts causally related sentences in the Choice of Plausible Alternatives (COPA) framework")
    parser.add_argument("--train_seqs", "-train", help="Specify filename (.csv) containing text used as training data.", type=str, required=True)
    parser.add_argument("--val_items", "-val", help="Specify filename (XML) containing COPA items in validation set.", type=str, required=True)
    parser.add_argument("--test_items", "-test", help="Specify filename (XML) containing COPA items in test set.", type=str, required=True)
    parser.add_argument("--save_filepath", "-save", help="Specify the directory filepath where the trained model should be stored.", type=str, required=True)
    parser.add_argument("--min_freq", "-freq", help="Specify frequency threshold for including words in model lexicon, such that only words that appear in the training sequences at least\
                                                    this number of times will be added (all other words will be mapped to a generic <UNKNOWN> token). Default is 5.", required=False, type=int, default=5)
    parser.add_argument("--segment_sents", "-sent", help="Specify if the segments in the input-output pairs should be sentences rather than intrasentential clauses (see paper).\
                                                            If not given, clause-based segmentation will be used.", required=False, action='store_true')
    parser.add_argument("--max_length", "-len", help="Specify the maximum length of the input and output segements in the training data (in terms of number of words).\
                                                    Pairs with longer sequences will be filtered. Default is 20.", required=False, type=int, default=20)
    parser.add_argument("--max_pair_distance", "-dist", help="Specify the distance window in which neighboring segments will be joined into input-output pairs.\
                                                            For example, if this parameter is 3, all segments that are separated by 3 or fewer segments in a particular training text will be added as pairs.\
                                                            Default is 4.", required=False, type=int, default=4)
    parser.add_argument("--recurrent", "-rec", help="Specify if the model should use RNN (GRU) layers. If not specified, feed-forward layers will be used,\
                                                    and the sequential ordering of words in the segments will be ignored.", required=False, action='store_true')
    parser.add_argument("--batch_size", "-batch", help="Specify number of sequences in batch during training. Default is 100.", required=False, type=int, default=100)
    parser.add_argument("--n_hidden_nodes", "-hid", help="Specify number of dimensions in the encoder and decoder layers. Default is 500.", required=False, type=int, default=500)
    parser.add_argument("--n_epochs", "-epoch", help="Specify the number of epochs the model should be trained for. Default is 50.", required=False, type=int, default=50)
    parser.add_argument("--chunk_size", "-chunk", help="If dataset is large, specify this parameter to load training sequences in chunks of this size instead of all at once to avoid memory issues.\
                                                         For smaller datasets (e.g. the ROCStories corpus), it is much faster to load entire dataset prior to training. This will be done by default\
                                                         if chunk size is not given.", required=False, type=int, default=0)
    args = parser.parse_args()

    transformer = SequenceTransformer(min_freq=args.min_freq, lemmatize=True, filepath=args.save_filepath, 
                                    include_tags=['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'RB', 'RBR', 'RBS', 'RP', #lemmatize segments and filter grammatical words
                                                    'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])

    classifier = EncoderDecoder(filepath=args.save_filepath, recurrent=args.recurrent, batch_size=args.batch_size, n_hidden_nodes=args.n_hidden_nodes)
    model = EncoderDecoderPipeline(transformer, classifier)

    if args.chunk_size: #load training data in chunks
        if not transformer.lexicon:
            for seqs in get_seqs(args.train_seqs, chunk_size=args.chunk_size):
                transformer.make_lexicon(seqs)

        for epoch in range(args.n_epochs):
            print("EPOCH:", epoch + 1)
            for seqs in get_seqs(args.train_seqs, chunk_size=args.chunk_size):
                seq_pairs = get_adj_sent_pairs(seqs, segment_clauses=False if args.segment_sents else True, max_distance=args.max_pair_distance, max_sent_length=args.max_length)
                model.fit(seqs1=[pair[0] for pair in seq_pairs], seqs2=[pair[1] for pair in seq_pairs], max_length=args.max_length,
                            eval_fn=lambda model: eval_copa(model, data_filepath=args.val_items), n_epochs=1)

    else: #load entire training data at once
        seqs = get_seqs(args.train_seqs, chunk_size=None)
        if not transformer.lexicon:
            transformer.make_lexicon(seqs)

        seq_pairs = get_adj_sent_pairs(seqs, segment_clauses=False if args.segment_sents else True, max_distance=args.max_pair_distance, max_sent_length=args.max_length)
        model.fit(seqs1=[pair[0] for pair in seq_pairs], seqs2=[pair[1] for pair in seq_pairs], max_length=args.max_length,
                    eval_fn=lambda model: eval_copa(model, data_filepath=args.val_items), n_epochs=args.n_epochs)

    #Evaluate model on test set after training
    print("\ntest accuracy:")
    test_accuracy = eval_copa(model, data_filepath=args.test_items)







