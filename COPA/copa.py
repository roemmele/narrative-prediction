import sys, numpy, timeit, cPickle, itertools, argparse, random
import xml.etree.cElementTree as et

sys.path.append('../')

from models.pipeline import *
from models.transformer import *
from models.classifier import *
from models.pmi import *

copa_dev_filepath = "dataset/copa-dev.xml"
copa_test_filepath = "dataset/copa-test.xml"

rng = numpy.random.RandomState(123)

def load(filepath=None):
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
        premise = unicode(item.find("p").text, 'utf-8')
        premises.append(premise)
        alt1 = unicode(item.find("a1").text, 'utf-8')
        alt2 = unicode(item.find("a2").text, 'utf-8')
        alts.append([alt1, alt2])
    answers = numpy.array(answers)
    return premises, alts, answers, modes

def analyze_item(model, input_seq, output_seq, analyze_seq='output'):
    
    probs = []
    
    if analyze_seq == 'output':
        trunc_outputs = []
        input_seq, _ = model.transformer.transform(X=[input_seq])
        output_seq = tokenize(output_seq)
        for word_idx in range(len(output_seq)):
            trunc_output = " ".join(output_seq[:word_idx + 1])
            trunc_outputs.append(trunc_output)
            trunc_output, _ = model.transformer.transform(X=[trunc_output])
            prob = model.classifier.predict(X=input_seq, y_seqs=[trunc_output])[0][0]
            probs.extend(prob)
        
        return trunc_outputs, probs
    
    elif analyze_seq == 'input':
        trunc_inputs = []
        output_seq, _ = model.transformer.transform(X=[output_seq])
        input_seq = tokenize(input_seq)
        for word_idx in range(len(input_seq)):
            trunc_input = " ".join(input_seq[:word_idx + 1])
            trunc_inputs.append(trunc_input)
            trunc_input, _ = model.transformer.transform(X=[trunc_input])
            prob = model.classifier.predict(X=trunc_input, y_seqs=[output_seq])[0][0]
            probs.extend(prob)
        
        return trunc_inputs, probs

def evaluate_sim(emb_model, premises, alts, answers, modes, combine_emb='max'):
    
    n_sequences = len(premises)
    pred_alts = []
    for premise, (alt1, alt2), mode in zip(premises, alts, modes): 
        #predict one by one since mode changes order of X and y_seqs
        if mode == 'cause':
            alt1_score = similarity_score.score(alt1, premise, emb_model, combine_emb=combine_emb)
            alt2_score = similarity_score.score(alt2, premise, emb_model, combine_emb=combine_emb)
        else:
            alt1_score = similarity_score.score(premise, alt1, emb_model, combine_emb=combine_emb)
            alt2_score = similarity_score.score(premise, alt2, emb_model, combine_emb=combine_emb)
        
        pred_alts.append(numpy.argmax([alt1_score, alt2_score]))

    accuracy = numpy.mean(numpy.array(pred_alts) == numpy.array(answers))

    return accuracy

def replace_alts(alts, answers, replacement='random'):
    '''replace the wrong alternative with a random alternative or empty string'''
    for idx, (alt, answer) in enumerate(zip(alts, answers)):
        if replacement == 'random':
            alts[idx][numpy.logical_not(answer).astype(int)] = random.choice(alts)[0]
    return alts

def evaluate_pmi(model_filepath, premises, alts, answers, modes):
    transformer = load_transformer(model_filepath)
    pmi_model = PMIModel(model_filepath)
    
    pred_alts = []
    for premise, (alt1, alt2), mode in zip(premises, alts, modes): 
        #predict one by one since mode changes order of X and y_seqs
        premise = transformer.text_to_nums([premise])[0]
        alt1 = transformer.text_to_nums([alt1])[0]
        alt2 = transformer.text_to_nums([alt2])[0]
        if mode == 'cause':
            alt1_score = pmi_model.score(alt1, premise)
            alt2_score = pmi_model.score(alt2, premise)
        else:
            alt1_score = pmi_model.score(premise, alt1)#
            alt2_score = pmi_model.score(premise, alt2)
        
        pred_alts.append(numpy.argmax([alt1_score, alt2_score]))

    accuracy = numpy.mean(numpy.array(pred_alts) == numpy.array(answers))

    return accuracy

def get_copa_scores(model, premises, alts, modes, model_scheme, pred_method='multiply', pmi=False, causal=False, unigram_probs=None):
    alt1_pairs = []
    alt2_pairs = []
    for premise, (alt1, alt2), mode in zip(premises, alts, modes):
        if (mode == 'cause' and model_scheme == 'forward') or (mode == 'effect' and model_scheme == 'reverse'):
            alt1_pairs.append([alt1, premise])
            alt2_pairs.append([alt2, premise])
        else:
            alt1_pairs.append([premise, alt1])
            alt2_pairs.append([premise, alt2])

    if pmi: #model is PMI model
        alt1_scores = model.predict(seqs1=[pair[0] for pair in alt1_pairs], seqs2=[pair[1] for pair in alt1_pairs], 
                                    causal=causal)
        alt2_scores = model.predict(seqs1=[pair[0] for pair in alt2_pairs], seqs2=[pair[1] for pair in alt2_pairs], 
                                    causal=causal)
    else:
        alt1_scores = model.predict(seqs1=[pair[0] for pair in alt1_pairs], seqs2=[pair[1] for pair in alt1_pairs])
        alt2_scores = model.predict(seqs1=[pair[0] for pair in alt2_pairs], seqs2=[pair[1] for pair in alt2_pairs])

    pred_alts = numpy.argmax(numpy.array(zip(alt1_scores, alt2_scores)), axis=1)
    
    return alt1_scores, alt2_scores, pred_alts

def show_copa_accuracy(premises, alts, answers, modes, alt1_scores, alt2_scores, pred_alts):
    for premise, (alt1, alt2), alt1_score, alt2_score, pred_alt, answer, mode\
                in zip(premises, alts, alt1_scores, alt2_scores, pred_alts, answers, modes):
        print "PREMISE:", premise, "( mode =", mode, ", correct =", pred_alt == answer, ")"
        print "*" if answer == 0 else " ", "ALT1:", alt1, "(%.2f)" % alt1_score
        print "*" if answer == 1 else " ", "ALT2:", alt2, "(%.2f)" % alt2_score, "\n"

    accuracy = get_copa_accuracy(pred_alts, answers)
    print "COPA accuracy: %.3f" % (accuracy)

def get_copa_accuracy(pred_alts, answers):
    pred_is_correct = numpy.array(pred_alts) == numpy.array(answers)
    accuracy = numpy.mean(pred_is_correct)
    return accuracy

def combine_copa_scores(alt1_scores_list, alt2_scores_list, weights=[1.0, 1.0], combine_mode='mean'):

    assert(len(alt1_scores_list) == len(alt2_scores_list) == len(weights))

    combined_alt1_scores = []
    combined_alt2_scores = []

    alt1_scores_list = numpy.array(zip(*alt1_scores_list)) * weights
    alt2_scores_list = numpy.array(zip(*alt2_scores_list)) * weights

    for alt1_scores, alt2_scores in zip(alt1_scores_list, alt2_scores_list):
        alt1_score = numpy.sum(alt1_scores)
        alt2_score = numpy.sum(alt2_scores)

        combined_alt1_scores.append(alt1_score)
        combined_alt2_scores.append(alt2_score)

    combined_alt1_scores = numpy.array(combined_alt1_scores)
    combined_alt2_scores = numpy.array(combined_alt2_scores)

    pred_alts = numpy.argmax(numpy.array(zip(combined_alt1_scores, combined_alt2_scores)), axis=1)

    return combined_alt1_scores, combined_alt2_scores, pred_alts

def evaluate_copa_with_lm(model, lm, filepath, show_items=False):
    premises, alts, answers, modes = load(filepath)
    pred_alts = []
    for idx, (premise, (alt1, alt2), mode) in enumerate(zip(premises, alts, modes)):
        if mode == 'cause':
            premise_lm_score = numpy.sum(lm.get_probs([premise]))
            alt1_score = model.predict(seq1=alt1, seq2=premise)
            alt1_score = alt1_score - premise_lm_score
            alt2_score = model.predict(seq1=alt2, seq2=premise)
            alt2_score = alt2_score - premise_lm_score
        else:
            alt1_lm_score = numpy.sum(lm.get_probs([alt1]))
            alt1_score = model.predict(seq1=premise, seq2=alt1)
            alt1_score = alt1_score - alt1_lm_score
            alt2_lm_score = numpy.sum(lm.get_probs([alt2]))
            alt2_score = model.predict(seq1=premise, seq2=alt2)
            alt2_score = alt2_score - alt2_lm_score
        
        pred_alt = numpy.argmax([alt1_score, alt2_score])
        pred_alts.append(pred_alt)

        if show_items:
            print "PREMISE:", premise, "( mode =", mode, ", correct =", pred_alts[idx] == answers[idx], ")"
            print "*" if answers[idx] == 0 else " ", "ALT1:", alt1, "(%.2f)" % alt1_score
            print "*" if answers[idx] == 1 else " ", "ALT2:", alt2, "(%.2f)" % alt2_score, "\n"

    correct = numpy.array(pred_alts) == numpy.array(answers)
    accuracy = numpy.mean(correct)
    print "COPA accuracy: %.3f" % (accuracy)
    return accuracy

def evaluate_copa_with_sim(model, sim_model, filepath, n_best=10, show_items=False):
    premises, alts, answers, modes = load(filepath)
    sim_premises_scores = []
    pred_alts = []
    for idx, (premise, (alt1, alt2), mode) in enumerate(zip(premises, alts, modes)):
        sim_seqs, scores = sim_model.get_sim_seqs(seqs=[premise, alt1, alt2], n_best=n_best)
        sim_premises, sim_alts1, sim_alts2 = sim_seqs
        if mode == 'cause':
            alt1_premise_score = model.predict(seq1=alt1, seq2=premise)
            alt1_sim_premises_scores = [model.predict(seq1=alt1, seq2=sim_premise) 
                                                                for sim_premise in sim_premises]
            
            alt2_premise_score = model.predict(seq1=alt2, seq2=premise)
            alt2_sim_premises_scores = [model.predict(seq1=alt2, seq2=sim_premise) 
                                                                for sim_premise in sim_premises]
        else:
            alt1_premise_score = model.predict(seq1=premise, seq2=alt1)
            alt1_sim_premises_scores = [model.predict(seq1=sim_premise, seq2=alt1) 
                                                            for sim_premise in sim_premises]
            
            alt2_premise_score = model.predict(seq1=premise, seq2=alt2)
            alt2_sim_premises_scores = [model.predict(seq1=sim_premise, seq2=alt2)
                                                            for sim_premise in sim_premises]
        
        sim_premises_scores.append(numpy.mean(scores))
        alt1_score = numpy.average([alt1_premise_score] + alt1_sim_premises_scores)
        alt2_score = numpy.average([alt2_premise_score] + alt2_sim_premises_scores)
        pred_alt = numpy.argmax([alt1_score, alt2_score])
        pred_alts.append(pred_alt)
        

        if show_items:
            print "PREMISE:", premise, "( mode =", mode, ", correct =", pred_alts[idx] == answers[idx], ")"
            print "*" if answers[idx] == 0 else " ", "ALT1:", alt1, "(%.2f)" % alt1_score
            print "*" if answers[idx] == 1 else " ", "ALT2:", alt2, "(%.2f)" % alt2_score, "\n"

    correct = numpy.array(pred_alts) == numpy.array(answers)
    correct_sim_score = numpy.mean(numpy.array(sim_premises_scores)[correct])
    incorrect_sim_score = numpy.mean(numpy.array(sim_premises_scores)[numpy.logical_not(correct)])
    accuracy = numpy.mean(correct)
    print "correct similarity:", correct_sim_score
    print "incorrect similarity:", incorrect_sim_score
    print "COPA accuracy: %.3f" % (accuracy)

    return accuracy



