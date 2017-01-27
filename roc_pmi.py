
# coding: utf-8

# In[49]:

import os
import roc
reload(roc)
from roc import *
from dine import *
sys.path.append('../')
from models.transformer import segment_and_tokenize, tokenize
sys.path.append("../generation")
# sys.path.append("../AvMaxSim")
import models.pmi
reload(models.pmi)
from models.pmi import PMI_Model as ROCPMI_Model
#import pmi
#from pmi import PMI_Model as NarrativePMI_Model
#from narrative_dataset import Narrative_Dataset
# import similarity_score
# reload(similarity_score)
# import similarity_score as sim_score

from models.transformer import segment_and_tokenize, tokenize

# if os.path.isdir("/Volumes/G-DRIVE mobile with Thunderbolt/"):
#     model_filepath = "/Volumes/G-DRIVE mobile with Thunderbolt/" + model_filepath


# In[2]:

def load_transformer(filepath):
    with open(filepath + '/transformer.pkl', 'rb') as f:
        transformer = pickle.load(f)
    return transformer

def predict_avemax(model, X, y_choices):
    #import pdb;pdb.set_trace()
    choice_scores = []
    for x in X:
        #scores = [sim_score.score(x, choice, model, tail=0, head=-1) for choice in y_choices]
        scores = [sim_score.score(x, choice, model) for choice in y_choices]
        choice_scores.append(scores)
    choice_scores = numpy.array(choice_scores)
    return choice_scores
        


# In[3]:

if __name__ == "__main__":
    val_input_seqs,    val_output_choices, val_output_gold = get_cloze_data(filepath='cloze_test_ALL_val.tsv', 
                                                         mode='concat', flatten=True)


# In[50]:

'''CREATE NEW PMI MODEL FROM ROC STORIES'''

def make_pmi(stories, filepath):
    transformer = SequenceTransformer(min_freq=1, verbose=1, 
                                      replace_ents=False, filepath=filepath)
    stories, _, _ = transformer.fit_transform(X=stories)
    #transformer = load_transformer(filepath)
    #stories, _ = transformer.transform(X=stories)
    pmi_model = ROCPMI_Model(dataset_name=filepath)
    import pdb; pdb.set_trace()
    pmi_model.count_unigrams(stories)
    pmi_model.count_all_bigrams(stories)
    return pmi_model

if __name__ == '__main__':
    train_stories = get_train_stories(filepath='ROC-Stories.tsv', flatten=True) +                    get_train_stories(filepath='ROCStories_winter2017.csv', flatten=True)
    #train_stories = train_stories[:550]
    filepath = 'roc_pmi' + str(len(train_stories)) + "minfreq1"
    pmi_model = make_pmi(train_stories, filepath)


# In[51]:

'''EVALUATE NEW PMI MODEL FROM ROC STORIES'''

def eval_pmi(transformer, input_seqs, output_choices, output_gold):
    choice_scores = []
    index = 0
    input_seqs, output_choices = transformer.transform(X=input_seqs, y_seqs=output_choices)
    for input_seq, output_choices in zip(input_seqs, output_choices):
        choice1_score = pmi_model.score(sequences=[input_seq, output_choices[0]])
        choice2_score = pmi_model.score(sequences=[input_seq, output_choices[1]])
        choice_scores.append([choice1_score, choice2_score])
        index += 1
        if index % 200 == 0:
            print "predicted", index, "inputs"
        #print choice_scores
    choice_scores = numpy.array(choice_scores)
    pred_choices = numpy.argmax(choice_scores, axis=1)
    accuracy = numpy.mean(numpy.array(pred_choices) == output_gold)
    return choice_scores, pred_choices, accuracy
    
if __name__ == "__main__":
    import pdb;pdb.set_trace()
    #filepath = 'roc_pmi97027'
    transformer = load_transformer(filepath)
    pmi_model = ROCPMI_Model(filepath)
    choice_scores, pred_y, accuracy = eval_pmi(transformer, val_input_seqs, val_output_choices, val_output_gold)
    print("accuracy:", accuracy)
    show_predictions(X=val_input_seqs[-40:], y=val_output_gold[-40:],
                     prob_y=choice_scores[-40:], y_choices=val_output_choices[-40:])


# In[42]:

'''EVALUATE EXISTING PMI MODEL FOR NARRATIVE DATASET'''

def eval_pmi(input_seqs, output_choices, output_gold):
    choice_scores = []
    index = 0
    for input_seq, output_choices in zip(input_seqs, output_choices):
        input_seq = narrative_dataset.encode_sequence(tokenize(input_seq))
        output_choices = [narrative_dataset.encode_sequence(tokenize(choice)) for choice in output_choices]
        choice1_score = pmi_model.score(sequences=[input_seq, output_choices[0]])
        choice2_score = pmi_model.score(sequences=[input_seq, output_choices[1]])
        choice_scores.append([choice1_score, choice2_score])
        index += 1
        if index % 200 == 0:
            print "predicted", index, "inputs"
        #print choice_scores
    choice_scores = numpy.array(choice_scores)
    pred_choices = numpy.argmax(choice_scores, axis=1)
    accuracy = numpy.mean(numpy.array(pred_choices) == output_gold)
    return choice_scores, pred_choices, accuracy
    
if __name__ == "__main__":
    import pdb;pdb.set_trace()
    #import pdb; pdb.set_trace()
    filepath = "../generation/narrative_dataset_1million"
    narrative_dataset = Narrative_Dataset(filepath)
    pmi_model = NarrativePMI_Model(narrative_dataset)
    choice_scores, pred_y, accuracy = eval_pmi(val_input_seqs, val_output_choices, val_output_gold)
    print("accuracy:", accuracy)
    show_predictions(X=val_input_seqs[-40:], y=val_output_gold[-40:],
                     prob_y=choice_scores[-40:], y_choices=val_output_choices[-40:])


# In[40]:

def eval_avemax(embeddings, input_seqs, output_choices, output_gold):
    choice_scores = []
    index = 0
    for input_seq, output_choices in zip(input_seqs, output_choices):
        choice1_score = sim_score.score(input_seq, output_choices[0], embeddings, tail=0, head=-1)
        choice2_score = sim_score.score(input_seq, output_choices[1], embeddings, tail=0, head=-1)
        choice_scores.append([choice1_score, choice2_score])
        index += 1
        if index % 200 == 0:
            print "predicted", index, "inputs"
        #print choice_scores
    choice_scores = numpy.array(choice_scores)
    pred_choices = numpy.argmax(choice_scores, axis=1)
    accuracy = numpy.mean(numpy.array(pred_choices) == output_gold)
    return choice_scores, pred_choices, accuracy

if __name__ == '__main__':
    import pdb;pdb.set_trace()
    embeddings = sim_score.load_model('../AvMaxSim/vectors')
    choice_scores, pred_y, accuracy = eval_avemax(embeddings, val_input_seqs, val_output_choices, val_output_gold)
    print("accuracy:", accuracy)
    show_predictions(X=val_input_seqs[-40:], y=val_output_gold[-40:],
                     prob_y=choice_scores[-40:], y_choices=val_output_choices[-40:])
    


# In[41]:

# if __name__ == "__main__":
#     #val_input_seqs = [seqs[-1] for seqs in val_input_seqs]
#     sim_model = sim_score.load_model()
#     choice_scores = []
#     index = 0
#     for input_seqs, output_choices, output_gold in zip(val_input_seqs, val_output_choices, val_output_gold):
#         max_choice1_score = -numpy.inf
#         max_choice2_score = -numpy.inf
#         for seq in input_seqs:
#             choice1_score = sim_score.score(seq, output_choices[0], sim_model, tail=0, head=-1)
#             if choice1_score > max_choice1_score:
#                 max_choice1_score = choice1_score
#             choice2_score = sim_score.score(seq, output_choices[1], sim_model, tail=0, head=-1)
#             if choice2_score > max_choice2_score:
#                 max_choice2_score = choice2_score
#         choice_scores.append([max_choice1_score, max_choice2_score])
#         index += 1
#         if index % 200 == 0:
#             print "predicted", index, "inputs"
#         #print choice_scores
#     choice_scores = numpy.array(choice_scores)
#     pred_choices = numpy.argmax(choice_scores, axis=1)


# In[42]:

# if __name__ == "__main__":
#     choice_scores = []
#     index = 0
#     for input_seqs, output_choices, output_gold in zip(val_input_seqs, val_output_choices, val_output_gold):
#         max_choice1_score = -numpy.inf
#         max_choice2_score = -numpy.inf
#         for seq in input_seqs:
#             choice1_score = pmi_model.score(sequences=[seq, output_choices[0]])
#             if choice1_score > max_choice1_score:
#                 max_choice1_score = choice1_score
#                 print "choice 1:", max_choice1_score
#             choice2_score = pmi_model.score(sequences=[seq, output_choices[1]])
#             if choice2_score > max_choice2_score:
#                 max_choice2_score = choice2_score
#                 print "choice 2:", max_choice2_score
#         choice_scores.append([max_choice1_score, max_choice2_score])
#         index += 1
#         print "\n"
#         if index % 200 == 0:
#             print "predicted", index, "inputs"
#         #print choice_scores
#     choice_scores = numpy.array(choice_scores)
#     pred_choices = numpy.argmax(choice_scores, axis=1)

