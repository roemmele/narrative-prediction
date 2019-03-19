# lm-generation
This repository contains Python code that uses Keras to train a Recurrent Neural Network language model and generate new sequences using the model. This work specifically focuses on models that generate a continuation of given sequence (e.g. given the beginning of a story, generate the next sentence in the story). This code is associated with the models described in the papers: [Evaluating Story Generation Systems Using Automated Linguistic Analyses](https://roemmele.github.io/publications/fiction_generation.pdf), [Automated Assistance for Creative Writing with an RNN Language Model](https://roemmele.github.io/publications/creative-help-demo.pdf), and [Linguistic Features of Helpfulness in Automated Support for Creative Writing](https://roemmele.github.io/publications/creative-help-evaluation.pdf). This code is also currently integrated into the [Creative Help](https://fiction.ict.usc.edu/creativehelp/) application that provides automated assistance for story writing. See below for instructions on using the already-trained Creative Help model to generate sequences.

## Dependencies

To run this, clone the entire narrative-prediction repository and navigate to this directory (lm-generation/). 

This code should run in both Python 2 (>= 2.7) and Python 3. It requires these libraries: [Keras](keras.io), [numpy](numpy.org), [pandas](http://pandas.pydata.org/), [h5py](http://www.h5py.org/), and [spaCy](https://spacy.io/). For spaCy, you will also need to download the en_core_web_md model by running "python -m spacy download en_core_web_md" (see [here](https://spacy.io/models/en#en_core_web_md)). 

<!-- \*<sup>Ideally, you'd be able to use the TensorFlow backend of Keras instead of Theano, but the code that does the generation by sampling from the LM probability distribution is written directly in Theano for the purpose of speed (see the pred_batch_next_words() method for the RNNLM class in models/classifier.py). Eventually I will try to remove this dependency since Theano is no longer actively being developed.</sup> -->

## Training

You can train a model from the command line by running train_generation.py. The --train_seqs parameter should be the path to a text file with one sequence per line, as can be seen in the file example_stories.csv included here. It should be possible to run this using a very large training set, as the sequences will be loaded in chunks. You must also supply --save_filepath indicating the folder where the trained model will be saved.

```
python train_generation.py --train_seqs TRAIN_SEQS [--use_features]
                           [--use_pos] --save_filepath SAVE_FILEPATH
                           [--min_freq MIN_FREQ] [--generalize_ents]
                           [--batch_size BATCH_SIZE]
                           [--n_timesteps N_TIMESTEPS]
                           [--n_hidden_layers N_HIDDEN_LAYERS]
                           [--n_embedding_nodes N_EMBEDDING_NODES]
                           [--n_hidden_nodes N_HIDDEN_NODES]
                           [--n_pos_nodes N_POS_NODES]
                           [--n_feature_nodes N_FEATURE_NODES]
                           [--n_epochs N_EPOCHS]
```
### Parameters:
```
  --train_seqs TRAIN_SEQS, -train TRAIN_SEQS
                        Specify filename containing training sequences.
  --use_features, -ufeat
                        If given, the model will be trained on noun features
                        in additon to word sequences.
  --use_pos, -upos      If given, the model will be trained on POS tags in
                        addition to word sequences.
  --save_filepath SAVE_FILEPATH, -save SAVE_FILEPATH
                        Specify the directory filepath where the trained model
                        should be stored.
  --min_freq MIN_FREQ, -freq MIN_FREQ
                        Specify frequency threshold for including words in
                        model lexicon, such that only words that appear in the
                        training sequences at least this number of times will
                        be added. Default is 5.
  --generalize_ents, -ents
                        Specify that named entities should be replaced with a
                        general entity type token (e.g. ENT_PERSON_0,
                        ENT_ORG_1). If not given, entities will be treated
                        like all other tokens.
  --batch_size BATCH_SIZE, -batch BATCH_SIZE
                        Specify number of sequences in batch during training.
                        Default is 25.
  --n_timesteps N_TIMESTEPS, -step N_TIMESTEPS
                        Specify number of timesteps (tokens) in a batch of
                        sequences that should be read at a single time between
                        updates. Default is 15.
  --n_hidden_layers N_HIDDEN_LAYERS, -lay N_HIDDEN_LAYERS
                        Specify number of recurrent hidden layers in model.
                        Default is 2.
  --n_embedding_nodes N_EMBEDDING_NODES, -emb N_EMBEDDING_NODES
                        Specify number of nodes in word embedding layer that
                        feeds into recurrent hidden layer. Default is 300.
  --n_hidden_nodes N_HIDDEN_NODES, -hid N_HIDDEN_NODES
                        Specify number of nodes in recurrent hidden layer.
                        Default is 500.
  --n_pos_nodes N_POS_NODES, -pos N_POS_NODES
                        For model with POS tags, specify number of nodes in
                        POS hidden layer. Default is 100.
  --n_feature_nodes N_FEATURE_NODES, -feat N_FEATURE_NODES
                        For model with features, specify number of nodes in
                        feature hidden layer. Default is 100.
  --n_epochs N_EPOCHS, -epoch N_EPOCHS
                        Specify the number of epochs the model should be
                        trained for. Default is 10.
```
### Example
```
python train_generation.py --train_seqs example_stories.csv --save_filepath example_model/
```
After training, the example_model/ folder will contain the trained model files transformer.pkl, classifier.pkl, and classifier.h5.

This is equivalent to running this Python code:
```
from train_generation import *
model = create_model(save_filepath='example_model', use_features=False, use_pos=False, min_freq=5, 
                    generalize_ents=False, batch_size=25, n_timesteps=15, n_hidden_layers=2, 
                    n_embedding_nodes=300, n_hidden_nodes=500, n_pos_nodes=100, n_feature_nodes=100)
train_model(train_seqs='example_stories.txt', model, n_epochs=10)
```
## Generating
Generation works by taking an initial sequence (context) and producing one or more sequences intended to follow it in the text. To use a trained model to perform this task for a given set of context sequences, you can run generate_sequences.py. The --context_seqs parameter should be the path to a text file with one sequence per line (same format as the train sequences file), as can be seen in the example_contexts.txt file included here. You'll also need to supply --model_filepath, the folder where the trained model files were saved, as well as --save_filepath, the .csv filepath where the generated sequences will be saved. In the case that multiple sequences per context are generated, they will be comma-separated on a single line such that each line of generated sequences corresponds to the position their context sequence appears in the file. 

We trained a model on approximately 8,000 books in the Toronto BookCorpus dataset. This is the model that is currently demoed in Creative Help. You can download this model [here](https://drive.google.com/file/d/1nnaBzAoAJnFXXwHMVNyADMtiRr8UcAAB/view?usp=sharing) and use it in place of the example model included here.


```
python generate_sequences.py [-h] --context_seqs CONTEXT_SEQS --model_filepath
                             MODEL_FILEPATH --save_filepath SAVE_FILEPATH
                             [--gen_mode {random,max}]
                             [--temperature TEMPERATURE]
                             [--n_gen_per_context N_GEN_PER_CONTEXT]
                             [--n_sents_per_seq N_SENTS_PER_SEQ]
                             [--n_context_sents N_CONTEXT_SENTS]
                             [--eos_tokens EOS_TOKENS [EOS_TOKENS ...]]
```
### Parameters:
```
  --context_seqs CONTEXT_SEQS, -cont CONTEXT_SEQS
                        Specify filename (.csv) containing context sequences.
  --model_filepath MODEL_FILEPATH, -modfp MODEL_FILEPATH
                        Specify the filepath directory where the trained model
                        is stored.
  --save_filepath SAVE_FILEPATH, -save SAVE_FILEPATH
                        Specify the .csv filename where the generated
                        sequences should be saved.
  --gen_mode {random,max}, -gmode {random,max}
                        Specify what method should be used to generate
                        sequences: either through random sampling (random) or
                        by taking the max probability (max). Default is
                        random.
  --temperature TEMPERATURE, -temp TEMPERATURE
                        When generation mode is random, specify the
                        temperature variable for sampling. Default is 1 (most
                        random).
  --n_gen_per_context N_GEN_PER_CONTEXT, -ngen N_GEN_PER_CONTEXT
                        Specify how many sequences should be generated for
                        each context sequence. Default is one sequence per
                        context sequence.
  --n_sents_per_seq N_SENTS_PER_SEQ, -nsents N_SENTS_PER_SEQ
                        Specify the length of generated sequences in terms of
                        the number of sentences. Default is one sentence per
                        sequence.
  --n_context_sents N_CONTEXT_SENTS, -ncont N_CONTEXT_SENTS
                        Specify if the context should be truncated so that
                        only the N most recent sentences are taken into
                        account when generating the next sequence. If the
                        model uses feature vectors, the features will still
                        take into account the whole context. Default is -1,
                        which means all context sentences will be included.
  --eos_tokens EOS_TOKENS [EOS_TOKENS ...], -eos EOS_TOKENS [EOS_TOKENS ...]
                        If sentence boundaries should be determined by the
                        occurence of specific end-of-sentence tokens, specify
                        these tokens. Default is None, in which case sentence
                        boundaries will be automatically inferred by spaCy's
                        segmenter.
```
### Example
```
python generate_sequences.py --context_seqs example_contexts.txt --model_filepath example_model/ --save_filepath example_generated.csv
```
This is equivalent to running the Python code:
```
from generate_sequences import *
model = RNNLMPipeline.load(filepath=example_model)
gen_seqs = generate(context_seqs='example_contexts.csv', model=model, save_filepath='example_generated.csv', 
                    gen_mode='random', temperature=1.0, n_gen_per_context=1, n_sents_per_seq=1, 
                    n_context_sents=-1, eos_tokens=None)
```
After generating the sequences, the generate() function will print a sample of the generated sequences for each context.
