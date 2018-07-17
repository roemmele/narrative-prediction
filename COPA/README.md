# Encoder-decoder Model for the Choice of Plausible Alternatives (COPA)
This repository contains Python code that trains and evaluates a model that predicts causally related sentences in the Choice of Plausible Alternatives (COPA) evaluation. See [here](http://people.ict.usc.edu/~gordon/copa.html) for a description of the COPA task and to download the associated items. The code here implements the best-performing model described in the paper [An Encoder-decoder Approach to Predicting Causal Relations in Stories](https://roemmele.github.io/publications/copa.pdf). To summarize, the model implemented here is a neural network-based encoder-decoder where the encoded input is a single segment (i.e. sentence or clause) in a text and the decoded output is a subsequent segment in that text. The overall idea is that the model is learning to predict "what happens next" from a given segment. Analogously, COPA items provide a sentence (the premise) and then elicit a prediction for the most likely cause or effect of the sentence given two candidate sentences (the alternatives). When provided with two segments, the model encodes the first segment and outputs probability scores that the second segment will appear after it in a text (suggesting the first sentence causes the second sentence). This can then be applied to prediction for a given COPA item by selecting the alternative with the higher mean score given by this model. See the paper for a more thorough explanation of this.

## Dependencies

To run this, clone the entire narrative-prediction repository and navigate to this directory (COPA/). 

This code should run in both Python 2 and Python 3. It requires these libraries: [Keras](keras.io) with either the [TensorFlow](https://www.tensorflow.org/) or [Theano](http://deeplearning.net/software/theano/) backend, [numpy](numpy.org), [pandas](http://pandas.pydata.org/), [h5py](http://www.h5py.org/), and [spaCy](https://spacy.io/). For spaCy, you will also need to download the en_core_web_md model by running "python -m spacy download en_core_web_md" (see [here](https://spacy.io/models/en#en_core_web_md)).

## Training

You can train a model from the command line by running encoder_decoder.py. In terms of required parameters, --train_seqs should be the path to a file with one text per line, as shown in the example file dataset/stories-example.csv. This example dataset come from the ROCStories corpus, which can be accessed [here](http://cs.rochester.edu/nlp/rocstories/). The ROCStories corpus provided the best performance on COPA for this particular model relative to other corpora, as conveyed in the paper. Additionally you must specify the filepaths --val_items and --test_items corresponding to the validation and test sets of the COPA evaluation, respectively. These files are already given in the datasets/ folder (they must be in the exact same format as the files downloaded from task description [page](http://people.ict.usc.edu/~gordon/copa.html)). You must supply --save_filepath indicating the folder where the trained model will be saved. There are several optional parameters whose default values are set to those that acheived the best COPA accuracy when the model was trained on the full ROCStories corpus (these values are explained further in the paper): the frequency threshold for including words in the lexicon (--min_freq), whether input-output segments correspond to sentences or intrasentential clauses (--segment_sents), the maximum number of words in the segments (--max_length), the distance between segments within which input-output pairs will be joined (--max_pair_distance), whether the encoder-decoder uses RNN or feed-forward layers (--recurrent), the number of dimensions in the encoding and decoding layers (--n_hidden_nodes), the number of iterations through the training data (--n_epochs), and the training batch size (--batch_size). Additionally, when training on large datasets, --chunk_size can be used to load the texts into chunks of this size rather than all at once in order to avoid memory errors. This was not needed when we trained this model on the ROCStories dataset of ~100,000 five-sentence stories with a 32GB-memory machine.

The script will evaluate the model on the validation COPA items periodically throughout training, and will save the model parameters each time the accuracy on these items is increased. After training is complete, the script will report the accuracy of the model on the COPA test set.

```
python encoder_decoder.py [-h] --train_seqs TRAIN_SEQS --val_items VAL_ITEMS
                          --test_items TEST_ITEMS --save_filepath
                          SAVE_FILEPATH [--min_freq MIN_FREQ]
                          [--segment_sents] [--max_length MAX_LENGTH]
                          [--max_pair_distance MAX_PAIR_DISTANCE]
                          [--recurrent] [--batch_size BATCH_SIZE]
                          [--n_hidden_nodes N_HIDDEN_NODES]
                          [--n_epochs N_EPOCHS] [--chunk_size CHUNK_SIZE]
```
### Parameters:
```
  --train_seqs TRAIN_SEQS, -train TRAIN_SEQS
                        Specify filename (.csv) containing text used as
                        training data.
  --val_items VAL_ITEMS, -val VAL_ITEMS
                        Specify filename (XML) containing COPA items in
                        validation set.
  --test_items TEST_ITEMS, -test TEST_ITEMS
                        Specify filename (XML) containing COPA items in test
                        set.
  --save_filepath SAVE_FILEPATH, -save SAVE_FILEPATH
                        Specify the directory filepath where the trained model
                        should be stored.
  --min_freq MIN_FREQ, -freq MIN_FREQ
                        Specify frequency threshold for including words in
                        model lexicon, such that only words that appear in the
                        training sequences at least this number of times will
                        be added (all other words will be mapped to a generic
                        <UNKNOWN> token). Default is 5.
  --segment_sents, -sent
                        Specify if the segments in the input-output pairs
                        should be sentences rather than intrasentential
                        clauses (see paper). If not given, clause-based
                        segmentation will be used.
  --max_length MAX_LENGTH, -len MAX_LENGTH
                        Specify the maximum length of the input and output
                        segements in the training data (in terms of number of
                        words). Pairs with longer sequences will be filtered.
                        Default is 20.
  --max_pair_distance MAX_PAIR_DISTANCE, -dist MAX_PAIR_DISTANCE
                        Specify the distance window in which neighboring
                        segments will be joined into input-output pairs. For
                        example, if this parameter is 3, all segments that are
                        separated by 3 or fewer segments in a particular
                        training text will be added as pairs. Default is 4.
  --recurrent, -rec     Specify if the model should use RNN (GRU) layers. If
                        not specified, feed-forward layers will be used, and
                        the sequential ordering of words in the segments will
                        be ignored.
  --batch_size BATCH_SIZE, -batch BATCH_SIZE
                        Specify number of sequences in batch during training.
                        Default is 100.
  --n_hidden_nodes N_HIDDEN_NODES, -hid N_HIDDEN_NODES
                        Specify number of dimensions in the encoder and
                        decoder layers. Default is 500.
  --n_epochs N_EPOCHS, -epoch N_EPOCHS
                        Specify the number of epochs the model should be
                        trained for. Default is 50.
  --chunk_size CHUNK_SIZE, -chunk CHUNK_SIZE
                        If dataset is large, specify this parameter to load
                        training sequences in chunks of this size instead of
                        all at once to avoid memory issues. For smaller
                        datasets (e.g. the ROCStories corpus), it is much
                        faster to load entire dataset prior to training. This
                        will be done by default if chunk size is not given.
```
### Example
```
python encoder_decoder.py --train_seqs dataset/stories-example.csv --val_items dataset/copa-dev.xml --test_items dataset/copa-test.xml --save_filepath example_model
```

After training, the example_model/ folder will contain the trained model files transformer.pkl (which represents the lexicon), and classifier.pkl and classifier.h5 (the encoder-decoder model itself).

## Loading a trained model

After training a model, you can reload it in a different Python session by specifying the filepath of the model in the load_model() function. For example, to load the example model from above and evaluate it on the COPA test items, run:

```
from encoder_decoder import *
model = load_model(filepath="example_model")
test_accuracy = eval_copa(model, data_filepath="dataset/copa-test.xml")
```

## Results

As reported in the paper, when trained on all 97,027 stories in the ROCStories corpus, this approach with the default parameters defined here obtained 66.0% accuracy on the validation set of COPA and 66.2% on the test set.
