# RNN-based Binary Classifier for the ROC Story Cloze Test
This repository contains Python code that trains and evaluates a model that predicts the endings of stories in the Story Cloze Test, using the corresponding ROCStories corpus for training (see [here](http://cs.rochester.edu/nlp/rocstories/) for details about the task and dataset). The code here implements the best-performing model described in the paper [An RNN-based Binary Classifier for the Story Cloze Test](https://roemmele.github.io/publications/eacl2017_storyclozetest_cameraready.pdf). Very broadly speaking, this model represents stories according to vector (specifically, skipthought vector) representations of their sentences which are then encoded by a recurrent (GRU) neural layer. The top layer of the model is a binary classifier that is trained to distinguish "correct" endings for stories given in the ROCStories corpus from "incorrect" endings that are artificially generated from the same corpus. Here the incorrect endings are generated using the "random" and "backward" methods described in the paper. When applied to an item in the Story Cloze Test, the model predicts the probability of each candidate ending being correct for the given story, and selects the ending with the higher score. See the paper for more details.

## Dependencies

This code should run in both Python 2 and Python 3. It requires these libraries: [Keras](keras.io) with either the [TensorFlow](https://www.tensorflow.org/) or [Theano](http://deeplearning.net/software/theano/) backend, [numpy](numpy.org), [pandas](http://pandas.pydata.org/), [h5py](http://www.h5py.org/), and [spaCy](https://spacy.io/).

The model makes use of [skipthought vectors](https://github.com/ryankiros/skip-thoughts) to represent story sentences. You will need to download this repository. Follow the instructions given in their README. If you download the model and embedding files to the main directory of the repository, then specifically set the paths in skipthoughts.py to the following:

```
path_to_models = os.path.dirname(os.path.realpath(__file__)) + "/"
path_to_tables = os.path.dirname(os.path.realpath(__file__)) + "/"
```

When you run the code, you'll need to supply the location of this directory (see below).

## Training

You can train a model from the command line by running story_cloze_test.py. In terms of required parameters, --train_seqs should be the path to a tab-separated (TSV) file containing the ROCStories dataset in same format as the original dataset that can be accessed [here]((http://cs.rochester.edu/nlp/rocstories/). The file ROC-Stories-example.tsv is included here in the datasets/ folder and shows an example of this format for 10 stories. Additionally you must specify the filepaths --val_seqs and --test_seqs corresponding to the validation and test sets of the cloze evaluation, respectively (also in TSV format). These can also be accessed through the given link, and there are examples given in datasets/. You must supply --save_filepath indicating the folder where the trained model will be saved. Finally, as mentioned above, you must indicate the filepath of the skipthoughts directory that contains all code and models associated with the skipthought vectors. Optionally, you can specify the number of "backward" sampled endings and "random" sampled endings that are used as negative training instances (see the above paper for what this means), and then the batch size, number of hidden layers, number of hidden dimensions, and number of training epochs for the classifier. By default, these values are set to those that produced the best result reported in the paper (see below).

During training, the skipthought vectors of the sentences in the training set are saved to disk (seqs1.npy and seqs2.npy in --save_filepath) in order to make it more efficient to dynamically load them into the model. Because of the size of the skipthought vectors, these are huge files (~15GB and ~4GB, respectively), so make sure you have enough disk space.

The script will evaluate the model on the validation cloze items periodically throughout training, and will save the model parameters each time the accuracy on these items is increased. After training is complete, the script will report the accuracy of the model on the test cloze items.

```
python story_cloze_test.py [-h] --train_seqs TRAIN_SEQS --val_items VAL_ITEMS
                           --test_items TEST_ITEMS --save_filepath
                           SAVE_FILEPATH --skip_filepath SKIP_FILEPATH
                           [--n_backward N_BACKWARD] [--n_random N_RANDOM]
                           [--batch_size BATCH_SIZE]
                           [--n_hidden_layers N_HIDDEN_LAYERS]
                           [--n_hidden_nodes N_HIDDEN_NODES]
                           [--n_epochs N_EPOCHS]
```
### Parameters:
```
  --train_seqs TRAIN_SEQS, -train TRAIN_SEQS
                        Specify filename (.tsv) containing ROCStories used as
                        training data.
  --val_items VAL_ITEMS, -val VAL_ITEMS
                        Specify filename (.tsv) containing cloze items in
                        validation set.
  --test_items TEST_ITEMS, -test TEST_ITEMS
                        Specify filename (.tsv) containing cloze items in test
                        set.
  --save_filepath SAVE_FILEPATH, -save SAVE_FILEPATH
                        Specify the directory filepath where the trained model
                        should be stored.
  --skip_filepath SKIP_FILEPATH, -skip SKIP_FILEPATH
                        Specify the directory filepath where the model for the
                        skipthought vectors is located.
  --n_backward N_BACKWARD, -bkwrd N_BACKWARD
                        Specify number of "backward" generated endings (i.e.
                        sentences selected from initial story) to include in
                        incorrect training samples. Default is 2.
  --n_random N_RANDOM, -rand N_RANDOM
                        Specify number of "random" generated endings (i.e.
                        endings randomly selected from other stories) to
                        include in incorrect training samples. Default is 4.
  --batch_size BATCH_SIZE, -batch BATCH_SIZE
                        Specify number of sequences in batch during training.
                        Default is 100.
  --n_hidden_layers N_HIDDEN_LAYERS, -lay N_HIDDEN_LAYERS
                        Specify number of recurrent hidden layers in model.
                        Default is 1.
  --n_hidden_nodes N_HIDDEN_NODES, -hid N_HIDDEN_NODES
                        Specify number of nodes in each recurrent hidden
                        layer. Default is 1000.
  --n_epochs N_EPOCHS, -epoch N_EPOCHS
                        Specify the number of epochs the model should be
                        trained for. Default is 10.
```
### Example
```
python story_cloze_test.py --train_seqs dataset/ROC-Stories-example.tsv --val_items dataset/cloze_val_example.tsv --test_items dataset/cloze_test_example.tsv --save_filepath example_model --skip_filepath skip-thoughts
```

Here, skip-thoughts/ is in the same directory as where the script is being run, for example. After training, the example_model/ folder will contain the trained model files classifier.pkl and classifier.h5.

### Loading a trained model

After training a model, you can reload it in a different Python session by specifying the filepath of the model as well as the skipthoughts model filepath:

```
from story_cloze_test import *
model = load_model(save_filepath, skip_filepath)
```

You can then load the cloze items and evaluate the model on them, just as done in story_cloze_test.py. For example:

```
model = load_model("example_model", "skip-thoughts").
test_input_seqs, test_output_choices, test_output_gold = get_cloze_data("dataset/cloze_test_example.tsv")
test_accuracy = evaluate_roc_cloze(model, test_input_seqs, test_output_choices, test_output_gold)

### Performance on Story Cloze Test

As reported in the paper, this approach with the default parameters defined here obtained 66.2% accuracy on the validation set of the Story Cloze Test and 66.9% on the test set.




