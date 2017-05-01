
# coding: utf-8

# In[1]:

from __future__ import unicode_literals
import sys, pandas, numpy, random
from gensim.models import Word2Vec
from ast import literal_eval

sys.path.append('../')
import models.transformer
reload(models.transformer)
from models.transformer import SequenceTransformer
import models.pipeline
reload(models.pipeline)
from models.pipeline import RNNPipeline
import models.classifier
reload(models.classifier)
from models.classifier import RNNLM

numpy.set_printoptions(precision=3, suppress=True)
rng = numpy.random.RandomState(0)


# In[7]:

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

def check_punct(stories, eos_markers=["\"", "\'", ".", "?", "!"]):
    #make sure all stories end with an end-of-sentence marker; if not, insert period
    for story_idx, story in enumerate(stories):
        for sent_idx, sent in enumerate(story):
            if sent[-1] not in eos_markers:
                #import pdb;pdb.set_trace()
                sent += '.' #add period
                stories[story_idx][sent_idx] = sent
    return stories      

def get_train_input_outputs(filepath, mode='adjacent', flatten=False):
    stories = get_train_stories(filepath)
    input_seqs, output_seqs = prep_input_outputs(stories, mode)
    if flatten:
        #combine sents for each story into single string
        input_seqs = [" ".join(seq) for seq in input_seqs]
    return input_seqs, output_seqs

def get_cloze_gold_stories(input_seqs, output_choices, output_gold):
    stories = [seqs + [choices[gold]] for seqs, choices, gold in zip(input_seqs, output_choices, output_gold)]
    return stories  
                                    
def get_cloze_data(filepath, mode='adjacent', flatten=False):
    cloze_test = pandas.DataFrame.from_csv(filepath, sep='\t', encoding='utf-8')
    #ensure readable encoding
    cloze_test = cloze_test[cloze_test[['InputSentence1', 'InputSentence2', 'InputSentence3', 
                            'InputSentence4', 'RandomFifthSentenceQuiz1', 'RandomFifthSentenceQuiz2']].apply(
                lambda sentences: numpy.all([type(sentence) is unicode for sentence in sentences]), axis=1)]
    input_seqs, output_choices, output_gold = prep_cloze_test(cloze_test, mode)
    #a few stories are missing end-of-sentence punctuation, so fix it
    input_seqs = check_punct(input_seqs)
    if flatten:
        #combine sents for each story into single string
        input_seqs = [" ".join(seq) for seq in input_seqs]
    return input_seqs, output_choices, output_gold

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

def prep_cloze_test(cloze_test, mode='adjacent'):
    #import pdb;pdb.set_trace()
    #if mode == 'adjacent':
    input_seqs = cloze_test[['InputSentence1', 'InputSentence2',
                            'InputSentence3', 'InputSentence4']].values.tolist()
    output_choices = pandas.concat([cloze_test[['RandomFifthSentenceQuiz1',
                                               'RandomFifthSentenceQuiz2']]], axis=1).values.tolist()
    #substract 1 from ending choice index so indices start at 0
    output_gold = pandas.concat([cloze_test["AnswerRightEnding"]], axis=1).values.flatten() - 1
    assert(len(input_seqs) == len(output_choices) == len(output_gold))
    return input_seqs, output_choices, output_gold

def prep_train_seqs(seqs, y_seqs, neg_y_seqs, trunc_seqs=None, trunc_y_seqs=None, return_pairs=False):
    #import pdb;pdb.set_trace()
    n_neg_per_seq = len(neg_y_seqs[0])
        
    if return_pairs:
        seqs = [seq_ for seq in seqs for seq_ in [seq] * n_neg_per_seq]
        y_seqs = [[neg_y_seq, y_seq] for y_seq, neg_y_seq in zip(y_seqs, neg_y_seqs)
                                        for y_seq in neg_y_seqs]
        y = rng.randint(low=0, high=2, size=len(y_seqs))
        y_seqs = [[seq[not_y], seqs[y]]
                    for seqs, y, y_not in zip(y_seqs, y, numpy.logical_not(y))]

    else:
        seqs = list(seqs) + [seq_ for seq in seqs for seq_ in [seq] * n_neg_per_seq]
        y = [1] * len(y_seqs) + [0] * (len(y_seqs) * n_neg_per_seq)
        y_seqs = list(y_seqs) + [neg_seq for neg_seqs in neg_y_seqs for neg_seq in neg_seqs]
        if trunc_seqs is not None:
            #import pdb;pdb.set_trace()
            seqs.extend(trunc_seqs)
            y_seqs.extend(trunc_y_seqs)
            y.extend([1] * len(trunc_seqs))  
        
    assert(len(seqs) == len(y_seqs) == len(y))
    y = numpy.array(y)
    seqs, y_seqs, y = shuffle_seqs(seqs, y_seqs, y)
        
    if type(seqs[0]) in [numpy.ndarray, numpy.memmap]:
        seqs = numpy.array(seqs)
    if type(y_seqs[0]) in [numpy.ndarray, numpy.memmap]:
        y_seqs = numpy.array(y_seqs)
    
    #print "training on total of", len(seqs), "instances"

    return seqs, y_seqs, y    


# In[1]:

def encode_skipthought_seqs(seqs, encoder_module, encoder, encoder_dim=4800, memmap=False, filepath=None):
    #provide module because skipthoughts has a different encoder function for their pre-trained model
    #import pdb;pdb.set_trace()
    n_seqs = len(seqs)
    if type(seqs[0]) in (list, tuple):
        seq_length = len(seqs[0])
        #seqs_shape = (n_seqs, len(seqs[0]), encoder_dim)
        #flatten seqs
        seqs = [sent for seq in seqs for sent in seq]
    else:
        seq_length = 1
    seqs_shape = (len(seqs), encoder_dim)
    if memmap:
        assert(filepath is not None)
        encoded_seqs = numpy.memmap(filepath, dtype='float64',
                                    mode='w+', shape=seqs_shape)
    else:
        encoded_seqs = numpy.zeros(seqs_shape)

    chunk_size = 500000
    for seq_idx in range(0, len(seqs), chunk_size):
        #memory errors if encoding a large number of stories
        #encoded_chunk = encoder_module.encode(encoder, seqs[seq_idx:seq_idx + chunk_size])
        #encoded_chunk = encoded_chunk.reshape(-1, seqs_shape[1], seqs_shape[-1])
        encoded_seqs[seq_idx:seq_idx + chunk_size] = encoder_module.encode(encoder, 
                                                                           seqs[seq_idx:seq_idx + chunk_size])
    if seq_length > 1:
        encoded_seqs = encoded_seqs.reshape(n_seqs, seq_length, encoder_dim)
    del encoded_seqs


# In[12]:

def shuffle_seqs(seqs, y_seqs, y):
    shuffled_idxs = rng.permutation(len(seqs))
    seqs = [seqs[idx] for idx in shuffled_idxs]
    y_seqs = [y_seqs[idx] for idx in shuffled_idxs]
    y = y[shuffled_idxs]
    return seqs, y_seqs, y

def make_chunks(seqs, y_seqs, y, seqs_per_chunk):
    chunks_seqs = [seqs[seq_index:seq_index + seqs_per_chunk]
                   for seq_index in range(0, len(seqs), seqs_per_chunk)]
    chunks_y_seqs = [y_seqs[seq_index:seq_index + seqs_per_chunk] 
                     for seq_index in range(0, len(seqs), seqs_per_chunk)]
    chunks_y = [y[seq_index:seq_index + seqs_per_chunk] 
                for seq_index in range(0, len(seqs), seqs_per_chunk)]
    return chunks_seqs, chunks_y_seqs, chunks_y


# In[1]:

def load_train_val_data(flatten=False):
    train_input_seqs1, train_output_seqs1 = get_train_input_outputs(filepath='ROC-Stories.tsv', mode='concat',
                                                                   flatten=flatten)
    train_input_seqs2, train_output_seqs2 = get_train_input_outputs(filepath='ROCStories_winter2017.csv',
                                                                  mode='concat', flatten=flatten)
    train_input_seqs = train_input_seqs1 + train_input_seqs2
    train_output_seqs = train_output_seqs1 + train_output_seqs2
    #get cloze items in validation set
    val_input_seqs,    val_output_choices, val_output_gold = get_cloze_data(filepath='cloze_test_ALL_val.tsv', 
                                                         mode='concat', flatten=flatten)
    return train_input_seqs, train_output_seqs, val_input_seqs, val_output_choices, val_output_gold


# In[52]:

if __name__ == "__main__":
    n_train_seqs = 10000
    #embeddings.save('roc_embeddings')
    #embeddings = Word2Vec.load('roc_embeddings')
    #embeddings = similarity_score.load_model("../AvMaxSim/vectors")
    #lm = load_lm('roc_lm97027_batchtrained_clio')
    filepath = 'roc_seqbinary' + str(n_train_seqs)
    #import pdb;pdb.set_trace()
    seq_binary = RNNPipeline(steps=[("transformer", SequenceTransformer(min_freq=1, verbose=1,#pad_seq=True, 
                                                                   word_embeddings=embeddings, 
                                                                    #sent_encoder=sent_encoder, 
                                                                    embed_y=True,
                                                                   replace_ents=False, reduce_emb_mode='mean',
                                                                   filepath=filepath)),
                                ("classifier", SeqBinaryClassifier(batch_size=50, nb_epoch=10, verbose=1,
                                                                   n_hidden_layers=1, n_embedding_nodes=300,
                                                                   #n_embedding_nodes=embeddings.vector_size,
                                                                   n_hidden_nodes=1000, context_size=4,
                                                                   embedded_input=False,
                                                                   filepath=filepath, use_dropout=False))])
                                                                    #, #pairs=True,
                                                                    #clipvalue=10.0))])
            
    seq_binary.named_steps['transformer'].fit(X=train_input_seqs[:n_train_seqs], 
                                              y_seqs=train_output_seqs[:n_train_seqs])
    train_seqs, train_y_seqs, train_y = get_pos_neg_seqs(train_input_seqs[:n_train_seqs], 
                                                         train_output_seqs[:n_train_seqs], 
                                                         n_neg_per_seq=3, lm=lm, gen_temps=(0.75,))
    seq_binary.fit(X=train_seqs, y_seqs=train_y_seqs, y=train_y)

    import pdb;pdb.set_trace()
    prob_y, pred_y, accuracy = seq_binary.predict(X=val_input_seqs,
                                                  y_seqs=val_output_choices, 
                                                  y=val_output_gold)
    print "accuracy:", accuracy
    show_predictions(X=val_input_seqs[-20:], y=val_output_gold[-20:],
                     prob_y=prob_y[-20:], y_choices=val_output_choices[-20:])


# In[49]:

def train_wordemb_by_chunk(input_seqs, output_seqs, embeddings, n_neg_per_seq=4, n_chunks=4, nb_epoch=10):
    
#     input_seqs, output_seqs, _ = seq_binary.named_steps['transformer'].fit_transform(X=input_seqs,
#                                                                            y_seqs=output_seqs)
    seq_binary.named_steps['transformer'].fit(X=input_seqs, y_seqs=output_seqs)

    seqs_per_chunk = len(input_seqs) / n_chunks
    print "training seq_binary classifier on a total of ", len(input_seqs),             "positive examples and", len(input_seqs) * n_neg_per_seq, " negative examples, for", nb_epoch,             "epochs (sequences divided into", n_chunks, "chunks)"

    nb_epoch = 5
    import pdb;pdb.set_trace()
    for epoch in range(nb_epoch):
        #reseed generator
        rng.seed(123)
        print "TRAINING EPOCH {}/{}".format(epoch + 1, nb_epoch)
        for chunk_idx in range(n_chunks):
            train_seqs, train_y_seqs,             train_y = get_pos_neg_seqs(input_seqs[chunk_idx * seqs_per_chunk: (chunk_idx + 1) * seqs_per_chunk],
                                       output_seqs[chunk_idx * seqs_per_chunk: (chunk_idx + 1) * seqs_per_chunk],
                                        n_neg_per_seq=n_neg_per_seq, lm=lm, gen_temps=(0.75,))
                                                            #, return_pairs=True) #lm=lm_pipeline)

#             seq_binary.named_steps['classifier'].fit(X=train_seqs,
#                                                      y_seqs=train_y_seqs,
#                                                      y=train_y, nb_epoch=1)
            seq_binary.fit(X=train_seqs, y_seqs=train_y_seqs, y=train_y, classifier__nb_epoch=1)

if __name__ == "__main__":
    #embeddings.save('roc_embeddings')
    embeddings = similarity_score.load_model("../AvMaxSim/vectors")
    #embeddings = Word2Vec.load('roc_embeddings')
    n_embedding_nodes = embeddings.vector_size
    n_train_seqs = 100
    lm = load_lm('roc_lm97027_batchtrained_clio')
    filepath = 'roc_seqbinary' + str(n_train_seqs)
    seq_binary = RNNPipeline(steps=[("transformer", SequenceTransformer(min_freq=1, verbose=1, 
                                                                   word_embeddings=embeddings, embed_y=True,
                                                                   replace_ents=False, reduce_emb_mode='mean',
                                                                   filepath=filepath)),
                                    ("classifier", SeqBinaryClassifier(batch_size=50, verbose=1,
                                                                   n_hidden_layers=1,
                                                                   n_embedding_nodes=n_embedding_nodes,
                                                                   n_hidden_nodes=1000, context_size=4,
                                                                   filepath=filepath, 
                                                                  #pairs=True,
                                                                   use_dropout=False))])
    
    train_wordemb_by_chunk(input_seqs=train_input_seqs[-n_train_seqs:],
                           output_seqs=train_output_seqs[-n_train_seqs:],
                           embeddings=embeddings, n_neg_per_seq=3)
    import pdb;pdb.set_trace()
    prob_y, pred_y, accuracy = seq_binary.predict(X=val_input_seqs, 
                                                  y_seqs=val_output_choices, 
                                                  y=val_output_gold)
    print "accuracy:", accuracy
    show_predictions(X=val_input_seqs[-20:], y=val_output_gold[-20:],
                     prob_y=prob_y[-20:], y_choices=val_output_choices[-20:])


# In[48]:

if __name__ == "__main__":
    #lm = load_pipeline('roc_lm44362_batchtrained_clio')
    #train_stories = get_train_stories(filepath='ROC-Stories.tsv')[:10]
#     samp_stories = train_stories[-20:]
    context_seqs = [story[:-1] for story in train_stories]
    import pdb;pdb.set_trace()
    gen_sents, p_sents = generate_sents(lm, context_seqs, n_best=1, batch_size=1, 
                                        mode='random', temp=0.9, n_words=35)
    show_gen_sents(context_seqs, gen_sents, p_sents)


# In[36]:

def train_skipthoughts_classifier(input_seqs, output_seqs, n_neg_per_seq=4, n_chunks=4, nb_epoch=10):
    
    input_seqs = encode_skipthought_seqs(input_seqs, skipthoughts, sent_encoder)
    output_seqs = encode_skipthought_seqs(output_seqs, skipthoughts, sent_encoder)
    #import pdb;pdb.set_trace()
#     train_seqs, train_y_seqs, train_y = get_pos_neg_seqs(input_seqs,
#                                            output_seqs,
#                                            n_neg_per_seq=n_neg_per_seq)#, return_pairs=True) #lm=lm_pipeline)

    seqs_per_chunk = len(input_seqs) / n_chunks
#     seqs_per_chunk = len(input_seqs)
#     n_chunks = len(train_seqs) / seqs_per_chunk
    print "training seq_binary classifier for", nb_epoch, "epochs (sequences divided into", n_chunks, "chunks)"
    #train_seqs, train_y_seqs, train_y = shuffle_seqs(train_seqs, train_y_seqs, train_y)
    #chunks_seqs, chunks_y_seqs, chunks_y = make_chunks(train_seqs, train_y_seqs, train_y, seqs_per_chunk)

    for epoch in range(nb_epoch):
        rng.seed(123)
        print "TRAINING EPOCH {}/{}".format(epoch + 1, nb_epoch)
        #for chunk_seqs, chunk_y_seqs, chunk_y in zip(chunks_seqs, chunks_y_seqs, chunks_y):
        for chunk_idx in range(n_chunks):
            train_seqs, train_y_seqs,             train_y = get_pos_neg_seqs(input_seqs[chunk_idx * seqs_per_chunk: (chunk_idx + 1) * seqs_per_chunk],
                                       output_seqs[chunk_idx * seqs_per_chunk: (chunk_idx + 1) * seqs_per_chunk],
                                        n_neg_per_seq=n_neg_per_seq) #, return_pairs=True) #lm=lm_pipeline)
            seq_binary_classifier.fit(X=train_seqs, y_seqs=train_y_seqs, y=train_y, nb_epoch=1)

#         print "TRAINING EPOCH {}/{}".format(epoch + 1, nb_epoch)
#         for chunk_seqs, chunk_y_seqs, chunk_y in zip(chunks_seqs, chunks_y_seqs, chunks_y):
#             seq_binary_classifier.fit(X=chunk_seqs,
#                                       y_seqs=chunk_y_seqs,
#                                       y=chunk_y, nb_epoch=1)
            
def eval_skipthoughts_classifier(input_seqs, output_choices, output_gold):
    input_seqs = encode_skipthought_seqs(input_seqs, skipthoughts, sent_encoder)
    output_choices = encode_skipthought_seqs(output_choices, skipthoughts, sent_encoder)
    #import pdb;pdb.set_trace()
    prob_y, pred_y, accuracy = seq_binary_classifier.predict(X=input_seqs, y_seqs=output_choices, y=output_gold)
    return prob_y, pred_y, accuracy

if __name__ == '__main__':
    #sent_encoder = skipthoughts.load_model()
    print "loaded skipthoughts encoder"
    n_train_seqs = 100
    filepath = 'roc_seqbinary_skipthoughts' + str(n_train_seqs)
    seq_binary_classifier = SeqBinaryClassifier(batch_size=100, verbose=1,
                                       n_hidden_layers=1,
                                       n_embedding_nodes=4800,
                                       n_hidden_nodes=1000, context_size=4,
                                       filepath=filepath, use_dropout=False)#, pairs=True)
    train_skipthoughts_classifier(input_seqs=train_input_seqs[:n_train_seqs],
                                  output_seqs=train_output_seqs[:n_train_seqs])
    prob_y, pred_y, accuracy = eval_skipthoughts_classifier(val_input_seqs, val_output_choices, val_output_gold)
    print "accuracy:", accuracy
    show_predictions(X=val_input_seqs[-20:], y=val_output_gold[-20:],
                     prob_y=prob_y[-20:], y_choices=val_output_choices[-20:])


# In[5]:

if __name__ == "__main__":
    #import pdb;pdb.set_trace()
    lm_transformer, lm_classifier = load_pipeline(filepath='roc_lm44362_clio')
    lm_transformer.sent_encoder = None
    lm_transformer.word_embeddings = None
    lm_pipeline = RNNPipeline(steps=[('transformer', lm_transformer),
                                    ('classifier', lm_classifier)])


# In[9]:

def make_embeddings(stories):
    sents = [segment_and_tokenize(sent) for story in stories for sent in story]
    embeddings = Word2Vec(sentences=sents, min_count=1, size=500, window=10, negative=25, iter=10)
    embeddings.save('roc' + str(len(stories)) + '_embeddings')
#     n_embedding_nodes = embeddings.vector_size
# #     embeddings = lm_classifier.get_embeddings()
# #     n_embedding_nodes = embeddings.shape[-1]
# #     embeddings = {lm_transformer.lexicon_lookup[idx]: values for idx, values in enumerate(embeddings)}
if __name__ == '__main__':
    train_stories = get_train_stories(filepath='ROC-Stories.tsv') +                     get_train_stories(filepath='ROCStories_winter2017.csv')
    make_embeddings(train_stories)


# In[54]:

if __name__ == "__main__":
    #import pdb;pdb.set_trace()
    n_train_seqs = 30000
    filepath = 'roc_seqbinary' + str(n_train_seqs)
    seq_binary = RNNPipeline(steps=[("transformer", SequenceTransformer(min_freq=1, verbose=1,#pad_seq=True, 
                                                                       word_embeddings=embeddings, 
                                                                        #sent_encoder=sent_encoder, 
                                                                        embed_y=True,
                                                                       replace_ents=False, reduce_emb_mode='mean',
                                                                       filepath=filepath)),
                                    ("classifier", SeqBinaryClassifier(batch_size=100, nb_epoch=6, verbose=1,
                                                                       n_hidden_layers=1,
                                                                       n_embedding_nodes=n_embedding_nodes,
                                                                       n_hidden_nodes=1000, context_size=4,
                                                                       filepath=filepath, use_dropout=True))])


# In[143]:

if __name__ == '__main__':
    from types import MethodType

    def create_encoder(self):

        self.encoder_model = self.create_model(pred_layer=False)
        self.encoder_model.set_weights(self.model.get_weights()[:-2])

    def encode_sents(self, seqs):
        #convert sentences to vectors
        encoded_seqs = []
        for seq in seqs:
            seq = numpy.array(seq)[None]
            seq = self.sent_encoder.predict(seq)[0][-1]
            encoded_seqs.append(seq)
            self.sent_encoder.reset_states()
        encoded_seqs = numpy.array(encoded_seqs)
        return encoded_seqs

    lm_classifier.stateful = True
    lm_classifier.create_encoder = MethodType(create_encoder, lm_classifier, RNNLM)
    lm_classifier.create_encoder()
    lm_transformer.sent_encoder = lm_classifier.encoder_model
    lm_transformer.encode_sents = MethodType(encode_sents, lm_transformer, SequenceTransformer)
    #import pdb;pdb.set_trace()
    lm_transformer.transform(X=train_input_seqs[:5])


# In[17]:

if __name__ == "__main__":
    import pdb;pdb.set_trace()
    embeddings = classifier.get_weights()[0]


# In[24]:

# if __name__ == "__main__":
#     #import pdb;pdb.set_trace()
#     #get word vectors
#     #sents = [segment_and_tokenize(sent) for seq in get_train_stories('ROC-Stories.tsv') for sent in seq]
#     embeddings_ = similarity_score.load_model("../AvMaxSim/vectors")
#     #embeddings = Word2Vec(sentences=sents, min_count=2, size=300, sg=1)#negative=20
#     n_embedding_nodes = embeddings.vector_size


# In[ ]:

# if __name__ == '__main__':
#     transformer, classifier = load_pipeline(filepath=filepath)
#     #provide embeddings for transformer, since they weren't pickled
#     transformer.embeddings = embeddings
#     seq_binary = RNNPipeline(steps=[('transformer', transformer), ('classifier', classifier)])


# In[17]:

# if __name__ == "__main__":
#     #import pdb;pdb.set_trace()
#     #get word vectors
#     embeddings = sim_score.load_model("../AvMaxSim/vectors")
#     n_embedding_nodes = embeddings.vector_size
#     #n_embedding_nodes = 250
#     seq2seq = RNNPipeline(steps=[("transformer", SequenceTransformer(pad_seq=True, min_freq=2, verbose=1,
#                                                                     embeddings=embeddings, replace_ents=True)),
#                                  ("classifier", Seq2SeqClassifier(batch_size=100, nb_epoch=3, verbose=1,
#                                                                   n_hidden_nodes=500, stateful=True))])
#     #import pdb;pdb.set_trace()
#     seq2seq.fit(X=train_x, y=)


# In[ ]:

# if __name__ == "__main__":
#     #sim_stories = get_stories(get_trusted_story_ids(n_stories=100))
#     #supplement training data with similar stories
#     train_seqs = [" ".join(story) for story in train_stories]
#     val_seqs = [" ".join(story) for story in val_stories]
#     #import pdb;pdb.set_trace()
#     #seqs = [" ".join(seq) for seq in stories_to_seqs(stories)]
# #     seqs.extend(sim_stories)
#     #import pdb;pdb.set_trace()
#     #assert(os.isdir("../DINE/val_stories"))
#     sim_index = SimilarityIndex(filepath="../DINE/val_stories", stories=train_seqs)
#     sim_seqs = []
#     print "retrieving matching stories for", len(val_seqs), "sequences..."
#     #combine input seqs and output sentence as one string
#     for seq in val_seqs[:50]:
#         print "input story:".upper(), seq, "\n"
#         #import pdb;pdb.set_trace()
#         sim_seq = sim_index.get_similar_seqs(seq)[0]
#         print "most similar:".upper(), sim_seq, "\n"
#         sim_seqs.extend(sim_seq)
#         #print "most similar:", "\n".join([sim_stories[id] for id in sim_ids])
#     #import pdb;pdb.set_trace()
# #     sim_input_seqs, sim_output_seqs = prep_input_output_seqs(sim_seqs)
# #     train_input_seqs.extend(sim_input_seqs)
# #     train_output_seqs.extend(sim_output_seqs)
# #     assert(len(train_input_seqs) == len(train_output_seqs))
    


# In[9]:

# if __name__ == "__main__":
#     import pdb;pdb.set_trace()
#     #prob_y = autoencoder.predict(X=val_input_seqs, y_choices=val_output_choices)
#     prob_y = seq2seq.predict(X=val_input_seqs, y_choices=val_output_choices)
#     pred_y = numpy.argmax(prob_y, axis=1)
#     #import pdb;pdb.set_trace()
#     #take majority of predictions for pairs in a story
#     #pred_y = numpy.round(numpy.mean(pred_y.reshape(-1, 4), axis=1))
#     accuracy = numpy.mean(pred_y == val_output_gold)
#     print accuracy
#     #accuracy = numpy.mean(pred_y == val_output_gold.reshape(-1, 4)[:, 0])

