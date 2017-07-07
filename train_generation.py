import pandas, argparse, numpy
import models.pipeline
reload(models.pipeline)
from models.pipeline import *

def load_train_seqs(train_seqs_file):
	#create generator object to read sequences in chunks (in case too many to fit in memory)
	train_seqs = (seqs[0].values.tolist() for seqs in pandas.read_csv(train_seqs_file, encoding='utf-8', header=None, chunksize=10000))
	return train_seqs

def create_model(save_filepath, use_features, use_pos, min_freq, generalize_ents, batch_size, n_timesteps, n_hidden_layers, 
				n_embedding_nodes, n_hidden_nodes, n_pos_nodes=None, n_feature_nodes=None):

	if os.path.exists(save_filepath + '/transformer.pkl'): #if transformer already exists, load it
	    transformer = load_transformer(save_filepath)
	else:
	    transformer = SequenceTransformer(min_freq=min_freq, generalize_ents=generalize_ents, verbose=1, filepath=save_filepath)

	classifier = RNNLM(verbose=1, use_features=use_features, use_pos=use_pos, n_pos_tags=len(pos_tag_idxs), batch_size=batch_size, n_timesteps=n_timesteps, n_feature_nodes=n_feature_nodes, 
						n_pos_nodes=n_pos_nodes, n_hidden_layers=n_hidden_layers, n_embedding_nodes=n_embedding_nodes, n_hidden_nodes=n_hidden_nodes, filepath=save_filepath)

	model = RNNLMPipeline(transformer, classifier)

	return model

def train_model(train_seqs_file, model, n_epochs):

	if not model.transformer.lexicon: #make lexicon if not already created
		for seqs in load_train_seqs(train_seqs_file):
			model.transformer.make_lexicon(seqs)

	for epoch in range(n_epochs):
	    print "training epoch {}/{}...".format(epoch + 1, n_epochs)
	    for seqs in load_train_seqs(train_seqs_file):
			model.fit(seqs=seqs)
			#sample some training sequences to show progress, generating final sentence in each sequence
			samp_seqs = [segment(seq) for seq in random.sample(seqs, min(25, len(seqs)))]
			context_seqs = [" ".join(seq[:-1]) for seq in samp_seqs]
			gold_seqs = [seq[-1] for seq in samp_seqs]
			gen_seqs = model.predict(seqs=context_seqs, n_best=1, mode='random', batch_size=len(samp_seqs),
			                          temp=1.0, n_sents_per_seq=1, detokenize=True, adapt_ents=True)
			for context_seq, gold_seq, gen_seq in zip(context_seqs, gold_seqs, gen_seqs):
			    print "CONTEXT:", context_seq
			    print "GOLD:", gold_seq
			    print "GENERATED:", gen_seq, "\n"


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Given a file of context sequences, generate continuations of these sequences.")
	parser.add_argument("--train_seqs", "-train", help="Specify filename (.csv) containing training sequences.", type=str, required=True)
	#parser.add_argument("--model_type", "-mod", help="Specify the type of model that should be used to generate the sequences. See list of choices.", choices=['feature_rnn', 'rnn'], required=True)
	parser.add_argument("--use_features", "-ufeat", help="If given, the model will be trained on noun features in additon to word sequences.", required=False, action='store_true')
	parser.add_argument("--use_pos", "-upos", help="If given, the model will be trained on POS tags in addition to word sequences.", required=False, action='store_true')
	parser.add_argument("--save_filepath", "-save", help="Specify the directory filepath where the trained model should be stored.", type=str, required=True)
	parser.add_argument("--min_freq", "-freq", help="Specify frequency threshold for including words in model lexicon, such that only words that appear in the training sequences at least\
													this number of times will be added. Default is 5.", required=False, type=int, default=5)
	parser.add_argument("--generalize_ents", "-ents", help="Specify that named entities should be replaced with a general entity type token (e.g. ENT_PERSON_0, ENT_ORG_1). If not given, entities will be treated like all other tokens.", 
													required=False, action='store_true')
	parser.add_argument("--batch_size", "-batch", help="Specify number of sequences in batch during training. Default is 25.", required=False, type=int, default=25)
	parser.add_argument("--n_timesteps", "-step", help="Specify number of timesteps (tokens) in a batch of sequences that should be read at a single time between updates. Default is 15.", 
												required=False, type=int, default=15)
	parser.add_argument("--n_hidden_layers", "-lay", help="Specify number of recurrent hidden layers in model. Default is 2.", required=False, type=int, default=2)
	parser.add_argument("--n_embedding_nodes", "-emb", help="Specify number of nodes in word embedding layer that feeds into recurrent hidden layer. Default is 300.", required=False, type=int, default=300)
	parser.add_argument("--n_hidden_nodes", "-hid", help="Specify number of nodes in recurrent hidden layer. Default is 500.", required=False, type=int, default=500)
	parser.add_argument("--n_pos_nodes", "-pos", help="For model with POS tags, specify number of nodes in POS hidden layer. Default is 100.", required=False, type=int, default=100)
	parser.add_argument("--n_feature_nodes", "-feat", help="For model with features, specify number of nodes in feature hidden layer. Default is 100.", required=False, type=int, default=100)
	parser.add_argument("--n_epochs", "-epoch", help="Specify the number of epochs the model should be trained for. Default is 10.", required=False, type=int, default=10)
	args = parser.parse_args()

 	model = create_model(save_filepath=args.save_filepath, use_features=args.use_features, use_pos=args.use_pos, min_freq=args.min_freq, generalize_ents=args.generalize_ents, 
 						batch_size=args.batch_size, n_timesteps=args.n_timesteps, n_hidden_layers=args.n_hidden_layers, 
 						n_embedding_nodes=args.n_embedding_nodes, n_hidden_nodes=args.n_hidden_nodes, n_feature_nodes=args.n_feature_nodes, n_pos_nodes=args.n_pos_nodes)
	train_model(train_seqs_file=args.train_seqs, model=model, n_epochs=args.n_epochs)


