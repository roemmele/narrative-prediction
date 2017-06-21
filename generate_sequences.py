import pandas, argparse, numpy
import models.pipeline
reload(models.pipeline)
from models.pipeline import *

def generate(context_seqs, model, save_prefix=None, gen_mode='random', temperature=1.0, n_gen_seqs=1, n_sents_per_seq=1, detokenize=True, capitalize_ents=True, adapt_ents=True):

	batch_size = min(1000, len(context_seqs))
	gen_seqs = model.predict([context_seqs[idx] for idx in numpy.arange(len(context_seqs)).repeat(n_gen_seqs)], mode=gen_mode,
							batch_size=batch_size, temp=temperature, n_sents_per_seq=n_sents_per_seq, detokenize=detokenize,
							capitalize_ents=capitalize_ents, adapt_ents=adapt_ents)
	gen_seqs = [gen_seqs[idx:idx + n_gen_seqs] for idx in range(0, len(context_seqs) * n_gen_seqs, n_gen_seqs)]
	if save_prefix:
		save_filepath = save_prefix + str(len(context_seqs)) + '_' + str(n_gen_seqs) + '.csv'
		pandas.DataFrame(gen_seqs).to_csv(save_filepath, header=False, index=False, encoding='utf-8')

	for context_seq, gen_seqs_ in zip(context_seqs, gen_seqs)[:100]: #print a sample of the generated sequences
		print "CONTEXT:", context_seq
		for gen_seq in gen_seqs_:
			print "GENERATED:", gen_seq, "\n"
	return gen_seqs


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Given a file of context sequences, generate continuations of these sequences.")
	parser.add_argument("--context_seqs", "-cont", help="Specify filename (.csv) containing context sequences.", type=str, required=True)
	#parser.add_argument("--model", "-mod", help="Specify the type of model that should be used to generate the sequences. See list of choices.", choices=['nneighors', 'rnn'], required=True)
	parser.add_argument("--model_filepath", "-modfp", help="Specify the filepath where the trained model is stored.", type=str, required=True)
	parser.add_argument("--save_prefix", "-save", help="Specify the prefix of the filepath where the generated sequences should be saved.\
													Number of context sequences and number of generated sequences will be appended to this prefix as full filepath.", type=str, required=False)
	parser.add_argument("--gen_mode", "-gmode", help="Specify what method should be used to generate sequences: either through random sampling (random) or by taking the max probability (max).\
													Default is random.", choices=['random', 'max'], required=False, default='random')
	parser.add_argument("--temperature", "-temp", help="When generation mode is random, specify the temperature variable for sampling. Default is 1 (most random).",
												required=False, type=float, default=1.0)
	parser.add_argument("--n_gen_seqs", "-nseqs", help="Specify how many sequences should be generated. Default is one sequence per context sequence.", type=int, required=False, default=1)
	parser.add_argument("--n_sents_per_seq", "-nsents", help="Specify the length of generated sequences in terms of the number of sentences.\
															Default is one sentence per sequence.", type=int, required=False, default=1)
	# parser.add_argument("--detokenize", "-detok", help="Specify whether generated sequences should be detokenized so that output is string with formatted punctuation. Default is True", 
	# 											required=False, default=True, type=bool)
	# parser.add_argument("--capitalize_ents", "-cap", help="Specify whether named entities should be capitalized in generated sequences. Default is True.", required=False, default=True, type=bool)
	# parser.add_argument("--adapt_ents", "-adapt", help="Specify whether generalized named entity tokens (e.g. those prefixed with \"ENT\") should be replaced with entities in the context sequence.\
	# 													Note that if trained model did not generalize entities, this will not apply. ", required=False, default=True, type=bool)
	args = parser.parse_args()

	context_seqs = pandas.read_csv(args.context_seqs, encoding='utf-8', header=None).loc[:,0].values.tolist()
	model = load_rnnlm_pipeline(args.model_filepath)
	gen_seqs = generate(context_seqs=context_seqs, model=model, save_prefix=args.save_prefix, gen_mode=args.gen_mode, temperature=args.temperature, n_gen_seqs=args.n_gen_seqs, 
						n_sents_per_seq=args.n_sents_per_seq)#, detokenize=args.detokenize, capitalize_ents=args.capitalize_ents, adapt_ents=args.adapt_ents)