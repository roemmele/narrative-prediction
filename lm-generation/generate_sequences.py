from __future__ import print_function
import pandas, argparse, numpy
from models.pipeline import *

def generate(context_seqs_file, model, save_filepath, gen_mode='random', temperature=1.0, n_gen_per_context=1, n_sents_per_seq=1, 
			n_context_sents=-1, eos_tokens=[], detokenize=True, capitalize_ents=True, adapt_ents=True, batch_size=1000):

	context_seqs = pandas.read_csv(context_seqs_file, encoding='utf-8', header=None).loc[:,0].values.tolist()
	batch_size = min(batch_size, len(context_seqs))
	gen_seqs = model.predict([context_seqs[idx] for idx in numpy.arange(len(context_seqs)).repeat(n_gen_per_context)], mode=gen_mode,
							batch_size=batch_size, temp=temperature, n_sents_per_seq=n_sents_per_seq, n_context_sents=n_context_sents,
							eos_tokens=eos_tokens, detokenize=detokenize, capitalize_ents=capitalize_ents, adapt_ents=adapt_ents)
	gen_seqs = [gen_seqs[idx:idx + n_gen_per_context] for idx in range(0, len(context_seqs) * n_gen_per_context, n_gen_per_context)]
	pandas.DataFrame(gen_seqs).to_csv(save_filepath, header=False, index=False, encoding='utf-8')
	print("saved generated sequences to", save_filepath)
	print("SAMPLE OF GENERATED SEQUENCES:")
	print("----------------------------------------------------------\n")
	for context_seq, gen_seqs_ in zip(context_seqs[:100], gen_seqs[:100]): #print a sample of the generated sequences
		print("CONTEXT:", context_seq)
		for gen_seq in gen_seqs_:
			print("GENERATED:", gen_seq)
		print("\n")
	return gen_seqs


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Given a file of context sequences, generate continuations of these sequences.")
	parser.add_argument("--context_seqs", "-cont", help="Specify filename (.csv) containing context sequences.", type=str, required=True)
	parser.add_argument("--model_filepath", "-modfp", help="Specify the filepath directory where the trained model is stored.", type=str, required=True)
	parser.add_argument("--save_filepath", "-save", help="Specify the .csv filename where the generated sequences should be saved.", type=str, required=True)
	parser.add_argument("--gen_mode", "-gmode", help="Specify what method should be used to generate sequences: either through random sampling (random) or by taking the max probability (max).\
													Default is random.", choices=['random', 'max'], required=False, default='random')
	parser.add_argument("--temperature", "-temp", help="When generation mode is random, specify the temperature variable for sampling. Default is 1 (most random).",
												required=False, type=float, default=1.0)
	parser.add_argument("--n_gen_per_context", "-ngen", help="Specify how many sequences should be generated for each context sequence. Default is one sequence per context sequence.", type=int, required=False, default=1)
	parser.add_argument("--n_sents_per_seq", "-nsents", help="Specify the length of generated sequences in terms of the number of sentences.\
															Default is one sentence per sequence.", type=int, required=False, default=1)
	parser.add_argument("--n_context_sents", "-ncont", help="Specify if the context should be truncated so that only the N most recent sentences are taken into account when generating the next sequence.\
															If the model uses feature vectors, the features will still take into account the whole context.\
															Default is -1, which means all context sentences will be included.", type=int, required=False, default=-1)
	parser.add_argument("--eos_tokens", "-eos", help="If sentence boundaries should be determined by the occurence of specific end-of-sentence tokens, specify these tokens.\
													Default is None, in which case sentence boundaries will be automatically inferred by spaCy's segmenter.", nargs='+', required=False, default=[])
	# parser.add_argument("--detokenize", "-detok", help="Specify whether generated sequences should be detokenized so that output is string with formatted punctuation. Default is True", 
	# 											required=False, default=True, type=bool)
	# parser.add_argument("--capitalize_ents", "-cap", help="Specify whether named entities should be capitalized in generated sequences. Default is True.", required=False, default=True, type=bool)
	# parser.add_argument("--adapt_ents", "-adapt", help="Specify whether generalized named entity tokens (e.g. those prefixed with \"ENT\") should be replaced with entities in the context sequence.\
	# 													Note that if trained model did not generalize entities, this will not apply. ", required=False, default=True, type=bool)
	args = parser.parse_args()

	model = RNNLMPipeline.load(args.model_filepath)
	gen_seqs = generate(context_seqs_file=args.context_seqs, model=model, save_filepath=args.save_filepath, gen_mode=args.gen_mode, temperature=args.temperature, n_gen_per_context=args.n_gen_per_context, 
						n_sents_per_seq=args.n_sents_per_seq, n_context_sents=args.n_context_sents, eos_tokens=args.eos_tokens)#, detokenize=args.detokenize, capitalize_ents=args.capitalize_ents, adapt_ents=args.adapt_ents)
