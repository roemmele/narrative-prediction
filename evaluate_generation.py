import sys, pandas, pprint, pickle, argparse

import analysis.generation_metrics
reload(analysis.generation_metrics)
from analysis.generation_metrics import *

import analysis.stats
reload(analysis.stats)
from analysis.stats import *

pandas.set_option('precision', 3)

def evaluate(context_seqs, gen_seqs, include_analyses=[], exclude_analyses=[], stat_sig=False, 
			verified_4gram_lexicon='data/blog_lexicon.pkl', verified_4gram_db='data/blog_ngrams.db'):

	#Sequence length
	if 'seq_length' in include_analyses and 'seq_length' not in exclude_analyses:
		seq_length = {'models':{},'p-values':{}}
		print "SEQUENCE LENGTHS:"
		for model in gen_seqs.keys():
		    seq_length['models'][model] = get_seq_lengths(gen_seqs[model])
		pprint.pprint(pandas.DataFrame.from_dict(seq_length['models'], orient='index')[['mean_length']])

		if stat_sig:
			seq_length['p-values']['lengths'] = eval_all_diffs({model:analysis['lengths']\
		                                                		for model,analysis\
		                                                		in seq_length['models'].items()})
			print "\np-values:"
			pprint.pprint(pandas.DataFrame.from_dict(seq_length['p-values'], orient='index'))

    #Lexical similarity between context and generated sequence
	if 'lexical_sim' in include_analyses and 'lexical_sim' not in exclude_analyses:
		lexical_sim = {'models':{},'p-values':{}}
		print "\nLEXICAL SIMILARITY:"
		for model in gen_seqs.keys():
		    lexical_sim['models'][model] = get_lexical_sim(context_seqs, gen_seqs[model])
		pprint.pprint(pandas.DataFrame.from_dict(lexical_sim['models'], orient='index')[['mean_word2vec', 'mean_jaccard']])
		if stat_sig:    	    
		    lexical_sim['p-values']['word2vec'] = eval_all_diffs({model:analysis['word2vec']\
		                                                    	for model,analysis\
		                                                        in lexical_sim['models'].items()})
		    lexical_sim['p-values']['jaccard'] = eval_all_diffs({model:analysis['jaccard']\
		                                                        for model,analysis\
		                                                        in lexical_sim['models'].items()})
		    print "\np-values:"
		    pprint.pprint(pandas.DataFrame.from_dict(lexical_sim['p-values'], orient='index'))


	# Skipthought (sentence vector) similarity between context and generated sequence
	if 'sent_sim' in include_analyses and 'sent_sim' not in exclude_analyses:
		sent_sim = {'models':{}, 'p-values':{}}
		print "\nSENTENCE (SKIPTHOUGHT) SIMILARITY:"
		for model in gen_seqs.keys():
		    #import pdb;pdb.set_trace()
		    sent_sim['models'][model] = get_skipthought_similarity(context_seqs, gen_seqs[model])
		pprint.pprint(pandas.DataFrame.from_dict(sent_sim['models'], orient='index')[['mean_skipthought_scores']])
		if stat_sig:
		    sent_sim['p-values']['skipthought_scores'] = eval_all_diffs({model:analysis['skipthought_scores']\
		                                                            					for model,analysis\
		                                                            					in sent_sim['models'].items()})
		    print "\np-values:"
		    pprint.pprint(pandas.DataFrame.from_dict(sent_sim['p-values'], orient='index'))

    #Type-token ratio
	if 'type_token_ratio' in include_analyses and 'type_token_ratio' not in exclude_analyses:
		type_token_ratio = {'models':{}, 'p-values':{}}
		print "\nTYPE-TOKEN RATIO:"
		for model in gen_seqs.keys():
		    type_token_ratio['models'][model] = get_type_token_ratio(gen_seqs[model])
		pprint.pprint(pandas.DataFrame.from_dict(type_token_ratio['models'], orient='index'))
		if stat_sig:
			type_token_ratio['p-values']['ratio'] = eval_all_proportion_diffs(model_pos={model:analysis['n_types']\
		                                            									for model,analysis\
		                                            									in type_token_ratio['models'].items()},
		                                            						model_samples={model:analysis['n_tokens']\
		                                            										for model,analysis\
		                                            										in type_token_ratio['models'].items()})
			print "\np-values:"
			pprint.pprint(pandas.DataFrame.from_dict(type_token_ratio['p-values'], orient='index'))

	#Unique trigram ratio (i.e. type-token ratio with trigrams instead of unigrams)
	if 'unique_trigram_ratio' in include_analyses and 'unique_trigram_ratio' not in exclude_analyses:
		unique_trigram_ratio = {'models':{}, 'p-values':{}}
		print "\nUNIQUE TRIGRAM RATIO:"
		for model in gen_seqs.keys():
		    unique_trigram_ratio['models'][model] = get_unique_ngram_ratio(gen_seqs[model])
		pprint.pprint(pandas.DataFrame.from_dict(unique_trigram_ratio['models'], orient='index')) 
		if stat_sig:
		    unique_trigram_ratio['p-values']['ratio'] = eval_all_proportion_diffs(model_pos={model:analysis['n_unique']\
		                                                    						for model,analysis\
		                                                    						in unique_trigram_ratio['models'].items()},
		                                                    			model_samples={model:analysis['n_total']\
		                                                    							for model,analysis\
		                                                    							in unique_trigram_ratio['models'].items()})
		    print "\np-values:"
		    pprint.pprint(pandas.DataFrame.from_dict(unique_trigram_ratio['p-values'], orient='index'))

	#Noun chunk complexity
	if 'noun_chunk_complexity' in include_analyses and 'noun_chunk_complexity' not in exclude_analyses:
		noun_chunk_complexity = {'models':{}, 'p-values':{}}
		print "\nNOUN CHUNK COMPLEXITY:"
		for model in gen_seqs.keys():
		    noun_chunk_complexity['models'][model] = get_noun_chunk_complexity(gen_seqs[model])
		pprint.pprint(pandas.DataFrame.from_dict(noun_chunk_complexity['models'], orient='index')[['mean_n_chunks', 'norm_mean_n_chunks', 'mean_chunk_lengths', 'norm_mean_chunk_lengths']])
		if stat_sig:
			noun_chunk_complexity['p-values']['n_chunks'] = eval_all_diffs({model:analysis['n_chunks']\
		                                                            		for model,analysis\
		                                                            		in noun_chunk_complexity['models'].items()})
			noun_chunk_complexity['p-values']['chunk_lengths'] = eval_all_diffs({model:analysis['chunk_lengths']\
			                                                                    for model,analysis\
			                                                                    in noun_chunk_complexity['models'].items()})
			noun_chunk_complexity['p-values']['norm_n_chunks'] = eval_all_diffs({model:analysis['norm_n_chunks']\
			                                                        			for model,analysis\
			                                                        			in noun_chunk_complexity['models'].items()})
			noun_chunk_complexity['p-values']['norm_chunk_lengths'] = eval_all_diffs({model:analysis['norm_chunk_lengths']\
			                                                                    	for model,analysis\
			                                                                    	in noun_chunk_complexity['models'].items()})
			print "\np-values:"
			pprint.pprint(pandas.DataFrame.from_dict(noun_chunk_complexity['p-values'], orient='index'))

	#Verb chunk complexity
	if 'verb_chunk_complexity' in include_analyses and 'verb_chunk_complexity' not in exclude_analyses:
		verb_phrase_complexity = {'models':{}, 'p-values':{}}
		print "\nVERB CHUNK COMPLEXITY:"
		for model in gen_seqs.keys():
		    verb_phrase_complexity['models'][model] = get_verb_phrase_complexity(gen_seqs[model])
		pprint.pprint(pandas.DataFrame.from_dict(verb_phrase_complexity['models'], orient='index')[['mean_n_phrases', 'mean_norm_n_phrases', 'mean_phrase_lengths', 'mean_norm_phrase_lengths']])
		if stat_sig:
			verb_phrase_complexity['p-values']['n_phrases'] = eval_all_diffs({model:analysis['n_phrases']\
			                                                        		for model,analysis\
			                                                        		in verb_phrase_complexity['models'].items()})
			verb_phrase_complexity['p-values']['phrase_lengths'] = eval_all_diffs({model:analysis['phrase_lengths']\
			                                                                    	for model,analysis\
			                                                                    	in verb_phrase_complexity['models'].items()})
			verb_phrase_complexity['p-values']['norm_n_phrases'] = eval_all_diffs({model:analysis['norm_n_phrases']\
			                                                        				for model,analysis\
			                                                        				in verb_phrase_complexity['models'].items()})
			verb_phrase_complexity['p-values']['norm_phrase_lengths'] = eval_all_diffs({model:analysis['norm_phrase_lengths']\
			                                                                    		for model,analysis\
			                                                                    		in verb_phrase_complexity['models'].items()})
			print "\np-values:"
			pprint.pprint(pandas.DataFrame.from_dict(verb_phrase_complexity['p-values'], orient='index'))

	#Subject-verb-object complexity (i.e. number of SVO triples per sequence)
	if 'svo_complexity' in include_analyses and 'svo_complexity' not in exclude_analyses:
		svo_complexity = {'models':{}, 'p-values':{}}
		print "\nSUBJECT-VERB-OBJECT COMPLEXITY (# PER SEQUENCE):"
		for model in gen_seqs.keys():
		    svo_complexity['models'][model] = get_svo_complexity(gen_seqs[model])
		pprint.pprint(pandas.DataFrame.from_dict(svo_complexity['models'], orient='index')[['mean_n_svos']])
		if stat_sig:
			svo_complexity['p-values']['n_svos'] = eval_all_diffs({model:analysis['n_svos']\
			                                                	for model,analysis\
			                                                	in svo_complexity['models'].items()})
			print "\np-values:"
			pprint.pprint(pandas.DataFrame.from_dict(svo_complexity['p-values'], orient='index'))

	#Phrases (two-word collocations in google word2vec model)
	if 'phrases' in include_analyses and 'phrases' not in exclude_analyses:
		phrases = {'models':{}, 'p-values':{}}
		print "\nPHRASES:"
		for model in gen_seqs.keys():
		    phrases['models'][model] = get_phrase_counts(gen_seqs[model])
		pprint.pprint(pandas.DataFrame.from_dict(phrases['models'], orient='index')[['mean_phrase_rates', 'n_bigrams', 'n_phrases', 'phrase_bigram_ratio']])
		if stat_sig:
			phrases['p-values']['phrase_rates'] = eval_all_diffs({model:analysis['phrase_rates']\
			                                                      for model,analysis\
			                                                      in phrases['models'].items()})
			phrases['p-values']['n_phrases'] = eval_all_proportion_diffs(model_pos={model:analysis['n_phrases']\
					                                                    				for model,analysis\
					                                                    				in phrases['models'].items()},
					                                                    		model_samples={model:analysis['n_bigrams']\
					                                                    					for model,analysis\
					                                                    					in phrases['models'].items()})
			print "\np-values:"
			pprint.pprint(pandas.DataFrame.from_dict(phrases['p-values'], orient='index'))

	#Word frequency (frequency statistics come from spaCy)
	if 'word_freq' in include_analyses and 'word_freq' not in exclude_analyses:
		word_freq = {'models':{}, 'p-values':{}}
		print "\nWORD FREQUENCY:"
		for model in gen_seqs.keys():
		    word_freq['models'][model] = get_frequency_scores(gen_seqs[model])
		pprint.pprint(pandas.DataFrame.from_dict(word_freq['models'], orient='index')[['mean_freq_scores']])
		if stat_sig:
			word_freq['p-values']['freq_scores'] = eval_all_diffs({model:analysis['freq_scores']\
			                                                       for model,analysis\
			                                                       in word_freq['models'].items()})
			print "\np-values:"
			pprint.pprint(pandas.DataFrame.from_dict(word_freq['p-values'], orient='index'))

	#Language style matching (POS token similarity between context and generated sequence)
	if 'pos_token_sim' in include_analyses and 'pos_token_sim' not in exclude_analyses:
		pos_token_sim = {'models':{}, 'p-values':{}}
		print "\nPART-OF-SPEECH TOKEN SIMILARITY (LANGUAGE STYLE MATCHING):"
		for model in gen_seqs.keys():
		    pos_token_sim['models'][model] = get_lsm_scores(context_seqs, gen_seqs[model])
		dataframe = pandas.DataFrame.from_dict(pos_token_sim['models'], orient='index')
		pprint.pprint(dataframe[[analysis for analysis in dataframe.keys() if analysis.endswith('mean')]])
		if stat_sig:
			pos_token_sim['p-values']['all'] = eval_all_diffs({model:analysis['all']\
			                                        			for model,analysis\
			                                        			in pos_token_sim['models'].items()})
			print "\np-values:"
			pprint.pprint(pandas.DataFrame.from_dict(pos_token_sim['p-values'], orient='index'))

	#Part-of-speech trigram similarity between context and generated sequence
	if 'pos_trigram_sim' in include_analyses and 'pos_trigram_sim' not in exclude_analyses:
		pos_trigram_sim = {'models':{}, 'p-values':{}}
		print "\nPART-OF-SPEECH TRIGRAM SIMILARITY:"
		for model in gen_seqs.keys():
		    pos_trigram_sim['models'][model] = get_pos_ngram_similarity(context_seqs, gen_seqs[model], n=3)
		pprint.pprint(pandas.DataFrame.from_dict(pos_trigram_sim['models'], orient='index')[['mean_pos_sim_scores']])
		if stat_sig:
			pos_trigram_sim['p-values']['pos_sim_scores'] = eval_all_diffs({model:analysis['pos_sim_scores']\
			                                            					for model,analysis\
			                                            					in pos_trigram_sim['models'].items()})
			print "\np-values:"
			pprint.pprint(pandas.DataFrame.from_dict(pos_trigram_sim['p-values'], orient='index'))

	#Verified 4-grams (i.e. number of 4-grams in generated sequences that also appear in a different corpus)
	if 'verified_ngrams' in include_analyses and 'verified_ngrams' not in exclude_analyses:
		verified_4grams = {'models':{}, 'p-values':{}}
		print "\nVERIFIED 4-GRAMS (I.E. 4-GRAMS THAT APPEAR IN HELD-OUT CORPUS):"
		for model in gen_seqs.keys():
		    verified_4grams['models'][model] = get_n_ngrams(gen_seqs[model], n=4, lexicon_filepath=verified_4gram_lexicon, db_filepath=verified_4gram_db)
		pprint.pprint(pandas.DataFrame.from_dict(verified_4grams['models'], orient='index')[['n_verified', 'n_total', 'ratio']])
		if stat_sig:
			verified_4grams['p-values']['ratio'] = eval_all_proportion_diffs(model_pos={model:analysis['n_verified']\
					                                                    				for model,analysis\
					                                                    				in verified_4grams['models'].items()},
					                                                    	model_samples={model:analysis['n_total']\
					                                                    					for model,analysis\
					                                                    					in verified_4grams['models'].items()})
			print "\np-values:"
			pprint.pprint(pandas.DataFrame.from_dict(verified_4grams['p-values'], orient='index'))

	#Grammaticality
	if 'grammar' in include_analyses and 'grammar' not in exclude_analyses:
		grammaticality = {'models':{}, 'p-values':{}}
		print "\nGRAMMATICALITY:"
		for model in gen_seqs.keys():
		    grammaticality['models'][model] = get_grammaticality_scores(gen_seqs[model])
		pprint.pprint(pandas.DataFrame.from_dict(grammaticality['models'], orient='index')[['mean_gram_scores']])
		if stat_sig:
			grammaticality['p-values']['gram_scores'] = eval_all_diffs({model:analysis['gram_scores']\
			                                    						for model,analysis\
			                                    						in grammaticality['models'].items()})
			print "\np-values:"
			pprint.pprint(pandas.DataFrame.from_dict(grammaticality['p-values'], orient='index'))

	#Coreference - CoreNLP server must be running for this analysis
	if 'coreference' in include_analyses and 'coreference' not in exclude_analyses:
		coref_counts = {'models':{}, 'p-values':{}}
		print "\nCOREFERENCE"
		for model in gen_seqs.keys():
		    coref_counts['models'][model] = get_coref_counts(context_seqs, gen_seqs[model])
		    # with open('cbt_' + model + '_coref_counts.pkl', 'wb') as f:
		    #     pickle.dump(coref_counts['models'][model], f)
		pprint.pprint(pandas.DataFrame.from_dict(coref_counts['models'], orient='index')[['mean_ents', 'mean_corefs', 'mean_res_rates']])
		if stat_sig:
			coref_counts['p-values']['ents'] = eval_all_diffs({model:analysis['ents']\
			                                            		for model,analysis\
			                                            		in coref_counts['models'].items()})
			coref_counts['p-values']['corefs'] = eval_all_diffs({model:analysis['corefs']\
			                                                    for model,analysis\
			                                                    in coref_counts['models'].items()})
			coref_counts['p-values']['res_rates'] = eval_all_diffs({model:analysis['res_rates']\
			                                                        for model,analysis\
			                                                        in coref_counts['models'].items()})
			print "\np-values:"
			pprint.pprint(pandas.DataFrame.from_dict(coref_counts['p-values'], orient='index'))

	#Sentiment similarity
	if 'sentiment' in include_analyses and 'sentiment' not in exclude_analyses:
		sentiment_sim = {'models':{}, 'p-values':{}}
		print "\nSENTIMENT SIMILARITY:"
		for model in gen_seqs.keys():
		    sentiment_sim['models'][model] = get_sentiment_sim(context_seqs, gen_seqs[model])
		pprint.pprint(pandas.DataFrame.from_dict(sentiment_sim['models'], orient='index')[['mean_sentiment_sim_scores']])
		if stat_sig:
			sentiment_sim['p-values']['sentiment_sim_scores'] = eval_all_diffs({model:analysis['sentiment_sim_scores']\
			                                            						for model,analysis\
			                                            						in sentiment_sim['models'].items()})
			print "\np-values:"
			pprint.pprint(pandas.DataFrame.from_dict(sentiment_sim['p-values'], orient='index'))

	#Pointwise mutual information (PMI)
	if 'pmi' in include_analyses and 'pmi' not in exclude_analyses:
		pmi_scores = {'models':{}, 'p-values':{}}
		print "\nPMI SCORES:"
		for model in gen_seqs.keys():
		    pmi_scores['models'][model] = get_pmi_scores(context_seqs, gen_seqs[model])
		    print "computed PMI scores for model:", model
		pprint.pprint(pandas.DataFrame.from_dict(pmi_scores['models'], orient='index')[['mean_pmi_scores']])
		if stat_sig:
			pmi_scores['p-values']['pmi_scores'] = eval_all_diffs({model:analysis['pmi_scores']\
			                                            				for model,analysis\
			                                            				in pmi_scores['models'].items()})
			print "\np-values:"
			pprint.pprint(pandas.DataFrame.from_dict(pmi_scores['p-values'], orient='index'))



if __name__ == '__main__':

	analyses = ['perplexity', 'seq_length','lexical_sim','sent_sim','type_token_ratio','unique_trigram_ratio','noun_chunk_complexity','phrases',\
				'verb_chunk_complexity','svo_complexity','word_freq','pos_token_sim','pos_trigram_sim', 'verified_ngrams', 'grammar','coreference', 'sentiment', 'pmi']

	parser = argparse.ArgumentParser(description="Given a file of context sequences and one or more files of corresponding generated sequences,\
													apply various evaluation metrics to the generated sequences. The context sequence file must contain one sequence per line,\
													while the generated sequence files must each contain one or more tab-seperated sequences per line, each of which correspond to\
													 the context sequence in the same position.")
	parser.add_argument("--context_seqs", "-cont", help="Specify filename containing context sequences.", type=str, required=True)
	parser.add_argument("--gen_seqs", "-gen", help="Specify one or more filenames containing generated sequences to evaluate.\
											If analyzing gold standard sequences, this filename should also be given in this list, since gold sequences\
											are analyzed in the same way as generated sequences.", nargs='+', required=True)
	parser.add_argument("--include_analyses", "-inc", help="Specify which analyses should be performed. If not given, all analyses will be applied,\
														except for those that appear in --exclude_analyses.", nargs='+', required=False, default=analyses, choices=analyses)
	parser.add_argument("--exclude_analyses", "-ex", help="Specify which analyses should be omitted. If not given, all analyses in --include_analyses will be performed,\
														(or if --include_analyses is not specified, all analyses will be performed).", nargs='+', required=False, default=[], choices=analyses)
	parser.add_argument("--stat_sig", "-sig", help="Specify whether differences between scores for generated sequences should be tested for statistical significance.", 
											default=False, action='store_true')
	parser.add_argument("--ngram_lexicon", "-lex", help="If analyses include verified 4-grams, specify .pkl file containing lexicon used to represent 4-grams.", 
													type=str, required=False, default='data/blog_lexicon.pkl')
	parser.add_argument("--ngram_db", "-db", help="If analyses include verified 4-grams, specify sqlite .db file containing n-gram counts.", 
											type=str, required=False, default='data/blog_ngrams.db')
	args = parser.parse_args()

	# if 'verified_ngrams' in args.include_analyses:
	# 	assert(args.ngram_lexicon and args.ngram_db)

	context_seqs = pandas.read_csv(args.context_seqs, encoding='utf-8', header=None)[0].values.tolist()

	gen_seqs = {}
	for filename in args.gen_seqs:
		# model = os.path.basename(filename).split(".")[0]
		gen_seqs[filename] = pandas.read_csv(filename, encoding='utf-8', header=None).values.tolist()

	evaluate(context_seqs=context_seqs, gen_seqs=gen_seqs, include_analyses=args.include_analyses, exclude_analyses=args.exclude_analyses, stat_sig=args.stat_sig,
			verified_4gram_lexicon = args.ngram_lexicon, verified_4gram_db=args.ngram_db)

