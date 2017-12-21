from __future__ import print_function
import numpy, argparse, timeit, collections, os, sqlite3, pickle, cPickle
import theano
import theano.tensor as T
import theano.typed_list
from keras.preprocessing.sequence import pad_sequences

from transformer import *

numpy.set_printoptions(suppress=True)

#theano.config.compute_test_value = 'raise'

rng = numpy.random.RandomState(123)


class PMIModel(object):
	def __init__(self, filepath, transformer):
		self.filepath = filepath
		self.transformer = transformer
		self.lexicon_size = self.transformer.lexicon_size + 1
		self.n_bigram_counts = None
		self.count_window_bigrams = None
		self.unigram_counts = None

		if not os.path.isdir(self.filepath):
			os.mkdir(self.filepath)

	def count_unigrams(self, stories):

		stories = self.transformer.text_to_nums(stories)

		if self.unigram_counts is None:
			self.unigram_counts = numpy.zeros((self.lexicon_size,))

		for story in stories:
			for word in story:
				self.unigram_counts[word] += 1
		
		if not self.unigram_counts[1]:
			self.unigram_counts[1] = 1

		self.n_unigram_counts = sum(self.unigram_counts)

		with open(self.filepath + '/unigram_counts.pkl', 'wb') as f:
			pickle.dump(self.unigram_counts, f)

		print("Saved unigram counts to", self.filepath + "/unigram_counts.pkl")

	def count_bigrams_across_pairs(self, seqs1, seqs2):

		seqs1 = self.transformer.text_to_nums(seqs1)
		seqs2 = self.transformer.text_to_nums(seqs2)

		bigram_counts = [collections.defaultdict(int) for word in xrange(self.lexicon_size)]

		for seq1, seq2 in zip(seqs1, seqs2):
			for word1 in seq1:
				for word2 in seq2:
					bigram_counts[word1][word2] += 1

		#save remaining bigrams
		self.save_bigrams(bigram_counts=bigram_counts)
		#self.save_n_bigrams(n_bigram_counts=n_bigram_counts)
    

	def init_count_window_bigrams(self, train_stories, window_size, batch_size):

		window = T.matrix('window', dtype='int32')
		window.tag.test_value = rng.randint(self.lexicon_size, size=(window_size, 100)).astype('int32')
		window.tag.test_value[1, 10] = -1
		window.tag.test_value[:, 0] = -1
		window.tag.test_value[-1, 1] = -1

		words1 = window[0]
		words2 = window[1:].T

		word_index = T.scalar('word_index', dtype='int32')
		word_index.tag.test_value = 0
		batch_index = T.scalar('batch_index', dtype='int32')
		batch_index.tag.test_value = 0

		#select words in sequence and batch
		window_ = train_stories[word_index:word_index + window_size, batch_index:batch_index + batch_size]
		#filter stories with all empty words from this batch
		window_ = window_[:, T.argmin(window_[0] < 0):]

		self.count_window_bigrams = theano.function(inputs=[word_index, batch_index],\
													outputs=[words1, words2],\
													givens={window: window_},\
													on_unused_input='ignore',\
													allow_input_downcast=True)

		
	def count_bigrams(self, stories, window_size=25, batch_size=10000):

		stories = self.transformer.text_to_nums(stories)
		#convert train sequences from list of arrays to matrix
		stories = pad_sequences(sequences=stories, padding='post')

		n_stories = len(stories)

		if not window_size: #if window size is None, window includes all words in story
			window_size = stories.shape[-1]

		#initialize shared stories with random data
		train_stories = theano.shared(rng.randint(self.lexicon_size, size=(window_size, n_stories)).astype('int32'), borrow=True)

		self.init_count_window_bigrams(train_stories, window_size, batch_size)

		start_time = timeit.default_timer()

		n_bigram_counts = 0

		#create list of dicts for storing bigram counts
		bigram_counts = [collections.defaultdict(int) for word in xrange(self.lexicon_size)]
			
		#sort stories
		lengths = numpy.sum(stories > 0, axis=1)
		stories = stories[numpy.argsort(lengths)]
		stories[stories == 0] = -1

		#make sure batch sizes are even
		if n_stories % batch_size != 0:
			padding = int(numpy.ceil((n_stories % batch_size) * 1. / batch_size)) * batch_size - (n_stories % batch_size)
			stories = numpy.append(numpy.ones((padding, stories.shape[-1]), dtype='int32') * -1, stories, axis=0)
			n_stories = len(stories)

		train_stories.set_value(stories.T)
		max_story_length = train_stories.get_value().shape[0]

		for batch_index in xrange(0, n_stories, batch_size):

			story_length = numpy.sum(train_stories.get_value()[:, batch_index + batch_size - 1] > -1)

			for word_index in xrange(story_length):

				words1, words2 = self.count_window_bigrams(word_index, batch_index)

				for word1_index, word1 in enumerate(words1):
					if numpy.any(words2[word1_index] == -1):
						word2_end_index = numpy.argmax(words2[word1_index] == -1)
					else:
						#no empty words in this set
						word2_end_index = words2.shape[1]
					for word2 in words2[word1_index, :word2_end_index]:
						bigram_counts[word1][word2] += 1
						n_bigram_counts += 1


			print("...processed through word %i/%i" % (story_length, max_story_length), "of %i/%i"\
					% (batch_index + batch_size, n_stories), "stories (%.2fm)" % ((timeit.default_timer() - start_time) / 60))


			#check size of bigram counts dict
			bigram_counts_size = sum([len(bigram_counts[word1]) for word1 in xrange(self.lexicon_size)])
			if bigram_counts_size >= 10000000: #200,000,000?
				#save bigrams from this file
				self.save_bigrams(bigram_counts=bigram_counts)
				self.save_n_bigrams(n_bigram_counts=n_bigram_counts)

				#reset bigram counts list
				bigram_counts = [collections.defaultdict(int) for word in xrange(self.lexicon_size)]

		#save remaining bigrams
		self.save_bigrams(bigram_counts=bigram_counts)
		self.save_n_bigrams(n_bigram_counts=n_bigram_counts)
		

	def save_bigrams(self, bigram_counts):

		connection = sqlite3.connect(self.filepath + "/bigram_counts.db")
		cursor = connection.cursor()

		#need to create bigram counts db if it hasn't been created
		cursor.execute("CREATE TABLE IF NOT EXISTS bigram(\
						word1 INTEGER,\
						word2 INTEGER,\
						count INTEGER DEFAULT 0,\
						PRIMARY KEY (word1, word2))")

		#create an index on count and words
		cursor.execute("CREATE INDEX IF NOT EXISTS count_index ON bigram(count)")
		cursor.execute("CREATE INDEX IF NOT EXISTS word1_index ON bigram(word1)")
		cursor.execute("CREATE INDEX IF NOT EXISTS word2_index ON bigram(word2)")


		#insert current counts into db
		for word1 in xrange(len(bigram_counts)):
			if bigram_counts[word1]:
				#insert words if they don't already exist
				cursor.executemany("INSERT OR IGNORE INTO bigram(word1, word2)\
								VALUES (?, ?)",\
								[(word1, int(word2)) for word2 in bigram_counts[word1]])
				#now update counts
				cursor.executemany("UPDATE bigram\
								SET count = (count + ?)\
								WHERE word1 = ? AND word2 = ?",
								[(bigram_counts[word1][word2], word1, int(word2)) for word2 in bigram_counts[word1]])

			if word1 > 0 and (word1 % 20000) == 0:
				print("Inserted bigram counts for words up to word", word1)

		#commit insert
		connection.commit()

		#close connection
		connection.close()

		print("Saved bigram counts to", self.filepath + "/bigram_counts.db")

	def save_n_bigrams(self, n_bigram_counts):
		'''since querying the bigram db to get the total number of bigram counts is way too slow, just 
		save the number of counts to a file'''

		with open(self.filepath + '/n_bigram_counts.pkl', 'wb') as f:
			pickle.dump(n_bigram_counts, f)

		print("Saved", n_bigram_counts, "bigram counts to", self.filepath + "/n_bigram_counts.pkl")

	def get_bigram_count(self, word1=None, word2=None):

		connection = sqlite3.connect(self.filepath + "/bigram_counts.db")
		cursor = connection.cursor()

		#assert(word1 is not None or word2 is not None)

		if word1 and word2:
			cursor.execute("SELECT count FROM bigram WHERE word1 = ? AND word2 = ?", (int(word1), int(word2)))
			bigram_count = cursor.fetchone()
			if not bigram_count:
				#count is 0, but smooth by tiny number so pmi is not NaN
				bigram_count = 1e-10
			else:
				#add one to existing bigram count
				bigram_count = bigram_count[0]
		elif word1: # count of all word pairs where first word is word1
			cursor.execute("SELECT count FROM bigram WHERE word1 = ?", (int(word1),))
			bigram_count = sum([count[0] for count in cursor.fetchall()])
		elif word2: # count of all word pairs where second word in word2
			cursor.execute("SELECT count FROM bigram WHERE word2 = ?", (int(word2),))
			bigram_count = sum([count[0] for count in cursor.fetchall()])
		else: #get total count of all bigrams
			cursor.execute("SELECT SUM(count) FROM bigram")
			bigram_count = cursor.fetchone()[0]

		connection.close()

		return bigram_count

	def compute_pmi(self, word1, word2):

		word1_count = self.unigram_counts[word1]
		if not word1_count:
			word1_count = 1e-10
		word2_count = self.unigram_counts[word2]
		if not word2_count:
			word2_count = 1e-10

		#get bigram count from db
		bigram_count = self.get_bigram_count(word1, word2)

		pmi = numpy.log(bigram_count) - numpy.log(word1_count) - numpy.log(word2_count) #+ numpy.log(self.lexicon_size)

		return pmi

	def compute_causal_pmi(self, word1, word2, alpha_weight=0.66, lambda_weight=0.9):
		if not self.n_unigram_counts:
			self.n_unigram_counts = sum(self.unigram_counts)
		if not self.n_bigram_counts:
			self.n_bigram_counts = self.get_bigram_count()
		word1_count = self.get_bigram_count(word1=word1)
		if not word1_count:
			word1_count = 1e-10
		word2_count = self.get_bigram_count(word2=word2)
		if not word2_count:
			word2_count = 1e-10

		#get bigram count from db
		bigram_count = self.get_bigram_count(word1, word2)

		#necessary_score = numpy.log(bigram_count) - (numpy.log(word1_count) * alpha_weight + numpy.log(word2_count))
		# sufficient_score = numpy.log(bigram_count) - (numpy.log(word1_count) + numpy.log(word2_count) * alpha_weight)
		p_word1 = numpy.log(word1_count) - numpy.log(self.n_bigram_counts)
	   	p_word2 = numpy.log(word2_count) - numpy.log(self.n_bigram_counts)
	   	p_bigram = numpy.log(bigram_count) - numpy.log(self.n_unigram_counts)
	   	necessary_score = p_bigram - (p_word1 * alpha_weight + p_word2)
	   	sufficient_score = p_bigram - (p_word1 + p_word2 * alpha_weight)

	   	causal_pmi = necessary_score * lambda_weight + sufficient_score * (1 - lambda_weight)

	   	return causal_pmi

	def predict(self, seqs1, seqs2, causal=False):
		'''compute total pmi for each ordered pair of words in a pair of sequences - result is score of association between sequence1 and sequence2'''

		seqs1 = self.transformer.text_to_nums(seqs1)
		seqs2 = self.transformer.text_to_nums(seqs2)

		pmi_scores = []

		for seq1, seq2 in zip(seqs1, seqs2):

			sum_pmi = 0
			pmis = []

			for word1 in seq1:
				for word2 in seq2:
					#get pmi of these words
					if causal:
						pmi = self.compute_causal_pmi(word1, word2)
					else:
						pmi = self.compute_pmi(word1, word2)
					sum_pmi += pmi
					pmis.append(pmi)

			#normalize score by length of sequences
			pmi_score = sum_pmi / (len(seq1) * len(seq2))
			pmi_scores.append(pmi_score)

		return pmi_scores

	@classmethod
	def load(cls, filepath):

		with open(filepath + '/transformer.pkl', 'rb') as f:
			transformer = pickle.load(f)

		model = cls(filepath, transformer)

		with open(filepath + '/unigram_counts.pkl', 'rb') as f:
			unigram_counts = cPickle.load(f)

		model.unigram_counts = unigram_counts
		model.n_unigram_counts = numpy.sum(model.unigram_counts)
		model.n_bigram_counts = None
		model.lexicon_size = transformer.lexicon_size + 1

		return model


def make_pmi(stories, filepath):
    transformer = SequenceTransformer(min_freq=1, verbose=1, 
                                      replace_ents=False, filepath=filepath)
    stories, _, _ = transformer.fit_transform(X=stories)
    #transformer = load_transformer(filepath)
    #stories, _ = transformer.transform(X=stories)
    pmi_model = PMI_Model(dataset_name=filepath)
    pmi_model.count_unigrams(stories)
    pmi_model.count_bigrams(stories)
    return pmi_model

def eval_pmi(transformer, model, input_seqs, output_choices):
    scores = []
    index = 0
    input_seqs, output_choices = transformer.transform(X=input_seqs, y_seqs=output_choices)
    for input_seq, choices in zip(input_seqs, output_choices):
    	choice_scores = [model.score(sequences=[input_seq, choice]) for choice in choices]
    	scores.append(choice_scores)
        # choice1_score = model.score(sequences=[input_seq, output_choices[0]])
        # choice2_score = model.score(sequences=[input_seq, output_choices[1]])
        # choice_scores.append([choice1_score, choice2_score])
        index += 1
        if index % 200 == 0:
            print("predicted", index, "inputs")
        #print choice_scores
    scores = numpy.array(scores)
    pred_choices = numpy.argmax(scores, axis=1)
    return scores, pred_choices


# if __name__ == "__main__":

# 	parser = argparse.ArgumentParser()
# 	parser.add_argument('-dataset', help='Specify name of dataset for PMI model.', required=True)
# 	parser.add_argument('-train', help='Specify train flag to create bigram counts file.', default=False, action='store_true', required=False)
# 	#parser.add_argument('-bigram_file', help='Provide name of file to save bigrams to.', required=True)
# 	args = parser.parse_args()

# 	narrative_dataset = Narrative_Dataset(dataset_name=args.dataset)

# 	#drop previous bigram counts file
# 	if args.train and os.path.isfile(narrative_dataset.dataset_name + "/" + narrative_dataset.bigram_counts_db):
# 		overwrite = raw_input("Are you sure you want to overwrite the bigram counts in " + narrative_dataset.dataset_name + "/" + narrative_dataset.bigram_counts_db + "? (y/n) ")
# 		if overwrite.lower() == "y":
# 			#print "Overwriting bigram counts in ", narrative_dataset.dataset_name + "/" + narrative_dataset.bigram_counts_db
# 			#give user time to cancel overwrite
# 			os.remove(narrative_dataset.dataset_name + "/" + narrative_dataset.bigram_counts_db)

# 	#specify limit on story length
# 	pmi_model = PMI_Model(dataset=narrative_dataset, max_retrieval_length=2000, train=args.train)



