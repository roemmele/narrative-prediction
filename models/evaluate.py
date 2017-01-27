import numpy, random

def evaluate_difference(model1_values, model2_values, num_trials=10000):
	'''run stats test to determine if difference between model accuracies is statistically significant'''
	#import pdb; pdb.set_trace()

	assert len(model1_values) == len(model2_values)
	model_difference = numpy.mean(model1_values) - numpy.mean(model2_values)
	counter = 0
	for trial in xrange(num_trials):
		#reshuffle all predictions between these 2 models
		shuffled_values = sorted(model1_values + model2_values, key=lambda *args: random.random())
		model1_sample = shuffled_values[:len(model1_values)]
		model2_sample = shuffled_values[len(model1_values):]
		sample_difference = numpy.mean(model1_sample) - numpy.mean(model2_sample)
		if sample_difference * numpy.sign(sample_difference) >= model_difference * numpy.sign(model_difference):
			counter += 1
	p_val = float(counter + 1) / (num_trials + 1)
	
	return p_val