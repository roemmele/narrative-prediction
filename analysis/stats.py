import numpy
import random


def eval_proportion_diff(model1_pos, model2_pos, model1_samples, model2_samples, num_trials=10000, verbose=False):
    '''compare proportion of positively predicted samples between two models'''

    model1_values = numpy.zeros((model1_samples))
    model1_values[:model1_pos] = 1

    model2_values = numpy.zeros((model2_samples))
    model2_values[:model2_pos] = 1

    p_val = eval_diff(model1_values, model2_values, num_trials=num_trials, verbose=verbose)
    return p_val


def eval_diff(model1_values, model2_values, num_trials=10000, verbose=False):
    '''return p-value of difference between model1_values and model2_values'''
    #import pdb; pdb.set_trace()

    model1_values = model1_values.flatten()
    model2_values = model2_values.flatten()
    # filter any NaN values
    model1_values = model1_values[~numpy.isnan(model1_values)]
    model2_values = model2_values[~numpy.isnan(model2_values)]
    model_difference = numpy.mean(model2_values) - numpy.mean(model1_values)
    counter = 0
    for trial in xrange(num_trials):
        # reshuffle all predictions between these 2 models
        values = numpy.concatenate((model1_values, model2_values))
        numpy.random.shuffle(values)
        model1_sample = values[:len(model1_values)]
        model2_sample = values[-len(model2_values):]
        sample_difference = numpy.mean(model2_sample) - numpy.mean(model1_sample)
        # if one_tailed and sample_difference >= model_difference:
        # 	counter += 1
        if numpy.abs(sample_difference) >= numpy.abs(model_difference):  # two tailed test
            counter += 1
        if verbose and trial % 500 == 0:
            print("completed", trial, "trials...")
    p_val = float(counter + 1) / (num_trials + 1)

    return p_val


def eval_all_diffs(model_values, num_trials=10000, verbose=False):
    '''takes a set of model values as input, figures out which pairs of models to evaluate 
    differences for based on the order of their values, and returns p-values for all tests'''

    sorted_models = sorted([(numpy.mean(values), model) for model, values in model_values.items()])
    model_pairs = [(sorted_models[idx][1], sorted_models[idx + 1][1]) for idx in range(len(sorted_models) - 1)]

    p_vals = {}

    for model1, model2 in model_pairs:
        p_vals[(model1, model2)] = eval_diff(model_values[model1], model_values[model2], num_trials=num_trials, verbose=verbose)

    return p_vals


def eval_all_proportion_diffs(model_pos, model_samples, num_trials=10000, verbose=False):
    '''takes a set of model proportion values as input, figures out which pairs of models to evaluate 
    differences for based on the order of their values, and returns p-values for all tests'''

    prop_values = {model: model_pos[model] * 1. / model_samples[model] for model in model_pos}
    sorted_models = sorted([(numpy.mean(values), model) for model, values in prop_values.items()])
    model_pairs = [(sorted_models[idx][1], sorted_models[idx + 1][1]) for idx in range(len(sorted_models) - 1)]

    p_vals = {}

    for model1, model2 in model_pairs:
        p_vals[(model1, model2)] = eval_proportion_diff(model_pos[model1], model_pos[model2],
                                                        model_samples[model1], model_samples[model2],
                                                        num_trials=num_trials, verbose=verbose)

    return p_vals
