from brian2 import *
import copy
import json
import numpy as np
import os
import pprint
import random
import yaml
import matplotlib.pyplot as plt
from analyse_sim_results import extract_firing_rates
from sklearn.linear_model import LogisticRegression


def extract_data(params, data_set, drift_iter, seed_iter, layer="output"):
	'''
	Extract data from the network activity for training a linear classifier
	on firing rates
	'''
	data_dic = dict()
	data_dic["times"] = np.fromfile("raw_data/" + str(seed_iter) + "/"
									+ data_set  + "_drift_iter_" + str(drift_iter) + "/raster_" + layer + "_layer_times.txt", dtype=np.float64, sep='\n')
	data_dic["ids"] = np.fromfile("raw_data/" + str(seed_iter) + "/"
									+ data_set  + "_drift_iter_" + str(drift_iter) + "/raster_" + layer + "_layer_IDs.txt", sep='\n')

	FR_array, _ = extract_firing_rates(params, data_dic, layer)

	number_data_points = np.shape(FR_array)[-1]

	# FR array is already structured in such a way that the first half of the data
	# is from the first object presentations, and the second half from the second object
	labels = np.concatenate((np.zeros(number_data_points),
							np.ones(number_data_points)), axis=0)

	output_features = np.transpose(np.concatenate(FR_array, axis=1))

	return output_features, labels


def train_classifier(seed_iter, drift_iter, stimuli_params, train_dataset, eval_dataset):


	classifier_training_features, classifier_training_labels = extract_data(stimuli_params, train_dataset,
																			drift_iter, seed_iter, layer="output")
	



	# As "number_of_eval_presentations" is used by a variety of down-stream analysis code
	# to e.g. appropriately extract firing rates, temporarily set this to the correct
	# value for the *classifier evaluation* data-set (i.e. "number_of_classifier_assessment_presentations")
	number_of_presents_backup = copy.copy(stimuli_params["number_of_eval_presentations"])
	stimuli_params["number_of_eval_presentations"] = stimuli_params["number_of_classifier_assessment_presentations"]

	classifier_eval_features, classifier_eval_labels = extract_data(stimuli_params, eval_dataset,
																	drift_iter, seed_iter, layer="output")
	
	stimuli_params["number_of_eval_presentations"] = copy.copy(number_of_presents_backup)

	clf = LogisticRegression(random_state=0).fit(classifier_training_features,
												 classifier_training_labels)
	
	results_dic = {}

	results_dic["training_acc"] = clf.score(classifier_training_features, classifier_training_labels)
	results_dic["eval_acc"] = clf.score(classifier_eval_features, classifier_eval_labels)
	results_dic["eval_prob_scores"] = (clf.predict_proba(classifier_eval_features)).tolist()

	plt.hist(np.asarray(results_dic["eval_prob_scores"]).flatten(), bins=10)
	plt.title("Prob scores: " + eval_dataset)
	plt.savefig("analysis_results/" + str(seed_iter) + "/prob_scores_" + eval_dataset)
	plt.clf()

	print("Results:")
	pprint.pprint(results_dic, depth=1)

	with open("analysis_results/" + str(seed_iter) + "/results_" + eval_dataset + ".json", 'w') as f:
		json.dump(results_dic, f)


if __name__ == '__main__':

	if os.path.exists("analysis_results") == 0:
		try:
			os.mkdir("analysis_results")
		except OSError:
			pass

	print("\n\nAssessing performance of linear classifiers trained on network activity")

	with open('config_TranslationInvariance.yaml') as f:
		params = yaml.load(f, Loader=yaml.FullLoader)

	stimuli_params = params["stimuli_params"]
	network_params = params["network_params"]

	for seed_iter in stimuli_params["seeds_list"]:

		# Set seed for both Brian and Numpy; re-set for each drift iter
		seed(seed_iter)
		random.seed(seed_iter)

		if os.path.exists("analysis_results/" + str(seed_iter)) == 0:
			try:
				os.mkdir("analysis_results/" + str(seed_iter))
			except OSError:
				pass


		for drift_iter in stimuli_params["drift_coef_list"]:


			print("\n\nEvaluating accuracy of classifier on different T's *BEFORE* STDP learning...")
			train_classifier(seed_iter, drift_iter, stimuli_params, train_dataset="untrained_spikepair_inputs",
							 eval_dataset="untrained_spikepair_inputs_classifier")

			print("\n\nEvaluating accuracy of classifier on different T's *FOLLOWING* STDP learning...")
			train_classifier(seed_iter, drift_iter, stimuli_params, train_dataset="spikepair_trained_spikepair_inputs",
							 eval_dataset="spikepair_trained_spikepair_inputs_classifier")


			print("\n\n\n\nEvaluating accuracy of classifier on NOISE vs objects *BEFORE* STDP learning...")
			train_classifier(seed_iter, drift_iter, stimuli_params, train_dataset="untrained_alternating_inputs",
							 eval_dataset="untrained_alternating_inputs_classifier")

			print("\n\nEvaluating accuracy of classifier on NOISE vs objects *FOLLOWING* STDP learning...")
			train_classifier(seed_iter, drift_iter, stimuli_params, train_dataset="spikepair_trained_alternating_inputs",
							 eval_dataset="spikepair_trained_alternating_inputs_classifier")

