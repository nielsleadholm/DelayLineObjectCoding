import json
import numpy as np
import os
import pprint
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


def train_classifier(stimuli_params, train_dataset, eval_dataset):

	print("===TEMPORARILY USING HARD-CODED *DATA-SETS*===")
	drift_iter = stimuli_params["drift_coef_list"][0]
	seed_iter = 0



	classifier_training_features, classifier_training_labels = extract_data(stimuli_params, train_dataset,
																			drift_iter, seed_iter, layer="output")
	


	# print("Training features")
	# print(np.shape(classifier_training_features))
	# print(classifier_training_features)

	# print("Training labels:")
	# print(classifier_training_labels)
	# print(np.shape(classifier_training_labels))


	# *** randomize the order of the training data **

	rand_idx = np.random.permutation(len(classifier_training_labels))
    classifier_training_features = classifier_training_features[rand_idx]
    classifier_training_labels = classifier_training_labels[rand_idx]

    print("\n\n\n\n=== TESTING THE INFLUENCE OF SHUFFLED DATA! ===")

	classifier_eval_features, classifier_eval_labels = extract_data(stimuli_params, eval_dataset,
																	drift_iter, seed_iter, layer="output")
	
	# Normalize evaluation data based on training data values
	#classifier_eval_features = classifier_eval_features / normalization_max

	# print("Eval data")
	# print(classifier_eval_features)
	# print(classifier_eval_labels)
	# #print(np.amax(classifier_eval_features, axis=0))

	#print("\nClassifier")
	clf = LogisticRegression(random_state=0).fit(classifier_training_features,
												 classifier_training_labels)
	
	results_dic = {}

	results_dic["training_acc"] = clf.score(classifier_training_features, classifier_training_labels)
	results_dic["eval_acc"] = clf.score(classifier_eval_features, classifier_eval_labels)
	results_dic["eval_prob_scores"] = clf.predict_proba(classifier_eval_features)

	print("Original prob scores")
	print(np.shape(results_dic["eval_prob_scores"]))
	print(results_dic["eval_prob_scores"])

	print("Flattened prob scores:")
	print(np.shape(np.flatten(results_dic["eval_prob_scores"])))
	print(np.flatten(results_dic["eval_prob_scores"]))

	plt.hist(np.flatten(results_dic["eval_prob_scores"]), bins=10)
	plt.title("Prob scores: " + eval_dataset)
	plt.savefig("analysis_results/prob_scores_" + eval_dataset)
	plt.clf()

	print("\nResults:")
	pprint.pprint(results_dic, depth=1)

	with open("analysis_results/" + str(seed_iter) "/results_" + eval_dataset + ".json", 'w') as f:
    	json.dump(results_dic, f)


if __name__ == '__main__':

	if os.path.exists("analysis_results") == 0:
		try:
			os.mkdir("analysis_results")
		except OSError:
			pass

	print("\nAssessing performane of linear classifiers trained on network activity")
	with open('config_TranslationInvariance.yaml') as f:
		params = yaml.load(f, Loader=yaml.FullLoader)

	stimuli_params = params["stimuli_params"]
	network_params = params["network_params"]

	print("\n\nEvaluating accuracy of classifier *BEFORE* STDP learning...")
	train_classifier(stimuli_params, train_dataset="untrained_spikepair_inputs",
					 eval_dataset="untrained_spikepair_inputs_classifier")

	print("\n\nEvaluating accuracy of classifier following STDP learning...")
	train_classifier(stimuli_params, train_dataset="spikepair_trained_spikepair_inputs",
					 eval_dataset="spikepair_trained_spikepair_inputs_classifier")

	# print("\n\nEvaluating accuracy of classifier following STDP learning *ON NONSENSE INPUTS*...")
	# print("Poor accuracy expected")
	# train_classifier(stimuli_params, train_dataset="spikepair_trained_spikepair_inputs",
	# 				 eval_dataset="spikepair_trained_spikepair_inputs_nonsense")

