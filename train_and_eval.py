from brian2 import *
import copy
import numpy as np
import os
import pprint
import random
import yaml
import run_simulation
from generate_input_stims import (create_underlying_spike_assemblies, plot_input_raster,
								  generate_spikes_fixed_pairs, visualize_spike_slopes)

def make_directories(dir_name, seed_iter, sub_dir_list, drift_iter='NA'):
	if os.path.exists(dir_name) == 0:
		try:
			os.mkdir(dir_name)
		except OSError:
			pass

	if os.path.exists(dir_name + "/" + str(seed_iter)) == 0:
		try:
			os.mkdir(dir_name + "/" + str(seed_iter))
		except OSError:
			pass

	[make_sub_directories(dir_name + "/" + str(seed_iter),
						  sub_name + "_drift_iter_" + str(drift_iter)) for
						  sub_name in sub_dir_list]


def make_sub_directories(upper_name, sub_name):
	if os.path.exists(upper_name + "/" + sub_name) == 0:
		try:
			os.mkdir(upper_name + "/" + sub_name)
		except OSError:
			pass


if __name__ == '__main__':

	with open('config_TranslationInvariance.yaml') as f:
		params = yaml.load(f, Loader=yaml.FullLoader)

	print("\nTraining and evaluating networks\nSetup parameters:")
	pprint.pprint(params)

	stimuli_params = params['stimuli_params']
	network_params = params['network_params']


	for seed_iter in stimuli_params["seeds_list"]:

		print("\n\n==NEW SEED== : " + str(seed_iter))

		[make_directories(dir_name, seed_iter, sub_dir_list=["input_stimuli"]) for dir_name in ["figures"]]

		for drift_iter in stimuli_params["drift_coef_list"]:

			# =============================================================================
			# SETUP     
			# =============================================================================

			# Set seed for both Brian and Numpy; re-set for each drift iter
			seed(seed_iter)
			random.seed(seed_iter)

			print("\nCurrent drift coefficient limit: " + str(drift_iter))
			[make_directories(dir_name, seed_iter, sub_dir_list=[				
				"untrained_spikepair_inputs",
				"untrained_spikepair_inputs_classifier",

				"untrained_alternating_inputs",
				"untrained_alternating_inputs_classifier",

				"during_spikepair_training",

				"spikepair_trained_spikepair_inputs",
				"spikepair_trained_spikepair_inputs_classifier",

				"spikepair_trained_alternating_inputs",
				"spikepair_trained_alternating_inputs_classifier"],
				drift_iter=drift_iter) for dir_name in ["weights", "figures", "raw_data"]]


			# Generate the underlying spike-timing slopes that will form the basis of all the
			# input stimuli
			assembly_IDs, relative_times_vertical, relative_times_horizontal, neuron_drift_coefs_dic = create_underlying_spike_assemblies(stimuli_params,
					drift_iter, seed_iter)

		    # Plot spike slopes
			visualize_spike_slopes(minimum(3, stimuli_params["input_layer_size"]), stimuli_params, relative_times_vertical,
				relative_times_horizontal, neuron_drift_coefs_dic, seed_iter)



			# =============================================================================
			# PRE TRAINING - UPRIGHT AND INVERTED INPUTS 
			# =============================================================================

			# Generate pre-training spike IDs; note that due to eval_bool, these will be generated
			# for a different total duration than the training inputs
			pre_training_spike_IDs, pre_training_spike_times, spike_pair_differences = generate_spikes_fixed_pairs(stimuli_params, assembly_IDs,
			    relative_times_vertical, relative_times_horizontal, seed_iter, eval_bool=True)
			
			plot_input_raster(stimuli_params, assembly_IDs, pre_training_spike_IDs, pre_training_spike_times,
					neuron_drift_coefs_dic, seed_iter, eval_bool=True, input_name="pre_training")

			# EVALUATE the network on the spatio-temporal input before training, and initialize weights
			run_params = {"weight_file" : "weights/" + str(seed_iter) + "/rand",
						  "STDP_on_bool" : False,
						  "input_stim" : [pre_training_spike_IDs, pre_training_spike_times],
				  		  "output_dir" : "/untrained_spikepair_inputs" + "_drift_iter_" + str(drift_iter)
						  }
			run_simulation.main_run(stimuli_params, network_params, run_params, seed_iter,
				spike_pair_differences, initialize_weights_bool=True)


			# Generate pre-STDP-training EVALUATION data for the *LINEAR CLASSIFIER* (i.e. to deteremine
			# the benefit of STDP for the classifier)

			# As "number_of_eval_presentations" is used by a variety of down-stream analysis code
			# to e.g. appropriately extract firing rates, temporarily set this to the correct
			# value for this particular data-set (number_of_classifier_assessment_presentations)
			# This enables evaluating the classifier on more data than it is trained on
			number_of_presents_backup = copy.copy(stimuli_params["number_of_eval_presentations"])
			stimuli_params["number_of_eval_presentations"] = copy.copy(stimuli_params["number_of_classifier_assessment_presentations"])
			classifier_pre_training_spike_IDs, classifier_pre_training_spike_times, spike_pair_differences = generate_spikes_fixed_pairs(stimuli_params, assembly_IDs,
			    relative_times_vertical, relative_times_horizontal, seed_iter, eval_bool=True)
			
			plot_input_raster(stimuli_params, assembly_IDs, classifier_pre_training_spike_IDs, classifier_pre_training_spike_times,
					neuron_drift_coefs_dic, seed_iter, eval_bool=True, input_name="classifier_pre_training")

			# EVALUATE the network on the spatio-temporal input before training
			run_params = {"weight_file" : "weights/" + str(seed_iter) + "/rand",
						  "STDP_on_bool" : False,
						  "input_stim" : [classifier_pre_training_spike_IDs, classifier_pre_training_spike_times],
				  		  "output_dir" : "/untrained_spikepair_inputs_classifier" + "_drift_iter_" + str(drift_iter)
						  }
			run_simulation.main_run(stimuli_params, network_params, run_params, seed_iter,
				spike_pair_differences)
			# Reset the number of eval presentations
			stimuli_params["number_of_eval_presentations"] = copy.copy(number_of_presents_backup)



			# =============================================================================
			# PRE TRAINING - ALTERNATING NOISE AND OBJECTS INPUTS
			# =============================================================================

			# GENERATE spikes that alternate stimuli and noise - BEFORE any STDP training
			number_of_presents_backup = copy.copy(stimuli_params["number_of_train_presentations"])
			# We set eval_bool to False to alternate stimuli and noise, but to ensure the
			# number of presentations is comparable, number_of_train_presentations is potentially set
			# NB for example that information theory code will always use number_of_eval_presentations
			stimuli_params["number_of_train_presentations"] = copy.copy(stimuli_params["number_of_eval_presentations"])
			alternating_spike_IDs_pre_train, alternating_spike_times_pre_train, spike_pair_differences = generate_spikes_fixed_pairs(stimuli_params, assembly_IDs,
			    relative_times_vertical, relative_times_horizontal, seed_iter)
			
			plot_input_raster(stimuli_params, assembly_IDs, alternating_spike_IDs_pre_train, alternating_spike_times_pre_train,
				neuron_drift_coefs_dic, seed_iter, eval_bool=False, input_name="alternating_pre_train")

			run_params = {"weight_file" : "weights/" + str(seed_iter) + "/rand",
						  "STDP_on_bool" : False,
						  "input_stim" : [alternating_spike_IDs_pre_train, alternating_spike_times_pre_train],
				  		  "output_dir" : "/untrained_alternating_inputs" + "_drift_iter_" + str(drift_iter)
						  }
			run_simulation.main_run(stimuli_params, network_params, run_params, seed_iter,
				spike_pair_differences)
			stimuli_params["number_of_train_presentations"] = copy.copy(number_of_presents_backup)


			# As above, but for EVALUATING the LINEAR CLASSIFIER on the alternating stimuli
			number_of_presents_backup = copy.copy(stimuli_params["number_of_train_presentations"])
			number_of_presents_backup_two = copy.copy(stimuli_params["number_of_eval_presentations"])
			# We set eval_bool to False to alternate stimuli and noise, but to ensure the
			# number of presentations is comparable, number_of_train_presentations is potentially set
			# NB for example that information theory code will always use number_of_eval_presentations
			# Because main_run will use number_of_eval_presentations unless STDP is active, also set this
			stimuli_params["number_of_train_presentations"] = copy.copy(stimuli_params["number_of_classifier_assessment_presentations"])
			stimuli_params["number_of_eval_presentations"] = copy.copy(stimuli_params["number_of_classifier_assessment_presentations"])
			classifier_alternating_spike_IDs_pre_train, classifier_alternating_spike_times_pre_train, spike_pair_differences = generate_spikes_fixed_pairs(stimuli_params, assembly_IDs,
			    relative_times_vertical, relative_times_horizontal, seed_iter)
			
			plot_input_raster(stimuli_params, assembly_IDs, classifier_alternating_spike_IDs_pre_train,
				classifier_alternating_spike_times_pre_train,
				neuron_drift_coefs_dic, seed_iter, eval_bool=False, input_name="classifier_alternating_pre_train")

			# TRAIN the network on spatiotemporally structured inputs
			run_params = {"weight_file" : "weights/" + str(seed_iter) + "/rand",
						  "STDP_on_bool" : False,
						  "input_stim" : [classifier_alternating_spike_IDs_pre_train,
						  		classifier_alternating_spike_times_pre_train],
				  		  "output_dir" : "/untrained_alternating_inputs_classifier" + "_drift_iter_" + str(drift_iter)
						  }
			run_simulation.main_run(stimuli_params, network_params, run_params, seed_iter,
				spike_pair_differences)
			stimuli_params["number_of_train_presentations"] = copy.copy(number_of_presents_backup)
			stimuli_params["number_of_eval_presentations"] = copy.copy(number_of_presents_backup_two)



			# =============================================================================
			# STDP TRAINING       
			# =============================================================================

			# # GENERATE spikes for training
			# training_spike_IDs, training_spike_times, spike_pair_differences = generate_spikes_fixed_pairs(stimuli_params, assembly_IDs,
			#     relative_times_vertical, relative_times_horizontal, seed_iter)
			
			# plot_input_raster(stimuli_params, assembly_IDs, training_spike_IDs, training_spike_times,
			# 	neuron_drift_coefs_dic, seed_iter, eval_bool=False, input_name="training")

			# #print("====TEMPORARILY INITIALIZING WEIGHTS HERE")

			# # TRAIN the network on spatiotemporally structured inputs
			# run_params = {"weight_file" : "weights/" + str(seed_iter) + "/rand",
			# 			  "STDP_on_bool" : True,
			# 			  "input_stim" : [training_spike_IDs, training_spike_times],
			# 	  		  "output_dir" : "/during_spikepair_training" + "_drift_iter_" + str(drift_iter)
			# 			  }
			# run_simulation.main_run(stimuli_params, network_params, run_params, seed_iter,
			# 	spike_pair_differences, initialize_weights_bool=False)



			# =============================================================================
			# POST TRAINING - UPRIGHT AND INVERTED INPUTS    
			# =============================================================================

			# # GENERATE the evaluation spikes, iterating (in blocks) through the possible translations
		 #    # eval_bool determines how translations are sampled (i.e. in blocks as opposed to in a random order)
			# print("\nGenerating spatio-temporal patterns for *evaluation*")
			# eval_spike_IDs, eval_spike_times, spike_pair_differences = generate_spikes_fixed_pairs(stimuli_params, assembly_IDs,
			#     relative_times_vertical, relative_times_horizontal, seed_iter, eval_bool=True)

			# plot_input_raster(stimuli_params, assembly_IDs, eval_spike_IDs, eval_spike_times,
			# 	neuron_drift_coefs_dic, seed_iter, eval_bool=True, input_name="evaluation_spikepairs")

			# # EVALUATE the spatio-temporally trained network on spatio-temporal inputs
			# run_params = {"weight_file" : ("weights/" + str(seed_iter) + "/during_spikepair_training"
			# 							   + "_drift_iter_" + str(drift_iter) + "/final"),
			# 			  "STDP_on_bool" : False,
			# 			  "input_stim" : [eval_spike_IDs, eval_spike_times],
			# 	  		  "output_dir" : "/spikepair_trained_spikepair_inputs" + "_drift_iter_" + str(drift_iter)
			# 			  }
			# run_simulation.main_run(stimuli_params, network_params, run_params, seed_iter,
			# 	spike_pair_differences)


			# # GENERATE the EVALUATION spikes for the *LINEAR CLASSIFIER*, where the network has now been trained with STDP
			# print("\nGenerating spatio-temporal patterns for *classifier evaluation*")
			# number_of_presents_backup = copy.copy(stimuli_params["number_of_eval_presentations"])
			# stimuli_params["number_of_eval_presentations"] = stimuli_params["number_of_classifier_assessment_presentations"]
			# classifier_eval_spike_IDs, classifier_eval_spike_times, spike_pair_differences = generate_spikes_fixed_pairs(stimuli_params, assembly_IDs,
			#     relative_times_vertical, relative_times_horizontal, seed_iter, eval_bool=True)

			# plot_input_raster(stimuli_params, assembly_IDs, classifier_eval_spike_IDs, classifier_eval_spike_times,
			# 	neuron_drift_coefs_dic, seed_iter, eval_bool=True, input_name="classifier_evaluation_spikepairs")

			# # EVALUATE the spatio-temporally trained network on spatio-temporal inputs for the Linear Classifier
			# run_params = {"weight_file" : ("weights/" + str(seed_iter) + "/during_spikepair_training"
			# 							   + "_drift_iter_" + str(drift_iter) + "/final"),
			# 			  "STDP_on_bool" : False,
			# 			  "input_stim" : [classifier_eval_spike_IDs, classifier_eval_spike_times],
			# 	  		  "output_dir" : "/spikepair_trained_spikepair_inputs_classifier" + "_drift_iter_" + str(drift_iter)
			# 			  }
			# run_simulation.main_run(stimuli_params, network_params, run_params, seed_iter,
			# 	spike_pair_differences)
			# stimuli_params["number_of_eval_presentations"] = copy.copy(number_of_presents_backup)



			# =============================================================================
			# POST TRAINING - ALTERNATING NOISE AND OBJECTS INPUTS
			# =============================================================================	

			# GENERATE spikes that alternate stimuli and noise - AFTER STDP training
			number_of_presents_backup = copy.copy(stimuli_params["number_of_train_presentations"])
			# We set eval_bool to False to alternate stimuli and noise, but to ensure the
			# number of presentations is comparable, number_of_train_presentations is potentially set
			# NB for example that information theory code will always use number_of_eval_presentations
			stimuli_params["number_of_train_presentations"] = copy.copy(stimuli_params["number_of_eval_presentations"])
			alternating_spike_IDs, alternating_spike_times, spike_pair_differences = generate_spikes_fixed_pairs(stimuli_params, assembly_IDs,
			    relative_times_vertical, relative_times_horizontal, seed_iter)
			
			plot_input_raster(stimuli_params, assembly_IDs, alternating_spike_IDs, alternating_spike_times,
				neuron_drift_coefs_dic, seed_iter, eval_bool=False, input_name="alternating_post_train")

			run_params = {"weight_file" : ("weights/" + str(seed_iter) + "/during_spikepair_training"
										   + "_drift_iter_" + str(drift_iter) + "/final"),
						  "STDP_on_bool" : False,
						  "input_stim" : [alternating_spike_IDs, alternating_spike_times],
				  		  "output_dir" : "/spikepair_trained_alternating_inputs" + "_drift_iter_" + str(drift_iter)
						  }
			run_simulation.main_run(stimuli_params, network_params, run_params, seed_iter,
				spike_pair_differences)
			stimuli_params["number_of_train_presentations"] = copy.copy(number_of_presents_backup)


			# As above, but for EVALUATING the LINEAR CLASSIFIER on the alternating stimuli
			number_of_presents_backup = copy.copy(stimuli_params["number_of_train_presentations"])
			number_of_presents_backup_two = copy.copy(stimuli_params["number_of_eval_presentations"])
			# We set eval_bool to False to alternate stimuli and noise, but to ensure the
			# number of presentations is comparable, number_of_train_presentations is potentially set
			# NB for example that information theory code will always use number_of_eval_presentations
			stimuli_params["number_of_train_presentations"] = copy.copy(stimuli_params["number_of_classifier_assessment_presentations"])
			stimuli_params["number_of_eval_presentations"] = copy.copy(stimuli_params["number_of_classifier_assessment_presentations"])
			classifier_alternating_spike_IDs, classifier_alternating_spike_times, spike_pair_differences = generate_spikes_fixed_pairs(stimuli_params, assembly_IDs,
			    relative_times_vertical, relative_times_horizontal, seed_iter)
			
			plot_input_raster(stimuli_params, assembly_IDs, classifier_alternating_spike_IDs, classifier_alternating_spike_times,
				neuron_drift_coefs_dic, seed_iter, eval_bool=False, input_name="classifier_alternating_post_train")

			# TRAIN the network on spatiotemporally structured inputs
			run_params = {"weight_file" : ("weights/" + str(seed_iter) + "/during_spikepair_training"
										   + "_drift_iter_" + str(drift_iter) + "/final"),
						  "STDP_on_bool" : False,
						  "input_stim" : [classifier_alternating_spike_IDs, classifier_alternating_spike_times],
				  		  "output_dir" : "/spikepair_trained_alternating_inputs_classifier" + "_drift_iter_" + str(drift_iter)
						  }
			run_simulation.main_run(stimuli_params, network_params, run_params, seed_iter,
				spike_pair_differences)
			stimuli_params["number_of_train_presentations"] = copy.copy(number_of_presents_backup)
			stimuli_params["number_of_eval_presentations"] = copy.copy(number_of_presents_backup_two)

