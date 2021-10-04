from brian2 import *
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


	for seed_iter in range(stimuli_params["num_seeds"]):

		# print("====TEMPORARILY SETTING A DIFFERENT SEED")

		# seed_iter = 1

		# === SETUP ===

		# Set seed for both Brian and Numpy
		print("\n\n==NEW SEED== : " + str(seed_iter))
		seed(seed_iter)
		random.seed(seed_iter)		

		[make_directories(dir_name, seed_iter, sub_dir_list=["input_stimuli", 
					 "untrained_poisson_inputs", "during_poisson_training",
					 "poisson_trained_poisson_inputs"]) for dir_name in ["weights", "figures", "raw_data"]]


		# # EVALUATE without any training, given Poisson input, and initialize starting weights
		# run_params = {"weight_file" : None,
		# 			  "STDP_on_bool" : False,
		# 			  "input_stim" : None,  # Options are None for Poisson input or
		# 			  # (spike_IDs, spike_times) for spike-pair inputs or spike_wave_input
		# 			  "output_dir" : "/untrained_poisson_inputs" + "_drift_iter_NA"
		# 			  }
		# run_simulation.main_run(stimuli_params, network_params, run_params, seed_iter,
		# 		initialize_weights_bool=True)


		# # TRAIN a copy of the network with Poisson input
		# run_params = {"weight_file" : "weights/" + str(seed_iter) + "/rand",
		# 			  "STDP_on_bool" : True,
		# 			  "input_stim" : None,
		# 			  "output_dir" : "/during_poisson_training" + "_drift_iter_NA"
		# 			  }
		# run_simulation.main_run(stimuli_params, network_params, run_params, seed_iter)


		# # EVALUATE the Poisson trained network with Poisson inputs
		# # NB that with the random seed as set, this will *not* be the same Poisson input
		# # used during pre-training or training
		# run_params = {"weight_file" : "weights/" + str(seed_iter) + "/during_poisson_training_drift_iter_NA/" + "final",
		# 			  "STDP_on_bool" : False,
		# 			  "input_stim" : None,
		# 			  "output_dir" : "/poisson_trained_poisson_inputs" + "_drift_iter_NA"
		# 			  }
		# run_simulation.main_run(stimuli_params, network_params, run_params, seed_iter)



		for drift_iter in stimuli_params["drift_coef_list"]:


			print("\nCurrent drift coefficient limit: " + str(drift_iter))
			[make_directories(dir_name, seed_iter, sub_dir_list=["untrained_spikepair_inputs",
					 "during_spikepair_training", "spikepair_trained_poisson_inputs",
					 "spikepair_trained_spikepair_inputs", "poisson_trained_spikepair_inputs",
					 "spikepair_trained_mixed_inputs", "poisson_trained_mixed_inputs", "untrained_mixed_inputs",
					 "spikepair_trained_spikepair_inputs_classifier",
					 "untrained_spikepair_inputs_classifier",
					 #"spikepair_trained_spikepair_inputs_nonsense",
					 "spikepair_trained_alternating_inputs"]
					 , drift_iter=drift_iter) for dir_name in ["weights", "figures", "raw_data"]]


			# Generate the underlying spike-timing slopes that will form the basis of all the
			# input stimuli
			assembly_IDs, relative_times_vertical, relative_times_horizontal, neuron_drift_coefs_dic = create_underlying_spike_assemblies(stimuli_params,
					drift_iter, seed_iter)

		    # Plot spike slopes
			visualize_spike_slopes(minimum(3, stimuli_params["input_layer_size"]), stimuli_params, relative_times_vertical,
				relative_times_horizontal, neuron_drift_coefs_dic, seed_iter)


			# # Generate pre-training spike IDs; note that due to eval_bool, these may be generated
			# # for a different total duration than the training ones
			# pre_training_spike_IDs, pre_training_spike_times, spike_pair_differences = generate_spikes_fixed_pairs(stimuli_params, assembly_IDs,
			#     relative_times_vertical, relative_times_horizontal, seed_iter, eval_bool=True)
			
			# plot_input_raster(stimuli_params, assembly_IDs, pre_training_spike_IDs, pre_training_spike_times,
			# 		neuron_drift_coefs_dic, seed_iter, eval_bool=True, input_name="pre_training")

			# print("====TEMPORARILY INITIALIZING WEIGHTS HERE")

			# # EVALUATE the network on the spatio-temporal input before training
			# run_params = {"weight_file" : "weights/" + str(seed_iter) + "/rand",
			# 			  "STDP_on_bool" : False,
			# 			  "input_stim" : [pre_training_spike_IDs, pre_training_spike_times],
			# 	  		  "output_dir" : "/untrained_spikepair_inputs" + "_drift_iter_" + str(drift_iter)
			# 			  }
			# run_simulation.main_run(stimuli_params, network_params, run_params, seed_iter,
			# 	spike_pair_differences, initialize_weights_bool=True)


			# #exit()

			# # Generate pre-training spike for the *LINEAR CLASSIFIER*; note that due to eval_bool, these may be generated
			# # for a different total duration than the training ones
			# classifier_pre_training_spike_IDs, classifier_pre_training_spike_times, spike_pair_differences = generate_spikes_fixed_pairs(stimuli_params, assembly_IDs,
			#     relative_times_vertical, relative_times_horizontal, seed_iter, eval_bool=True)
			
			# plot_input_raster(stimuli_params, assembly_IDs, classifier_pre_training_spike_IDs, classifier_pre_training_spike_times,
			# 		neuron_drift_coefs_dic, seed_iter, eval_bool=True, input_name="classifier_pre_training")

			# # EVALUATE the network on the spatio-temporal input before training
			# run_params = {"weight_file" : "weights/" + str(seed_iter) + "/rand",
			# 			  "STDP_on_bool" : False,
			# 			  "input_stim" : [classifier_pre_training_spike_IDs, classifier_pre_training_spike_times],
			# 	  		  "output_dir" : "/untrained_spikepair_inputs_classifier" + "_drift_iter_" + str(drift_iter)
			# 			  }
			# run_simulation.main_run(stimuli_params, network_params, run_params, seed_iter,
			# 	spike_pair_differences, initialize_weights_bool=False)



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



			# # # # # EVALUATE the spatio-temporally trained network on Poisson inputs
			# # # # run_params = {"weight_file" : ("weights/" + str(seed_iter) + "/during_spikepair_training"
			# # # # 							   + "_drift_iter_" + str(drift_iter) + "/final"),
			# # # # 			  "STDP_on_bool" : False,
			# # # # 			  "input_stim" : None,
			# # # # 	  		  "output_dir" : "/spikepair_trained_poisson_inputs" + "_drift_iter_" + str(drift_iter)
			# # # # 			  }
			# # # # run_simulation.main_run(stimuli_params, network_params, run_params, seed_iter,
			# # # #spike_pair_differences)



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

			# # # EVALUATE the Poisson trained network on spatio-temporal inputs
			# # run_params = {"weight_file" : "weights/" + str(seed_iter) + "/during_poisson_training_drift_iter_NA/" + "final",
			# # 			  "STDP_on_bool" : False,
			# # 			  "input_stim" : [eval_spike_IDs, eval_spike_times],
			# # 	  		  "output_dir" : "/poisson_trained_spikepair_inputs" + "_drift_iter_" + str(drift_iter)
			# # 			  }
			# # run_simulation.main_run(stimuli_params, network_params, run_params, seed_iter)



			# # GENERATE the evaluation spikes for the *LINEAR CLASSIFIER*, iterating (in blocks) through the possible translations
		 #    # eval_bool determines how translations are sampled (i.e. in blocks as opposed to in a random order)
			# print("\nGenerating spatio-temporal patterns for *classifier evaluation*")
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




			# GENERATE spikes that alternate stimuli and noise
			alternating_spike_IDs, alternating_spike_times, spike_pair_differences = generate_spikes_fixed_pairs(stimuli_params, assembly_IDs,
			    relative_times_vertical, relative_times_horizontal, seed_iter)
			
			plot_input_raster(stimuli_params, assembly_IDs, alternating_spike_IDs, alternating_spike_times,
				neuron_drift_coefs_dic, seed_iter, eval_bool=False, input_name="alternating")

			#print("====TEMPORARILY INITIALIZING WEIGHTS HERE")

			# TRAIN the network on spatiotemporally structured inputs
			run_params = {"weight_file" : ("weights/" + str(seed_iter) + "/during_spikepair_training"
										   + "_drift_iter_" + str(drift_iter) + "/final"),
						  "STDP_on_bool" : False,
						  "input_stim" : [alternating_spike_IDs, alternating_spike_times],
				  		  "output_dir" : "/spikepair_trained_alternating_inputs" + "_drift_iter_" + str(drift_iter)
						  }
			run_simulation.main_run(stimuli_params, network_params, run_params, seed_iter,
				spike_pair_differences)







			# # GENERATE the NONSENSE stimuli inputs
		 #    # eval_bool determines how translations are sampled (i.e. in blocks as opposed to in a random order)
			# print("\nGenerating spatio-temporal patterns for *nonsense stimulus evaluation*")
			# nonsense_eval_spike_IDs, nonsense_eval_spike_times, spike_pair_differences = generate_spikes_fixed_pairs(stimuli_params, assembly_IDs,
			#     relative_times_vertical, relative_times_horizontal, seed_iter, eval_bool=True, nonsense_cross_bool=True)

			# plot_input_raster(stimuli_params, assembly_IDs, nonsense_eval_spike_IDs, nonsense_eval_spike_times,
			# 	neuron_drift_coefs_dic, seed_iter, eval_bool=True, input_name="nonsense_evaluation_spikepairs")


			# # EVALUATE the spatio-temporally trained network on the nonsense spatio-temporal inputs
			# run_params = {"weight_file" : ("weights/" + str(seed_iter) + "/during_spikepair_training"
			# 							   + "_drift_iter_" + str(drift_iter) + "/final"),
			# 			  "STDP_on_bool" : False,
			# 			  "input_stim" : [nonsense_eval_spike_IDs, nonsense_eval_spike_times],
			# 	  		  "output_dir" : "/spikepair_trained_spikepair_inputs_nonsense" + "_drift_iter_" + str(drift_iter)
			# 			  }
			# run_simulation.main_run(stimuli_params, network_params, run_params, seed_iter,
			# 	spike_pair_differences)
