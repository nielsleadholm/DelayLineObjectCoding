from brian2 import *
import math
import numpy as np
import os
from analyse_sim_results import main_fr_analysis
import random
import yaml
import run_simulation
import train_and_eval
from generate_input_stims import (create_underlying_spike_assemblies, plot_input_raster,
								  generate_spikes_fixed_pairs, visualize_spike_slopes)


# Hyper-parameters specific to this script; NB only one seed is ever used
num_binary_search_steps = 10
seed_iter = 0
excluding_indices = [2, 4, 5, 7, 8, 9, 14, 16, 17, 19] # None # [2]


# def check_mean_rates_spread(target_rate, mean_rates):

# 	print("Mean rates by individual neuron:")
# 	print(mean_rates)

# 	if np.all(mean_rates > target_rate) or np.all(mean_rates < target_rate):
# 		pass

# 	else:
# 		print("Rates within network straddle the target firing rate; search likely at best weight value atainable.")
# 		exit()

def input_setup(stimuli_params, drift_iter):

	print("Generating the input stimuli")

	[train_and_eval.make_directories(dir_name, seed_iter, sub_dir_list=["input_stimuli"]
		 , drift_iter="NA") for dir_name in ["weights", "figures", "raw_data"]]

	# Generate the underlying spike-timing slopes that will form the basis of all the
	# input stimuli
	assembly_IDs, relative_times_vertical, relative_times_horizontal, neuron_drift_coefs_dic = create_underlying_spike_assemblies(stimuli_params,
			drift_iter, seed_iter)

    # Plot spike slopes
	visualize_spike_slopes(1, stimuli_params, relative_times_vertical,
		relative_times_horizontal, neuron_drift_coefs_dic, seed_iter)

	input_spike_IDs, input_spike_times, spike_pair_differences = generate_spikes_fixed_pairs(stimuli_params, assembly_IDs,
	    relative_times_vertical, relative_times_horizontal, seed_iter, eval_bool=True)
	
	plot_input_raster(stimuli_params, assembly_IDs, input_spike_IDs,
					  input_spike_times, neuron_drift_coefs_dic, seed_iter,
					  eval_bool=True, input_name="binary_search")


	return input_spike_IDs, input_spike_times, spike_pair_differences

def eval_network_info(stimuli_params, network_params, drift_iter_val,
					  input_spike_IDs, input_spike_times, current_wmax,
					  spike_pair_differences):

	# Re-set seed for both Brian and Numpy to ensure e.g. background inputs comparable across simulations
	seed(seed_iter)
	random.seed(seed_iter)

	# Set the value for w-max to be the desired one
	network_params["wmax"] = current_wmax

	print("\nUsing wmax:")
	print(network_params["wmax"])
	# exit()

	# Uses a slightly hacky naming system to repurpose the 'drift_iter' naming of files/directories
	drift_iter = str(drift_iter_val) + "_WMAX_" + str(current_wmax)

	[train_and_eval.make_directories(dir_name, seed_iter, sub_dir_list=["untrained_spikepair_inputs"]
			 , drift_iter=drift_iter) for dir_name in ["weights", "figures", "raw_data"]]


	# EVALUATE the network on the spatio-temporal input before training
	run_params = {"weight_file" : "weights/" + str(seed_iter) + "/rand_weights.txt",
				  "STDP_on_bool" : False,
				  "input_stim" : [input_spike_IDs, input_spike_times],
		  		  "output_dir" : "/untrained_spikepair_inputs" + "_drift_iter_" + str(drift_iter)
				  }
	run_simulation.main_run(stimuli_params, network_params, run_params, seed_iter,
		spike_pair_differences,
		initialize_weights_bool=True) # Note that we always initialize the weights, because
	 # these need to reflect wmax

	info_fr, info_binary, mean_rates = main_fr_analysis(stimuli_params,
			"untrained_spikepair_inputs", drift_iter, seed_iter, excluding_indices, save_fig=False)


	#info_fr, info_binary, mean_rates = np.asarray([0.8, 0.2, 0.3]), np.asarray([1.0, 0.7, 0.4]), np.asarray([2.1, 2.3, 2.4])

	# print("Iter results:")
	# print(info_fr)
	# print(info_binary)
	# print(mean_rates)

	# print("Output:")

	# print(info_mean)

	# if info_mean == np.nan:
	# 	print("Info is np nan")
	# 	info_mean = 0
	# print(info_mean)

	# if info_mean == math.nan:
	# 	print("Info is math nan")
	# 	info_mean = 0
	# print(info_mean)

	# Handle nan values of information when e.g. firing rates are all 0
	info_mean = info_binary.mean()

	if not (info_mean >= 0):
		info_mean = 0

	return info_mean, mean_rates.mean(), mean_rates


if __name__ == '__main__':

	with open('config_TranslationInvariance.yaml') as f:
		params = yaml.load(f, Loader=yaml.FullLoader)

	stimuli_params = params['stimuli_params']
	network_params = params['network_params']

	print("\nPerforming a binary search with number of steps : " + str(num_binary_search_steps))


	# if os.path.exists("binary_weight_search/") == 0:
	# 	try:
	# 		os.mkdir("binary_weight_search/")
	# 	except OSError:
	# 		pass

	# Track results from the search, including the two initial values, for platting
	search_iter_list = [0,0]
	search_weight_list = []
	search_FR_list = []
	search_info_list = []


	assert len(stimuli_params["drift_coef_list"]) == 1, "Only a single drift-value should be used during binary search"
	drift_iter_val = stimuli_params["drift_coef_list"][0]


	# Set seed for both Brian and Numpy
	print("\n\n==NEW SEED== : " + str(seed_iter))
	seed(seed_iter)
	random.seed(seed_iter)


	# Determine appropriate target firing rate for the search
	# Should divide by the number of stimuli
	target_rate = (1/(stimuli_params["duration_of_presentations"]/1000)) / len(stimuli_params["stimuli_names"])
	print("\nTarget firing rate for search: " + str(target_rate))

	w_lower = network_params["wmax_search_vals"][0]
	w_upper = network_params["wmax_search_vals"][1]
	best_info_val = 0  # Track the most amount of information that has been recovered

	print("\nInitial max weights:")
	print("w_lower: " + str(w_lower))
	print("w_upper: " + str(w_upper))
	search_weight_list.append(w_lower)
	search_weight_list.append(w_upper)


	input_spike_IDs, input_spike_times, spike_pair_differences = input_setup(stimuli_params, drift_iter_val)


	# Determine initial firing rates and information quantities
	low_info_result, low_fr_result, mean_rates = eval_network_info(stimuli_params, network_params, drift_iter_val,
			input_spike_IDs, input_spike_times,
			current_wmax=w_lower,
			spike_pair_differences=spike_pair_differences)
	print("\nStarting lower value results...")
	print("Information : " + str(low_info_result))
	print("Firing rate : " + str(low_fr_result))
	search_info_list.append(low_info_result)
	search_FR_list.append(low_fr_result)

	#check_mean_rates_spread(target_rate, mean_rates)


	high_info_result, high_fr_result, mean_rates = eval_network_info(stimuli_params, network_params, drift_iter_val,
			input_spike_IDs, input_spike_times,
			current_wmax=w_upper,
			spike_pair_differences=spike_pair_differences)
	print("Starting higher value results...")
	print("Information : " + str(high_info_result))
	print("Firing rate : " + str(high_fr_result))
	#check_mean_rates_spread(target_rate, mean_rates)
	search_info_list.append(high_info_result)
	search_FR_list.append(high_fr_result)

	# Make sure initial range of wmax will be condusive to a useful binary search
	assert not ((low_fr_result < target_rate) and (high_fr_result < target_rate) or
		(low_fr_result > target_rate) and (high_fr_result > target_rate)), "Initial wmax range does not result in spanning target rate"




	# If no information in either situation, then use mean firing rates to guide next weight value
	if (low_info_result == 0) and (high_info_result == 0):
		use_fr_for_search_bool = True
		print("Using firing rate to guide search...")
	else:
		use_fr_for_search_bool = False
		print("Using information values to guide search...")

	for search_iter in range(num_binary_search_steps):

		print("\nOn search iter : " + str(search_iter))
		wmax_iter = (w_upper + w_lower)/2
		print("Using max weight : " + str(wmax_iter))

		search_iter_list.append(search_iter+1) # Account for the first step being done outside the loop
		search_weight_list.append(wmax_iter)

		new_info_result, new_fr_result, mean_rates = eval_network_info(stimuli_params, network_params, drift_iter_val,
				input_spike_IDs, input_spike_times,
				current_wmax=wmax_iter,
				spike_pair_differences=spike_pair_differences)

		print("Information theory result: " + str(new_info_result))

		search_info_list.append(new_info_result)
		search_FR_list.append(new_fr_result)

		#check_mean_rates_spread(target_rate, mean_rates)

		# new_info_result = 0
		# new_fr_result = fake_fr_list[search_iter]
		
		# if (use_fr_for_search_bool == False) or (new_info_result > 0):
		# 	# Some information available at some level, therefore stop using firing rate
		# 	# to guide search
		# 	use_fr_for_search_bool = False
		# 	print("**Using information values to guide binary search**")

		# 	if new_info_result == 1:
		# 		print("! Achieved perfect information encoding with weight : " + str(wmax_iter))
		# 		exit()
		# 	elif new_info_result > best_info_val:
		# 		print("Amount of information improved:")
		# 		print(new_info_result)
		# 		w_upper = wmax_iter
		# 		best_info_val = new_info_result
		# 	elif new_info_result < best_info_val:
		# 		print("Less information available with latest weight:")
		# 		print(new_info_result)
		# 		w_lower = wmax_iter
		# 	else:
		# 		print("Information value not changed, reverting to firing rate to guide search:")
		# 		print(new_info_result)
		# 		use_fr_for_search_bool = True

		# # Use firing rate to search
		# if (use_fr_for_search_bool == True) and (new_info_result == 0):

		print("On-going target rate:")
		print(target_rate)

		if new_fr_result > target_rate:
			print("Firing rate above target:")
			print(new_fr_result)
			w_upper = wmax_iter
		elif new_fr_result < target_rate:
			print("Firing rate below target:")
			print(new_fr_result)
			w_lower = wmax_iter
		else:
			print("Firing rate of " + str(new_fr_result) + " exactly matches target with weight : " + str(wmax_iter))
			#print("Unfortunately, still no information at this firing rate.")
			break


	print("\nAfter " + str(search_iter) + " iterations, final weight value: " + str(wmax_iter))
	print("Final FR: " + str(new_fr_result))
	print("Final information value: " + str(new_info_result))

	np.savetxt("raw_data/search_iters.txt", search_iter_list)
	np.savetxt("raw_data/search_weights.txt", search_weight_list)
	np.savetxt("raw_data/search_info_vals.txt", search_info_list)
	np.savetxt("raw_data/search_FR_vals.txt", search_FR_list) 


	fig, axis_list = plt.subplots(1, 2)
	info_vis_scaling = 30  # Ensure the size of the info dots are reasonable
	fr_vis_scaling = 10

	# Plot the search results
	for search_iter in range(len(search_iter_list)):
		axis_list[0].scatter(search_iter_list[search_iter], search_weight_list[search_iter],
							 s=(search_info_list[search_iter]+0.1)*info_vis_scaling, color='k', alpha=0.5)
		axis_list[0].annotate(str(round(search_info_list[search_iter],2)), (search_iter_list[search_iter], search_weight_list[search_iter]))
		axis_list[1].scatter(search_iter_list[search_iter], search_weight_list[search_iter],
							 s=(search_FR_list[search_iter]+0.1)*fr_vis_scaling, color='k', alpha=0.5)
		axis_list[1].annotate(str(round(search_FR_list[search_iter],2)), (search_iter_list[search_iter], search_weight_list[search_iter]))


	for ax in axis_list:
		ax.set(xlabel='Search Iteration')
		ax.set(xlim=(0, len(search_iter_list)+1))

	axis_list[0].set(ylabel='Maximum Weight')
	axis_list[0].set_title("Information")
	axis_list[1].set_title("Firing Rates")


	savefig("figures/" + str(seed_iter) + "binary_search_results", dpi=300)
	clf()

	# CONSIDER IMPLEMENTING If any information by the final num_binary_search_steps iter, do an additional
	# 5 steps of search; otherwise abandon it

