from brian2 import *
import copy
import json
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy

def exclude_indices(input_array, excluding_indices):
	
	input_array = np.ma.array(input_array, mask=False)
	input_array.mask[excluding_indices] = True
	input_array = input_array.compressed()
	
	return input_array


def isi_analysis(params, data_set, drift_iter, seed_iter, layer):

	print("\nISI, CV and Fano-Factor analysis for layer: " + str(layer))
	data_dic = dict()
	data_dic["times"] = np.fromfile("raw_data/" + str(seed_iter) + "/"
									+ data_set  + "_drift_iter_" + str(drift_iter) + "/raster_"
									+ layer + "_layer_times.txt", dtype=np.float64, sep='\n')
	data_dic["ids"] = np.fromfile("raw_data/" + str(seed_iter) + "/"
									+ data_set  + "_drift_iter_" + str(drift_iter) + "/raster_"
									+ layer + "_layer_IDs.txt", sep='\n')

	all_isi = []  # Data combined from all neurons
	cv_of_isi = []  # Data represented for each individual neuron

	# Iterate through each neuron, masking the spike times, and then determining the ISIs
	# and the CV of the ISI
	for ID_iter in range(0, params[layer + '_layer_size']):

		ID_mask = data_dic["ids"]==ID_iter

		current_times = data_dic["times"][ID_mask]

		# Take the temporal difference between two adjacent spikes
		current_isi = current_times[1:] - current_times[:-1]

		all_isi.extend(current_isi)

		# Track the CV of the ISI for each individual neuron
		# NB CV is calculated using the standard deviation, while the Fano Factor
		# is calculated using the variance
		cv_of_isi.append(np.std(current_isi)/(np.mean(current_isi)+0.00001))


	# Plot the ISI distribution across all neurons
	plt.hist(all_isi)
	plt.xlabel("Interspike Interval (ms)")
	plt.xlim(0)
	plt.title("ISI Distribution in Layer : " + layer)
	plt.savefig("analysis_results/isi_distribution_" + layer + "_" + data_set + ".png", dpi=300)
	plt.clf()

	# Plot the distribution of CV of the ISI across the network
	plt.hist(cv_of_isi)
	plt.xlabel("CV of the Interspike Interval")
	plt.xlim(0,2)
	plt.title("CV of the ISI in Layer : " + layer)
	plt.savefig("analysis_results/cv_of_isi_distribution_" + layer + "_" + data_set + ".png", dpi=300)
	plt.clf()

	fano_factor_layer = calculate_fano_factor(params, data_dic, layer)

	# Plot the distribution of Fano Factor across the network
	plt.hist(cv_of_isi)
	plt.xlabel("Fano Factor")
	plt.xlim(0,2)
	plt.title("Distribution of Fano Factor in Layer : " + layer)
	plt.savefig("analysis_results/fano_factor_" + layer + "_" + data_set + ".png", dpi=300)
	plt.clf()

def main_fr_analysis(params, data_set, drift_iter, seed_iter, excluding_indices=None, save_fig=True, layer="output"):
	'''
	Visualize the firing rates of neurons from a simulation, as well as to evaluate the
	information content in the firing rates of the neurons

	excluding_indices (None or list) : if list, exclude these indices when calculating 
		mean firing rate; this can be useful e.g. when performing a binary search if it is
		known that some neurons do not have delay lines that align with any of the inputs
	'''
	data_dic = dict()
	data_dic["times"] = np.fromfile("raw_data/" + str(seed_iter) + "/"
									+ data_set  + "_drift_iter_" + str(drift_iter) + "/raster_" + layer + "_layer_times.txt", dtype=np.float64, sep='\n')
	data_dic["ids"] = np.fromfile("raw_data/" + str(seed_iter) + "/"
									+ data_set  + "_drift_iter_" + str(drift_iter) + "/raster_" + layer + "_layer_IDs.txt", sep='\n')


	# == INFORMATION THEORY analysis ==

	FR_array, binary_activation_array = extract_firing_rates(params, data_dic, layer)

	# Information theory based on whether a neuron fires or not during presentation
	information_theory_results_binary_1st = binary_information_theory_calculation(params, binary_activation_array, stimulus_info_calc=0)
	information_theory_results_binary_2nd = binary_information_theory_calculation(params, binary_activation_array, stimulus_info_calc=1)
	UnitTest_binary_information_theory_calculation(params, binary_activation_array, layer)

	# Information theory based on a neurons firing rate
	(mid_threshold) = information_theory_discretize(params, FR_array, layer)
	information_theory_counts = information_theory_counting(params, FR_array, mid_threshold, layer)
	information_theory_results_fr = fr_information_theory_calculation(params, information_theory_counts)
	UnitTest_FR_information_theory_calculation(params, information_theory_counts)


	# == RAW FIRING RATE analysis ==

	mean_FR_list = []

	# Iterate through the stimuli to extract mean firing rates
	for stimuli_iter in range(len(params["stimuli_names"])):

		mean_FR_list.append(find_mean_firing_rates(params, FR_array[stimuli_iter], layer))

	# Take the difference in the mean firing rates
	mean_FR_difference = mean_FR_list[1] - mean_FR_list[0]

	# Sort FR by the difference in firing rates
	sorting_indices = np.argsort(mean_FR_difference)
	mean_FR_difference = np.take_along_axis(mean_FR_difference, sorting_indices, axis=0)
	stim1_FR = np.take_along_axis(mean_FR_list[1], sorting_indices, axis=0)
	stim2_FR = np.take_along_axis(mean_FR_list[0], sorting_indices, axis=0)

	# Mean firing rates regardless of the stimulus
	mean_rate_across_stim = np.mean(np.asarray(mean_FR_list), axis=0)


	# == PLOT firing rate results ==

	#Plot the results
	plt.bar(np.arange(0,(params[layer + '_layer_size'])), height=stim1_FR+stim2_FR, bottom=-stim2_FR, alpha=0.5, color="dodgerblue")
	plt.scatter(np.arange(0,(params[layer + '_layer_size'])), mean_FR_difference, marker='x', color='k')
	plt.plot(np.arange(0,(params[layer + '_layer_size'])),np.zeros((params[layer + '_layer_size'])),linestyle='dashed', color='k') # Plot a horizontal line to indicate 0
	plt.ylabel("Difference in Firing Rate (Hz)")
	plt.xlabel("Neurons, Sorted by Firing Rate Difference")
	plt.title(data_set + " : Difference in Firing Rates Across Stimuli, Layer " + layer)
	plt.ylim(-15, 15)
	if save_fig:
		plt.savefig("analysis_results/difference_in_FR_rates_" + layer + "_" + data_set + ".png", dpi=300)
	plt.clf()


	# == MASK any required neurons known not to carry information (i.e. in hand-crafted networks) ==

	if excluding_indices is not None:

		mean_rate_across_stim = exclude_indices(mean_rate_across_stim, excluding_indices)
		information_theory_results_fr = exclude_indices(information_theory_results_fr, excluding_indices)
		information_theory_results_binary = exclude_indices(information_theory_results_binary, excluding_indices)

	return (information_theory_results_fr, information_theory_results_binary_1st,
			information_theory_results_binary_2nd, mean_rate_across_stim)

def calculate_fano_factor(params, data_dic, layer):

	no_math_error = 0.000000000001

	duration_of_simulation = (stimuli_params['number_of_eval_presentations']
			 				   * stimuli_params['duration_of_presentations']
			 				   * len(stimuli_params['stimuli_names']))

	# Determine the upper limit of simulation time used for calculating the Fano Factor in 1 second bins
	fano_duration = (duration_of_simulation // 1000) * 1000 

	print("\nCalculating Fano Factor in 1 second bins over a simulation period of seconds: " + str(fano_duration))

	assert (fano_duration % 1000) == 0, "Duration is " + str(fano_duration) + " but is not a multiple of 1000ms"

	fano_factor = []

	# Iterate through the neurons
	for ID_iter in range(params[layer + '_layer_size']):

		ID_window_rates = []

		# Iterate through the temporal windows 
		for presentation_iter in range(0, fano_duration, 1000):

			mask = np.nonzero((data_dic["times"]>=presentation_iter) & (data_dic["times"]<(presentation_iter+1000)))

			# NB that because we're looking at 1000msec windows, the number of spikes is equivalent to the rate in Hz
			ID_window_rates.append(np.sum(data_dic["ids"][mask]==ID_iter))

		fano_factor.append(np.var(ID_window_rates)/(np.mean(ID_window_rates)+no_math_error))

	return fano_factor



def extract_firing_rates(params, data_dic, layer):

	#Initialize a vector to hold the firing rates of each neuron
	FR_array = np.zeros([len(params["stimuli_names"]), params[layer + '_layer_size'], params['number_of_eval_presentations']])
	binary_activation_array = np.zeros([len(params["stimuli_names"]), params[layer + '_layer_size']])
	fano_factor = np.zeros([len(params["stimuli_names"]), params[layer + '_layer_size']])

	for presentation_iter in range(0,params['number_of_eval_presentations']):

		# print("\nPresentation iter:")
		# print(presentation_iter)

		for stimuli_iter in range(len(params["stimuli_names"])):

			# print("Stimuli iter:")
			# print(stimuli_iter)
			# print("Lower bound:")
			# print(((len(params["stimuli_names"])*presentation_iter+stimuli_iter)*params['duration_of_presentations']))
			# print("Upper bound:")
			# print(((len(params["stimuli_names"])*presentation_iter+stimuli_iter+1)*params['duration_of_presentations']))

			#Apply a mask to the times data to extract spikes in the period of interest
			mask = ((data_dic["times"] > ((len(params["stimuli_names"])*presentation_iter+stimuli_iter)*params['duration_of_presentations'])) & 
				(data_dic["times"] <= ((len(params["stimuli_names"])*presentation_iter+stimuli_iter+1)*params['duration_of_presentations'])))

			#Iterate through each neuron ID, counting the total number of appearances in the masked-array
			for ID_iter in range(0, params[layer + '_layer_size']):

				FR_array[stimuli_iter][ID_iter][presentation_iter] = np.sum(data_dic["ids"][mask]==ID_iter)

			#Divide these values by the duration of the presentation, in seconds
			FR_array[stimuli_iter,:,presentation_iter] = FR_array[stimuli_iter,:,presentation_iter] / (params['duration_of_presentations']/1000)

	
	# print("\nNew extraction")

	for stimuli_iter in range(len(params["stimuli_names"])):

		# print("Stimulus specific array")
		# print(FR_array[stimuli_iter,:,:])
		# print(np.shape(FR_array[stimuli_iter,:,:]))
		# print(np.shape(np.sum(FR_array[stimuli_iter,:,:]>0,axis=1)))
		binary_activation_array[stimuli_iter,:] = np.sum(FR_array[stimuli_iter,:,:]>0,axis=1)

	# print("Final arrays")
	# print(FR_array)
	# print(binary_activation_array)

	return FR_array, binary_activation_array


def find_mean_firing_rates(params, FR_array, layer):
	mean_FR = np.zeros(params[layer + '_layer_size'])

	mean_FR = np.sum(FR_array, axis = 1)
	mean_FR = mean_FR / params['number_of_eval_presentations']

	return mean_FR


#Find the firing rate thresholds that determine if a firing rate is low, medium or high
def information_theory_discretize(params, FR_array, layer):
	#Note that as used in the Hutter thesis (2018), each neuron has its own thresholds
	#These are based on the minimal and maximal firing rate obtained across all presentations, the difference of which is divided into three equal bins

	#Vector of minimum firing rates for each neuron (across presentations of all stimuli)
	#Minimum is first taken for each particular stimulus (and so iterating through them), and then across all stimuli
	temp_min_array = np.zeros([params[layer + '_layer_size'], len(params["stimuli_names"])])
	for stimuli_iter in range(0, len(params["stimuli_names"])):
		temp_min_array[:, stimuli_iter] = np.amin(FR_array[stimuli_iter], axis=1)
	min_vector = np.amin(temp_min_array, axis = 1)
	# print("Minimum firing rates:")
	# print(min_vector)

	#Vector of maximum firing rates for each neuron (across presentations of all stimuli)
	temp_max_array = np.zeros([params[layer + '_layer_size'], len(params["stimuli_names"])])
	for stimuli_iter in range(0, len(params["stimuli_names"])):
		temp_max_array[:, stimuli_iter] = np.amax(FR_array[stimuli_iter], axis=1)
	max_vector = np.amax(temp_max_array, axis = 1)
	# print("Maximum firing rates:")
	# print(max_vector)

	#Generate the vector containing the thresholds for separating low-medium and medium-high for each neuron
	mid_threshold = (max_vector - min_vector)*(0.5)
	# upper_threshold = (max_vector - min_vector)*(2/3)

	# print("Lower and upper thresholds")
	# print(lower_threshold)
	# print(upper_threshold)

	return (mid_threshold)


def information_theory_counting(params, FR_array, mid_threshold, layer):
	#Information can be encoded in firing rates by discretizing the rates into e.g. low, medium, and high rates, which will be done here

	information_theory_dic = dict()	
	#For each stimulus, find the number of times that a particular neuron's firing rate was low, medium, or high
	for stimuli_iter in range(0, len(params["stimuli_names"])):
		firing_rate_counter = np.zeros([params[layer + '_layer_size'], 2]) #Array to store these counts

		#Apply a mask such that all firing rates relative to a particula threshold return a 1, then sum their values
		firing_rate_counter[:, 0] = np.sum(FR_array[stimuli_iter]<=mid_threshold[:, None], axis=1) #lower counts
		firing_rate_counter[:, 1] = np.sum(FR_array[stimuli_iter]>mid_threshold[:, None], axis=1) #upper counts
		# firing_rate_counter[:, 1] = params['number_of_eval_presentations'] - (firing_rate_counter[:, 0] + firing_rate_counter[:, 2]) #mid firing rate counts

		#Check that all of the separate counts sum appropriately
		assert np.all((firing_rate_counter[:, 0]+firing_rate_counter[:, 1]) == (np.ones(params[layer + '_layer_size'])*params['number_of_eval_presentations']))

		information_theory_dic[stimuli_iter] = firing_rate_counter

	#Return an array containing the number of presentations where the neuron activity was low, medium, and high respectively
	return information_theory_dic


def fr_information_theory_calculation(params, information_theory_dic):
	#Information_theory_dic contains an array for each stimulus presentation
	#This array contains the number of counts of low and high firing rates for each neuron
	# NB that in this code-base, the low and high e.g. firing rates are calculated separately (this is because
	# the older code actually had three different possible responses - low, medium or high)
	# For binary_information_theory_calculation, because there were always only two responses, the probability can
	# be found as p(R_b) = 1 - p(R_a)

	no_math_error = 0.000000000001 #Prevent division by zero


	# *** Initially find the result for stimulus 1 presentation
	# print("\nFiring rate info theory")
	# print(information_theory_dic)


	#The conditional probabilities of a particular neuron having low or high activity for a particular stimulus
	conditional_prob_array = information_theory_dic[0]/params['number_of_eval_presentations']
	# print("Conditional")
	# print(np.shape(conditional_prob_array))
	# print(conditional_prob_array[0:3, 0:3])

	#The marginal propabailities of a particular neuron having low or high activity 
	marginal_prob_array = (information_theory_dic[0]+information_theory_dic[1])/(params['number_of_eval_presentations']*len(params["stimuli_names"]))
	# print("Marginal")
	# print(np.shape(marginal_prob_array))
	# print(marginal_prob_array)
	# print(marginal_prob_array[0:3, 0:3])

	information_low = np.multiply(conditional_prob_array[:, 0], np.log2(np.divide(conditional_prob_array[:, 0], marginal_prob_array[:, 0]+no_math_error)+no_math_error))
	#information_mid = np.multiply(conditional_prob_array[:, 1], np.log2(np.divide(conditional_prob_array[:, 1], marginal_prob_array[:, 1]+no_math_error)+no_math_error))
	information_high = np.multiply(conditional_prob_array[:, 1], np.log2(np.divide(conditional_prob_array[:, 1], marginal_prob_array[:, 1]+no_math_error)+no_math_error))
	

	information_theory_results = information_low + information_high
	# print("FR information results")
	# print(information_theory_results)

	assert np.all(information_theory_results>=-(10**(-8))), "Some information theory values are negative!"

	information_theory_results = np.clip(information_theory_results, 0, 1)

	# print("Clipped info results:")
	# print(information_theory_results)


	return information_theory_results


def binary_information_theory_calculation(params, information_theory_data, stimulus_info_calc=0):
	'''
	stimulus_info_calc should be set to 0 or 1, depending on whether information should specifically
	be calculated for the first or second stimulus
	'''

	#Information_theory_data is indexed by [dataset_iter, assembly_iter]; thus the row indicates which stimulus was presented, and the 
	#column value indicates how many presentations were associated with at least one activation of that assembly

	no_math_error = 0.000000000001 #Prevent division by zero

	# print(information_theory_data)

	#The probabilities of a particular assembly being active for each stimulus
	conditional_prob_array = information_theory_data/params['number_of_eval_presentations']
	# print("Conditional")
	# print(np.shape(conditional_prob_array))
	# print(conditional_prob_array)

	# The marginal probability of a particular Response, regardless of the Stimulus
	marginal_prob_array = np.sum(information_theory_data, axis=0)/(params['number_of_eval_presentations']*len(params["stimuli_names"]))
	# print("Marginal")
	# print(np.shape(marginal_prob_array))
	# print(marginal_prob_array)

	# print("\nThe inverse probs (i.e. probability of *no* response)")
	# print("Conditional")
	# print(1-conditional_prob_array)
	# print("Marginal")
	# print(1-marginal_prob_array)

	# Calculate information theory for the first stimulus (i.e. indexed with 0) across all of the neurons
	# print("\nDiv values")
	# print(np.divide(conditional_prob_array[0, :], marginal_prob_array+no_math_error))
	# print(np.divide(1-conditional_prob_array[0, :], (1-marginal_prob_array+no_math_error)))

	# print("\nLog values")
	# print(np.log2(np.divide(conditional_prob_array[0, :], marginal_prob_array+no_math_error)+no_math_error))
	# print(np.log2(np.divide(1-conditional_prob_array[0, :], (1-marginal_prob_array+no_math_error))+no_math_error))

	information1 = np.multiply(conditional_prob_array[stimulus_info_calc, :], np.log2(np.divide(conditional_prob_array[stimulus_info_calc, :], marginal_prob_array+no_math_error)+no_math_error))
	# By taking (1-p) where p is the above probabilities, we look at the probabilities for the alternative response (i.e. no spike)
	information2 = np.multiply(1-conditional_prob_array[stimulus_info_calc, :], np.log2(np.divide(1-conditional_prob_array[stimulus_info_calc, :], (1-marginal_prob_array+no_math_error))+no_math_error))

	information_theory_results = information1+information2
	# print("\nFinal info results")
	# print(information_theory_results)

	# print("\nRounded info results")
	# print(np.round(information_theory_results, decimals=8))

	# print("Checking values are greater than : " + str(-(10**(-8))))


	assert np.all(information_theory_results>=-(10**(-8))), "Some information theory values are negative!"

	information_theory_results = np.clip(information_theory_results, 0, 1)

	# print("Clipped info results:")
	# print(information_theory_results)


	return information_theory_results


# Test information theory calculation by analysing idealised data
def UnitTest_FR_information_theory_calculation(params, information_theory_dic):
		temp_information_theory_data = information_theory_dic  # Copy of information theory data

		# print(np.shape(information_theory_dic))
		# print(information_theory_dic)
		# print(np.shape(temp_information_theory_data[0][:,0]))
		# # exit()

		# Set every assembly (second dimension) to have a high firing rate for stimulus 1 only
		temp_information_theory_data[0][:,0] = 0
		temp_information_theory_data[0][:,1] = params['number_of_eval_presentations']
		temp_information_theory_data[1][:,0] = params['number_of_eval_presentations']
		temp_information_theory_data[1][:,1] = 0
		#print(temp_information_theory_data)
		temp_information_theory_results = fr_information_theory_calculation(params, temp_information_theory_data)
		#print(temp_information_theory_results)
		assert np.sum(temp_information_theory_results>=0.80) == len(temp_information_theory_results), "Unit Test Failure: Idealized data does not have perfect information."

		# Set every assembly to be highly active for presentation of stimulus 2 only
		temp_information_theory_data[0][:,0] = params['number_of_eval_presentations']
		temp_information_theory_data[0][:,1] = 0
		temp_information_theory_data[1][:,0] = 0
		temp_information_theory_data[1][:,1] = params['number_of_eval_presentations']
		temp_information_theory_results = fr_information_theory_calculation(params, temp_information_theory_data)
		assert np.sum(temp_information_theory_results>=0.80) == len(temp_information_theory_results), "Unit Test Failure: Idealized data does not have perfect information."

		# Test in a case of no information (activation for either stimulus equally likely)
		temp_information_theory_data[0][:,0] = params['number_of_eval_presentations']/2
		temp_information_theory_data[0][:,1] = params['number_of_eval_presentations']/2
		temp_information_theory_data[1][:,0] = params['number_of_eval_presentations']/2
		temp_information_theory_data[1][:,1] = params['number_of_eval_presentations']/2
		temp_information_theory_results = fr_information_theory_calculation(params, temp_information_theory_data)
		assert np.sum(temp_information_theory_results>=0.80) == 0, "Unit Test Failure: Artificially uninformative data still has information."

		print("Unit tests passed: Firing rate information theory.")

		return None


# Test information theory calculation by analysing idealised data
def UnitTest_binary_information_theory_calculation(params, information_theory_data, layer):
		temp_information_theory_data = np.asarray(information_theory_data) #Copy of information theory data

		#Set every neuron (second dimension) to be active for every presentation of stimulus 1 (first dimension) only
		temp_information_theory_data[0, :] = params['number_of_eval_presentations']
		temp_information_theory_data[1, :] = 0
		temp_information_theory_results = binary_information_theory_calculation(params, temp_information_theory_data)
		assert np.sum(temp_information_theory_results>=0.80) == len(temp_information_theory_results), "Unit Test Failure: Idealized data does not have perfect information."

		#Set half the neurons to always be active for stimulus 1, and the other half for stimulus 2
		temp_information_theory_data[0, :int(params[layer + "_layer_size"]/2)] = params['number_of_eval_presentations']
		temp_information_theory_data[0, int(params[layer + "_layer_size"]/2):] = 0
		temp_information_theory_data[1, :int(params[layer + "_layer_size"]/2)] = 0
		temp_information_theory_data[1, int(params[layer + "_layer_size"]/2):] = params['number_of_eval_presentations']
		temp_information_theory_results = binary_information_theory_calculation(params, temp_information_theory_data)
		assert np.sum(temp_information_theory_results>=0.80) == len(temp_information_theory_results), "Unit Test Failure: Idealized data does not have perfect information."


		#Set every neuron to be active for presentation of stimulus 2 only
		temp_information_theory_data[0, :] = 0
		temp_information_theory_data[1, :] = params['number_of_eval_presentations']
		temp_information_theory_results = binary_information_theory_calculation(params, temp_information_theory_data)
		assert np.sum(temp_information_theory_results>=0.80) == len(temp_information_theory_results), "Unit Test Failure: Idealized data does not have perfect information."

		# Test in a case of no information (activation for either stimulus equally likely)
		temp_information_theory_data[0, :] = params['number_of_eval_presentations']/2
		temp_information_theory_data[1, :] = params['number_of_eval_presentations']/2
		temp_information_theory_results = binary_information_theory_calculation(params, temp_information_theory_data)
		assert np.sum(temp_information_theory_results>=0.80) == 0, "Unit Test Failure: Artificially uninformative data still has information."

		# Test in a case of no information (activate to every presentation)
		temp_information_theory_data[0, :] = params['number_of_eval_presentations']
		temp_information_theory_data[1, :] = params['number_of_eval_presentations']
		temp_information_theory_results = binary_information_theory_calculation(params, temp_information_theory_data)
		assert np.sum(temp_information_theory_results>=0.80) == 0, "Unit Test Failure: Artificially uninformative data still has information."

		print("Unit tests passed: Binary activity information theory.")

		return None


def perform_primary_analyses(stimuli_params, layer):


	# List of evaluation conditions to look at data from (excludes data collected during training)
	eval_data_list_with_drift =  [
		"untrained_spikepair_inputs",
		#"untrained_alternating_inputs",
		"spikepair_trained_spikepair_inputs",
		#"spikepair_trained_alternating_inputs"
		]

	information_theory_fr_dic = {}
	information_theory_binary_dic_1st = {}
	information_theory_binary_dic_2nd = {}
	mean_rate_dic = {}

	for data_set in eval_data_list_with_drift:

		information_theory_fr_dic[data_set] = []
		mean_rate_dic[data_set] = []
		information_theory_binary_dic_1st[data_set] = []
		information_theory_binary_dic_2nd[data_set] = []

		for seed_iter in stimuli_params["seeds_list"]:

			for drift_iter in stimuli_params["drift_coef_list"]:

				# print("\n=====Temporarily setting drift iter=====")
				# drift_iter = "30_WMAX_1.8"

				info_fr_temp, info_binary_temp_1st, info_binary_temp_2nd, rate_temp = main_fr_analysis(stimuli_params, data_set,
						drift_iter, seed_iter, layer=layer)
				information_theory_fr_dic[data_set].append(info_fr_temp)
				information_theory_binary_dic_1st[data_set].append(info_binary_temp_1st)
				information_theory_binary_dic_2nd[data_set].append(info_binary_temp_2nd)
				mean_rate_dic[data_set].append(rate_temp)

				isi_analysis(stimuli_params, data_set, drift_iter, seed_iter, layer)


	# ==== PLOT RESULTS ====

	# Information theory for binary values
	for key, val in information_theory_binary_dic_1st.items():
		# Histogram
		plt.hist(val, bins=np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]), alpha=0.5, label=key)
		plt.legend()
		plt.xlabel("Information (bits)")
		plt.ylim(0, stimuli_params["output_layer_size"])
		plt.title("Information in Binary Activity - 1st stimulus")
		plt.savefig("analysis_results/hist_info_1_binary_" + layer + "_" + key + ".png", dpi=300)
		plt.clf()

		# Display indices associated with the most information
		print("\nOn 1st object for : " + key)
		ind = np.argpartition(val[0], -5)[-5:]
		top_vals = val[0][ind]
		sorting_top = np.argsort(val[0][ind])
		print("Top neuron IDs:")
		print(np.flip(ind[sorting_top]))
		print("Associated info:")
		print(np.flip(top_vals[sorting_top]))

		# Ranked information
		# Sorts the array (ascending) and then reverses the order
		plt.plot(np.arange(stimuli_params["output_layer_size"]), np.flip(np.sort(val))[0,:],
					color="dodgerblue", alpha=0.5, label=key)
		plt.legend()
		plt.xlabel("Cell Rank")
		plt.ylabel("Information (bits)")
		plt.ylim(0, 1)
		plt.xlim(0, stimuli_params["output_layer_size"])
		plt.title("Information in Binary Activity - 1st stimulus")
		plt.savefig("analysis_results/ranked_info_1_binary_" + layer + "_" + key + ".png", dpi=300)
		plt.clf()

	for key, val in information_theory_binary_dic_2nd.items():
		# Histogram
		plt.hist(val, bins=np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]), alpha=0.5, label=key)
		plt.legend()
		plt.xlabel("Information (bits)")
		plt.ylim(0, stimuli_params["output_layer_size"])
		plt.title("Information in Binary Activity - 2nd stimulus")
		plt.savefig("analysis_results/hist_info_2_binary_" + layer + "_" + key + ".png", dpi=300)
		plt.clf()

		# Display indices associated with the most information
		print("\nOn 2nd object for : " + key)
		ind = np.argpartition(val[0], -5)[-5:]
		top_vals = val[0][ind]
		sorting_top = np.argsort(val[0][ind])
		print("Top neuron IDs:")
		print(np.flip(ind[sorting_top]))
		print("Associated info:")
		print(np.flip(top_vals[sorting_top]))

		# Ranked information
		# Sorts the array (ascending) and then reverses the order
		plt.plot(np.arange(stimuli_params["output_layer_size"]), np.flip(np.sort(val))[0,:],
					color="dodgerblue", alpha=0.5, label=key)
		plt.legend()
		plt.xlabel("Cell Rank")
		plt.ylabel("Information (bits)")
		plt.ylim(0, 1)
		plt.xlim(0, stimuli_params["output_layer_size"])
		plt.title("Information in Binary Activity - 2nd stimulus")
		plt.savefig("analysis_results/ranked_info_2_binary_" + layer + "_" + key + ".png", dpi=300)
		plt.clf()

	# Mean rates
	for key, val in mean_rate_dic.items():
		plt.hist(val, alpha=0.3, label=key)
	plt.legend()
	plt.xlabel("Mean FR (Hz)")
	plt.title("Distributions of Firing Rates Across Stimuli")
	plt.savefig("analysis_results/mean_FR.png", dpi=300)
	plt.clf()


	# Typical sparsity in a presentation window
	for key, val in mean_rate_dic.items():
		plt.hist(np.minimum(np.asarray(val)*stimuli_params["duration_of_presentations"]/1000,1)[0], 
				 bins=np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]), alpha=0.3, label=key)
	plt.legend()
	plt.xlabel("Average Proportion of Stimulus Windows With a Spike")
	plt.title("Distributions of Sparsity Across Stimuli")
	plt.savefig("analysis_results/temporal_sparsity" + layer + ".png", dpi=300)
	plt.clf()


def dummy_visualise_cellular_properties():
	'''
	Basic provisional code to test loading of saved
	cellular property data such as membrane voltage
	'''

	# # Test loading of membrane variables
	# drift_iter = 32
	# seed_iter = 0
	# fname = (str(seed_iter) + "/untrained_spikepair_inputs"
	# 		 + "_drift_iter_" + str(drift_iter) + "/g_e_output_layer_both_aligned")
	# variable = np.load("raw_data/" + fname + "_values.npy")
	# times = np.load("raw_data/" + fname + "_times.npy")


def weights_across_epochs(stimuli_params, network_params, data_set, seed_iter, drift_iter):
	'''
	Visualization of the distribtuion of all weights across epochs
	'''


	initial_weights = np.loadtxt("weights/" + str(seed_iter) + "/rand_weights.txt")

	learning_weights_list = np.loadtxt("weights/" + str(seed_iter) + "/" + data_set + "_drift_iter_"
						+ str(drift_iter) + "/weights_over_epochs_vals.txt")


	learning_weights_list = np.insert(learning_weights_list, 0, initial_weights, axis=0)

	epoch_markers_str = np.loadtxt("raw_data/" + str(seed_iter) + "/" + data_set + "_drift_iter_"
						+ str(drift_iter) + "/epoch_markers.txt")
	epoch_markers_str = np.insert(epoch_markers_str, 0, 0)

	epoch_markers_int = list(range(stimuli_params["num_intervals_for_weight_saving"]+1))

	df_weights_across_epochs = pd.DataFrame(np.transpose(learning_weights_list))
	df_weights_across_epochs = pd.melt(df_weights_across_epochs, var_name="epoch", value_name="weight")

	sns.violinplot(x="epoch", y="weight", data=df_weights_across_epochs,
				   scale="area", color="dodgerblue", cut=0, inner=None)
	# see more options at https://seaborn.pydata.org/generated/seaborn.violinplot.html
	# including options for "inner"

	xlabel("Duration of Training (sec)")
	xticks(epoch_markers_int, epoch_markers_str)
	xlim(-0.5)
	ylabel("Weights")
	title("Weights across epochs of STDP training")
	savefig("analysis_results/" + str(seed_iter) + "/violins_weights_across_epochs.png", dpi=300)
	clf()


def weights_across_epochs_by_alignment(stimuli_params, network_params, data_set, seed_iter, drift_iter):
	'''
	Distribution of weights across epochs, broken down by whether the input has a delay-line alignment
	'''

	# Load alignment results
	with open("raw_data/" + str(seed_iter) + "/" + data_set + "_drift_iter_"
						+ str(drift_iter) + "/alignment_results.json") as f:
		alignment_results = json.load(f)

	all_aligned = copy.copy(alignment_results["upright_aligned_weights"])
	all_aligned.extend(alignment_results["inverted_aligned_weights"])
	all_aligned.extend(alignment_results["both_aligned_weights"])

	assert ((len(all_aligned) + len(alignment_results["non_aligned_weights"]))
			== stimuli_params["input_layer_size"]*stimuli_params["output_layer_size"]), "Total number of alignment-checked weight indices should equal total number of weights"

	# Load weights across epochs
	initial_weights = np.loadtxt("weights/" + str(seed_iter) + "/rand_weights.txt")

	learning_weights_list = np.loadtxt("weights/" + str(seed_iter) + "/" + data_set + "_drift_iter_"
						+ str(drift_iter) + "/weights_over_epochs_vals.txt")


	learning_weights_list = np.insert(learning_weights_list, 0, initial_weights, axis=0)

	epoch_markers_str = np.loadtxt("raw_data/" + str(seed_iter) + "/" + data_set + "_drift_iter_"
						+ str(drift_iter) + "/epoch_markers.txt")
	epoch_markers_str = np.insert(epoch_markers_str, 0, 0)

	epoch_markers_int = list(range(stimuli_params["num_intervals_for_weight_saving"]+1))


	# Mask the weights for the different conditions
	learning_weights_list_aligned = learning_weights_list[:,all_aligned]
	learning_weights_list_nonaligned = learning_weights_list[:,alignment_results["non_aligned_weights"]]

	# Process results
	df_weights_across_epochs_aligned = pd.DataFrame(np.transpose(learning_weights_list_aligned))
	df_weights_across_epochs_aligned = pd.melt(df_weights_across_epochs_aligned, var_name="epoch", value_name="weight")

	df_weights_across_epochs_nonaligned = pd.DataFrame(np.transpose(learning_weights_list_nonaligned))
	df_weights_across_epochs_nonaligned = pd.melt(df_weights_across_epochs_nonaligned, var_name="epoch", value_name="weight")


	# Plot aligned results
	sns.violinplot(x="epoch", y="weight", data=df_weights_across_epochs_aligned,
				   scale="area", color="dodgerblue", cut=0, inner=None)
	xlabel("Duration of Training (sec)")
	xticks(epoch_markers_int, epoch_markers_str)
	xlim(-0.5)
	ylabel("Weights")
	title("Weights across epochs of STDP training - Aligned")
	savefig("analysis_results/" + str(seed_iter) + "/violins_weights_across_epochs_aligned.png", dpi=300)
	clf()

	# Plot non-aligned results
	sns.violinplot(x="epoch", y="weight", data=df_weights_across_epochs_nonaligned,
				   scale="area", color="crimson", cut=0, inner=None)
	xlabel("Duration of Training (sec)")
	xticks(epoch_markers_int, epoch_markers_str)
	xlim(-0.5)
	ylabel("Weights")
	title("Weights across epochs of STDP training - Non-Aligned")
	savefig("analysis_results/" + str(seed_iter) + "/violins_weights_across_epochs_nonaligned.png", dpi=300)
	clf()


	# For the aligned weights, plot each one before and after training, drawing a line
	# to make it clear how each particular weight evolved
	number_of_weights_to_vis = len(all_aligned)

	for current_weight_iter in range(number_of_weights_to_vis):

		plot([0, 1], [learning_weights_list[0, all_aligned[current_weight_iter]],
					  learning_weights_list[-1, all_aligned[current_weight_iter]]],
		 	 alpha=0.5)
	
	xlabel("Evolution of Individual Weights")
	xticks([0,1], ["Pre-StDP", "Post-STDP"])
	ylabel("Weights")
	title("Tracking Evolution of Specific Weights - Aligned")
	savefig("analysis_results/" + str(seed_iter) + "/specific_weights_evolution_aligned.png", dpi=300)
	clf()

	# As above, but for non-aligned weights; NB that typically not all non-aligned weights are visualized,
	# i.e. instead the same number as there are aligned weights are (in effect randomly) selected
	for current_weight_iter in range(number_of_weights_to_vis):

		plot([0, 1], [learning_weights_list[0, alignment_results["non_aligned_weights"][current_weight_iter]],
					  learning_weights_list[-1, alignment_results["non_aligned_weights"][current_weight_iter]]],
		 	 alpha=0.5)
	
	xlabel("Evolution of Individual Weights")
	xticks([0,1], ["Pre-StDP", "Post-STDP"])
	ylabel("Weights")
	title("Tracking Evolution of Specific Weights - Non-Aligned")
	savefig("analysis_results/" + str(seed_iter) + "/specific_weights_evolution_nonaligned.png", dpi=300)
	clf()

	# Scatter plot for aligned weights --> weight pre and post STDP
	# Looking for how much of a benefit there is to starting with an an initially higher weight
	scatter(learning_weights_list[0, all_aligned],learning_weights_list[-1, all_aligned])
	xlabel("Initial Weight, Aligned Neurons")
	ylabel("Final Weight, Aligned Neurons")
	title("Effect of Initial Weight - Aligned")
	savefig("analysis_results/" + str(seed_iter) + "/effect_of_initial_weight.png", dpi=300)
	clf()


def strong_delays_across_epochs(stimuli_params, network_params, data_set, seed_iter, drift_iter):
	'''
	Visualize the distribution of delay lines associated with strong (wmax/2)
	weights across the epochs of training
	'''

	initial_weights = np.loadtxt("weights/" + str(seed_iter) + "/rand_weights.txt")

	delays = np.loadtxt("weights/" + str(seed_iter) + "/rand_delays.txt")

	learning_weights_list = np.loadtxt("weights/" + str(seed_iter) + "/" + data_set + "_drift_iter_"
						+ str(drift_iter) + "/weights_over_epochs_vals.txt")


	learning_weights_list = np.insert(learning_weights_list, 0, initial_weights, axis=0)


	strong_delays_list = []

	for epoch_iter in range(len(learning_weights_list)):

		mask = learning_weights_list[epoch_iter] >= network_params["wmax"]/2

		# As arrays associated with strong delays will be of different sizes, pad with NaN values
		current_delays = np.empty(np.shape(learning_weights_list[epoch_iter]))
		current_delays[:] = numpy.nan

		current_delays[mask] = delays[mask]

		strong_delays_list.append(current_delays)


	epoch_markers_str = np.loadtxt("raw_data/" + str(seed_iter) + "/" + data_set + "_drift_iter_"
						+ str(drift_iter) + "/epoch_markers.txt")
	epoch_markers_str = np.insert(epoch_markers_str, 0, 0)

	epoch_markers_int = list(range(stimuli_params["num_intervals_for_weight_saving"]+1))

	df_delays_across_epochs = pd.DataFrame(np.transpose(strong_delays_list))
	df_delays_across_epochs = pd.melt(df_delays_across_epochs, var_name="epoch", value_name="delay")

	sns.violinplot(x="epoch", y="delay", data=df_delays_across_epochs,
				   scale="count", color="dodgerblue", cut=0, inner=None)
	# see more options at https://seaborn.pydata.org/generated/seaborn.violinplot.html
	# including options for "inner"

	xlabel("Duration of Training (sec)")
	xticks(epoch_markers_int, epoch_markers_str)
	xlim(-0.5)
	ylabel("Delays (ms)")
	title("Delay with strong weights across epochs of STDP training")
	savefig("analysis_results/" + str(seed_iter) + "/violins_delays_across_epochs.png", dpi=300)
	clf()


def FR_across_epochs(stimuli_params, network_params, data_set, seed_iter, drift_iter):

	# Load firing rate data, including pre-STDP firing rates
	initial_FR = np.loadtxt("raw_data/" + str(seed_iter) + "/untrained_alternating_inputs_drift_iter_"
		+ str(drift_iter) + "/fr_output_layer.txt")

	learning_FR_list = np.loadtxt("raw_data/" + str(seed_iter) + "/" + data_set + "_drift_iter_"
						+ str(drift_iter) + "/rates_over_epochs_vals.txt")

	learning_FR_list = np.insert(learning_FR_list, 0, initial_FR, axis=0)

	# Load epoch markers
	epoch_markers_str = np.loadtxt("raw_data/" + str(seed_iter) + "/" + data_set + "_drift_iter_"
						+ str(drift_iter) + "/epoch_markers.txt")
	epoch_markers_str = np.insert(epoch_markers_str, 0, 0)

	# Load alignment results; this will enable visualising the change in firing rates
	# the the correspondence (if any) to a particular type of alignment
	with open("raw_data/" + str(seed_iter) + "/" + data_set + "_drift_iter_"
						+ str(drift_iter) + "/alignment_results.json") as f:
		alignment_results = json.load(f)
	all_aligned = copy.copy(alignment_results["upright_aligned_indices"])
	all_aligned.extend(alignment_results["inverted_aligned_indices"])

	alignment_type_results = [all_aligned, alignment_results["both_aligned_indices"], alignment_results["non_aligned_indices"]]
	alignment_type_labels = ["Single Aligned", "Both Aligned", "None Aligned"]
	alignment_type_colors = ["dodgerblue", "purple", "crimson"]

	assert ((len(all_aligned) + len(alignment_results["both_aligned_indices"])
			+ len(alignment_results["non_aligned_indices"])) == stimuli_params["output_layer_size"]), "Aligned indices do not match layer size"

	for epoch_iter in range(len(learning_FR_list)):

		current_fr = learning_FR_list[epoch_iter]

		for current_alignment_iter in range(len(alignment_type_results)):

			mask = alignment_type_results[current_alignment_iter]

			# Only include legend on the first plotting
			if epoch_iter == 0:
				labels = alignment_type_labels[current_alignment_iter]
			else:
				labels = None

			plt.scatter(np.ones(len(mask))*epoch_markers_str[epoch_iter],
						current_fr[mask], color=alignment_type_colors[current_alignment_iter],
						alpha=0.2, label=labels)
		
		plt.scatter(epoch_markers_str[epoch_iter], np.mean(current_fr), s=60, marker='x', color='k')

	plt.axhline((1/(stimuli_params["duration_of_presentations"]/1000)) / len(stimuli_params["stimuli_names"]),
			    label="Ideal Rate",
			    linestyle='--', alpha=0.5, color='k')

	xlabel("Duration of Training (sec)")
	xlim(-10)
	ylabel("Firing Rates (Hz)")
	legend()
	title("Firing rates across epochs of STDP training")
	savefig("analysis_results/" + str(seed_iter) + "/violins_rates_across_epochs.png", dpi=300)
	clf()


def visualize_strong_weights(stimuli_params, network_params, data_set, seed_iter, drift_iter):

	delays = np.loadtxt("weights/" + str(seed_iter) + "/rand_delays.txt")

	initial_weights = np.loadtxt("weights/" + str(seed_iter) + "/rand_weights.txt")

	final_weights = np.loadtxt("weights/" + str(seed_iter) + "/" + data_set + "_drift_iter_"
						+ str(drift_iter) + "/final_weights.txt")

	num_bins = 6


	# Histogram of weights exceeding the threshold of interest
	mask = initial_weights >= network_params["wmax"]/2
	hist(np.asarray(delays)[mask], bins=num_bins, alpha=0.5, color="dodgerblue")
	xlabel("Synapse Delay (ms)")
	title("Initial Delays with Strong Weight")
	savefig("analysis_results/strong_weights_initial.png")
	clf()

	mask = final_weights >= network_params["wmax"]/2
	hist(np.asarray(delays)[mask], bins=num_bins, alpha=0.5, color="dodgerblue")
	xlabel("Synapse Delay (ms)")
	title("Final Delays with Strong Weight")
	savefig("analysis_results/strong_weights_final.png")
	clf()

	# Scatterplot showing correlation (if any) between delay and weight
	scatter(delays, initial_weights, label="Initial", color="crimson", alpha=0.5)
	scatter(delays, final_weights, label="Final", color="dodgerblue", alpha=0.5)
	xlabel("Synapse Delay (ms)")
	ylabel("Weight")
	title("Weight vs Delay")
	savefig("analysis_results/delay_vs_weight_scatter.png")
	clf()


	# Use Spearman as data generally not very normally distributed
	coef, p_val = scipy.stats.spearmanr(delays, initial_weights)
	print("\nCorrelation before STDP")
	print("Coef : " + str(coef))
	print("p-value : " + str(p_val))

	coef, p_val = scipy.stats.spearmanr(delays, final_weights)
	print("\nCorrelation AFTER STDP")
	print("Coef : " + str(coef))
	print("p-value : " + str(p_val))


def specific_weights_across_epochs(stimuli_params, network_params, data_set, seed_iter, drift_iter):
	'''
	Visualize the change in aligned weights across epochs, specifically looking at those associated
	with aligned inputs/delays, vs. those that are not
	'''


	# Load firing rate data, including pre-STDP firing rates
	initial_FR = np.loadtxt("raw_data/" + str(seed_iter) + "/untrained_alternating_inputs_drift_iter_"
		+ str(drift_iter) + "/fr_output_layer.txt")

	learning_FR_list = np.loadtxt("raw_data/" + str(seed_iter) + "/" + data_set + "_drift_iter_"
						+ str(drift_iter) + "/rates_over_epochs_vals.txt")

	learning_FR_list = np.insert(learning_FR_list, 0, initial_FR, axis=0)

	# Load epoch markers
	epoch_markers_str = np.loadtxt("raw_data/" + str(seed_iter) + "/" + data_set + "_drift_iter_"
						+ str(drift_iter) + "/epoch_markers.txt")
	epoch_markers_str = np.insert(epoch_markers_str, 0, 0)

	# Load alignment results; this will enable visualising the change in firing rates
	# the the correspondence (if any) to a particular type of alignment
	with open("raw_data/" + str(seed_iter) + "/" + data_set + "_drift_iter_"
						+ str(drift_iter) + "/alignment_results.json") as f:
		alignment_results = json.load(f)
	all_aligned = copy.copy(alignment_results["upright_aligned_indices"])
	all_aligned.extend(alignment_results["inverted_aligned_indices"])

	alignment_type_results = [all_aligned, alignment_results["both_aligned_indices"], alignment_results["non_aligned_indices"]]
	alignment_type_labels = ["Single Aligned", "Both Aligned", "None Aligned"]
	alignment_type_colors = ["dodgerblue", "purple", "crimson"]

	assert ((len(all_aligned) + len(alignment_results["both_aligned_indices"])
			+ len(alignment_results["non_aligned_indices"])) == stimuli_params["output_layer_size"]), "Aligned indices do not match layer size"

	for epoch_iter in range(len(learning_FR_list)):

		current_fr = learning_FR_list[epoch_iter]

		for current_alignment_iter in range(len(alignment_type_results)):

			mask = alignment_type_results[current_alignment_iter]

			# Only include legend on the first plotting
			if epoch_iter == 0:
				labels = alignment_type_labels[current_alignment_iter]
			else:
				labels = None

			plt.scatter(np.ones(len(mask))*epoch_markers_str[epoch_iter],
						current_fr[mask], color=alignment_type_colors[current_alignment_iter],
						alpha=0.2, label=labels)
		
		plt.scatter(epoch_markers_str[epoch_iter], np.mean(current_fr), s=60, marker='x', color='k')

	plt.axhline((1/(stimuli_params["duration_of_presentations"]/1000)) / len(stimuli_params["stimuli_names"]),
			    label="Ideal Rate",
			    linestyle='--', alpha=0.5, color='k')

	xlabel("Duration of Training (sec)")
	xlim(-10)
	ylabel("Firing Rates (Hz)")
	legend()
	title("Firing rates across epochs of STDP training")
	savefig("analysis_results/" + str(seed_iter) + "/violins_rates_across_epochs.png", dpi=300)
	clf()


if __name__ == '__main__':

	if os.path.exists("analysis_results") == 0:
		try:
			os.mkdir("analysis_results")
		except OSError:
			pass

	print("\nPerforming analysis of simulation results")
	with open('config_TranslationInvariance.yaml') as f:
		params = yaml.load(f, Loader=yaml.FullLoader)

	stimuli_params = params['stimuli_params']
	network_params = params["network_params"]

	layers_to_analyse = ["output"] # "input"

	# for layer in layers_to_analyse:

	# 	perform_primary_analyses(stimuli_params, layer)


	# Analyses/plotting specific to during training
	data_set = "during_spikepair_training"

	for seed_iter in stimuli_params["seeds_list"]:

		if os.path.exists("analysis_results/" + str(seed_iter)) == 0:
			try:
				os.mkdir("analysis_results/" + str(seed_iter))
			except OSError:
				pass

		for drift_iter in stimuli_params["drift_coef_list"]:

			weights_across_epochs_by_alignment(stimuli_params, network_params, data_set, seed_iter, drift_iter)

			visualize_strong_weights(stimuli_params, network_params, data_set, seed_iter, drift_iter)

			FR_across_epochs(stimuli_params, network_params, data_set, seed_iter, drift_iter)

			weights_across_epochs(stimuli_params, network_params, data_set, seed_iter, drift_iter)

			strong_delays_across_epochs(stimuli_params, network_params, data_set, seed_iter, drift_iter)


