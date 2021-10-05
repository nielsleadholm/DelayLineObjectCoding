import numpy as np
import os
import yaml
import matplotlib.pyplot as plt

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

def weights_across_time(stimuli_params, network_params, data_set, seed_iter, drift_iter):

	simulation_duration = (stimuli_params['number_of_train_presentations']
			 				   * stimuli_params['duration_of_presentations']
			 				   * len(stimuli_params['stimuli_names']))

	delays = np.loadtxt("raw_data/" + str(seed_iter) + "/" + data_set + "_drift_iter_"
						+ str(drift_iter) + "/ff_delays.txt")
	current_weights = np.loadtxt("weights/" + str(seed_iter) + "/" + data_set + "_drift_iter_" + str(drift_iter) + "/weights"
				   + str(stimuli_params["intervals_for_weight_saving"]/1000) + "_seconds_of_sim.txt")

	# Plot initial weights vs delays
	plt.scatter(delays, current_weights, color='red', alpha=0.3)

	plt.ylabel("Weight values")
	plt.xlabel("Delays (ms)")
	plt.ylim(-0.1, network_params["wmax"]+0.1)
	plt.title("Weights vs Delays Before Learning")
	plt.savefig("analysis_results/weights_vs_delays_before_learning" + data_set, dpi=300)
	plt.clf()


	for learning_period in range(0, simulation_duration, stimuli_params["intervals_for_weight_saving"]):
		current_weights = np.loadtxt("weights/" + str(seed_iter) + "/" + data_set + "_drift_iter_" + str(drift_iter) + "/weights"
				   + str((learning_period+stimuli_params["intervals_for_weight_saving"])/1000) + "_seconds_of_sim.txt")

		# print(np.shape(current_weights))
		# print(learning_period)
		# print(current_weights[0:5])
		# exit()

		plt.scatter(np.ones(len(current_weights))*learning_period+np.random.normal(0,100,len(current_weights)), current_weights, alpha=0.2)
		plt.scatter(learning_period, np.mean(current_weights), s=60, marker='x', color='k')

	plt.ylabel("Weight values")
	plt.xlabel("Time (ms)")
	plt.ylim(-0.1, network_params["wmax"]+0.1)
	plt.title("Weights across learning intervals")
	plt.savefig("analysis_results/weights_over_time_" + data_set, dpi=300)
	plt.clf()

	# Plot the final weights as a function of delay
	plt.scatter(delays, current_weights, color='dodgerblue', alpha=0.3)

	plt.ylabel("Weight values")
	plt.xlabel("Delays (ms)")
	plt.ylim(-0.1, network_params["wmax"]+0.1)
	plt.title("Weights vs Delays After Learning")
	plt.savefig("analysis_results/weights_vs_delays_after_learning" + data_set, dpi=300)
	plt.clf()

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

		for seed_iter in range(1):

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

	for layer in layers_to_analyse:

		perform_primary_analyses(stimuli_params, layer)


	# List of training conditions to look at (excludes data collected during evaluation)
	# - change of weight distributions --> double check that the 'final' weights are equivalent to the final time point weights
	# - change of firing rates --> NB this can be done by simply binning the spiking activity collected at the end of training,
	# binning it as desired 


	# data_set = "during_spikepair_training"
	# for seed_iter in range(1):

	# 	for drift_iter in stimuli_params["drift_coef_list"]:

	# 		weights_across_time(stimuli_params, network_params, data_set, seed_iter, drift_iter)


