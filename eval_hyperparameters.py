from brian2 import *
import numpy as np
import os
import random
import yaml
#from run_simulation import visualise_cellular_property


def custom_visualise_cellular_property(variable_monitor, variable_name, spike_monitor,
								neuron_IDs, fig_title, fname, y_limits=None,
								x_limits=None, threshold=None, jitter_vis=None,
								effective_ISI_point=None):
	
	line_styles = ['-', 'dotted', 'dashed', 'dashdot']

	# print(getattr(variable_monitor, variable_name))

	# print(np.shape(getattr(variable_monitor, variable_name)))

	max_val = np.amax(getattr(variable_monitor, variable_name)) # For plotting the occurance of spikes

	# print(max_val)
	# exit()
	fig, [ax1, ax2] = plt.subplots(2)

	for ii in range(len(neuron_IDs)):
		if len(neuron_IDs) == 4:
			linestyle_temp=line_styles[ii]
		else:
			linestyle_temp='-'

		ax1.plot(variable_monitor.t/ms, getattr(variable_monitor, variable_name)[ii],
			alpha=0.5, linestyle=linestyle_temp)

		# Add an indicator when spikes have occured
		mask = spike_monitor.i==neuron_IDs[ii]
		ax1.scatter(np.asarray(spike_monitor.t/ms)[mask],
				np.multiply(np.ones(np.sum(mask)), max_val), alpha=0.5)

	if threshold is not None:
		ax1.plot(variable_monitor.t/ms, np.ones(len(variable_monitor.t/ms))*threshold,
			 linestyle='--', alpha=0.5, color='k')

	print("Annotation at:")
	print(effective_ISI_point)
	ax1.scatter(effective_ISI_point, -0.070, marker="^", color="crimson")
	ax1.annotate("ISI", (effective_ISI_point, -0.070))

	if x_limits is not None:
		ax1.set_xlim(x_limits[0],x_limits[1])
	else:
		ax1.set_xlim(0,600)
	ax1.set_ylabel(variable_name)
	if y_limits is not None:
		ax1.set_ylim(y_limits[0],y_limits[1])
	#ax1.legend()
	#ax1.title(fig_title)


	# Plot the probability distribution of the spike time difference
	# of two perfectly aligned spikes based on the current jitter

	# Any given sample has mean of 0 (because the mean of both samples is 0)
	# The standard deviation is presumably the sum of the two, or more be the same at
	# exepctation

	# Simulated data
	number_samples = 100000
	half_bins = 100
	first_spike_times = np.random.normal(0,jitter_vis,number_samples)
	#print(np.shape(first_spike_times))
	#print(first_spike_times[0:10])
	second_spike_times = np.random.normal(0,jitter_vis,number_samples)
	#print(second_spike_times[0:10])
	spike_time_differences = first_spike_times - second_spike_times
	#print(spike_time_differences[0:10])
	abs_spike_time_differences = np.absolute(spike_time_differences)
	#print(abs_spike_time_differences)
	# ax2.hist(spike_time_differences, label='half', alpha=0.5, bins=half_bins*2)
	ax2.hist(abs_spike_time_differences, label="absolute", alpha=0.5, bins=half_bins)
	# ax2.hist(np.random.normal(0,jitter_vis,number_samples), label="single_spike", alpha=0.5, bins=half_bins*2)
	# ax2.hist(np.random.normal(0,sqrt(2*jitter_vis),number_samples), label="double_jitter", alpha=0.5, bins=half_bins*2)
	ax2.set_xlim(x_limits[0],x_limits[1]/100)
	#ax2.legend()

	ax2.set_ylabel('Frequency')
	ax2.set_xlabel('Time (ms)')




	savefig("figures/" + fname + ".png", dpi=300)
	# np.savetxt("raw_data/" + fname + "_times.txt", variable_monitor.t/ms)
	# np.savetxt("raw_data/" + fname + "_values.txt", getattr(variable_monitor, variable_name))
	clf()
	#exit()

def calc_ISI(stimuli_params):
	'''
	Determine the effective interspike interval for spikes received by the output neurons
	'''

	effective_rate = (stimuli_params["input_layer_size"]
					  * (1/((stimuli_params["duration_of_presentations"]
					  		 - stimuli_params["duration_of_saccades"])/1000)))

	print("\nThe effective input rate (Hz) to each output neuron is : " + str(effective_rate))
	print("The effective ISI (ms) for each output neuron is : " + str(1000/effective_rate))

	return 1000/effective_rate

def plot_STDP_curve(network_params, effective_ISI):
	'''
	Visualize the decay of the STDP curve, in comparison to effective ISI
	'''

	# STDP curve
	A_post = -network_params["Apre"]*network_params["Apost_scaling"]
	delta_t = linspace(-50, 50, 100)
	W = where(delta_t>0, network_params["Apre"]*exp(-delta_t/network_params["taupre"]), A_post*exp(delta_t/network_params["taupost"]))
	
	# Expected values of the exponential distribution of the ISI
	specific_vals = np.asarray([-effective_ISI, effective_ISI])
	expected_ISI = where(specific_vals>0, network_params["Apre"]*exp(-specific_vals/network_params["taupre"]), A_post*exp(specific_vals/network_params["taupost"]))

	# PDF of the exponential distribution of the ISI
	delta_t_prob = linspace(0, 50, 50)
	prob_exp = (1/effective_ISI)*exp((-1/effective_ISI)*delta_t_prob)
	#print(prob_exp)


	fig, ax1 = plt.subplots()

	#print("Rate in mHz : " + str(1/effective_ISI))

	color = 'tab:blue'
	ax1.plot(delta_t, W, color=color, alpha=0.5, label="STDP Weight Change")
	ax1.scatter([-effective_ISI, effective_ISI], expected_ISI, marker='+', color='k', s=80, label="Expected ISI")
	ax1.set_xlabel(r'$\Delta t$ (ms)')
	ax1.set_ylabel('W', color=color)
	ax1.legend()
	ax1.set_ylim(A_post,network_params["Apre"])
	ax1.tick_params(axis='y', labelcolor=color)
	ax1.axhline(0, ls='-', c='k');
	
	ax2 = ax1.twinx()

	color = 'tab:red'
	ax2.plot(delta_t_prob, prob_exp, linestyle='dashed', label="PDF of ISI", color=color, alpha=0.5)
	ax2.legend(loc="lower right")
	ax2.set_ylabel('Probability', color=color)
	ax2.set_ylim(-network_params["Apost_scaling"],1.0)
	ax2.tick_params(axis='y', labelcolor=color)

	fig.tight_layout()
	fig.savefig("hyperparameter_results/STDP_cruve.png", dpi=300)
	fig.clf()

	# cdf_exponential = 1 - exp((-1/effective_ISI)*delta_t_prob)
	# plot(delta_t_prob, cdf_exponential)
	# show()


def plot_g_e_curve(stimuli_params, network_params, effective_ISI, current_weight):
	'''
	Visualize the decay of the g_e curve, in comparison to effective ISI
	'''

def run_basic_simulation(stimuli_params, network_params, effective_ISI, current_delay):

	start_scope()

	seed(0)
	random.seed(0)

	# ==== RUN BASIC SIMULATION OF SINGLE NEURON ===

	mean_membrane_voltage = [-70*mV, -73*mV] # User set hyperparameter; range
	# over which to consider the values of interest

	# --> ** RATHER THAN THIS, CAN JUST ADD THE BACKGROUND INPUT NEURONS **

	print("==== SETTING THE DELAY OF THE FEED-FORWARD INHIBITION ===")

	inhibitory_delay = current_delay*ms


	# print("==== TEMPORARILY USING PYRAMIDAL CELL PARAMETERS ===")

	defaultclock.dt = network_params["time_step"]*ms

	E_l = -81.6*mV         # v_rest **
	g_l = 18*nS            # leak conductance 18**
	E_e = 0*mV          # excitatory synaptic reversal potential **
	E_i = -70*mV          # inhibitory synaptic reversal potential **
	C_m = 0.214*nF         # membrane capacitance; 0.214**; with 180, the effective tau-membrane is 10ms, as used in Masquelier 2009
	tau_e = 1.75*ms         # excitatory synaptic time constant; NB that beyond Hausser 1997, Troyer 1998 appear to use a value of 1.75 --> given that I'm not modelling the rise, 
	                        #   does it make sense to use a slightly smalelr value (e.g. 1.7 rather than 1.75 here, 5 rather than 5.25 for tau_i?)
	tau_i = 5.25*ms           # inhibitory synaptic time constant  **NB this is the value used in Troyer i.e. tau_in(fall)
	tau_r = 1.0*ms           # refractory period **
	V_th = -52.5*mV          # firing threshold **
	V_r = -57.8*mV           # reset potential **


	# # ** pyramidal cell properties **
	# E_l = -73.6*mV         # v_rest **
	# g_l = 25*nS            # leak conductance 18**
	# E_e = 0*mV          # excitatory synaptic reversal potential **
	# E_i = -70*mV          # inhibitory synaptic reversal potential **
	# C_m = 0.500*nF         # membrane capacitance; 0.214**; with 180, the effective tau-membrane is 10ms, as used in Masquelier 2009
	# tau_e = 1.75*ms         # excitatory synaptic time constant; NB that beyond Hausser 1997, Troyer 1998 appear to use a value of 1.75 --> given that I'm not modelling the rise, 
	#                         #   does it make sense to use a slightly smalelr value (e.g. 1.7 rather than 1.75 here, 5 rather than 5.25 for tau_i?)
	# tau_i = 5.25*ms           # inhibitory synaptic time constant  **NB this is the value used in Troyer i.e. tau_in(fall)
	# tau_r = 1.5*ms           # refractory period **
	# V_th = -52.5*mV          # firing threshold **
	# V_r = -56.5*mV           # reset potential **



	#print("\n===TEMPORARILY INITIALIZING CAPACITANCE AT 0===")
	g_init = 10*nS  # Maximum initial synapse conductance value
	scaling_constant = g_l

	print("\nEffective tau membrane:")
	print(C_m/g_l)

	# Main neuron group

	LIF_neurons_eqs='''
		idx = i                                                  : 1                        # store index of each neuron (for tracking synaptic connections)
		dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) + g_i*(E_i-v))/C_m    : volt (unless refractory) # membrane potential (LIF equation)               
		dg_e/dt = -g_e/tau_e                                     : siemens                  # post-synaptic exc. conductance (incremented when excitatory spike arrives at neuron - see synapse equations)
		dg_i/dt = -g_i/tau_i                                     : siemens                  # post-synaptic inh. conductance (incremented when inhibitory spike arrives at neuron - see synapse equations)
		'''
	output_neuron = NeuronGroup(1, LIF_neurons_eqs, threshold='v > V_th', reset='v = V_r', refractory='tau_r', method='euler')
	output_neuron.g_e = g_init
	output_neuron.g_i = g_init

	# inhibitory_neuron = NeuronGroup(1, LIF_neurons_eqs, threshold='v > V_th', reset='v = V_r', refractory='tau_r', method='euler')
	# inhibitory_neuron.g_e = g_init
	# inhibitory_neuron.g_i = 0


	# Create the input spikes

	input_times = [12,13]#[2*network_params["time_step"],4*network_params["time_step"]]
	time_iter = 1

	# Spread out paired spikes to observe their interactions
	# This assumes that 100ms >> the membrane time constant
	while time_iter < effective_ISI:

		# print(input_times)
		# print(time_iter)

		input_times.append(100*time_iter)
		input_times.append(100*time_iter+time_iter)

		time_iter += 1

	# Spikes separated by effective ISI
	input_times.append(100*time_iter)
	input_times.append(100*time_iter+effective_ISI)

	# Final, singular spike
	input_times.append(100*(time_iter+1))

	# print(input_times)
	# exit()

	# input_times = [0,0+network_params["time_step"],100,100+1,200,202,300,303,400,404,500,505,600,606,
	# 			   700,707,800,800+effective_ISI,900]
	input_IDs = np.zeros(np.shape(input_times))
	input_neuron = SpikeGeneratorGroup(1, input_IDs, input_times*ms)



	# Create synapses

	S_ff = Synapses(input_neuron, output_neuron, 'w : 1', on_pre='g_e += scaling_constant*w')
	S_ff.connect(p=1.0)
	S_ff.w = network_params["wmax"]


	print("==== CURRENTLY USING MONO-SYNAPTIC FEED-FORWARD INHIBITION ====")

	S_ff_inhibition = Synapses(input_neuron, output_neuron, 'w : 1',
			on_pre='g_i += scaling_constant*w',
			delay=inhibitory_delay)
	S_ff_inhibition.connect(p=1.0)
	S_ff_inhibition.w = network_params["wmax"]*(9/3)



	# # **di-synaptic feed-forward inhibition**
	# S_ff_inhibition = Synapses(input_neuron, inhibitory_neuron, 'w : 1',
	# 		on_pre='g_e += scaling_constant*w',
	# 		delay=inhibitory_delay)
	# S_ff_inhibition.connect(p=1.0)
	# S_ff_inhibition.w = network_params["wmax"]*(2)
	# # Set this multiplier to x2, as this should, by definition, be sufficient to induce a 
	# # spike in the post-synaptic neuron


	print("==== CURRENTLY DISABLING LATERAL INHIBITION ====")

	# S_lateral_inhibition = Synapses(inhibitory_neuron, output_neuron, 'w : 1',
	# 		on_pre='g_i += scaling_constant*w',
	# 		delay=inhibitory_delay)
	# S_lateral_inhibition.connect(p=1.0)
	# S_lateral_inhibition.w = network_params["wmax"]*(9/3)




	# Background input

	background_w = network_params["background_weight"]

	background_input = PoissonInput(output_neuron, 'g_e',
				network_params["background_layer_width"],
				network_params["background_base_rate"]*Hz,
				weight="rand()*background_w*g_l")

	# # Background to the inhibitory neurons
	# inh_background_input = PoissonInput(inhibitory_neuron, 'g_e',
	# 						network_params["background_layer_width"],
	# 						network_params["background_base_rate"]*Hz,
	# 						weight="rand()*background_w*g_l")


	store()

	initial_membrane_voltage = mean_membrane_voltage[0]

	#for initial_membrane_voltage in mean_membrane_voltage:

	restore()

	spike_M = SpikeMonitor(output_neuron)
	properties_M = StateMonitor(output_neuron, ('v', 'g_e', 'g_i'), record=0)


	# inh_spike_M = SpikeMonitor(inhibitory_neuron)
	# inh_properties_M = StateMonitor(inhibitory_neuron, ('v', 'g_e', 'g_i'), record=0)


	output_neuron.v = initial_membrane_voltage
	#inhibitory_neuron.v = initial_membrane_voltage


	run((max(input_times)+30)*ms)


	custom_visualise_cellular_property(properties_M, "v",
		spike_M, neuron_IDs=[0], fig_title="Membrane Voltage",
		fname=("toy_OUTPUT_membrane_voltage_" + str(C_m/nF)
			   + "_ff_inhibition_" + str(inhibitory_delay)),
		y_limits=[-0.075,-0.05],
		x_limits=[0,max(input_times)+30],
		threshold=(V_th/mV)/1000,
		jitter_vis=stimuli_params["jitter_std"],
		effective_ISI_point=input_times[-2])

	# visualise_cellular_property(properties_M, "g_e",
	# 	spike_M, neuron_IDs=[0], fig_title="g_e Conductance",
	# 	fname="toy_OUTPUT_g_e_" + str(C_m/nF),
	# 	x_limits=[0,max(input_times)+30])

	# visualise_cellular_property(properties_M, "g_i",
	# 	spike_M, neuron_IDs=[0], fig_title="g_i Conductance",
	# 	fname="toy_OUTPUT_g_i_" + str(C_m/nF),
	# 	x_limits=[0,max(input_times)+30])


	# visualise_cellular_property(inh_properties_M, "v",
	# 	inh_spike_M, neuron_IDs=[0], fig_title="Membrane Voltage",
	# 	fname="toy_INH_membrane_voltage_" + str(C_m/nF), y_limits=[-0.075,-0.05],
	# 	x_limits=[0,max(input_times)+30],
	# 	threshold=(V_th/mV)/1000)

	# visualise_cellular_property(inh_properties_M, "g_e",
	# 	inh_spike_M, neuron_IDs=[0], fig_title="g_e Conductance",
	# 	fname="toy_INH_g_e_" + str(C_m/nF),
	# 	x_limits=[0,max(input_times)+30])


# def estimate_alignments_across_noise(stimuli_params, network_params):
	'''
	Never properly implemented
	'''
# 	print("=== TEMPORARILY DEFINING THE SPIKE TIMING DELAY DIFFERNECE ==")
# 	base_line_diff = [-8, 8]

# 	num_random_samples = 1000  # Number of random samples to perform for estimating the distribution
# 	delay_line_upper = 15
# 	jitter_levels = [2,3] #list(range(6))
# 	integration_window_list = [0.2, 1, 2, 3, 4, 5, 6]
# 	gain_in_individual_list = [] # Track how many more individual avereage connections there are than
# 	# both

# 	for current_integration_window in integration_window_list:

# 		# Track the mean results for each jitter level used
# 		all_jitters_both_result = []
# 		all_jitters_both_SD = []
# 		all_jitters_individual_result = []
# 		all_jitters_individual_SD = []

# 		for current_jitter_level in jitter_levels:

# 			# Track the results for each random sample
# 			current_jitter_both = []
# 			current_jitter_individual = []

# 			# Iterate through the cells in the output, determining which ones 
# 			# have an individual or both alignments
# 			for output_cell in range(stimuli_params["output_layer_size"]):

# 				# Track the number of cells with an alignment
# 				alignment_counter_both = 0
# 				alignment_counter_individual = 0

# 				diff = base_line_diff + np.random.uniform(-stimuli_params["initial_times_limit"],
# 						stimuli_params["initial_times_limit"], np.shape(base_line_diff))
# 				# print("Current diff")
# 				# print(diff)

# 				copy_first_delay_lines = np.random.uniform(0,delay_line_upper,stimuli_params['input_layer_size'])
# 				copy_second_delay_lines = np.random.uniform(0,delay_line_upper,stimuli_params['input_layer_size'])

# 				#print(copy_second_delay_lines)

# 				for sample_iter in range(num_random_samples):

# 					# Sample the delay lines for this cell, plus random noise from jitter
# 					first_delay_lines = (copy_first_delay_lines
# 						+ np.random.normal(0, current_jitter_level, stimuli_params['input_layer_size']))
# 					second_delay_lines = (copy_second_delay_lines
# 						+ np.random.normal(0, current_jitter_level, stimuli_params['input_layer_size']))

# 					#print(second_delay_lines)

# 					delay_diff = second_delay_lines - first_delay_lines

# 					# Check if, for the current cell, any of the delay lines align
# 					stim_1_alignments = np.any((delay_diff >= (diff[0] - current_integration_window)) &
# 						(delay_diff <= (diff[0] + current_integration_window)))
# 					#exit()
# 					# print("Checking delay results:")
# 					# print(delay_diff)
# 					# print(delay_diff >= (diff[0] - current_integration_window))
# 					# print(delay_diff <= (diff[0] + current_integration_window))
# 					# print((delay_diff >= (diff[0] - current_integration_window)) &
# 					# 	(delay_diff <= (diff[0] + current_integration_window)))
# 					# print(np.any((delay_diff >= (diff[0] - current_integration_window)) &
# 					# 	(delay_diff <= (diff[0] + current_integration_window))))

# 					stim_2_alignments = np.any((delay_diff >= (diff[1] - current_integration_window)) &
# 						(delay_diff <= (diff[1] + current_integration_window)))

# 					# print("Raw alignments")
# 					# print(stim_1_alignments)
# 					# print(stim_2_alignments)

# 					# Check if, for the current cell, there are aligning delay lines for both stimuli
# 					if stim_1_alignments and stim_2_alignments:

# 						alignment_counter_both += 1

# 					# Otherwise if there is alignment for at least one stimuli
# 					elif stim_1_alignments or stim_2_alignments:

# 						alignment_counter_individual += 1

# 				#assert (alignment_counter_both + alignment_counter_individual) <= stimuli_params["output_layer_size"], "Cannot have more alignment results than cells"

# 				current_jitter_both.append(alignment_counter_both/num_random_samples)
# 				current_jitter_individual.append(alignment_counter_individual/num_random_samples)

# 			all_jitters_both_result.append(np.mean(current_jitter_both))
# 			all_jitters_both_SD.append(np.std(current_jitter_both))
# 			all_jitters_individual_result.append(np.mean(current_jitter_individual))
# 			all_jitters_individual_SD.append(np.std(current_jitter_both))

# 		all_jitters_both_result = np.asarray(all_jitters_both_result)
# 		all_jitters_both_SD = np.asarray(all_jitters_both_SD)
# 		all_jitters_individual_result = np.asarray(all_jitters_individual_result)
# 		all_jitters_individual_SD = np.asarray(all_jitters_individual_SD)

# 		plot(jitter_levels, all_jitters_both_result,
# 			 label="Both Alignments", color="crimson", alpha=0.5, linestyle="--")
# 		fill_between(jitter_levels, all_jitters_both_result-all_jitters_both_SD,
# 				all_jitters_both_result+all_jitters_both_SD, color='crimson', alpha=0.1)
		
# 		plot(jitter_levels, all_jitters_individual_result,
# 			label="Individual Alignments", color="dodgerblue", alpha=0.5)
# 		fill_between(jitter_levels, all_jitters_individual_result-all_jitters_individual_SD,
# 				all_jitters_individual_result+all_jitters_individual_SD, color='dodgerblue', alpha=0.1)
		
# 		ylim(0,stimuli_params["output_layer_size"])
# 		legend()
# 		savefig("hyperparameter_results/jitter_integration_estimates_window_"
# 				+ str(current_integration_window) + ".png", dpi=300)
# 		clf()


def estimate_alignments_across_delays(stimuli_params, network_params):

	print("=== TEMPORARILY DEFINING THE SPIKE TIMING DELAY DIFFERNECE ==")
	base_line_diff = [-8, 8]

	num_random_samples = 100  # Number of random samples to perform for estimating the distribution
	max_delay_upper = 50
	integration_window_list = [0.2, 1, 2, 3, 4, 5, 6, 7, 8]
	gain_in_individual_list = [] # Track how many more individual avereage connections there are than
	# both

	for current_integration_window in integration_window_list:

		# Track the mean results for each delay line used
		all_delays_both_result = []
		all_delays_individual_result = []

		for delay_line_upper in range(max_delay_upper):

			# Track the results for each random sample
			current_delay_both = []
			current_delay_individual = []

			for sample_iter in range(num_random_samples):

				# Track the number of cells with an alignment
				alignment_counter_both = 0
				alignment_counter_individual = 0

				# Iterate through the cells in the output, determining which ones 
				# have an individual or both alignments
				for output_cell in range(stimuli_params["output_layer_size"]):

					diff = base_line_diff + np.random.uniform(-stimuli_params["initial_times_limit"],
							stimuli_params["initial_times_limit"], np.shape(base_line_diff))
					# print("Current diff")
					# print(diff)

					# Sample the delay lines for this cell
					first_delay_lines = np.random.uniform(0,delay_line_upper,stimuli_params['input_layer_size'])
					second_delay_lines = np.random.uniform(0,delay_line_upper,stimuli_params['input_layer_size'])
					delay_diff = second_delay_lines - first_delay_lines

					# Check if, for the current cell, any of the delay lines align
					stim_1_alignments = np.any((delay_diff >= (diff[0] - current_integration_window)) &
						(delay_diff <= (diff[0] + current_integration_window)))

					# print("Checking delay results:")
					# print(delay_diff)
					# print(delay_diff >= (diff[0] - current_integration_window))
					# print(delay_diff <= (diff[0] + current_integration_window))
					# print((delay_diff >= (diff[0] - current_integration_window)) &
					# 	(delay_diff <= (diff[0] + current_integration_window)))
					# print(np.any((delay_diff >= (diff[0] - current_integration_window)) &
					# 	(delay_diff <= (diff[0] + current_integration_window))))

					stim_2_alignments = np.any((delay_diff >= (diff[1] - current_integration_window)) &
						(delay_diff <= (diff[1] + current_integration_window)))

					# print("Raw alignments")
					# print(stim_1_alignments)
					# print(stim_2_alignments)

					# Check if, for the current cell, there are aligning delay lines for both stimuli
					if stim_1_alignments and stim_2_alignments:

						alignment_counter_both += 1

					# Otherwise if there is alignment for at least one stimuli
					elif stim_1_alignments or stim_2_alignments:

						alignment_counter_individual += 1

				assert (alignment_counter_both + alignment_counter_individual) <= stimuli_params["output_layer_size"], "Cannot have more alignment results than cells"

				current_delay_both.append(alignment_counter_both)
				current_delay_individual.append(alignment_counter_individual)

			all_delays_both_result.append(np.mean(current_delay_both))
			all_delays_individual_result.append(np.mean(current_delay_individual))

		gain_in_individual_list.append(np.asarray(all_delays_individual_result)
									   - np.asarray(all_delays_both_result))

		plot(list(range(max_delay_upper)), all_delays_both_result,
			 label="Both Alignments", color="crimson", alpha=0.5, linestyle="--")
		plot(list(range(max_delay_upper)), all_delays_individual_result,
			label="Individual Alignments", color="dodgerblue", alpha=0.5)
		axvline(abs(base_line_diff[0]), label="Stimulus off-set",
			 linestyle='--', alpha=0.5, color='k')
		legend()
		savefig("hyperparameter_results/delay_integration_estimates_window_"
				+ str(current_integration_window) + ".png", dpi=300)
		clf()


	# Plot the advantage (if any) of individual vs. both alignments
	
	iter_counter = 0
	for current_integration_window in integration_window_list:

		plot(list(range(max_delay_upper)), gain_in_individual_list[iter_counter],
			label="Integration window:" + str(current_integration_window),
			alpha=0.5)
		axvline(abs(diff[0]), label="Stimulus off-set",
			 linestyle='--', alpha=0.5, color='k')
		iter_counter += 1
	
	legend()
	savefig("hyperparameter_results/integration_individual_advantage.png", dpi=300)
	clf()

if __name__ == '__main__':

	with open('config_TranslationInvariance.yaml') as f:
		params = yaml.load(f, Loader=yaml.FullLoader)

	stimuli_params = params['stimuli_params']
	network_params = params['network_params']

	if os.path.exists("hyperparameter_results") == 0:
		try:
			os.mkdir("hyperparameter_results")
		except OSError:
			pass

	if os.path.exists("figures") == 0:
		try:
			os.mkdir("figures")
		except OSError:
			pass

	estimate_alignments_across_noise(stimuli_params, network_params)
	exit()

	estimate_alignments_across_delays(stimuli_params, network_params)
	exit()

	effective_ISI = calc_ISI(stimuli_params)

	plot_STDP_curve(network_params, effective_ISI)

	ff_inh_delays = list(range(network_params["ff_inhibition_delay"]+1))
	ff_inh_delays.append(50)
	# print(ff_inh_delays)
	# exit()

	for current_delay in ff_inh_delays:

		run_basic_simulation(stimuli_params, network_params, effective_ISI,
							 current_delay)


