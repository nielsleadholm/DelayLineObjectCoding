from brian2 import *
import json
import numpy as np
import pprint
import seaborn as sns
import time

def visualise_raster(spike_monitor, fig_title, fname):
	'''
	Basic visualization of spiking raster
	'''
	plot(spike_monitor.t/ms, spike_monitor.i, '.k')
	xlabel('Time (ms)')
	xlim(0,2000)
	ylabel('IDs')
	title(fig_title)
	savefig("figures/" + fname + ".png")
	np.savetxt("raw_data/" + fname + "_times.txt", spike_monitor.t/ms)
	np.savetxt("raw_data/" + fname + "_IDs.txt", spike_monitor.i)
	clf()


def input_drive_visualization(input_spike_monitor, output_spike_monitor, fig_title, fname):
	'''
	Visualize the association between input (stimuli) spikes and output spikes
	on a raster plot
	'''
	plot(input_spike_monitor.t/ms, input_spike_monitor.i, '.k', marker='x')
	plot(output_spike_monitor.t/ms, output_spike_monitor.i, 'red', alpha=0.5,
			marker='.', linewidth=0)
	xlabel('Time (ms)')
	xlim(0,2000)
	ylabel('IDs')
	title(fig_title)
	savefig("figures/" + fname + "_sim_beginning.png")
	clf()

	end_of_sim = np.amax(input_spike_monitor.t)*1000
	plot(input_spike_monitor.t/ms, input_spike_monitor.i, '.k', marker='x')
	plot(output_spike_monitor.t/ms, output_spike_monitor.i, 'red', alpha=0.5,
			marker='.', linewidth=0)
	xlabel('Time (ms)')
	xlim(max(0,end_of_sim-1000),end_of_sim)
	ylabel('IDs')
	title(fig_title)
	savefig("figures/" + fname + "_sim_end.png")
	clf()

def visualise_histo(array_vals, variable_name, fig_title, fname, num_bins=20):

	hist(np.asarray(array_vals), bins=num_bins, alpha=0.5)
	xlabel(variable_name)
	title(fig_title)
	savefig("figures/" + fname + ".png")
	np.savetxt("raw_data/" + fname + ".txt", array_vals)
	clf()


def visualise_violins_over_epochs(list_of_results, epoch_markers, fig_title, fname, y_label):
	'''
	Visualize the distribution of a cellular property such as membrane voltage 
	for all neurons, across the entire recording duration
	'''

	violinplot(np.transpose(list_of_results), positions=epoch_markers, showmeans=True)

	xlabel("Duration of Training (sec)")
	xlim(0)
	ylabel(y_label)

	title(fig_title)
	savefig("figures/" + fname + "/" + fig_title + ".png", dpi=300)
	clf()


def visualise_cellular_property(variable_monitor, variable_name, spike_monitor,
								neuron_IDs, fig_title, fname, y_limits=None,
								x_limits=None, threshold=None):
	'''
	Visualize the change of a cellular property such as membrane voltage 
	over time, for a series of specified neurons
	'''

	max_val = np.amax(getattr(variable_monitor, variable_name)) # For plotting the occurance of spikes

	for ii in neuron_IDs:

		plot(variable_monitor.t/ms, getattr(variable_monitor, variable_name)[ii],
			alpha=0.5, linestyle="-", label="Index : " + str(ii))

		# Add an indicator when spikes have occured
		mask = spike_monitor.i==ii
		scatter(np.asarray(spike_monitor.t/ms)[mask],
				np.multiply(np.ones(np.sum(mask)), max_val), alpha=0.5)

	if threshold is not None:
		plot(variable_monitor.t/ms, np.ones(len(variable_monitor.t/ms))*threshold,
			 linestyle='--', alpha=0.5, color='k')

	xlabel('Time (ms)')
	if x_limits is not None:
		xlim(x_limits[0],x_limits[1])
	else:
		xlim(0,700)

	ylabel(variable_name)
	if y_limits is not None:
		ylim(y_limits[0],y_limits[1])

	title(fig_title)
	legend()
	savefig("figures/" + fname + ".png", dpi=300)
	clf()

	# Save raw data in binary format to take up less disk space
	np.save("raw_data/" + fname + "_times.txt", variable_monitor.t/ms)
	np.save("raw_data/" + fname + "_values.txt", getattr(variable_monitor, variable_name))


def visualise_cellular_distribution(variable_monitor, variable_name,
								fig_title, fname,
								vertical_markers=None,
								units=1):
	'''
	Visualize the distribution of a cellular property such as membrane voltage 
	for all neurons, across the entire recording duration
	'''

	all_varaible_values = np.asarray(getattr(variable_monitor, variable_name)/units).flatten()

	sns.distplot(all_varaible_values, hist=True, kde=True, 
		bins=int(180/5), color = 'dodgerblue', 
		hist_kws={'edgecolor':'black'},
		kde_kws={'linewidth': 4})

	if vertical_markers is not None:
		axvline(vertical_markers["threshold"]/units, label="V-threshold",
			 linestyle='--', alpha=0.5, color='k')
		axvline(vertical_markers["rest"]/units, label="V-rest",
			 linestyle='--', alpha=0.5, color='dodgerblue')
		legend()

	xlabel(variable_name)

	title(fig_title)
	savefig("figures/" + fname + "/" + fig_title + ".png", dpi=300)
	clf()


def array_of_cellular_properties(variable_monitor, spike_monitor, neuron_IDs, title, fname, v_threshold):
	'''
	Combine visualization of a variety of cellular properties
	'''

	# Only plot if something actually recorded
	if len(getattr(variable_monitor, "g_e")) > 0:

		visualise_cellular_property(variable_monitor, "g_e",
			spike_monitor, neuron_IDs, fig_title="Sample of " + title,
			fname=fname + "/g_e_" + title)

		visualise_cellular_property(variable_monitor, "v",
			spike_monitor, neuron_IDs, fig_title="Sample of " + title,
			fname=fname + "/v_" + title, y_limits=[-0.075,-0.05],
			threshold=v_threshold)

		visualise_cellular_property(variable_monitor, "g_i",
			spike_monitor, neuron_IDs, fig_title="Sample of " + title,
			fname=fname + "/g_i_" + title)



def basic_visualise_connectivity(S, fname):
	'''
	Basic visualization code for connectivity, e.g. lateral inhibitory
	connections (where relevant)
	'''
	Ns = len(S.source)
	Nt = len(S.target)
	figure(figsize=(10, 4))
	subplot(121)
	plot(zeros(Ns), arange(Ns), 'ok', ms=10)
	plot(ones(Nt), arange(Nt), 'ok', ms=10)
	for i, j in zip(S.i, S.j):
		plot([0, 1], [i, j], '-k')
	xticks([0, 1], ['Source', 'Target'])
	ylabel('Neuron index')
	xlim(-0.1, 1.1)
	ylim(-1, max(Ns, Nt))
	subplot(122)
	plot(S.i, S.j, 'ok')
	xlim(-1, Ns)
	ylim(-1, Nt)
	xlabel('Source neuron index')
	ylabel('Target neuron index')
	savefig("figures/" + fname + ".png", dpi=300)
	clf()


def visualise_connectivity(S, fname, spike_pair_differences, 
						   integration_window=0,
						   plot_weights_bool=False):
	'''
	More advanced visualization code intended for feed-forward 
	excitatory synapses; also checks for e.g. delay-line alignments
	'''
	Ns = len(S.source)
	Nt = len(S.target)
	figure(figsize=(10, 4))
	subplot(111)
	yvals = arange(Ns)
	# print(arange(Ns))
	color_list = ["crimson", "gold", "dodgerblue", "k", "darkmagenta",
		"forestgreen", "darkorange", "aqua", "hotpink", "greenyellow"]
	# print(yvals)
	#exit()
	# print(zip(S.i, S.j))
	# print(S.i)
	# print(S.j)
	# exit()

	alignment_results = {
	# Track the post-synaptic neurons associated with different alignments
	"upright_aligned_indices":[],
	"inverted_aligned_indices":[],
	"both_aligned_indices":[],
	"non_aligned_indices":[],
	# Track the specific weight indices associated with different alignments
	"upright_aligned_weights":[],
	"inverted_aligned_weights":[],
	"both_aligned_weights":[],
	"non_aligned_weights":[]
	}



	for jj in range(Nt):
	# Create a new plot for each post-synaptic neuron

		spike_pair_delays_dic = {}  # Track the delay lines associated with each pair of neurons
		spike_pair_indices_dic = {}  # Track the connection indices associated with each pair of neurons
		
		figure(figsize=(8, 6))
		file_append = ''

		for ii in range(Ns):

			# print("\nii")
			# print(ii)
			# print("Post neuron: " + str(jj))
			# print(ii*Ns+jj)
			delay_temp = round(S.delay[ii*Ns+jj]/ms,2)

			# print("Delay temp")
			# print(delay_temp)

			# Track which spike-pair is currently being visualized
			spike_pair_iter = int(ii%(Ns/2))

			# print("Spike pair iter and dic")
			# print(spike_pair_iter)
			# print(spike_pair_delays_dic)
			try:
				spike_pair_delays_dic[spike_pair_iter].append(delay_temp)
				spike_pair_indices_dic[spike_pair_iter].append(ii*Ns+jj)

			except:
				spike_pair_delays_dic[spike_pair_iter] = [delay_temp]
				spike_pair_indices_dic[spike_pair_iter] = [ii*Ns+jj]

			# if len(spike_pair_delays_dic[spike_pair_iter]) == 1:
			# 	spike_pair_delays_dic[spike_pair_delays_dic].append(delay_temp)

			# else:
			# 	# Initialize the list tracking the delay values associated
			# 	# with each spike pair
			# 	spike_pair_delays_dic[spike_pair_iter] = [delay_temp]


			# After both values of the spike pair delays have been
			# found, determine the relative differences, and compare them
			# to those associated with the input
			if len(spike_pair_delays_dic[spike_pair_iter]) == 2:
				# NB we compare for a delay/input match to within 1 ms at the minimum
				# Also note that we take the difference in the opposite order from spike_pair_differences,
				# as an alignment at the post-synaptic neuron implies that the delays are making up for 
				# any difference in pre-synaptic firing
				delay_difference = round(spike_pair_delays_dic[spike_pair_iter][0] - spike_pair_delays_dic[spike_pair_iter][1],0)
				# print("Delay difference:")
				# print(delay_difference)

				upright_t_diff = round(spike_pair_differences["t"][spike_pair_iter],0)
				inverted_t_diff = round(spike_pair_differences["inverted_t"][spike_pair_iter],0)


				# print("Comparison spike time differences for t and inverted t")
				# print(upright_t_diff)
				# print(inverted_t_diff)

				# For t stimulus; NB that the spike_pair_differences are also indexed by the
				# particular spike pair of interest

				# print("\nDelay difference:")
				# print(delay_difference)
				# print("Upright range:")
				# print(upright_t_diff-integration_window)
				# print(upright_t_diff+integration_window)
				# print("Inverted range:")
				# print(inverted_t_diff-integration_window)
				# print(inverted_t_diff+integration_window)

				# With a large integration window, may align with both inputs
				if (((delay_difference >= upright_t_diff-integration_window)
					and (delay_difference <= upright_t_diff+integration_window)) and 
					((delay_difference >= inverted_t_diff-integration_window)
					and (delay_difference <= inverted_t_diff+integration_window))):
					current_linestyle = "dashdot"
					file_append = "_BOTH_alignment"
					alignment_results["both_aligned_weights"].extend(spike_pair_indices_dic[spike_pair_iter])

				elif ((delay_difference >= upright_t_diff-integration_window)
					and (delay_difference <= upright_t_diff+integration_window)):

					# print("Alignment with T stimulus")
					current_linestyle = "dashed"
					# Keep track of neurons that align with both stimuli
					if file_append == "_BOTH_alignment":
						alignment_results["both_aligned_weights"].extend(spike_pair_indices_dic[spike_pair_iter])
					elif file_append == "_invert_t_alignment":
						file_append = "_BOTH_alignment"
						alignment_results["both_aligned_weights"].extend(spike_pair_indices_dic[spike_pair_iter])
					else:
						file_append = "_upright_t_alignment"
						alignment_results["upright_aligned_weights"].extend(spike_pair_indices_dic[spike_pair_iter])

				elif ((delay_difference >= inverted_t_diff-integration_window)
					and (delay_difference <= inverted_t_diff+integration_window)):

					# print("Alignment with inverted t")
					current_linestyle = "dotted"
					if file_append == "_BOTH_alignment":
						alignment_results["both_aligned_weights"].extend(spike_pair_indices_dic[spike_pair_iter])
					elif file_append == "_upright_t_alignment":
						file_append = "_BOTH_alignment"
						alignment_results["both_aligned_weights"].extend(spike_pair_indices_dic[spike_pair_iter])
					else:
						file_append = "_invert_t_alignment"
						alignment_results["inverted_aligned_weights"].extend(spike_pair_indices_dic[spike_pair_iter])

				else:
					current_linestyle = "-"
					alignment_results["non_aligned_weights"].extend(spike_pair_indices_dic[spike_pair_iter])
				# 	legend_append = ''

					# if file_append == '':
					# 	pass


			else:
				current_linestyle = "-"
				# file_append = ''
				# legend_append = ''

			# if delay_temp == 1:
			# 	current_linestyle = "dashed"
			# elif delay_temp == 9:
			# 	current_linestyle = "dotted"
			# else:
			# 	current_linestyle = "-"

			if plot_weights_bool:
				width_determinent = round(S.w[ii*Ns+jj],2)
			else:
				width_determinent = delay_temp

			plot([0, 1], [ii, jj], linestyle=current_linestyle, linewidth=0.1+width_determinent,
				 alpha=0.5, label=str(width_determinent),
				 color=color_list[spike_pair_iter])

		# print("append status")
		# print(file_append)

		# Determine the status of the neuron's input alignments
		if file_append == "_upright_t_alignment":
			alignment_results["upright_aligned_indices"].append(jj)

		elif file_append == "_invert_t_alignment":
			alignment_results["inverted_aligned_indices"].append(jj)

		elif file_append == "_BOTH_alignment":
			alignment_results["both_aligned_indices"].append(jj)

		else:
			alignment_results["non_aligned_indices"].append(jj)


		# print("indices")
		# print(aligned_indices)
		# print(non_aligned_indices)

		# exit()


		plot(zeros(Ns), yvals, 'ok', ms=10)
		plot(1, jj, 'ok', ms=10)
		xticks([0, 1], ['Source', 'Target'])
		ylabel('Neuron index')
		xlim(-0.1, 1.1)
		ylim(-1, max(Ns, Nt))
		legend()
		# subplot(122)
		# plot(S.i, S.j, 'ok', alpha=0.05)
		# xlim(-1, Ns)
		# ylim(-1, Nt)
		# xlabel('Source neuron index')
		# ylabel('Target neuron index')
		if plot_weights_bool:
			savefig("figures/" + fname + "_weights_post_index_" + str(jj) + file_append + ".png", dpi=300)
		else:
			savefig("figures/" + fname + "_delays_post_index_" + str(jj) + file_append + ".png", dpi=300)
		clf()
		#exit()

	alignment_results["num_single_aligned"] = (len(alignment_results["upright_aligned_indices"])
		  		+ len(alignment_results["inverted_aligned_indices"]))
	alignment_results["num_both_aligned"] = len(alignment_results["both_aligned_indices"])
	alignment_results["num_non_aligned"] = len(alignment_results["non_aligned_indices"])

	assert (len(alignment_results["upright_aligned_indices"])
	  		+ len(alignment_results["inverted_aligned_indices"])
	  		+ len(alignment_results["both_aligned_indices"])
	  		+ len(alignment_results["non_aligned_indices"])) == Nt, "Number of sorted indices should match number of output neurons"

	return alignment_results


def main_run(stimuli_params, network_params, run_params, seed_iter, spike_pair_differences,
			 drift_name='', initialize_weights_bool=False):

	# =============================================================================
	# INITIAL SETUP       
	# =============================================================================

	print("\n\n\n*New Simulation*")
	defaultclock.dt = network_params["time_step"]*ms

	start_scope() # Exclude any Brian objects (e.g. neuron groups) created before this call

	# Calculate how long to present for, even if inputting Poisson activity
	if run_params["STDP_on_bool"]:
		simulation_duration = (stimuli_params['number_of_train_presentations']
			 				   * stimuli_params['duration_of_presentations']
			 				   * len(stimuli_params['stimuli_names']))

	else:
		simulation_duration = (stimuli_params['number_of_eval_presentations']
			 				   * stimuli_params['duration_of_presentations']
			 				   * len(stimuli_params['stimuli_names']))


	print("Run parameters:")
	pprint.pprint(run_params, depth=1)

	print("\nSimulation duration : " + str(simulation_duration))



	# =============================================================================
	# NEURON GROUPS       
	# =============================================================================

	input_layer_width = stimuli_params['input_layer_size']
	output_layer_width = stimuli_params['output_layer_size']
	inhibitory_layer_width = round(output_layer_width/4)

	# Define the input neurons
	if run_params["input_stim"] is not None:
		input_neurons = SpikeGeneratorGroup(input_layer_width, run_params["input_stim"][0], run_params["input_stim"][1]*ms)

	else: 
		input_neurons = PoissonGroup(input_layer_width, rates=stimuli_params["poisson_input_rate"]*Hz)

	# ==== LIF NEURON PARAMETERS ====
	# Parameters with ** indicate the value is as used in Troyer et al 1998 for fast spiking neurons
	# NB that fast spiking cells were associated with stellate cells in McCormick 1985, a common
	# receiving neuron in L4 

	#** stellate cell parameters **
	E_l = -81.6*mV         # v_rest **
	g_l = 18*nS            # leak conductance 18**
	E_e = 0*mV          # excitatory synaptic reversal potential **
	E_i = -70*mV          # inhibitory synaptic reversal potential **
	C_m = 0.214*nF         # membrane capacitance; 214**; with 180, the effective tau-membrane is 10ms, as used in Masquelier 2009
	tau_e = 1.75*ms         # excitatory synaptic time constant; NB that beyond Hausser 1997, Troyer 1998 appear to use a value of 1.75 --> given that I'm not modelling the rise, 
	                        #   does it make sense to use a slightly smalelr value (e.g. 1.7 rather than 1.75 here, 5 rather than 5.25 for tau_i?)
	tau_i = 5.25*ms           # inhibitory synaptic time constant  **NB this is the value used in Troyer i.e. tau_in(fall)
	tau_r = 1.0*ms           # refractory period **
	V_th = -52.5*mV          # firing threshold **
	V_r = -57.8*mV           # reset potential **

	g_init = 10*nS  # Maximum initial synapse conductance value

	print("\nEffective tau membrane:")
	print(C_m/g_l)

	# =============================================================================
	# definitions        
	# =============================================================================

	# LIF neuron equations (only difference between excitatory and inhibitory is spatial locations)
	LIF_neurons_eqs='''
	idx = i                                                  : 1                        # store index of each neuron (for tracking synaptic connections)
	dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) + g_i*(E_i-v))/C_m    : volt (unless refractory) # membrane potential (LIF equation)               
	dg_e/dt = -g_e/tau_e                                     : siemens                  # post-synaptic exc. conductance (incremented when excitatory spike arrives at neuron - see synapse equations)
	dg_i/dt = -g_i/tau_i                                     : siemens                  # post-synaptic inh. conductance (incremented when inhibitory spike arrives at neuron - see synapse equations)
	'''

	# Define output layer; NB I did attempt to refactor this but there were issues around Brian's name-space scoping
	output_neurons = NeuronGroup(output_layer_width, LIF_neurons_eqs, threshold='v > V_th', reset='v = V_r', refractory='tau_r', method='euler')

	# Set the initial values
	assert output_neurons.g_e[0]==0, "Before initialization, conductance should be 0"
	output_neurons.v = 'rand()*(V_r-E_l)+E_l'  # random initial membrane potentials
	output_neurons.g_e = 'rand()*g_init'        # random initial excitatory conductances
	output_neurons.g_i = 'rand()*g_init'    # random initial inhibitory conductances; 0 for inhibitory neurons

	background_w = network_params["background_weight"] # background weights; not plastic

	# Define background neurons layer
	# Use the efficient Poisson background input described at :
	# https://brian2.readthedocs.io/en/stable/user/input.html?highlight=poisson#poisson-inputs
	# NB there is no number of synapses; instead, each neuron receives the number of *independent*
	# poisson spike trains equvialent to network_params["background_layer_width"]; thus if one wanted
	# 100k poisson neurons with 0.1 probability of connection (i.e. 10k synapses), then set
	# network_params["background_layer_width"]=10,000
	background_input = PoissonInput(output_neurons, 'g_e',
									network_params["background_layer_width"],
									network_params["background_base_rate"]*Hz,
									weight="rand()*background_w*g_l")

	# Define inhibitory neurons
	inhibitory_neurons = NeuronGroup(inhibitory_layer_width, LIF_neurons_eqs,
			threshold='v > V_th', reset='v = V_r', refractory='tau_r', method='euler')

	# Set the initial values
	inhibitory_neurons.v = 'E_l'  # random initial membrane potentials
	inhibitory_neurons.g_e = 'rand()*g_init'        # random initial excitatory conductances
	inhibitory_neurons.g_i = 0    # 0 for inhibitory neurons

	# Background to the inhibitory neurons
	inh_background_input = PoissonInput(inhibitory_neurons, 'g_e',
							network_params["background_layer_width"],
							network_params["background_base_rate"]*Hz,
							weight="rand()*background_w*g_l")



	# =============================================================================
	# SYNAPSES       
	# =============================================================================

	# synaptic parameters
	taupre = network_params["taupre"]*ms
	taupost = network_params["taupost"]*ms
	scaling_constant = g_l  # Biological scaling constant that ensures the weights are in the correct unit, and that synaptic conductance is modified relative to membrane conductance

	wmax = network_params["wmax"] # max upper (and intial) weight; NB that because scaling_constant = g_l (i.e. in nS unit), these weights are defined unitless

	if run_params["STDP_on_bool"]:
		Apre = network_params["Apre"]
	else:
		Apre = 0

	Apost = -Apre*network_params["Apost_scaling"]

	Apre *= wmax
	Apost *= wmax

	print("\nSimulation Apre and Apost values")
	print("Apre: " + str(Apre))
	print("Apost: " + str(Apost))


	# ===== FEED-FORWARD SYNAPSES =====

	# Define feed-forward connections, which have STDP learning
	S_feedforward = Synapses(input_neurons, output_neurons,
	             '''
	             w : 1
	             dapre/dt = -apre/taupre : 1 (event-driven)
	             dapost/dt = -apost/taupost : 1 (event-driven)
	             ''',
	             on_pre='''
	             g_e_post += scaling_constant*w
	             apre += Apre
	             w = clip(w+apost, 0, wmax)
	             ''',
	             on_post='''
	             apost += Apost
	             w = clip(w+apre, 0, wmax)
	             ''')

	S_feedforward.connect(p=network_params["primary_connection_prob"])

	# ===== FEED-FORWARD WEIGHTS AND DELAYS =====

	# Distribution of delays
	delay_upper = network_params["delay_upper"]
	small_timestep = 0.000001 #Helps avoid numerical issues

	delays_list = []

	#print("\n===TEMPORARILY INITIALIZING AT MAX WEIGHTS ** ===")

	if initialize_weights_bool:
		S_feedforward.w = 'rand()*(wmax-wmax/2)+wmax/2' # 'wmax'

		np.savetxt("weights/" + str(seed_iter) + "/rand_weights.txt", S_feedforward.w)

		#print("\n===TEMPORARILY USING DISCRETE RANDOM DELAYS ** SKIPPING 1ms INTERVALS ** ===")
		
		for ii in range(len(S_feedforward.w)):
			#Discrete delay np.random.choice((0, delay_upper)) + np.random.uniform(0,0.1)
			setting_val = np.random.uniform(0, delay_upper) + small_timestep
			#setting_val = np.random.choice(range(0, delay_upper+1, 2)) + small_timestep
			delays_list.append(setting_val)


		S_feedforward.delay = delays_list*ms
		np.savetxt("weights/" + str(seed_iter) + "/rand_delays.txt", S_feedforward.delay/ms)

	else:
		S_feedforward.w = np.fromfile(run_params["weight_file"] + "_weights.txt", sep="\n")
		S_feedforward.delay = np.fromfile(run_params["weight_file"] + "_delays.txt", sep="\n")*ms

	# VISUALIZE AND ANALYSE feed-forward connectivity

	# Visualize delay-line alignments
	alignment_results = visualise_connectivity(S_feedforward,
						   fname=str(seed_iter) + run_params["output_dir"] + "/ff_connectivity",
						   spike_pair_differences=spike_pair_differences,
						   integration_window=network_params["estimated_integration_window"])

	print("\nAlignment results:")
	pprint.pprint(alignment_results, depth=1)

	with open("raw_data/" + str(seed_iter) + run_params["output_dir"] + "/alignment_results.json", 'w') as f:
		json.dump(alignment_results, f)


	# Visualize weights associated with ff-connectivity
	visualise_connectivity(S_feedforward,
						   fname=str(seed_iter) + run_params["output_dir"] + "/ff_connectivity",
						   spike_pair_differences=spike_pair_differences,
						   integration_window=network_params["estimated_integration_window"],
						   plot_weights_bool=True)


	# ===== FEED-FORWARD MONOSYNAPTIC INHIBITORY SYNAPSES =====

	S_input_ff_inh = Synapses(input_neurons, output_neurons,                         '''
	                        w : 1
	                        ''',
	                        on_pre='''g_i_post += scaling_constant*w''')
	S_input_ff_inh.connect(p=network_params["primary_connection_prob"])
	S_input_ff_inh.w = wmax*(9/3)

	#print("=== TEMPORARILY DISABLING FF INHIBITION ===")

	S_input_ff_inh.delay = S_feedforward.delay + network_params["ff_inhibition_delay"]*ms


	# ===== COMPETITIVE, LATERAL INHIBITORY SYNAPSES =====

	S_exc_to_inh = Synapses(output_neurons, inhibitory_neurons,                         '''
	                        w : 1
	                        ''',
	                        on_pre='''g_e_post += scaling_constant*w''',
	                        delay=network_params["lateral_delay"]*ms)
	S_exc_to_inh.connect(p=network_params["inh_connection_prob"])
	S_exc_to_inh.w = wmax*network_params["exc_to_inh_scaling"]


	S_inh_to_exc = Synapses(inhibitory_neurons, output_neurons,                         '''
	                        w : 1
	                        ''',
	                        on_pre='''g_i_post += scaling_constant*w''',
	                        delay=network_params["lateral_delay"]*ms) 
	S_inh_to_exc.connect(p=network_params["inh_connection_prob"])
	S_inh_to_exc.w = wmax*network_params["exc_to_inh_scaling"]

	basic_visualise_connectivity(S_exc_to_inh, fname=str(seed_iter) + run_params["output_dir"] + "/exc_to_inh_connectivity")
	basic_visualise_connectivity(S_inh_to_exc, fname=str(seed_iter) + run_params["output_dir"] + "/inh_to_exc_connectivity")



	# =============================================================================
	# MONITORS       
	# =============================================================================

	# Initialize monitors
	input_neurons_spikes_M = SpikeMonitor(input_neurons)
	output_neurons_spikes_M = SpikeMonitor(output_neurons)
	inhibitory_neurons_spikes_M = SpikeMonitor(inhibitory_neurons)

	assert np.all(output_neurons_spikes_M.count==0) # Spike count for new monitor should be zero

	# When running for the (shorter) non-training periods, monitor neurons
	# This is avoided during training as it is quite memory intensive
	if not run_params["STDP_on_bool"]:

		output_neurons_vars_M = StateMonitor(
			output_neurons, ('v', 'g_e', 'g_i'), record=list(range(output_layer_width)))


	# =============================================================================
	# RUN SIMULATION       
	# =============================================================================

	pre_time = time.time()
	print("\nStarting simulation run...")
	prev_count = 0 # Keep a running track of new out-put layer spikes to determine the 
	# firing rate as a function of training epoch

	# When performing learning, periodically save the output weights
	if run_params["STDP_on_bool"]:

		assert (stimuli_params["number_of_train_presentations"]
				% stimuli_params["num_intervals_for_weight_saving"] == 0), "Num training presentations must be divisible by number of weight-save intervals"

		elapsed_time = 0
		duration_of_section = simulation_duration/stimuli_params["num_intervals_for_weight_saving"]

		assert (duration_of_section % (stimuli_params['duration_of_presentations']
 				   * len(stimuli_params['stimuli_names'])) == 0), "Section duration must not split stimulus presentations"

		learning_fr_list = []
		learning_weights_list = []
		epoch_markers = []

		for learning_iter in range(stimuli_params["num_intervals_for_weight_saving"] ):

			run(duration_of_section*ms)
			elapsed_time += duration_of_section
			epoch_markers.append(elapsed_time/1000)

			np.savetxt("weights/" + str(seed_iter) + run_params["output_dir"] + "/weights"
					   + str(elapsed_time/1000) + "_seconds_of_sim.txt",
				   	   S_feedforward.w)

			# As deepcopy() does not work, iteratively create a copy of the original weights to 
			# ensure STDP behaving as expected
			current_weights = [weight_iter for weight_iter in S_feedforward.w]
			learning_weights_list.append(current_weights)

			# print("\nPrevious count")
			# print(prev_count)
			# print("Unsubtracted count")
			# print(output_neurons_spikes_M.count)
			# print("Adjusted count and rate")
			# print(output_neurons_spikes_M.count-prev_count)
			# print((output_neurons_spikes_M.count-prev_count)/(duration_of_section/1000))
			learning_fr_list.append((output_neurons_spikes_M.count-prev_count)/(duration_of_section/1000))

			np.savetxt("raw_data/" + str(seed_iter) + run_params["output_dir"] + "/rates"
					   + str(elapsed_time/1000) + "_seconds_of_sim.txt",
				   	   (output_neurons_spikes_M.count-prev_count)/(duration_of_section/1000))
			
			# As deepcopy() does not work, iteratively create a copy of the previous spike counts
			prev_count = [count_iter for count_iter in output_neurons_spikes_M.count]

		assert elapsed_time==simulation_duration, "Cumulative run-time should equal total defined run-time"

		visualise_violins_over_epochs(learning_fr_list, epoch_markers,
			fig_title="violins_FR_over_learning",
			fname=str(seed_iter) + run_params["output_dir"],
			y_label="Firing Rate (Hz)")

		visualise_violins_over_epochs(learning_weights_list, epoch_markers,
			fig_title="violins_weights_over_learning",
			fname=str(seed_iter) + run_params["output_dir"],
			y_label="Synapse Weights")

	else:

		# As deepcopy() does not work, iteratively create a copy of the original weights to 
		# ensure STDP behaving as expected
		pre_weights_copy = [pre_weight for pre_weight in S_feedforward.w]

		run(simulation_duration*ms)

		# Check weights static if STDP was inactive
		num_weights_to_check = 5
		for ii in range(5):
			assert pre_weights_copy[ii] == S_feedforward.w[ii], "\nWeights do not match despite no STDP! \nPre: " + str(pre_weights_copy[ii]) + "\nPost: " + str(S_feedforward.w[ii])

	print("\nMinutes to run the simulation:")
	print(round((time.time()-pre_time)/60,2))

	# Save final weights and delays
	np.savetxt("weights/" + str(seed_iter) +  run_params["output_dir"] + "/final_weights.txt", S_feedforward.w)
	np.savetxt("weights/" + str(seed_iter) +  run_params["output_dir"] + "/final_delays.txt", S_feedforward.delay/ms)



	# =============================================================================
	# PLOT RESULTS       
	# =============================================================================

	# Plot various results and save the outputs for later analysis


	# == Spike raster plots == 

	input_drive_visualization(input_neurons_spikes_M, output_neurons_spikes_M, fig_title="Simulation Beginning",
					 fname=str(seed_iter) + run_params["output_dir"] + "/raster_inputs_and_output_layer")

	visualise_raster(input_neurons_spikes_M, fig_title="Simulation Beginning",
					 fname=str(seed_iter) + run_params["output_dir"] + "/raster_input_layer")

	visualise_raster(output_neurons_spikes_M, fig_title="Simulation Beginning",
					 fname=str(seed_iter) + run_params["output_dir"] + "/raster_output_layer")



	# == Histograms of weights, delays and firing rates == 

	visualise_histo(S_feedforward.w, "Weights", "Feed Forward Synapses",
					fname=str(seed_iter) + run_params["output_dir"] + "/ff_weights_final", num_bins=20)

	visualise_histo(S_feedforward.delay, "Delays", "Feed Forward Delays",
					fname=str(seed_iter) + run_params["output_dir"] + "/ff_delays", num_bins=20)

	visualise_histo(input_neurons_spikes_M.count/(simulation_duration/1000), "Firing Rate", "Input Layer",
					fname=str(seed_iter) + run_params["output_dir"] + "/fr_input_layer", num_bins=10)

	visualise_histo(output_neurons_spikes_M.count/(simulation_duration/1000), "Firing Rate", "Output Layer",
					fname=str(seed_iter) + run_params["output_dir"] + "/fr_output_layer", num_bins=10)

	visualise_histo(inhibitory_neurons_spikes_M.count/(simulation_duration/1000), "Firing Rate", "Inhibitory Layer",
					fname=str(seed_iter) + run_params["output_dir"] + "/fr_inhibitory_layer", num_bins=10)


	# When running shorter, non-training simulations, visualize cellular properties
	if not run_params["STDP_on_bool"]:

		# Sample which output neurons to plot based on alignment analysis, maximum of 2 each
		upright_aligned_record_IDs = alignment_results["upright_aligned_indices"][
			:min(2,len(alignment_results["upright_aligned_indices"]))]
		inverted_aligned_record_IDs = alignment_results["inverted_aligned_indices"][
			:min(2,len(alignment_results["inverted_aligned_indices"]))]
		both_aligned_record_IDs = alignment_results["both_aligned_indices"][
			:min(2,len(alignment_results["both_aligned_indices"]))]
		nonaligned_record_IDs = alignment_results["non_aligned_indices"][
			:min(2,len(alignment_results["non_aligned_indices"]))]

		# == Cellular variables over time == 

		array_of_cellular_properties(output_neurons_vars_M,
				output_neurons_spikes_M, neuron_IDs=upright_aligned_record_IDs, title="output_layer_upright_aligned",
				fname=str(seed_iter) + run_params["output_dir"],
				v_threshold=(V_th/mV)/1000)

		array_of_cellular_properties(output_neurons_vars_M,
				output_neurons_spikes_M, neuron_IDs=inverted_aligned_record_IDs, title="output_layer_inverted_aligned",
				fname=str(seed_iter) + run_params["output_dir"],
				v_threshold=(V_th/mV)/1000)

		array_of_cellular_properties(output_neurons_vars_M,
				output_neurons_spikes_M, neuron_IDs=both_aligned_record_IDs, title="output_layer_both_aligned",
				fname=str(seed_iter) + run_params["output_dir"],
				v_threshold=(V_th/mV)/1000)

		array_of_cellular_properties(output_neurons_vars_M,
				output_neurons_spikes_M, neuron_IDs=nonaligned_record_IDs, title="output_layer_nonaligned",
				fname=str(seed_iter) + run_params["output_dir"],
				v_threshold=(V_th/mV)/1000)


		# == Distributions of cellular variables ==

		visualise_cellular_distribution(output_neurons_vars_M, "v",
								fig_title="v_distribution_output_layer",
								fname=str(seed_iter) + run_params["output_dir"],
								vertical_markers={"threshold":V_th,
									"rest":E_l},
								units=mV)

		visualise_cellular_distribution(output_neurons_vars_M, "g_e",
								fig_title="g_e_distribution_output_layer",
								fname=str(seed_iter) + run_params["output_dir"],
								units=nS)
