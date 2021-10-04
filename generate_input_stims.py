from brian2 import *
import numpy as np
import random
import copy
import math
from PIL import Image


def sample_positions(current_stimulus, params, eval_bool, position_iter):

    # Possible positions of the stimulus, determined by the position of the vertical bar
    # NB the extreme position of the vertical bar is constrained by the ultimate positions of the
    # horizontal bar (i.e. stimulus needs to remain int he receptive field)
    positions = [0 - params["relative_distances_dic"]["t"], params["x_rf_range"]/2,
                                params["x_rf_range"] - params["relative_distances_dic"]["inverted_t"]]

    if eval_bool:
        # Iterate through each possible training position in blocks; this is to simplify information theory
        # analysis later
        vertical_pos = positions[position_iter]

    else:
        # Randomly sample the position of the vertical bar, constrained by both bars needing to be in the receptive field
        # NB that since we eventually want to test the ability to perform interpolation, only three possible locations are sampled from during training
        vertical_pos = np.random.choice(positions)


    # ** Not Implemented : for testing novel (e.g. interpolated) locations later **
    # vertical_pos = np.random.uniform(0, x_rf_range - relative_distances_dic["inverted_t"])


    # Depending on the stimulus (T or inverted_t), find the position of the horizontal bar
    horizontal_pos = vertical_pos + params["relative_distances_dic"][current_stimulus]

    return vertical_pos, horizontal_pos


def add_vertical_bar(image_canvas, vertical_offset):

    image_canvas[0+int(64*(vertical_offset-0.25)):int(64/2)+int(64*(vertical_offset-0.25)),
                 int(64/2-3):int(64/2+3)] = 1.0

    return image_canvas


def add_horizontal_bar(image_canvas, horizontal_offset):
    
    image_canvas[int(64*(horizontal_offset)):6+int(64*(horizontal_offset)),
                 int(64/2-64/4):int(64/2+64/4)] = 1.0

    return image_canvas


def visualize_stimulus(presentation_iter, vertical_offset, horizontal_offset, current_stimulus, seed_iter, eval_bool):

    image_canvas = np.zeros((64,64))

    image_canvas = add_vertical_bar(image_canvas, vertical_offset)
    image_canvas = add_horizontal_bar(image_canvas, horizontal_offset)

    imsave("figures/" + str(seed_iter) + "/input_stimuli_drift_iter_NA/eval_" + str(eval_bool) + "_presentation_"
            + str(presentation_iter) + "_" + current_stimulus + ".png", image_canvas)


def setup_inputs(params, neuron_drift_coefs_dic):

    initial_times_dic = {"vertical":np.random.uniform(-params["initial_times_limit"],params["initial_times_limit"],int(params["input_layer_size"]/2)),
                             "horizontal":np.random.uniform(-params["initial_times_limit"],params["initial_times_limit"],int(params["input_layer_size"]/2))}

    relative_times_vertical = simulate_diffusion(neuron_drift_coefs_dic["vertical"],
                                                 params["diffusion_coef"], initial_times_dic["vertical"],
                                                 params["spatial_step"], params["x_rf_range"])

    if params["correlated_neuron_pairs_bool"]:
        neuron_drift_coefs_dic["horizontal"] = neuron_drift_coefs_dic["vertical"]
    else:
        neuron_drift_coefs_dic["horizontal"] = neuron_drift_coefs_dic["horizontal"]

    # NB even with 'fixed'/correlated pairs of neurons, the initial relative spike times are always sampled independently
    relative_times_horizontal = simulate_diffusion(neuron_drift_coefs_dic["horizontal"], params["diffusion_coef"],
                                                    initial_times_dic["horizontal"],
                                                    params["spatial_step"], params["x_rf_range"])

    return relative_times_vertical, relative_times_horizontal, neuron_drift_coefs_dic


def simulate_diffusion(drift_coefs, diffusion_coef, initial_times, spatial_step, x_rf_range):
    '''
    drift_coefs : list determining drift in the Brownian simulation; NB these don't have to be the same values for each neuron
    initial_times : array corresponding to the spike time at the upper-most extreme of the receptive field
    spatial_step : the spatial resolution of the steps used when generating the diffusion process
    '''

    relative_times = np.zeros(shape=(len(initial_times), int(x_rf_range/spatial_step)+1))

    relative_times[:,0] = initial_times

    for ii in range(1, int(x_rf_range/spatial_step)+1): # Add index because I would like to sample all the way to RF end
        # Brownian motion/Wiener Process with drift implemented as described in OU course book 5, page 95
        # Note that the mean and SD are modified in order to account for the size of the spatial-step (equivalent to time dt in 
        # Brownian motion)
        relative_times[:,ii] = (relative_times[:, ii-1] +
                                np.random.normal(loc=drift_coefs*spatial_step,
                                                 scale=diffusion_coef*sqrt(spatial_step), size=(len(initial_times))))

    return relative_times


def visualize_spike_slopes(num_neurons_to_visualize, params, relative_times_vertical,
                           relative_times_horizontal, neuron_drift_coefs_dic, seed_iter):

    fig, axis_list = plt.subplots(1, max(num_neurons_to_visualize,2)) # The max here
    # ensures a list is still created if num_neurons_to_visualize=1

    for sample_neuron in range(num_neurons_to_visualize):

        # Vertical bar neurons
        axis_list[sample_neuron].plot(np.arange(0, 1 + params["spatial_step"], params["spatial_step"]),
             relative_times_vertical[sample_neuron,:],
             label="Vertical_" + str(sample_neuron) + "; Drift : " + str(round(neuron_drift_coefs_dic["vertical"][sample_neuron], 2)),
             color='dodgerblue', alpha=0.5, linestyle='solid')
        # for current_loc in spatial_samples:
        #     axis_list[sample_neuron].scatter(current_loc,
     #                relative_times_vertical[sample_neuron,int(current_loc/params["spatial_step"])],
        #             color='dodgerblue', alpha=0.5, marker='o')
        
        # Horizontal bar neurons
        axis_list[sample_neuron].plot(np.arange(0, 1 + params["spatial_step"], params["spatial_step"]),
             relative_times_horizontal[sample_neuron,:],
             label="Horizontal_" + str(sample_neuron) + "; Drift : " + str(round(neuron_drift_coefs_dic["horizontal"][sample_neuron], 2)),
             color='crimson', alpha=0.5, linestyle='dashed')
        # for current_loc in spatial_samples:
        #     axis_list[sample_neuron].scatter(current_loc+params["relative_distances_dic"]['inverted_t'], relative_times_horizontal[sample_neuron,int((current_loc+relative_distances_dic['inverted_t'])/params["spatial_step"])],
        #             color="crimson", alpha=0.5, marker='x')

        # Difference
        axis_list[sample_neuron].plot(np.arange(0, 1 + params["spatial_step"], params["spatial_step"]),
             relative_times_vertical[sample_neuron,:] - relative_times_horizontal[sample_neuron,:],
             label="Difference_" + str(sample_neuron),
             color="k", alpha=0.5, linestyle='-.')

    for ax in axis_list:
        ax.set(xlabel='Position in Recepive Field')
        ax.legend(prop={'size': 12})
        ax.set(ylim=(-35,35))

    axis_list[0].set(ylabel='Relative Spike Time (ms)')

    #fig.set_size_inches(18.5, 10.5)
    savefig("figures/" + str(seed_iter) + "spike_slope_samples", dpi=300)
    clf()



# For each assembly, generate the spikes that occur within the given presentation window
def generate_window_spikes(params, num_assemblies, presentation_iter, assembly_IDs,
                           assembly_times, current_stimulus, random_iter, eval_bool):

    temp_spike_times = []
    temp_spike_IDs = []

    # Jitter is always individual to each neuron, although it is not needed on truly random
    # stimulus presentations
    if not random_iter:
        jitter_array = np.random.normal(0, params['jitter_std'], size=np.shape(assembly_times))
    if random_iter:
        jitter_array = np.zeros(shape=np.shape(assembly_times))

    # Determine offset of the stimulus in the whole simulation
    # Multiplying presentation_iter by up to 4 accounts for the two different stimuli, plus the alternating random presentations
    # (when doing training as opposed to evaluation)
    # The saccade offset ensures that after the stimulus has been presented, there is a period of time
    # for the receiving neurons to integrate the input --> otherwise activity resulting from one stimulus window
    # might incorrectly be thought to have been a result of the subsequent stimulus presentation
    window_offset = ((presentation_iter*len(params["stimuli_names"])*(1+int(not eval_bool))
                     + (1+int(not eval_bool))*int(current_stimulus=='inverted_t') + random_iter)
                     * params['duration_of_presentations'])

    # print("\npresentation_iter")
    # print(presentation_iter)
    # print("current_stimulus")
    # print(current_stimulus)
    # print("random_iter")
    # print(random_iter)
    # print("no eval")
    # print(not eval_bool)
    # print("Window offset")
    # print(window_offset)

    # Spike pairs
    if params["correlated_neuron_pairs_bool"]:

        # Find earliest and latest spike times for each spike *pair*, accounting for jitter
        max_time = np.amax(assembly_times+jitter_array, axis=1) + params["duration_of_saccades"]
        min_time = np.amin(assembly_times+jitter_array, axis=1)

        # Create the uniform offsets that are shared by each spike pair, i.e. this is not a single
        # offset shared ainverted_t all neurons
        uniform_offset = np.random.uniform(0 - min_time, params['duration_of_presentations'] - max_time)

    # Synchronous waves/pulse packets
    else:

        if bool(random_iter):
            # Generate a random wave-like packet of activity in place of true stimulus
            assembly_times = np.random.normal(0, params["noisy_pulse_packet_std"], size=np.shape(assembly_times))

        # Find earliest and latest spike times ainverted_t *all* the assembly neurons, accounting for jitter
        max_time = np.amax(np.amax(assembly_times+jitter_array, axis=0), axis=0) + params["duration_of_saccades"]
        min_time = np.amin(np.amin(assembly_times+jitter_array, axis=0), axis=0)

        # Create a single spike-wave offset shared ainverted_t all neurons
        uniform_offset = np.random.uniform(0 - min_time, params['duration_of_presentations'] - max_time, size=1)

    #Iterate through each assembly
    for assembly_iter in range(num_assemblies):

        if params["correlated_neuron_pairs_bool"] and (not bool(random_iter)):
            # Each pair of neurons shares an offset with its partner
            stimulus_offset = uniform_offset[assembly_iter] + assembly_times[assembly_iter,:]

        elif params["correlated_neuron_pairs_bool"] and bool(random_iter):
            # Each neuron's spike is sampled from a uniform distribution, i.e. no temporal structure
            uniform_offset = np.random.uniform(0, params['duration_of_presentations']-params["duration_of_saccades"],
                                               size=len(assembly_IDs[assembly_iter,:]))
            stimulus_offset = uniform_offset

        elif params["correlated_neuron_pairs_bool"] is False:
            # Generate stimulus offset when the input is a synchronous wave/pulse packet
            stimulus_offset = uniform_offset + assembly_times[assembly_iter,:]


        assembly_final_times = (stimulus_offset +
            window_offset +
            jitter_array[assembly_iter,:])

        # NB that the uniform distribution determining spike times includes the lower bound,
        # but excludes the upper bound, hence the following should hold
        assert (np.all(assembly_final_times >= window_offset)
                & np.all(assembly_final_times < window_offset+params["duration_of_presentations"])), "Spike times fall outside allowable window"


        temp_spike_times.extend(assembly_final_times)
        
        temp_spike_IDs.extend(assembly_IDs[assembly_iter,:])

    return temp_spike_IDs, temp_spike_times


def sort_spikes(temp_spike_IDs, temp_spike_times):

    sorting_indices = np.argsort(temp_spike_times)
    sorted_spike_IDs = np.take_along_axis(np.asarray(temp_spike_IDs), sorting_indices, axis=0)
    sorted_spike_times = np.take_along_axis(np.asarray(temp_spike_times), sorting_indices, axis=0)

    return sorted_spike_IDs, sorted_spike_times


# #Randomly sample and add a single spike for each neuron
# def add_random_firing(params, presentation_iter, sorted_spike_IDs, sorted_spike_times, current_stimulus):

#     for neuron_iter in range(params['input_layer_size']):

#         # Generate a spike time from a uniform random distrubtion, over the interval of interest (i.e. Poisson-like)
#         # This is off-set by the appropriate time given the current presentation
#         spike_time = (np.random.random()*params['duration_of_presentations'] +
#                       (presentation_iter*2 + int(current_stimulus=='inverted_t')) * params['duration_of_presentations'])


#         # NB not concerned with refactory period, as Poisson inputs that are the alternative do not respect this
#         # current_times = sorted_spike_times[np.where(sorted_spike_IDs==neuron_iter)]

#         # #If a randomly sampled neuron fires at the same time as a neuron that is already firing (within the refractory period), then shift the sample
#         # while np.any((current_times >= spike_time-params['refractory_duration']) & (current_times <= spike_time+params['refractory_duration'])):
#         #     spike_time = spike_time + 3*params['refractory_duration']
        
#         insertion_index = np.searchsorted(sorted_spike_times, spike_time)

#         sorted_spike_times = np.insert(sorted_spike_times, insertion_index, spike_time)
#         sorted_spike_IDs = np.insert(sorted_spike_IDs, insertion_index, neuron_iter)

#         assert len(sorted_spike_IDs) == len(sorted_spike_times), "Number of insertions unequal."

#     return sorted_spike_IDs, sorted_spike_times


def generate_spikes_fixed_pairs(params, assembly_IDs, relative_times_vertical, relative_times_horizontal,
                                seed_iter, eval_bool=False, save_images_bool=True, nonsense_cross_bool=False):

    if eval_bool:
        assert params['number_of_eval_presentations']%3==0, "Recommend number_of_presentations is divisible by 3 (corresponding translations) when evaluating"

    # Present the 'nonsense' cross on both presentations, which should 
    # not trigger significant learned upright or inverted T representations
    if nonsense_cross_bool:
        print("\nPresenting the network with the nonsense cross to assess robustness")
        stimuli_list = ["nonsense_cross", "nonsense_cross"]

    else:
        stimuli_list = params['stimuli_names']

    # Gather all spikes for the stimulus presentations
    spike_IDs = []
    spike_times = []
    spike_pair_differences = {}  # Dictionary storing example spike-time differences

    if eval_bool:
        number_of_presentations = params["number_of_eval_presentations"] 
    else:
        number_of_presentations = params["number_of_train_presentations"] 

    # Iterate through the total number of presentations
    for presentation_iter in range(number_of_presentations):

        # print("\nPresentation")
        # print(presentation_iter)

        # Alternate between the two diffferent stimuli types
        for current_stimulus in stimuli_list:

            # Randomly sample the location of the stimulus
            vertical_pos, horizontal_pos = sample_positions(current_stimulus, params,
                                                            eval_bool, position_iter=math.floor((3*presentation_iter) / 
                                                                number_of_presentations))

            # Use the location to determine the relative spike times (relative to 'internal cortical activity' --> as in Havenith et al)
            # NB this is done for all the neurons of each bar type, as the stimulus position is fixed for a given presentation period (e.g. 250ms)
            vertical_times = relative_times_vertical[:,int(vertical_pos/params["spatial_step"])]
            horizontal_times = relative_times_horizontal[:,int(horizontal_pos/params["spatial_step"])]
            # Given that the distance steps will be of a fixed interval, I directly calculate an estimate
            # of what the index will be given the x position (e.g. x / size_of_step)

            # The relative stimulus times that will be used for this presentation/translation
            assembly_times = np.concatenate((np.expand_dims(vertical_times,axis=1),
                                np.expand_dims(horizontal_times,axis=1)),axis=1)

            num_assemblies = len(assembly_times)

            # print("current_stimulus")
            # print(current_stimulus)

            spike_pair_differences[current_stimulus] = assembly_times[:,1]-assembly_times[:,0]

            # print("vertical and horizontal")
            # print(vertical_pos)
            # print(horizontal_pos)

            # print("assemblies")
            # print(assembly_times)
            # print("differences")
            # print(assembly_times[:,1]-assembly_times[:,0])

            if save_images_bool and (presentation_iter<=16):
                visualize_stimulus(presentation_iter, vertical_pos,
                                   horizontal_pos, current_stimulus,
                                   seed_iter, eval_bool)

            # Alternate presenting a stimulus with presenting random noise if doing training
            # (as opposed to evaluation of e.g. information theory)
            for random_iter in range(1+int(not eval_bool)):

                temp_spike_IDs, temp_spike_times = generate_window_spikes(params, num_assemblies, presentation_iter,
                                                                        assembly_IDs, assembly_times, current_stimulus,
                                                                        random_iter, eval_bool)

                sorted_spike_IDs, sorted_spike_times = sort_spikes(temp_spike_IDs, temp_spike_times)

                # print("Random iter")
                # print(random_iter)
                # print("Spikes:")
                # print(sorted_spike_times)
                # print(sorted_spike_IDs)

                # NB no need to sort spikes outside the loop, as random spikes are inserted at the appropriate position,
                # and presentations are constrained to their temporal window to prevent overlap
                spike_IDs.extend(sorted_spike_IDs)
                spike_times.extend(sorted_spike_times)

    # print("Spike pair differences")
    # print(spike_pair_differences)

    return np.asarray(spike_IDs), np.asarray(spike_times), spike_pair_differences


# Generating stimuli assemblies with spike times deteremined by drift
def create_underlying_spike_assemblies(stimuli_params, drift_iter, seed_iter):

    # Pair up the vertical and horizontal bar neurons
    assembly_IDs = np.concatenate((np.expand_dims(np.arange(0, stimuli_params['input_layer_size']/2), axis=1),
                                  np.expand_dims(np.arange(stimuli_params['input_layer_size']/2, stimuli_params['input_layer_size']), axis=1)), axis=1)

    # neuron_drift_coefs_dic = {"vertical":np.random.uniform(-drift_iter, drift_iter, int(stimuli_params["input_layer_size"]/2)),
    #                              "horizontal":np.random.uniform(-drift_iter, drift_iter, int(stimuli_params["input_layer_size"]/2))}

    print("\n===TEMPORARILY CREATING STIMULI WITH A FIXED UPPER AND LOWER BOUND ON DRIFT===")
    neuron_drift_coefs_dic = {"vertical":np.random.choice((-drift_iter, drift_iter), int(stimuli_params["input_layer_size"]/2)),
                                 "horizontal":np.random.choice((-drift_iter, drift_iter), int(stimuli_params["input_layer_size"]/2))}


    relative_times_vertical, relative_times_horizontal, neuron_drift_coefs_dic = setup_inputs(stimuli_params, neuron_drift_coefs_dic)

    return assembly_IDs, relative_times_vertical, relative_times_horizontal, neuron_drift_coefs_dic


def plot_input_raster(stimuli_params, assembly_IDs, spike_IDs, spike_times, neuron_drift_coefs_dic,
                      seed_iter, eval_bool, input_name):

    # Plot raster of input activity
    visualization_interval = 2000  # ms

    for presentation_iter in range(0, visualization_interval, stimuli_params["duration_of_presentations"]):

        mask = np.nonzero((spike_times>=presentation_iter) & (spike_times<(presentation_iter+stimuli_params["duration_of_presentations"])))

        if eval_bool:
            color='crimson'
            vlines(spike_times[mask],
                spike_IDs[mask], ymax=1, alpha=0.5)

        # For training stimuli, determine if the input is noise or signal and plot appropriately
        else: 
            if (presentation_iter % (stimuli_params["duration_of_presentations"]*2)) == 0:
                color='crimson'
                vlines(spike_times[mask],
                    spike_IDs[mask], ymax=1, alpha=0.5)
            else:
                color='k'

        scatter(spike_times[mask],
                spike_IDs[mask], s=14, alpha=0.5, color=color)

    ylabel("Neuron ID")
    xlabel("Time (ms)")
    xlim(0, visualization_interval)
    title("Spatio-temporal Inputs")
    savefig("figures/" + str(seed_iter) + "input_spike_pairs_raster_"
           + input_name + ".png", dpi=300)
    clf()


