import os
import numpy as np
import matplotlib.pyplot as plt

from raves import raves, run_ART, run_MoDART


if __name__ == '__main__':
    # The environment to process.
    environment_folder = os.path.join('..', 'example environments',
                                      'DampenedMiddle_20_patches')
  
    # If `MoD-ART.csv` exists, a full analysis has already been carried out.
    # If it doesn't exist, some or all of the analysis needs to be run.
    if not os.path.isfile(os.path.join(environment_folder, 'MoD-ART.csv')):
        raves(environment_folder)
    else:
        print('The environment at', environment_folder,
              'has already been analyzed. Existing results will be used.')
    
    # Results of the echogram comparison will be saved to this subfolder.
    echograms_subfolder = os.path.join(environment_folder, 'Echograms')
    os.makedirs(echograms_subfolder, exist_ok=True)
    
    # Duration of the echograms to be displayed, in seconds.
    shown_duration = 0.5
    # All frequency band plots are saved to files; this one is also displayed.
    shown_band = 3
    # Sample rate used for the echograms. Mostly relevant to avoid rounding
    #   errors in the propagation delays.
    echogram_sample_rate = 1e4
    
    # Source and listener positions used for the generated echograms.
    source_positions = np.array([[2.1, 1.9, 1.5],
                                 [5.8, 4.1, 1.5],
                                 [7.2, 6.5, 1.5]])
    listener_positions = np.array([[3., 3.5, 1.75],
                                   [5., 3.5, 1.75],
                                   [7., 3.5, 1.75],
                                   [9., 3.5, 1.75],
                                   [9., 5.5, 1.75],
                                   [9., 7.5, 1.75],])
    num_sources = len(source_positions)
    num_listeners = len(listener_positions)
    
    # Generate the echograms with TD-ART.
    ART_echograms, frequencies = run_ART(environment_folder,
                                         source_positions,
                                         listener_positions,
                                         echogram_sample_rate=echogram_sample_rate,
                                         echogram_duration=shown_duration,
                                         output_folder_path=echograms_subfolder)
    
    # Generate the echograms with MoD-ART.
    MoDART_echograms, _, _ = run_MoDART(environment_folder,
                                        source_positions,
                                        listener_positions,
                                        echogram_sample_rate=echogram_sample_rate,
                                        echogram_duration=shown_duration,
                                        output_folder_path=echograms_subfolder)
    
    # Clip the echograms above 0 and convert to dB.
    ART_echograms = np.clip(ART_echograms, 1e-20, None)
    MoDART_echograms = np.clip(MoDART_echograms, 1e-20, None)
    ART_echograms = 10 * np.log10(ART_echograms)
    MoDART_echograms = 10 * np.log10(MoDART_echograms)

    # Prepare a time axis.
    time_axis = np.linspace(0, shown_duration, ART_echograms.shape[-1])
    num_bands = len(frequencies)
    
    print('Plotting echograms.')
    
    # Generate a separate figure for each frequency band.
    for b in range(num_bands):
        max_extent = max(np.max(ART_echograms[:, :, b, :]),
                         np.max(MoDART_echograms[:, :, b, :]))
        min_extent = min(np.min(ART_echograms[:, :, b, -1]),
                         np.min(MoDART_echograms[:, :, b, -1]))
        
        fig, ax = plt.subplots(num_sources, num_listeners,
                               figsize=(3*num_listeners, 3*num_sources))
    
        for s in range(num_sources):
            for l in range(num_listeners):
                ax[s, l].plot(time_axis,
                              ART_echograms[s, l, b],
                              label='TD-ART', marker='o', fillstyle='none',
                              markevery=int(3e-2 * echogram_sample_rate))
                ax[s, l].plot(time_axis,
                              MoDART_echograms[s, l, b],
                              label='MoD-ART', marker='x',
                              markevery=int(3e-2 * echogram_sample_rate))
                
                ax[s, l].legend()
                ax[s, l].set_xlim(0, shown_duration)
                ax[s, l].set_ylim(min_extent, max_extent + 3)
                
                ax[s, l].set_title('S{}, L{}'.format(s+1, l+1))
                if l == 0:
                    ax[s, l].set_ylabel('Intensity response [W/m2]')
                if s == num_sources-1:
                    ax[s, l].set_xlabel('Time [s]')
    
        plt.suptitle('Frequency band {} ({:.2f}Hz)'.format(b+1, frequencies[b]))
        plt.tight_layout()

        plt.savefig(os.path.join(echograms_subfolder,
                                 'Echograms for band {}.png'.format(b+1)))

        if b == shown_band:
            plt.show()
        plt.close()
    