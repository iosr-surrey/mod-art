import os
import warnings
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as pathefx

from numpy.random import default_rng
from scipy.io.wavfile import write
from scipy.signal import butter, sosfilt
from scipy.interpolate import make_interp_spline

from raves import raves, run_MoDART


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

    # Results of the response generation will be saved to this subfolder.
    responses_subfolder = os.path.join(environment_folder, 'Responses')
    os.makedirs(responses_subfolder, exist_ok=True)

    # Sample rate used for the echograms. Mostly relevant to avoid rounding
    #   errors in the propagation delays.
    echogram_sample_rate = 1e4
    # Audio sample rate used by the generated responses.
    audio_sample_rate = 48000.
    # Duration of the responses to be generated, in seconds.
    response_duration = 1.5

    # Source and listener positions used for the generated echograms.
    source_positions = np.array([[2.1, 1.9, 1.5],
                                 [5.8, 4.1, 1.5],
                                 [7.2, 6.5, 1.5]])
    listener_positions = np.array([[3., 3.5, 1.75],
                                   [9., 3.5, 1.75],
                                   [9., 9.5, 1.75],])
    num_sources = len(source_positions)
    num_listeners = len(listener_positions)

    # Generate the echograms with MoD-ART.
    MoDART_echograms, frequencies, _ = run_MoDART(environment_folder,
                                                  source_positions,
                                                  listener_positions,
                                                  echogram_sample_rate=echogram_sample_rate,
                                                  echogram_duration=response_duration,
                                                  output_folder_path=responses_subfolder)
    
    print('Generating responses.')
    
    # Take note of the echogram energy, to compare it after upsampling.
    old_energy = np.sum(MoDART_echograms, axis=-1)
    
    # Prepare the audio-rate time intervals at which we'll evaluate the upsampled echogram.
    echogram_time_axis = np.arange(0, response_duration, 1 / echogram_sample_rate)
    audio_time_axis = np.arange(0, response_duration, 1 / audio_sample_rate)
    # We use a linear interpolation, because any other upsampling algorithm risks introducing negative values.
    linear_spline = make_interp_spline(echogram_time_axis, MoDART_echograms, k=1, axis=-1)
    upsampled_echograms = linear_spline(audio_time_axis)
    
    # Normalize w.r.t. the new sample rate, to preserve the energy-per-second definition of echogram values.
    upsampled_echograms *= echogram_sample_rate / audio_sample_rate
    
    # Compare the new energy to the old one.
    new_energy = np.sum(upsampled_echograms, axis=-1)
    # The ratio (averaged over all frequency bands) should be close to 1 for all sources and listeners.
    print(np.mean(old_energy / new_energy, axis=-1))

    # Random number generator for the stochastic signal to be modulated.
    rng = default_rng()
    
    # White noise
    #   noise_signal = rng.normal(size=len(audio_time_axis))
    # Poisson process
    noise_signal = rng.poisson(lam=0.5, size=len(audio_time_axis)).astype(float)

    # Ensure the noise signal has unit energy per second, matching the
    #   convention used to generate the echograms.
    noise_signal *= np.sqrt(response_duration / np.sum(noise_signal**2))
    
    # Factor for octave-band boundaries.
    band_bound = np.sqrt(2)
    # Consider the frequency band centers provided alongside the input data.
    band_centers = frequencies
    num_bands = len(frequencies)
    
    # Ensure that all frequencies support band-pass filtering.
    if np.any(band_centers * band_bound >= audio_sample_rate):
        print('Warning: the audio sample rate is too low for some frequency bands.')
        # Select only acceptable bands.
        band_centers = band_centers[band_centers * band_bound < audio_sample_rate]
        # Update the number of rendered bands.
        num_bands = len(band_centers)
        # Drop unused bands from the echogram, to preserve the right shape.
        upsampled_echograms = upsampled_echograms[:, :, :num_bands]
    
    # Prepare an array for the band-pass filtered signals.
    filtered_noise_signals = np.zeros((num_bands, len(audio_time_axis)))
    
    for b in range(num_bands):
        # Prepare the suitable band-pass filter...
        sos = butter(6, (band_centers[b] / band_bound,
                         band_centers[b] * band_bound),
                     btype='bandpass', output='sos',
                     fs=audio_sample_rate)
        # ...and apply it to the stochastic signal.
        filtered_noise_signals[b] = sosfilt(sos, noise_signal)
    
    # Translate the energy envelopes to amplitude envelopes.
    envelopes = np.sqrt(upsampled_echograms)
    
    # The envelope array has shape (S, L, B, T), the noise signals have shape (B, T):
    #   we need to add two "leading" dimensions, which is done using [None, None].
    modulated_noise_signals = envelopes * filtered_noise_signals[None, None]
    
    # The dimension of index 2 holds the separate frequency bands.
    # Sum the array along that dimension to obtain the complete room impulse responses.
    responses = np.sum(modulated_noise_signals, axis=2)

    print('Saving response files.')
    
    for s in range(num_sources):
        for l in range(num_listeners):
            if np.any(np.abs(responses[s, l]) > 1.):
                warnings.warn('The response "S{}, L{}.wav" is clipped.'.format(s+1, l+1))
                responses[s, l] /= np.max(np.abs(responses[s, l]))

            file_name = 'S{}, L{}.wav'.format(s+1, l+1)
            write(os.path.join(responses_subfolder, file_name),
                  int(audio_sample_rate), responses[s, l])
    
    try:
        from librosa import amplitude_to_db
        from librosa.core import cqt
        from librosa.display import specshow
    except ImportError:
        print('Install librosa to plot the spectrograms.')
    else:
        print('Plotting constant-Q spectrograms.')
        
        bins_per_octave = 24

        fmin = frequencies[0] / 2
        fmax = audio_sample_rate / 2
        n_octaves = np.log2(fmax / fmin)
        n_bins = int(np.floor(n_octaves * bins_per_octave))
            
        band_boundaries = np.append(band_centers / band_bound,
                                    band_centers[-1] * band_bound)
        
        spectrograms = cqt(y=responses, sr=audio_sample_rate,
                           bins_per_octave=bins_per_octave,
                           n_bins=n_bins, fmin=fmin)
        
        spectrograms_dB = amplitude_to_db(np.abs(spectrograms), ref=1.0)
        
        max_value = np.max(spectrograms_dB)
        min_value = max_value - 50
        
        fig, ax = plt.subplots(num_sources, num_listeners,
                               figsize=(4*num_listeners, 3*num_sources),
                               squeeze=False, constrained_layout=True)
            
        cs = None
        for s in range(num_sources):
            for l in range(num_listeners):
                cs = specshow(spectrograms_dB[s, l], sr=audio_sample_rate,
                              x_axis='time', y_axis='cqt_hz',
                              ax=ax[s, l], cmap='magma',
                              fmin=fmin, bins_per_octave=bins_per_octave,
                              vmin=min_value, vmax=max_value)
                
                line = ax[s, l].hlines(band_boundaries, 0, response_duration,
                                        color='white', ls='--', linewidth=1)
                line.set_path_effects([pathefx.Stroke(linewidth=1.5, foreground='black'),
                                       pathefx.Normal()])
        
                ax[s, l].set_xlim(0, response_duration)
                ax[s, l].set_ylim(fmin, fmax)
                
                ax[s, l].yaxis.set_major_locator(ticker.FixedLocator(band_centers))
                ax[s, l].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: (f'{int(x)}'
                                                                                        if x < 1e3 else
                                                                                        f'{int(x / 1e3)}k')))
                ax[s, l].yaxis.set_minor_locator(ticker.NullLocator())
                ax[s, l].yaxis.set_minor_formatter(ticker.NullFormatter())
                
                ax[s, l].set_title('S{}, L{}'.format(s+1, l+1))
                if l == 0:
                    ax[s, l].set_ylabel('Frequency [Hz]')
                else:
                    ax[s, l].set_ylabel('')
                if s == num_sources-1:
                    ax[s, l].set_xlabel('Time [s]')
                else:
                    ax[s, l].set_xlabel('')

        cbar = fig.colorbar(cs, ax=ax, format='{x:.0f}dB')

        plt.suptitle('Constant-Q spectrograms of room impulse responses for different source/listener configurations.'
                     + '\nDotted lines show frequency band limits.')
        
        plt.savefig(os.path.join(responses_subfolder, 'CQT spectrograms.png'))
                    
        plt.show()
