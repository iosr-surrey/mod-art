import os
import warnings
import numpy as np
from numpy.random import default_rng
from scipy.io.wavfile import write
from scipy.signal import resample_poly, butter, sosfilt

from raves import raves, run_MoDART


def closest_divisor(dividend: int, target: int
                    ) -> int:
    """
    Return a factor of `dividend` whose value is closest to `target`.

    Parameters
    ----------
    dividend : int
        Integer to factor.
    target : int
        Desired factor value.

    Returns
    -------
    factor: int
        A factor of `dividend` (i.e. `dividend % factor == 0`) such that
        `abs(factor - target)` is minimal among all factors of `dividend`.

    """
    # Start by looking in an arbitrary nonzero range around the target.
    search_range = max(1, int(target / 16))
    while True:
        checked_values = np.arange(target - search_range,
                                   target + search_range,
                                   dtype=int)

        # Factors have to be strictly positive.
        checked_values = checked_values[checked_values > 0]

        is_factor = ((dividend % checked_values) == 0)
        if np.any(is_factor):
            factors = checked_values[is_factor]
            closest = np.argmin(np.abs(factors - target))
            return factors[closest]

        # If no factor was found in the range, expand the range and try again.
        # The loop will eventually break, because checked_values will
        #   eventually include 1.
        search_range *= 2


def smart_upsample(envelope: np.ndarray,
                   low_fs: int, high_fs: int,
                   clip: bool = True
                   ) -> np.ndarray:
    """
    Upsamples an envelope for noise-shaping, minimizing upsampling artifacts.
    Guarantees the total energy is preserved.

    Parameters
    ----------
    envelope : numpy.ndarray
        the envelope to be upsampled
    low_fs : int
        The starting sample rate of the envelope.
    high_fs : int
        The target sample rate of the envelope (must be >= the starting rate).
    clip : bool, default: True
        If True, any negative values in the result are set to 0.

    Returns
    -------
    envelope : numpy.ndarray
        The upampled envelope.
    """
    if high_fs != low_fs:
        assert high_fs > low_fs

        # Resample envelope to the closest prime factor of high_fs (maintaining energy, and non-negativity)
        near_fs = closest_divisor(high_fs, low_fs)

        old_energy = np.sum(envelope ** 2, axis=0)

        envelope = resample_poly(envelope, up=near_fs, down=low_fs, axis=0)
        if clip:
            envelope = envelope.clip(0.)

        new_energy = np.sum(envelope ** 2, axis=0)
        envelope *= np.sqrt(old_energy / new_energy)[None]

        # Repeat envelope to achieve high_fs
        repetitions = high_fs // near_fs
        envelope = np.repeat(envelope, repetitions, axis=0)
        # Note: divide the energy by the number of repetitions to preserve energy-per-time
        envelope /= np.sqrt(repetitions)

    return envelope


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
    aural_sample_rate = 48000.
    # Duration of the responses to be generated, in seconds.
    response_duration = 2.5

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

    # Prepare band edges for bandpass filtering.
    num_bands = len(frequencies)
    lower_band_edges = frequencies / np.sqrt(2)
    upper_band_edges = frequencies * np.sqrt(2)

    # Random number generator for white noise.
    rng = default_rng()
    noise_signal = rng.normal(size=int(aural_sample_rate * response_duration))

    # Ensure the noise signal has unit energy per second, matching the
    #   convention used to generate the echograms.
    noise_signal *= np.sqrt(response_duration / np.sum(noise_signal**2))

    print('Generating responses.')

    responses_dict = dict()
    for s in range(num_sources):
        for l in range(num_listeners):
            response = np.zeros_like(noise_signal)

            for b in range(num_bands):
                sos = butter(7, (lower_band_edges[b], upper_band_edges[b]),
                             btype='bandpass',
                             fs=aural_sample_rate, output='sos')

                filtered_noise = sosfilt(sos, noise_signal)

                envelope = np.sqrt(MoDART_echograms[s, l, b])

                envelope = smart_upsample(envelope,
                                          echogram_sample_rate,
                                          aural_sample_rate)
                response += filtered_noise * envelope

            if np.any(np.abs(response) > 1.):
                warnings.warn('The response "S{}, L{}.wav" is clipped.'.format(s+1, l+1))
                response /= np.max(np.abs(response))

            file_name = 'S{}, L{}.wav'.format(s+1, l+1)
            write(os.path.join(responses_subfolder, file_name),
                  int(aural_sample_rate), response)
            
            responses_dict[(s, l)] = response

    try:
        from librosa import amplitude_to_db
        from librosa.core import cqt
        from librosa.display import specshow
        
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
    except ImportError:
        print('Install librosa to plot the spectrograms.')
    else:
        print('Plotting constant-Q spectrograms.')
        
        all_band_edges = np.append(lower_band_edges, upper_band_edges[-1])
        
        bins_per_octave = 24
        nyquist = aural_sample_rate / 2
        n_octaves = np.log2(nyquist / all_band_edges[0])
        n_bins = int(np.floor(n_octaves * bins_per_octave))
        
        spectrograms_dict = {key: cqt(y=response, sr=aural_sample_rate,
                                      bins_per_octave=bins_per_octave,
                                      n_bins=n_bins, fmin=all_band_edges[0])
                             for key, response in responses_dict.items()}
        
        max_linear_value = np.max([np.max(np.abs(spec))
                                   for spec in spectrograms_dict.values()])
        
        spectrograms_dict = {key: amplitude_to_db(spec, ref=max_linear_value)
                             for key, spec in spectrograms_dict.items()}
        
        max_value = np.max([np.max(spec)
                            for spec in spectrograms_dict.values()])
        min_value = max_value - 70
        
        fig, ax = plt.subplots(num_sources, num_listeners,
                               figsize=(4*num_listeners, 3*num_sources),
                               squeeze=False, constrained_layout=True)
        
        cs = None
        for s in range(num_sources):
            for l in range(num_listeners):
                cs = specshow(spectrograms_dict[s, l],
                              x_axis='time', y_axis='cqt_hz',
                              ax=ax[s, l], cmap='viridis',
                              sr=aural_sample_rate,
                              fmin=all_band_edges[0],
                              bins_per_octave=bins_per_octave,
                              vmin=min_value, vmax=max_value)
                
                line = ax[s, l].hlines(all_band_edges, 0, response_duration,
                                       color='white', ls='--', linewidth=1)
                line.set_path_effects([pe.Stroke(linewidth=1.5,
                                                 foreground='black'),
                                       pe.Normal(),])
     
                ax[s, l].set_xlim(0, response_duration)
                ax[s, l].set_ylim(all_band_edges[0] * 0.95, nyquist)
                
                ax[s, l].set_title('S{}, L{}'.format(s+1, l+1))
        
        cbar = fig.colorbar(cs, ax=ax)
        cbar.set_label('dB')

        plt.suptitle('Dotted lines show frequency band limits.')
        
        plt.savefig(os.path.join(responses_subfolder, 'CQT spectrograms.png'))
                    
        plt.show()
