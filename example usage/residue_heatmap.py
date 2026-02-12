import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from raves import raves, run_MoDART


# TODO: Write a version which isn't ad-hoc, and works for any environment.
def is_inside(point: np.ndarray,
              safeguard: float = 1e-3
              ) -> bool:
    """
    Ad-hoc function to detect whether a position is within the bounds of
    our three-room environment.

    Parameters
    ----------
    point : numpy.ndarray
        Position to be checked.
    safeguard : float, default: 1e-3
        Minimum distance from surface for the point to be considered in-bounds.

    Returns
    -------
    bool
        True if the given position is inside of the environment mesh.
    """
    if point[2] < safeguard or point[2] > 3-safeguard:
        return False

    if point[0] < safeguard:
        return False
    elif point[0] < 4-safeguard:
        if point[1] < safeguard or point[1] > 8-safeguard:
            return False
        else:
            return True
    elif point[0] < 4+safeguard:
        if point[1] < 2.75+safeguard:
            return False
        elif point[1] > 4.25-safeguard:
            return False
        else:
            return True
    elif point[0] < 10-safeguard:
        if point[1] < 2+safeguard:
            return False
        elif point[1] < 5-safeguard:
            return True
        elif point[1] < 5+safeguard:
            if point[0] < 8.5+safeguard:
                return False
            else:
                return True
        elif point[1] < 13-safeguard:
            if point[0] > 6+safeguard:
                return True
            else:
                return False
        else:
            return True
    else:
        return False


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

    # Results of the residue mapping will be saved to this subfolder.
    maps_subfolder = os.path.join(environment_folder, 'Residue_maps')
    os.makedirs(maps_subfolder, exist_ok=True)

    # Number of modes to plot in each frequency band (if present).
    max_shown_modes = 3
    # All frequency band plots are saved to files; this one is also displayed.
    shown_band = 3
    # Sample rate used for the echograms. Mostly relevant to avoid rounding
    #   errors in the propagation delays.
    echogram_sample_rate = 1e3

    # These source positions are fixed; listeners are mapped over a grid.
    source_positions = np.array([[2.1, 1.9, 1.5],
                                 [5.8, 4.1, 1.5],
                                 [7.2, 6.5, 1.5]])

    # Spacing of the mapped position grid.
    spacing = 0.25
    # Height of the plane on which positions are sampled.
    height = 1.75

    # Generate the position grid.
    X = np.arange(spacing, 10., spacing)
    Y = np.arange(spacing, 13., spacing)
    XX, YY = np.meshgrid(X, Y)
    flattened_grid = np.column_stack([XX.ravel(),
                                      YY.ravel(),
                                      np.full(XX.size, height)])
    # Determine which positions are within the bounds of the three rooms.
    grid_mask = np.array([is_inside(p) for p in flattened_grid], dtype=bool)
    listener_positions = flattened_grid[grid_mask]

    num_sources = len(source_positions)
    num_listeners = len(listener_positions)

    # Generate the source and listener residues. Disregard the echograms.
    _, frequencies, MoDART_data = run_MoDART(environment_folder,
                                             source_positions,
                                             listener_positions,
                                             echogram_sample_rate=echogram_sample_rate,
                                             output_folder_path=maps_subfolder)
    num_bands = len(frequencies)

    print('Plotting residue maps.')

    # Generate a separate figure for each frequency band.
    for b in range(num_bands):
        # Select the modes related to this frequency band.
        relevant_modes = np.flatnonzero(MoDART_data['Band idx'] == b)
        num_modes = min(max_shown_modes, len(relevant_modes))
        
        if num_modes == 0:
            continue
        
        relevant_modes = relevant_modes[:num_modes]

        # Generate the full residues by combining source and listener terms.
        source_residues = MoDART_data['Source residues'][:, relevant_modes]
        listener_residues = MoDART_data['Listener residues'][:, relevant_modes]
        residue_matrix = np.einsum('lm,sm->lsm',
                                   listener_residues,
                                   source_residues)

        # Prepare the heatmap range based on the residue extents.
        vmin = residue_matrix.min()
        vmax = residue_matrix.max()
        high_dB_extent = max(np.log10(np.abs(vmin)),
                             np.log10(np.abs(vmax)))
        contour_levels = list()
        contour_ticks = list()
        contour_labels = list()
        for l in np.arange(np.ceil(high_dB_extent),
                           np.ceil(high_dB_extent)-3.25,
                           -0.25):
            if l > high_dB_extent:
                continue
            contour_levels.append(-10 ** l)
            if np.isclose(l, round(l)):
                contour_ticks.append(-10 ** l)
                contour_labels.append(rf'$-10^{{{int(round(l))}}}$')
        contour_levels.append(0)
        contour_ticks.append(0)
        contour_labels.append(r'$0$')
        for l in np.arange(np.ceil(high_dB_extent)-3.0,
                           np.ceil(high_dB_extent)+0.25,
                           0.25):
            if l > high_dB_extent:
                continue
            contour_levels.append(10 ** l)
            if np.isclose(l, round(l)):
                contour_ticks.append(10 ** l)
                contour_labels.append(rf'$10^{{{int(round(l))}}}$')

        # Diverging colormap to differentiate positive and negative values.
        cmap = plt.get_cmap('RdBu', len(contour_levels) + 1)
        norm = mpl.colors.BoundaryNorm(contour_levels, ncolors=cmap.N, extend='both')

        fig, axes = plt.subplots(num_sources, num_modes,
                                 figsize=(3*num_modes, 3*num_sources),
                                 squeeze=False, constrained_layout=True)

        cs = None
        for s in range(num_sources):
            for m in range(num_modes):
                plot_map = np.ma.masked_all(XX.shape)
                plot_map.flat[grid_mask] = residue_matrix[:, s, m]

                cs = axes[s, m].contourf(XX, YY, plot_map,
                                         extend='both', norm=norm,
                                         cmap=cmap, levels=contour_levels,
                                         corner_mask=False)

                # Add a marker at the source position.
                axes[s, m].scatter(source_positions[s, 0],
                                   source_positions[s, 1],
                                   s=120, marker='o', linewidths=2.5,
                                   facecolors='white', edgecolors='black')

                axes[s, m].set_title('Mode {}, S{}'.format(m+1, s+1))

        for ax in axes.ravel():
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 13)
            ax.set_aspect('equal')

        cbar = fig.colorbar(cs, ax=axes, spacing='uniform')
        cbar.set_ticks(contour_ticks, labels=contour_labels)

        plt.suptitle('Frequency band {} ({:.2f}Hz)'.format(b+1, frequencies[b]))

        plt.savefig(os.path.join(maps_subfolder,
                                 'Residue maps for band {}.png'.format(b+1)))

        if b == shown_band:
            plt.show()
        else:
            plt.close()
