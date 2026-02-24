import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Approximate 2D positions of 10-20 electrodes
ELECTRODE_POSITIONS = {
    "Fp1": (-0.5, 1.0),
    "Fp2": (0.5, 1.0),
    "F7": (-1.0, 0.5),
    "F3": (-0.5, 0.5),
    "Fz": (0.0, 0.5),
    "F4": (0.5, 0.5),
    "F8": (1.0, 0.5),
    "T3": (-1.2, 0.0),
    "C3": (-0.5, 0.0),
    "Cz": (0.0, 0.0),
    "C4": (0.5, 0.0),
    "T4": (1.2, 0.0),
    "T5": (-1.0, -0.5),
    "P3": (-0.5, -0.5),
    "Pz": (0.0, -0.5),
    "P4": (0.5, -0.5)
}

def plot_topomap(channel_values, title="EEG Topomap"):
    """
    channel_values: dict of {channel_name: value}
    """

    x = []
    y = []
    z = []

    for ch, val in channel_values.items():
        if ch in ELECTRODE_POSITIONS:
            pos = ELECTRODE_POSITIONS[ch]
            x.append(pos[0])
            y.append(pos[1])
            z.append(val)

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # Create grid
    grid_x, grid_y = np.mgrid[-1.5:1.5:100j, -1.5:1.5:100j]

    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    fig, ax = plt.subplots()

    im = ax.imshow(
        grid_z.T,
        extent=(-1.5, 1.5, -1.5, 1.5),
        origin='lower',
        cmap='jet'
    )

    # Draw electrode markers
    ax.scatter(x, y, c='black')
    for i, ch in enumerate(channel_values.keys()):
        if ch in ELECTRODE_POSITIONS:
            ax.text(x[i], y[i], ch, fontsize=8)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.colorbar(im, ax=ax)
    return fig