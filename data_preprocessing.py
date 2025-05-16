import numpy as np


def find_ambiguous_windows(Xt, Xc, y, name):
    """
    Identify ambiguous windows based on label dominance.

    A window is considered ambiguous if no single class label accounts for 
    more than 50% of the labels within that window. This function returns 
    the indices of such windows.

    Parameters:
    ----------
    Xt : List of sensor data windows (e.g., acc,gyr..).
         Shape: (num_windows, window_length, num_features)

    Xc : List complementary input data per window.
         Shape: (num_windows, ...)

    y : List labels per window.
        Shape: (num_windows, window_length)

    name : (str) Name of the modality (e.g., "ACC", "GYR") used for logging.

    Returns:
    -------
    ambiguous_indices : list of int
        Indices of windows where no label has more than 50% frequency 
    """
    ambiguous_indices = []
    print(f"[{name}] Checking for ambiguous windows...")

    for i in range(len(y)):
        values, counts = np.unique(y[i], return_counts=True)
        idx = np.argmax(counts)
        if counts[idx] <= 0.5 * np.sum(counts):
            ambiguous_indices.append(i)

    print(f"[{name}] Found {len(ambiguous_indices)} ambiguous windows.")
    return ambiguous_indices
       

def remove_windows_by_indices(Xt, Xc, y, indices_to_remove):

    """
    Remove specified windows from the input data arrays.

    Parameters:
    ----------
    Xt : List of sensor data windows (e.g., acc,gyr..).
         Shape: (num_windows, window_length, num_features)

    Xc : List complementary input data per window.
         Shape: (num_windows, ...)

    y : List labels per window.
        Shape: (num_windows, window_length)

    indices_to_remove : List of indices of windows to be removed.

    Returns:
    -------
    Xt_clean : np.ndarray
        Filtered sensor data with specified windows removed.

    Xc_clean : np.ndarray
        Filtered contextual data with specified windows removed.

    y_clean : np.ndarray
        Filtered label array with specified windows removed.
    """
     
    indices_to_remove = set(indices_to_remove)
    mask = [i not in indices_to_remove for i in range(len(Xt))]
    return np.array(Xt)[mask], np.array(Xc)[mask], np.array(y)[mask]
