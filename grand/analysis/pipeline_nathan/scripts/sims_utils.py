import numpy as np

import grand.analysis.constants as cons

# -------------------------------
# Trigger function for ADC traces
# -------------------------------
dict_trigger_parameter = dict([
    # trigger parameters
    ("t_quiet", 512),
    #("t_quiet", 256),
    ("t_period", 512),
    ("t_sepmax", 50),

    # thresholds
    ("th1",70),
    ("th2", 60),
  ])

def trigger_1(trace, trigger_dict=dict_trigger_parameter, dt_ns=2):
    # t1 crossing
    index_t1_crossing = np.where((trace) > trigger_dict["th1"], np.arange(len(trace)), -1)
    if sum(index_t1_crossing != -1) == 0: 
        # print("No T1 crossing")
        return False # No T1 crossing
    else:
        idx_first_T1 = index_t1_crossing[index_t1_crossing != -1][0]
    
    # t_quiet condition
    if idx_first_T1 < trigger_dict["t_quiet"]//2:
        # print("Not enough data before T1 crossing to satisfy quiet time condition")
        return False # Not enough data before T1 crossing to satisfy quiet time condition

    # T2 crossings and Tsepmax condition
    period_after_T1_crossing = trace[idx_first_T1 : idx_first_T1+trigger_dict['t_period']//2]
    T2_crossings = np.where(
        np.diff((period_after_T1_crossing > trigger_dict['th2']).astype(int)) == 1
        )[0] + 1 # Indices of T2 crossings relative to the start of the period after T1 crossing
    indices = [0, *T2_crossings] # Include the first T1 crossing as a T2 crossing
    if not all((j - i) * dt_ns >= trigger_dict["t_sepmax"] for i, j in zip(indices[:-1], indices[1:])):
        # print("Violating Tsepmax condition")
        return False # Violating Tsepmax condition

    return True

def trigger_adc(trace, trigger_dict=dict_trigger_parameter, dt_ns=2):
    """ Trigger function for ADC traces.

    Returns the indices of the channels that satisfy the trigger condition.
    
    Parameters
    ----------
    trace : np.ndarray
        1D array of shape (n_samples,).
    trigger_dict : dict
        Dictionary containing the trigger parameters.
    dt_ns : float, optional
        Sampling time in nanoseconds (default is 2 ns).
    
    Returns
    -------
    trigg_channels : list
        List of indices of the triggered channels.
    """
    trigg_channels = []
    for i in range(trace.shape[0]):
        if trigger_1(trace[i], trigger_dict, dt_ns):
            trigg_channels.append(i)

    return trigg_channels

def get_antenna_positions_from_run(trun, du_indices):
    """
    Get the antenna positions (X, Y, Z) in the GRAND reference frame for a given run.

    Parameters
    ----------
    trun : object
        The run object containing the antenna information.

    Returns
    -------
    Xants : np.ndarray
        2D array of shape (n_antennas, 3) containing the X, Y, Z coordinates of the antennas.
    """
    Xants = np.asarray(
        trun.du_xyz,
        dtype=float,
    )[du_indices]

    Xants[:, 2] += cons.groundAltitude
    return Xants