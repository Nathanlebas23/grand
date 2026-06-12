import numpy as np


def uncertainties_from_file_name(file_name: str) -> tuple:
    """Determine the uncertainties (time jitter, background noise, amplitude uncertainty) based on the file name for simulation data. Gives out the Plain Simulation values if nothing is precised.

    Inputs:
        file_name: str
            The file name of the data.
    Outputs:
        tuple:
            A tuple containing the following uncertainties:
            - min_amplitude: float
                The minimum amplitude increment that can be resolved.
            - jitter_time: float
                The time jitter (in seconds).
            - background_noise: float
                The standard deviation of the background noise (in ADC counts or µV/m).
            - amplitude_uncertainty: float
                The relative uncertainty on the amplitude.
    """

    # Determine noise floor and amplitude uncertainty based on file path
    is_efield = 'efield'  in file_name
    is_gp300  = 'GP300'   in file_name
    is_gp289  = 'GP289'   in file_name
    is_nj_adc = '-NJ_adc' in file_name
    is_an_adc = '-AN_adc' in file_name

    jitter_time_min = 0.5e-9 if is_efield else 2e-9 # time step of the data (0.5 ns for Efield, 2 ns for GP300/GP289/CoREAS)

    if is_efield:
        min_amplitude    = 1e-3 # 1e-3 µV/m, minimal increment of values
        jitter_time      = 0.0 # No added time jitter en Efield
        background_noise = 0.0 # Std of background noise
        amplitude_uncertainty = 0.0 # Relative uncertainty on amplitude, e.g. due to calibration (0.075 = 7.5%)

    elif is_gp300:  # ZHAireS
        min_amplitude = 1.0 # 1 ADC count, minimal increment of values
        if is_nj_adc:
            jitter_time = 0.0
            background_noise = 0.0
            amplitude_uncertainty = 0.0
        elif is_an_adc:
            jitter_time = 10e-9
            background_noise = 15.0
            amplitude_uncertainty = 0.075
        else:
            jitter_time = 10e-9
            background_noise = 4.0
            amplitude_uncertainty = 0.075

    elif is_gp289:  # ZHAireS
        min_amplitude = 1.0 
        if is_nj_adc:
            jitter_time = 0.0
            background_noise = 0.0
            amplitude_uncertainty = 0.0
        elif is_an_adc:
            jitter_time = 10e-9
            background_noise = 12.0
            amplitude_uncertainty = 0.075
        else:
            jitter_time = 10e-9
            background_noise = 5.0
            amplitude_uncertainty = 0.075

    else:  # CoREAS
        min_amplitude = 1.0 # 1 ADC count, minimal increment of values
        if is_an_adc:
            jitter_time = 10e-9
            background_noise = 10.0
            amplitude_uncertainty = 0.075
    
    sigma_time = np.sqrt(jitter_time**2 + jitter_time_min**2)
    background_noise = np.sqrt(background_noise**2 + min_amplitude**2)

    return (sigma_time, background_noise, amplitude_uncertainty)
