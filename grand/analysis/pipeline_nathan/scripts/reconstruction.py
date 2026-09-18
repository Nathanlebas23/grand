"""
File to reconstuct events from the raw data. It uses the configuration file to get the paths and parameters for the reconstruction.
"""
import sys  
sys.path.append("/home/lpnhe/grand")

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from grand.aoi import EventList, Shower
from grand.dataio import TRecons
import grand.analysis.signals.extraction as ext
import grand.analysis.fitting as fit
import grand.analysis.constants as cons
import grand.analysis.geom as geom
import grand.analysis.energy_reco as en


logger = logging.getLogger("grand.process")


def reconstruct_event(e, antenna_position, nutrig_template, trecons: TRecons, run_number) -> None:
    """Run the PWF/SWF/ADF/energy reconstruction chain for a single event.

    Fills the real, persisting Shower fields on e.shower and prepares trecons's fields
    for this event. Does not call trecons.fill() or e.write() - the caller (process_file)
    does that, so a TRecons entry only exists once both reconstruction and writing succeed.
    """

    # -------------------------------
    # Convert voltage traces to ADC counts
    # At this stage, v.trace already contains only three components:
    #   v.trace[0] -> X
    #   v.trace[1] -> Y
    #   v.trace[2] -> Z
    # Therefore channels=[0,1,2] refers to X, Y, Z respectively
    # --------------------------------
    ADC_traces_list = []
    for v in e.voltages:
        ADC_traces_list.append(ext.convert_voltage_to_ADC(v.trace, channels=[0, 1, 2]))
    ADC_traces = np.array(ADC_traces_list)
    n_antennas = len(e.voltages)

    # ---------------------------------------------------------------
    # Compute peak amplitudes and times for each antenna
    # ---------------------------------------------------------------
    peak_amps = np.array([ext.get_peak_amplitude(ADC_traces[i], channels=[0, 1, 2])
                          for i in range(n_antennas)])

    t0_all = np.array([v.t0.astype('int64') for v in e.voltages])
    t0 = t0_all - t0_all.min()

    peak_times = np.array([ext.get_peak_time_adc(ADC_traces[i], nutrig_template, t0[i])
                           for i in range(n_antennas)])

    # ----------------------------------------------------------------
    # Map DU IDs to their positions (X, Y, Z) in GRAND reference frame
    # ----------------------------------------------------------------
    du_ids = np.array([v.du_id for v in e.voltages]).astype(int)

    du_positions_ev = antenna_position[antenna_position['DU_id'].isin(du_ids)]
    du_positions_ev = du_positions_ev.set_index('DU_id').loc[du_ids]
    x_coords = du_positions_ev['x'].astype(float).values
    y_coords = -du_positions_ev['y'].astype(float).values
    z_coords = du_positions_ev['z'].astype(float).values + cons.groundAltitude

    Xants = np.column_stack((x_coords, y_coords, z_coords))

    # -------------------------------
    # Plane Wave Fit (PWF)
    # -------------------------------
    theta_pwf, phi_pwf = fit.PWF_semianalytical(Xants, peak_times, verbose=False, c=cons.c_light, n=cons.n_atm, sigma=5e-9)
    # chi2_pwf is the raw (non-reduced) chi2, matching TRecons.chi2_pwf's documented convention
    # (divide by du_count downstream, exactly as grand/analysis/example/display.py does).
    chi2_pwf = fit.PWF_loss((theta_pwf, phi_pwf), Xants, peak_times, verbose=False, c=cons.c_light, n=cons.n_atm, sigma=5e-9)

    # -------------------------------
    # Spherical Wave Fit (SWF)
    # -------------------------------
    theta_swf, phi_swf, r_xmax, t_s = fit.recons_swf(theta_pwf, phi_pwf, peak_times, Xants, sigma=5e-9)
    Xsource = fit.compute_Xsource_cartesian_coords(theta_swf, phi_swf, r_xmax)

    # Xsource has shape (1, 3); extract the (3,) Cartesian coordinate vector
    Xsource = Xsource[0]

    l_ant = geom.distance_source_antenna(Xants, Xsource)
    chi2_swf = fit.SWF_loss(theta_swf, phi_swf, r_xmax, t_s, Xants, peak_times, sigma=5e-9)  # raw, see above

    # -------------------------------
    # Angular Distribution Function (ADF)
    # -------------------------------
    theta_adf, phi_adf, delta_omega, scaling_factor = fit.recons_ADF(theta_pwf, phi_pwf, peak_amps, Xants, Xsource)
    eta, omega, omega_cr, l_ant, amplitude_model = fit.ADF_parameters(theta_adf, phi_adf, delta_omega, scaling_factor, Xants, Xsource, groundAltitude=cons.groundAltitude, Bvec=cons.Bvec)
    best_params = theta_adf, phi_adf, delta_omega, scaling_factor
    chi2_adf = fit.ADF_loss(best_params, peak_amps, Xants, Xsource)  # raw, see above
    sin_alpha = geom.sin_geomag_angle(theta_adf, phi_adf, B=cons.Bn)
    energy_elm = en.recons_energy_from_voltage(scaling_factor, sin_alpha)

    # ---------------------------------------------------------------
    # Fill only the real, persisting Shower fields - no ad hoc duplicate
    # attributes. All detailed PWF/SWF/ADF/chi2/energy values live in
    # TRecons, which is what actually gets written to ROOT for them.
    # ---------------------------------------------------------------
    if e.shower is None:
        e.shower = Shower()
    e.shower.zenith = theta_adf
    e.shower.azimuth = phi_adf
    e.shower.energy_em = energy_elm

    # ---------------------------------------------------------------
    # Prepare the TRecons entry for this event (real field names).
    # trecons.fill() is called by process_file(), only after Event.write() succeeds.
    # ---------------------------------------------------------------
    trecons.run_number = run_number
    trecons.event_number = e.event_number

    trecons.peak_amps = peak_amps
    trecons.peak_time = peak_times
    trecons.Xants = Xants
    trecons.du_count = n_antennas

    trecons.zenith_pwf = theta_pwf
    trecons.azimuth_pwf = phi_pwf
    trecons.chi2_pwf = chi2_pwf

    trecons.zenith_swf = theta_swf
    trecons.azimuth_swf = phi_swf
    trecons.r_xmax = r_xmax
    trecons.t_s = t_s
    trecons.Xsource = Xsource
    trecons.chi2_swf = chi2_swf
    trecons.distance_source_antenna = l_ant

    trecons.zenith_adf = theta_adf
    trecons.azimuth_adf = phi_adf
    trecons.width = delta_omega
    trecons.scaling_factor = scaling_factor
    trecons.chi2_adf = chi2_adf
    trecons.omega = omega
    trecons.eta = eta
    trecons.omega_cr = omega_cr
    trecons.adf_amplitude = amplitude_model
    trecons.energy_elm_voltage = energy_elm
