"""
File to reconstuct events from the raw data. It uses the configuration file to get the paths and parameters for the reconstruction.
"""
import sys  
import logging
from pathlib import Path
import numpy as np
from grand.aoi import Shower
from grand.dataio import TRecons
import grand.analysis.signals.extraction as ext
import grand.analysis.fitting as fit
import grand.analysis.constants as cons
import grand.analysis.geom as geom
import grand.analysis.energy_reco as en

#-------------------------------
# Load configuration
#-------------------------------
from grand.analysis.pipeline_nathan.scripts.loading import load_config

config_path = Path(__file__).parent.parent / "config.yaml"
config = load_config(config_path)

sys.path.append(config['paths']['grandlib_path'])

logger = logging.getLogger("grand.process")


def reconstruct_event(e, antenna_position, trecons: TRecons, run_number, ADC_traces) -> None:
    """Run the PWF/SWF/ADF/energy reconstruction chain for a single event.

    Fills the Shower fields on e.shower and prepares trecons's fields
    for this event. Does not call trecons.fill() or e.write() - the caller (process_file)
    does that, so a TRecons entry only exists once both reconstruction and writing succeed.

    ADC_traces (shape (n_antennas, 3, n_samples), same order as e.voltages) is computed
    once by compute_nutrig_event() and reused here - this function does not know about
    NUTRIG/rho at all, it stays scoped to the physical reconstruction (peaks/PWF/SWF/ADF/energy).
    """
    n_antennas = len(e.voltages)

    # ---------------------------------------------------------------
    # Compute peak amplitudes and times for each antenna
    # ---------------------------------------------------------------
    peak_amps = np.array([ext.get_peak_amplitude(ADC_traces[i], channels=[0, 1, 2])
                          for i in range(n_antennas)])

    t0 = ext.compute_t0(e.tvoltage)  # t0 in ns
    logger.debug(f"Event {e.event_number} (run {run_number}): t0 = {t0}")

    # To investigate if the time here is consistent with the method used in the extraction module,
    # we can compute t0 using the compute_t0 function from the extraction module and compare 
    # it with the t0 computed above.
    # t0_method = ext.compute_t0(e.t0.astype('int64'))
    # logger.debug(f"t0_method (from compute_t0): {t0_method}, t0_method - t0_all.min(): {t0_method - t0_all.min()}")

    # On ADC
    peak_times = np.array([ext.get_peak_time_efield(ADC_traces[i], t0[i], channels=[0,1,2])
                           for i in range(n_antennas)])
    
    # peak_times = np.array([ext.get_peak_time_adc(ADC_traces[i], nutrig_template, t0[i])
    #                        for i in range(n_antennas)])

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
    logger.debug(f"Event {e.event_number} (run {run_number}): PWF: theta={np.rad2deg(theta_pwf):.3f}, phi={np.rad2deg(phi_pwf):.3f}, chi2={chi2_pwf:.3f}")
    
    # -------------------------------
    # Spherical Wave Fit (SWF)
    # -------------------------------
    theta_swf, phi_swf, r_xmax, t_s = fit.recons_swf(theta_pwf, phi_pwf, peak_times, Xants, sigma=5e-9)
    Xsource = fit.compute_Xsource_cartesian_coords(theta_swf, phi_swf, r_xmax)

    # TRecons.Xsource is a StdVectorListDesc("vector<float>") field and expects that same
    # (1, 3) shape (one outer entry holding the 3 coordinates) - assigning the flattened (3,)
    # vector to it instead silently corrupts the branch (verified on disk: reads back as
    # garbage). Keep Xsource itself untouched for the trecons assignment below, and use a
    # separate flattened variable for the geometry/ADF calls that need a plain (3,) vector.
    Xsource_vec = Xsource[0]

    l_ant = geom.distance_source_antenna(Xants, Xsource_vec)
    chi2_swf = fit.SWF_loss(theta_swf, phi_swf, r_xmax, t_s, Xants, peak_times, sigma=5e-9)  # raw, see above
    logger.debug(f"Event {e.event_number} (run {run_number}): SWF: theta={np.rad2deg(theta_swf):.3f}, phi={np.rad2deg(phi_swf):.3f}, r_xmax={r_xmax:.3f}, chi2={chi2_swf:.3f}")

    # -------------------------------
    # Angular Distribution Function (ADF)
    # -------------------------------
    theta_adf, phi_adf, delta_omega, scaling_factor = fit.recons_ADF(theta_pwf, phi_pwf, peak_amps, Xants, Xsource_vec)
    eta, omega, omega_cr, l_ant, amplitude_model = fit.ADF_parameters(theta_adf, phi_adf, delta_omega, scaling_factor, Xants, Xsource_vec, groundAltitude=cons.groundAltitude, Bvec=cons.Bvec)
    best_params = theta_adf, phi_adf, delta_omega, scaling_factor
    chi2_adf = fit.ADF_loss(best_params, peak_amps, Xants, Xsource_vec)  # raw, see above
    sin_alpha = geom.sin_geomag_angle(theta_adf, phi_adf, B=cons.Bn)
    energy_elm = en.recons_energy_from_voltage(scaling_factor, sin_alpha)

    logger.debug(f"Event {e.event_number} (run {run_number}): ADF: theta={np.rad2deg(theta_adf):.3f}, phi={np.rad2deg(phi_adf):.3f}, delta_omega={delta_omega:.3f}, scaling_factor={scaling_factor:.3e}, chi2={chi2_adf:.3f}, energy_elm={energy_elm:.3e}")


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

    trecons.peak_amps = peak_amps # ADC
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
