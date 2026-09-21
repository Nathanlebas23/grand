import sys
import numpy as np
import logging
from pathlib import Path

from grand.dataio import TRecons
from grand.analysis.coords import array_shower as co
from grand.analysis import geom, fitting as fit
import grand.analysis.constants as cons

#-------------------------------
# Load GRANDlib
#-------------------------------
from grand.analysis.pipeline_nathan.scripts.loading import load_config

config_path = Path(__file__).parent.parent / "config.yaml"
config = load_config(config_path)

sys.path.append(config['paths']['grandlib_path'])

logger = logging.getLogger("grand.process")

SIGMA_T_NS = 5.0  # matches reconstruction.py's sigma=5e-9 used throughout PWF/SWF fits

def analyze_event(t_recons: TRecons, ev_no, run_no, config: dict) -> dict:   
    K = co.shower_direction_vector(t_recons.zenith_adf, t_recons.azimuth_adf)
    Xsource = np.asarray(t_recons.Xsource)[0]
    Xcore = geom.compute_core(K, Xsource)
    Xants = np.asarray(t_recons.Xants)
    l_ant_mean = np.mean(geom.distance_source_antenna(Xants, Xsource))
    npts = 100
    omega_cr_mean = np.mean(t_recons.omega_cr)
    v_cone = geom.generate_cone_surface_vectors(K, omega_cr_mean, npts)
    logger.info("Computing footprint with opening angle w_c =", np.rad2deg(omega_cr_mean), "deg")
    x_ell = np.zeros((npts, 3))
    for i, v in enumerate(v_cone):
        x_ell[i, :] = geom.compute_core(v, Xsource)

    # ---------------------------------------------------------------
    # Compute mean ADF model (averaged over all eta angles)
    # ---------------------------------------------------------------
    w ,adf_f = fit.ADF_fun(l_ant_mean,t_recons.scaling_factor, omega_cr_mean,t_recons.width)

    # Reduced chi2 for ADF fit
    chi2_adf_reduced = t_recons.chi2_adf/(t_recons.du_count-4)

    fsuptit = (
    f"Event {ev_no} Run {run_no}"
    )
    ftit_adf = (
    rf"$\Theta = {np.rad2deg(t_recons.zenith_adf):.1f}^\circ, "
    rf"\Phi = {np.rad2deg(t_recons.azimuth_adf):.1f}^\circ, "
    rf"\chi^2_\mathrm{{adf}} / \mathrm{{ndf}} = {chi2_adf_reduced:.2f}$"
    )
    results = {
        "event_number": t_recons.event_number,
        "run_number": t_recons.run_number,
        "zenith_adf": t_recons.zenith_adf,
        "azimuth_adf": t_recons.azimuth_adf,
        "scaling_factor": t_recons.scaling_factor,
        "omega_cr_mean": omega_cr_mean,
        "l_ant_mean": l_ant_mean,
        "chi2_adf": t_recons.chi2_adf,
        "du_count": t_recons.du_count,
        "fsuptit": fsuptit,
        "ftit_adf": ftit_adf,
        "K": K,
        "Xsource": Xsource,
        "Xcore": Xcore,
        "Xants": Xants,
        "x_ell": x_ell,
        "w": w,
        "adf_f": adf_f,
    }
    return results


def compute_timing_residuals(t_recons: TRecons) -> dict:
    """Compare measured peak times to the PWF and SWF model predictions, per antenna.

    Uses the real GRAND model functions (fit.PWF_model / fit.SWF_model) rather than
    re-deriving propagation times by hand, so the residuals stay consistent with what
    chi2_pwf/chi2_swf actually measure.
    """
    peak_times = np.asarray(t_recons.peak_time)  # seconds
    Xants = np.asarray(t_recons.Xants)  # meters, (N, 3)
    n_antennas = len(peak_times)
    t_exp_ns = peak_times * 1e9

    # PWF_model has no absolute time anchor - fit one common offset from all antennas.
    t_pwf_model_ns = fit.PWF_model(
        (t_recons.zenith_pwf, t_recons.azimuth_pwf), Xants,
        c=cons.c_light, n=cons.n_atm, groundAltitude=cons.groundAltitude,
    ) * 1e9
    offset_pwf_ns = np.mean(t_exp_ns - t_pwf_model_ns)
    t_pwf_ns = t_pwf_model_ns + offset_pwf_ns

    # SWF_model already includes the fitted t_s anchor - the same model chi2_swf is built from.
    t_swf_ns = fit.SWF_model(
        t_recons.zenith_swf, t_recons.azimuth_swf, t_recons.r_xmax, t_recons.t_s,
        Xants, groundAltitude=cons.groundAltitude, cr=cons.c_light,
    ) * 1e9

    resid_pwf_ns = t_exp_ns - t_pwf_ns
    resid_swf_ns = t_exp_ns - t_swf_ns

    # Shift all three series by the same constant, purely for readable axes - a common
    # additive shift cancels exactly in both residual definitions above.
    t0_ns = t_exp_ns.min()
    t_exp_ns, t_pwf_ns, t_swf_ns = t_exp_ns - t0_ns, t_pwf_ns - t0_ns, t_swf_ns - t0_ns

    # ndf_pwf=3: PWF_residuals() implicitly fits away a reference time via `-= res.mean()`,
    # on top of the 2 explicit angular params (theta, phi) - this differs from the informal,
    # unused "nants-2" comment inside PWF_loss, which doesn't match what its own residual
    # computation actually does. ndf_swf=4: theta, phi, r_xmax, t_s, all explicit in SWF_loss,
    # no implicit centering there.
    ndf_pwf, ndf_swf = n_antennas - 3, n_antennas - 4
    return {
        "t_exp_ns": t_exp_ns,
        "t_pwf_ns": t_pwf_ns,
        "resid_pwf_ns": resid_pwf_ns,
        "t_swf_ns": t_swf_ns,
        "resid_swf_ns": resid_swf_ns,
        "chi2_pwf_reduced": t_recons.chi2_pwf / ndf_pwf if ndf_pwf > 0 else np.nan,
        "chi2_swf_reduced": t_recons.chi2_swf / ndf_swf if ndf_swf > 0 else np.nan,
        "antenna_index": np.arange(n_antennas),
        "sigma_t_ns": SIGMA_T_NS,
    }
