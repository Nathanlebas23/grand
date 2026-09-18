import sys
sys.path.append("/home/lpnhe/grand")

import numpy as np
import logging

from grand.dataio import TRecons
from grand.analysis.coords import array_shower as co
from grand.analysis import geom, fitting as fit

logger = logging.getLogger("grand.process")

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
