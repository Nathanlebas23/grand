'''
Nathan & Emma analysis about SWF prior 
30/10/2026

'''

# ------------------------------
# Imports
# ------------------------------
import os
import sys
sys.path.append('/home/lpnhe/grand/')
sys.path.append('/home/lpnhe/pipeline/')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.optimize import brentq

from grand.analysis.coords import array_shower as co
from grand.analysis import geom, fitting as fit
import grand.analysis.constants as cons

from scipy.optimize import differential_evolution

sys.path.append("/home/lpnhe/nl_style")
from nl_style import set_style , NL_COLORS

from scripts.analysis_utils import build_xants
from scripts.loading import load_antenna_positions, load_nutrig_template
from grand.dataio import TADC
import grand.analysis.signals.extraction as ext
import grand.analysis.constants as cons
import grand.analysis.fitting as fit
import grand.analysis.geom as geom

set_style()
# -------------------------------
# Paths
# ------------------------------
root_path = '/home/lpnhe/pipeline/CD_HIGH_THRESH_80_100/GP80_20261021_171907_RUN10483_CD_20dB-GP65-64DUs-512trace-FY2Float-unFiltered-Th10080-TC50-CD-100000-1_dat.root'
template_path = "/home/lpnhe/pipeline/templates_ZHAireS_DC2.1rc4_RAW_XY_THRESH_100_5.txt"
rtk_path = "/home/lpnhe/grand/grand/analysis/example/RTK_pos/rtk_pos_gp65_from_xls_gdl.csv"
output_dir = "/home/lpnhe/grand/grand/analysis/swf_analysis/"
print(f"Opening root file: {root_path}")

# -----------------------------
# Entry to plot
# -----------------------------
ENTRY_TO_PLOT = 0

# -----------------------------
# Gaussian prior on Xmax (g/cm²) for the SWF fit
# USE_PRIOR = False gives the reference fit without prior
# -----------------------------
USE_PRIOR = False
mu_xmax = 750. if USE_PRIOR else None
sigma_xmax = 100. if USE_PRIOR else None


# -----------------------------
# Fonctions 
# -----------------------------
def build_adc_traces(t_ana, n_antennas: int) -> np.ndarray:
    adc_traces = np.asarray([np.asarray(t_ana.trace_ch[i], dtype=float)[1:4] for i in range(n_antennas)],
        dtype=float,
    )
    return adc_traces

# For a given theta, phi, r_xmax, Xants and tants, 
# compute the chi2 of the SWF model with analytically optimized emission time t_s.
def swf_chi2_profile(theta, phi, r_xmax, Xants, tants, sigma=5e-9):
    """
    SWF chi2 with analytically optimized emission time t_s.
    """
    t_model = fit.SWF_model(
        theta, phi, r_xmax, 0., Xants
    )

    # Optimal emission time
    t_s = np.mean(tants - t_model)

    # Same bounds as recons_swf
    t_s = np.clip(
        t_s,
        -2e6 / cons.c_light,
        0.,
    )

    chi2 = np.sum(
        ((tants - t_model - t_s) / sigma)**2
    )

    return chi2, t_s

def plot_swf_minima(
    theta_pwf,
    phi_pwf,
    Xants,
    tants,
    first_fit,
    n_r=120,
    sigma=5e-9,
    ):
    """
    Explore the SWF chi2 landscape without prior.

    - Fixed angles: optimize t_s only.
    - Profile: optimize theta, phi and t_s for each r_xmax.
    - Repeat differential evolution with different seeds.
    """

    theta_best, phi_best, r_best, ts_best = first_fit

    # Same angular bounds as recons_swf
    bounds = [
        (theta_pwf - np.deg2rad(5), theta_pwf + np.deg2rad(5)),
        (phi_pwf - np.deg2rad(5), phi_pwf + np.deg2rad(5)),
    ]

    # Different r_xmax values to explore the chi2 landscape
    r_values = np.unique(np.r_[
        np.geomspace(100., 2e6, n_r),
        max(r_best, 100.),
    ])

    chi2_fixed = np.zeros(len(r_values))
    chi2_profile = np.zeros(len(r_values))
    ts_profile = np.zeros(len(r_values))

    for i, r in enumerate(r_values):
        chi2_fixed[i], ts_fixed = swf_chi2_profile(
            theta_best, phi_best, r,
            Xants, tants, sigma
        )

        best_chi2 = chi2_fixed[i]
        best_ts = ts_fixed

        for angles0 in [
            [theta_best, phi_best],
            [theta_pwf, phi_pwf],
        ]:

            result = minimize(
                lambda angles: swf_chi2_profile(
                    angles[0], angles[1], r,
                    Xants, tants, sigma
                )[0],
                x0=angles0,
                method="Powell",
                bounds=bounds,
                options={"maxiter": 60},
            )

            # result = differential_evolution(
            #     lambda angles: swf_chi2_profile(
            #         angles[0], angles[1], r,
            #         Xants, tants, sigma
            #     )[0],
            #     bounds=bounds,
            #     maxiter=60,
            #     tol=1e-6,
            #     mutation=(0.5, 1),
            #     recombination=0.7,
            #     seed=None,
            #     x0 = np.array(bounds, dtype=np.float64).mean(axis=1)
            # )

            if result.fun < best_chi2:
                best_chi2, best_ts = swf_chi2_profile(
                    result.x[0], result.x[1], r,
                    Xants, tants, sigma
                )

        chi2_profile[i] = best_chi2
        ts_profile[i] = best_ts


    chi2_min = np.min(chi2_profile)

    # ----------------------------------------
    # Plot
    # ----------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(8, 9),
        sharex=True,
    )

    # Chi2 as a function of r_xmax
    ax1.plot(
        r_values / 1e3,
        chi2_fixed - chi2_min,
        label=r"Fixed $\theta$, $\phi$",
        alpha=0.8,
    )

    ax1.plot(
        r_values / 1e3,
        chi2_profile - chi2_min,
        label=r"Profiled $\theta$, $\phi$, $t_s$",
        linewidth=2,
    )

    ax1.set_ylabel(r"$\Delta\chi^2_{\mathrm{SWF}}$")
    
    ax1.axhline()
    ax1.set_xscale("log")
    ax1.set_yscale("symlog", linthresh=1.)
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Optimal emission time
    ax2.plot(
        r_values / 1e3,
        ts_profile * 1e3,
    )

    ax2.set_xlabel(r"$r_{\max}$ (km)")
    ax2.set_ylabel(r"Optimal $t_s$ (ms)")
    ax2.set_xscale("log")
    ax2.grid(alpha=0.3)

    fig.suptitle("SWF chi2 landscape — No prior")
    plt.tight_layout()

    plt.show()



def scan_adf_rmax(theta_ref, phi_ref, peak_amps, Xants, r_values_km):

    results = []
    profiles = {}

    for r_km in r_values_km:

        r = r_km * 1e3

        # Source position for this distance
        Xsource = fit.compute_Xsource_cartesian_coords(
            theta_ref, phi_ref, r
        )[0]

        # ADF reconstruction
        theta_adf, phi_adf, delta_omega, scaling_factor = fit.recons_ADF(
            theta_ref,
            phi_ref,
            peak_amps,
            Xants,
            Xsource,
        )

        # ADF chi2
        params = (theta_adf, phi_adf, delta_omega, scaling_factor)

        chi2_adf = fit.ADF_loss(
            params,
            peak_amps,
            Xants,
            Xsource,
        )

        # Model amplitudes and angles
        eta, omega, omega_cr, l_ant, amplitude_model = fit.ADF_parameters(
            theta_adf,
            phi_adf,
            delta_omega,
            scaling_factor,
            Xants,
            Xsource,
            groundAltitude=cons.groundAltitude,
            Bvec=cons.Bvec,
        )

        # Atmospheric depth
        Xmax = fit.compute_Xmax_grammage(
            theta_ref, phi_ref, r
        )

        results.append({
            "r_km": r_km,
            "Xmax_gcm2": Xmax,
            "chi2_adf_red": chi2_adf / (len(peak_amps) - 4),
            "theta_adf": np.rad2deg(theta_adf),
            "phi_adf": np.rad2deg(phi_adf),
            "delta_omega": delta_omega,
        })

        profiles[r_km] = (
            np.rad2deg(omega),
            np.asarray(peak_amps),
            np.asarray(amplitude_model),
        )

    return pd.DataFrame(results), profiles


# ------------------------
# Main  
# ------------------------
data = TADC(str(root_path))
nutrig_template = load_nutrig_template(str(template_path))

event_number = int(data.event_number)
run_number = int(data.run_number)
n_entries = int(data.get_number_of_entries())

# ---------------------------
# Initialize 
# ---------------------------
event_du_ids = []
entry_list = []
event_number_list = []
n_antennas_list = []
thtas_pwf_list = []
phis_pwf_list = []
chi2_pwf_list = []

thetas_swf_list = []
phis_swf_list = []
r_xmax_list = []
xmax_list = []
t_s_list = []
chi2_swf_list = []

# -------------------------------------------
# Loop over all event in the TADC file
# -------------------------------------------
for entry in range(n_entries) :
    # if entry % 100 == 0:
    #    print(f"Processing  {entry}/{n_entries}")


        if entry != ENTRY_TO_PLOT:
            continue

        data.get_entry(entry)

        event_du_ids.extend(np.asarray(data.du_id, dtype=int).reshape(-1))
        seconds = np.asarray(data.du_seconds, dtype=np.int64).reshape(-1)
        nanoseconds = np.asarray(data.du_nanoseconds, dtype=np.int64).reshape(-1)

        du_ids = np.asarray(data.du_id, dtype=int).reshape(-1)
        n_antennas = du_ids.size
        if n_antennas < 5:
            print(f"Event {entry}: only {n_antennas} antennas, skipping SWF fit")
            continue

        # ADC traces for all antennas in the event, shape (n_antennas, 3)
        adc_traces = build_adc_traces(
                data,
                n_antennas
        )

        peak_amps = np.asarray(
            [
                ext.get_peak_amplitude(
                    adc_traces[i],
                    channels=[0, 1, 2],
                )
                for i in range(n_antennas)
            ],
            dtype=float,
        )

        # ------------------------------------------------------
        # Timing 
        # ------------------------------------------------------
        t0 = np.asarray(
            ext.compute_t0(data),
            dtype=float,
        )

        peak_times = np.asarray(
            [
                ext.get_peak_time_adc(
                    adc_traces[i],
                    nutrig_template,
                    t0[i],
                )
                for i in range(n_antennas)
            ],
            dtype=float,
        )

        # --------------------------
        # Position 
        # --------------------------
        antenna_position = load_antenna_positions(str(rtk_path))
        antenna_by_id = antenna_position.set_index("DU_id")


        Xants = build_xants(
            du_ids,
            antenna_by_id,
            event_number=event_number,
            is_simulation=False,
            list_lat=data.gps_lat,
            list_lon=data.gps_long,
            list_alt=data.gps_alt,
        )

        theta_pwf, phi_pwf = fit.PWF_semianalytical(
            Xants,
            peak_times,
            verbose=False,
            c=cons.c_light,
            n=cons.n_atm,
            sigma=5e-9,
        )
        thtas_pwf_list.append(theta_pwf)
        phis_pwf_list.append(phi_pwf)

        # Raw (non-reduced) chi2, matching the current TRecons convention.
        chi2_pwf = fit.PWF_loss(
            (theta_pwf, phi_pwf),
            Xants,
            peak_times,
            verbose=False,
            c=cons.c_light,
            n=cons.n_atm,
            sigma=5e-9,
        )

        chi2_pwf_red = chi2_pwf / (n_antennas - 3)
        chi2_pwf_list.append(chi2_pwf_red)

        # ---------------------------------------------------------------
        # Spherical Wave Fit (SWF)
        # ---------------------------------------------------------------
        theta_swf, phi_swf, r_xmax, t_s = fit.recons_swf(
            theta_pwf,
            phi_pwf,
            peak_times,
            Xants,
            sigma=5e-9,
            mu_xmax=mu_xmax,
            sigma_xmax=sigma_xmax,
        )

        Xmax = fit.compute_Xmax_grammage(
            theta_swf,
            phi_swf,
            r_xmax,
        )

        # Keep the (1, 3) shape for TRecons.Xsource.
        Xsource = np.asarray(
            fit.compute_Xsource_cartesian_coords(
                theta_swf,
                phi_swf,
                r_xmax,
            ),
            dtype=float,
        )

        if Xsource.shape != (1, 3):
            raise ValueError(
                f"Event {event_number}: unexpected Xsource shape "
                f"{Xsource.shape}, expected (1, 3)"
            )

        # Geometry/ADF functions expect the flattened source vector.
        Xsource_vec = Xsource[0]

        l_ant = geom.distance_source_antenna(
            Xants,
            Xsource_vec,
        )

        # Raw (non-reduced) chi2 stored in TRecons.
        chi2_swf = fit.SWF_loss(
            theta_swf,
            phi_swf,
            r_xmax,
            t_s,
            Xants,
            peak_times,
            sigma=5e-9,
        )

        # Preserve the exact normalization currently used by the pipeline.
        chi2_swf_red = chi2_swf / (n_antennas - 2)

        print(
            f"Event {entry}: "
            f'chi2_swf_red = {chi2_swf_red:.3f}, '
            f'chi2_pwf_red = {chi2_pwf_red:.3f}, '
            f'Xmax = {Xmax:.1f} g/cm², '
            f'theta_swf = {np.rad2deg(theta_swf):.3f} deg, '
            f'phi_swf = {np.rad2deg(phi_swf):.3f} deg, '
            f'r_xmax = {r_xmax:.1f} m, '
        )

        entry_list.append(entry)
        event_number_list.append(int(data.event_number))
        n_antennas_list.append(n_antennas)
        thetas_swf_list.append(theta_swf)
        phis_swf_list.append(phi_swf)
        r_xmax_list.append(r_xmax)
        xmax_list.append(Xmax)
        t_s_list.append(t_s)
        chi2_swf_list.append(chi2_swf_red)


# ------------------------
# Save results
# ------------------------
results = pd.DataFrame({
    "entry": entry_list,
    "event_number": event_number_list,
    "n_antennas": n_antennas_list,
    "theta_pwf_rad": thtas_pwf_list,
    "phi_pwf_rad": phis_pwf_list,
    "chi2_pwf_red": chi2_pwf_list,
    "theta_swf_rad": thetas_swf_list,
    "phi_swf_rad": phis_swf_list,
    "r_xmax_m": r_xmax_list,
    "Xmax_gcm2": xmax_list,
    "t_s": t_s_list,
    "chi2_swf_red": chi2_swf_list,
    "use_prior": USE_PRIOR,
    "mu_xmax_gcm2": mu_xmax if USE_PRIOR else np.nan,
    "sigma_xmax_gcm2": sigma_xmax if USE_PRIOR else np.nan,
})

root_name = os.path.splitext(os.path.basename(root_path))[0]
prior_tag = f"prior_mu{mu_xmax:.0f}_sig{sigma_xmax:.0f}" if USE_PRIOR else "noprior"
csv_path = f"{output_dir}swf_results_{root_name}_{prior_tag}.csv"
results.to_csv(csv_path, index=False)
print(f"SWF results saved to {csv_path}")


# --------------------------------------------------------------------------
# Analyse SWF minima and ADF reconstruction as a function of r_xmax
# --------------------------------------------------------------------------
print("Plotting SWF minima...")
print(f"theta_pwf = {np.rad2deg(theta_pwf):.3f} deg, phi_pwf = {np.rad2deg(phi_pwf):.3f} deg")

plot_swf_minima(
    theta_pwf,
    phi_pwf,
    Xants,
    peak_times,
    first_fit=(theta_swf, phi_swf, r_xmax, t_s),
    n_r=120,
    sigma=5e-9,
)

# Distances to explore (km)
r_values_km = np.arange(10, 80, 2)

adf_results, adf_profiles = scan_adf_rmax(
    theta_pwf,
    phi_pwf,
    peak_amps,
    Xants,
    r_values_km,
)

print(adf_results.to_string(index=False))
