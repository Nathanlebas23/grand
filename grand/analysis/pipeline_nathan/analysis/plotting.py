import sys
sys.path.append("/home/lpnhe/grand")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def plot_ADC_vs_omega(t_recons, omega_cr_mean, w, adf_f, fsuptit, ftit_adf, plot_dir):
    """Plot ADC peak amplitude vs angle ω and ADF model."""
    plt.figure()
    # Peak ADC values measured at each triggered DU
    plt.plot(np.rad2deg(t_recons.omega), np.asarray(t_recons.peak_amps), 'ob', label="max ADC @ DU")
    plt.errorbar(np.rad2deg(t_recons.omega), np.asarray(t_recons.peak_amps), 0.075 * np.asarray(t_recons.peak_amps), fmt='None', marker='o', markerfacecolor='b')
    # ADF amplitude at each DU
    plt.plot(np.rad2deg(t_recons.omega), t_recons.adf_amplitude, '+r', label="ADF fit @ DU")
    # Mean Cherenkov angle for all eta values
    plt.plot([np.rad2deg(omega_cr_mean), np.rad2deg(omega_cr_mean)], [0, max(np.asarray(t_recons.peak_amps)) * 1.5], label="mean Cherenkov angle")
    # Mean ADF model
    plt.plot(np.rad2deg(w), adf_f, "--r", label="mean ADF model")
    plt.ylim([0, max(t_recons.adf_amplitude) * 1.5])
    plt.xlim([0, 1.6])
    plt.xlabel("$\omega$ (deg)")
    plt.legend(loc="best")
    plt.suptitle(fsuptit, fontsize=10)
    plt.title(ftit_adf)
    plt.ylabel("Voltage (ADC)")
    plt.savefig(f"{plot_dir}/ADC_vs_omega_event_{t_recons.event_number}_run_{t_recons.run_number}.png")


def plot_footprint(t_recons, antenna_position, Xants, K, Xcore, x_ell, omega_cr_mean, fsuptit, ftit_adf, plot_dir):
    """Plot the 2D footprint of the event on the ground."""
    distm = 5000  # distance in meters
    xmin, xmax = -distm, distm
    ymin, ymax = -distm, distm

    plt.figure()
    # Triggered DUs: color and size proportional to amplitude
    sc = plt.scatter(-Xants[:, 1], Xants[:, 0], c=t_recons.adf_amplitude, cmap='viridis', s=np.asarray(t_recons.adf_amplitude), label="Triggered DUs")
    # All DUs on site
    plt.scatter(antenna_position['x'], antenna_position['y'], marker='+', color='red', label="DUs on site")
    plt.colorbar(sc, label='Peak amplitude (ADC)')
    # Shower direction arrow
    plt.arrow(-Xcore[1] + K[1] * 1e3, Xcore[0] - K[0] * 1e3, -K[1] * 1e3, K[0] * 1e3, head_width=1, head_length=1, fc='black', ec='black')
    # Footprint: core position (black dot) and Cherenkov ellipse (dashed line)
    plt.plot(-Xcore[1], Xcore[0], 'ko')
    plt.plot(-x_ell[:, 1], x_ell[:, 0], '--k', linewidth=2)
    plt.xlabel('Easting [m]')
    plt.ylabel('Northing [m]')
    plt.grid(True)
    plt.suptitle(fsuptit, fontsize=10)
    plt.title(ftit_adf)
    plt.legend()
    plt.axis('equal')
    plt.ylim([xmin, xmax])
    plt.xlim([ymin, ymax])
    plt.subplots_adjust(left=0.15)
    plt.savefig(f"{plot_dir}/footprint_event_{t_recons.event_number}_run_{t_recons.run_number}.png")

