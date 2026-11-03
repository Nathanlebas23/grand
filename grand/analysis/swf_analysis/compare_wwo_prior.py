'''
Compare SWF reconstructions with and without a Gaussian prior on Xmax.

Each CSV written by swf_analysis.py is one "run" of the fit. List them in RUNS:
the first one is the reference (the fit without prior), every other run is
compared to it event by event. To compare another prior setting, add a line.

'''

# ------------------------------
# Imports
# ------------------------------
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator

sys.path.append("/home/lpnhe/nl_style")
from nl_style import set_style, NL_COLORS

set_style()

# ------------------------------
# Settings
# ------------------------------
results_dir = "/home/lpnhe/grand/grand/analysis/swf_analysis/"
run_name = "GP80_20261021_171907_RUN10483_CD_20dB-GP65-64DUs-512trace-FY2Float-unFiltered-Th10080-TC50-CD-100000-1_dat"

# Legend label -> CSV file and colour. The first run is the reference.
RUNS = {
    "Without prior": {
        "file": f"swf_results_{run_name}_noprior.csv",
        "color": NL_COLORS["red"],
    },
    "Prior 750 ± 100 g/cm²": {
        "file": f"swf_results_{run_name}_prior_mu750_sig100.csv",
        "color": NL_COLORS["blue"],
    },
}

QUANTITIES = {
    "theta_swf_deg": (r"$\theta_{\rm swf}$ (deg)", False),
    "phi_swf_deg": (r"$\phi_{\rm swf}$ (deg)", False),
    "Xmax_gcm2": (r"$X_{\rm max}$ (g cm$^{-2}$)", True),
    "chi2_swf_red": (r"$\chi^2_{\rm swf}\,/\,n_{\rm dof}$", True),
}

DROP_FAILED_FITS = True  # remove events without a valid Xmax (see load_run)
SAVE_FIGURES = True      # save every figure as PDF in results_dir


# ------------------------------
# Functions
# ------------------------------
def angular_separation_deg(theta1, phi1, theta2, phi2):
    """Angle (deg) between two directions given by their zenith and azimuth (rad)."""
    cos_psi = np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2) + np.cos(theta1) * np.cos(theta2)
    return np.rad2deg(np.arccos(np.clip(cos_psi, -1, 1)))


def load_run(path, label):
    """Read one CSV from swf_analysis.py and add the columns used in the plots."""
    df = pd.read_csv(path)
    df["theta_swf_deg"] = np.rad2deg(df["theta_swf_rad"])
    df["phi_swf_deg"] = np.rad2deg(df["phi_swf_rad"]) % 360
    df["swf_pwf_sep_deg"] = angular_separation_deg(
        df["theta_swf_rad"], df["phi_swf_rad"], df["theta_pwf_rad"], df["phi_pwf_rad"]
    )

    # Xmax = 0 when the fit is stuck at the r_xmax boundary, NaN for upgoing directions (theta > 90 deg)
    failed = ~(df["Xmax_gcm2"] > 0)
    print(f"{label}: {len(df)} events, {failed.sum()} failed fits")
    if DROP_FAILED_FITS:
        df = df[~failed]
    return df.set_index("event_number")


def common_events(ref, df):
    """Keep only the events present in both runs, in the same order."""
    events = ref.index.intersection(df.index)
    return ref.loc[events], df.loc[events]


def shared_bins(arrays, log=False):
    """Bin edges computed on all runs together, so that the histograms are comparable."""
    values = np.concatenate(arrays)
    values = values[np.isfinite(values)]
    if log:
        return 10 ** np.histogram_bin_edges(np.log10(values[values > 0]), bins="auto")
    return np.histogram_bin_edges(values, bins="auto")


def hist_runs(ax, runs, column, log=False):
    """One histogram per run: the reference is filled, the others are outlined."""
    bins = shared_bins([df[column] for df in runs.values()], log=log)
    for i, (label, df) in enumerate(runs.items()):
        style = dict(histtype="stepfilled", alpha=0.35) if i == 0 else dict(histtype="step", lw=1.8)
        ax.hist(df[column].dropna(), bins=bins, color=RUNS[label]["color"], label=f"{label} ({len(df)})", **style)
    if log:
        ax.set_xscale("log")
    ax.set_ylabel("Events")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_minor_locator(NullLocator())


def draw_prior(ax, label, df, vertical=True):
    """Shade mu ± sigma if this run was fitted with a prior on Xmax."""
    if not df["use_prior"].iloc[0]:
        return
    mu, sigma = df["mu_xmax_gcm2"].iloc[0], df["sigma_xmax_gcm2"].iloc[0]
    color = RUNS[label]["color"]
    if vertical:
        ax.axvspan(mu - sigma, mu + sigma, color=color, alpha=0.15, lw=0)
        ax.axvline(mu, color=color, ls="--", lw=1)
    else:
        ax.axhspan(mu - sigma, mu + sigma, color=color, alpha=0.15, lw=0)
        ax.axhline(mu, color=color, ls="--", lw=1)


def compared_runs(runs):
    """(reference label, reference) and the list of the other runs."""
    ref_label, ref = next(iter(runs.items()))
    others = list(runs.items())[1:]
    return ref_label, ref, others


def save(fig, name):
    if SAVE_FIGURES:
        fig.savefig(f"{results_dir}compare_wwo_prior_{name}.pdf")


# ------------------------
# Plotting functions
# ------------------------
def plot_distributions(runs):
    """Histograms of the reconstructed parameters, one curve per run."""
    nrows = (len(QUANTITIES) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(10, 3.8 * nrows), layout="constrained")
    axes = axes.ravel()

    for ax, (column, (xlabel, log)) in zip(axes, QUANTITIES.items()):
        hist_runs(ax, runs, column, log)
        ax.set_xlabel(xlabel)
        if column == "Xmax_gcm2":
            for label, df in runs.items():
                draw_prior(ax, label, df)

    for ax in axes[len(QUANTITIES):]:
        ax.set_visible(False)

    fig.suptitle("SWF reconstruction: distributions")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=len(runs), frameon=False)
    save(fig, "distributions")


def plot_event_by_event(runs):
    """Each run against the reference, event by event: unchanged events sit on y = x."""
    ref_label, ref, others = compared_runs(runs)
    nrows = (len(QUANTITIES) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(9, 4.5 * nrows), layout="constrained")
    axes = axes.ravel()

    for ax, (column, (title, log)) in zip(axes, QUANTITIES.items()):
        for label, df in others:
            ref_c, df_c = common_events(ref, df)
            ax.scatter(ref_c[column], df_c[column], s=10, color=RUNS[label]["color"], alpha=0.6, lw=0,
                       label=label, zorder=3)

        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        # Same range on both axes, so that y = x is the diagonal
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.axline((1, 1), (2, 2), color=NL_COLORS["dark"], ls=":", lw=1, zorder=1)
        if column == "Xmax_gcm2":
            for label, df in others:
                draw_prior(ax, label, df, vertical=False)

        ax.set_title(title)
        ax.set_xlabel(ref_label)
        ax.set_ylabel("With prior")

    for ax in axes[len(QUANTITIES):]:
        ax.set_visible(False)

    fig.suptitle("SWF reconstruction: event by event")
    if len(others) > 1:
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=len(others), frameon=False)
    save(fig, "event_by_event")


def plot_xmax_vs_zenith(runs):
    """Xmax against zenith angle, one panel per run: Xmax is hardest to constrain for inclined showers."""
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4.5), sharex=True, sharey=True,
                             layout="constrained")

    for ax, (label, df) in zip(axes, runs.items()):
        draw_prior(ax, label, df, vertical=False)
        ax.scatter(df["theta_swf_deg"], df["Xmax_gcm2"], s=10, color=RUNS[label]["color"], alpha=0.6, lw=0)
        ax.set_yscale("log")
        ax.set_title(label)
        ax.set_xlabel(r"$\theta_{\rm swf}$ (deg)")

    axes[0].set_ylabel(r"$X_{\rm max}$ (g cm$^{-2}$)")
    save(fig, "xmax_vs_zenith")


def plot_prior_effect(runs):
    """What the prior changes for each event, against Xmax without prior."""
    ref_label, ref, others = compared_runs(runs)
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 9), sharex=True, layout="constrained")

    for label, df in others:
        ref_c, df_c = common_events(ref, df)
        shifts = [
            df_c["Xmax_gcm2"] - ref_c["Xmax_gcm2"],
            angular_separation_deg(ref_c["theta_swf_rad"], ref_c["phi_swf_rad"],
                                   df_c["theta_swf_rad"], df_c["phi_swf_rad"]),
            df_c["chi2_swf_red"] - ref_c["chi2_swf_red"],
        ]
        for ax, shift in zip(axes, shifts):
            ax.scatter(ref_c["Xmax_gcm2"], shift, s=10, color=RUNS[label]["color"], alpha=0.6, lw=0,
                       label=label, zorder=3)

    ylabels = [
        r"$\Delta X_{\rm max}$ (g cm$^{-2}$)",
        "Direction change (deg)",
        r"$\Delta\,(\chi^2_{\rm swf}/n_{\rm dof})$",
    ]
    # symlog: linear close to 0 (below linthresh), logarithmic beyond, to see small and large changes together
    linthresh = [1, 1e-3, 1e-2]
    for ax, ylabel, lin in zip(axes, ylabels, linthresh):
        for label, df in others:
            draw_prior(ax, label, df)
        ax.axhline(0, color=NL_COLORS["dark"], lw=0.8)
        ax.set_yscale("symlog", linthresh=lin)
        ax.set_ylabel(ylabel)

    axes[-1].set_xscale("log")
    axes[-1].set_xlabel(rf"$X_{{\rm max}}$, {ref_label.lower()} (g cm$^{{-2}}$)")
    if len(others) > 1:
        axes[0].legend()
    fig.suptitle(f"What the prior changes ($\\Delta$ = with prior $-$ {ref_label.lower()})")
    save(fig, "prior_effect")


def plot_swf_vs_pwf(runs):
    """Angle between the SWF and PWF directions, for each run."""
    fig, ax = plt.subplots(layout="constrained")
    hist_runs(ax, runs, "swf_pwf_sep_deg", log=True)
    ax.set_xlabel("Angle between SWF and PWF directions (deg)")
    ax.legend(loc="upper left")
    save(fig, "swf_vs_pwf")


def print_summary(runs, n_rows=10):
    """Print, for each run, the events whose Xmax moved the most with respect to the reference."""
    ref_label, ref, others = compared_runs(runs)
    for label, df in others:
        ref_c, df_c = common_events(ref, df)
        table = pd.DataFrame({
            "n_ant": ref_c["n_antennas"],
            "theta": ref_c["theta_swf_deg"],
            "Xmax_ref": ref_c["Xmax_gcm2"],
            "Xmax_run": df_c["Xmax_gcm2"],
            "dXmax": df_c["Xmax_gcm2"] - ref_c["Xmax_gcm2"],
            "d_dir_deg": angular_separation_deg(ref_c["theta_swf_rad"], ref_c["phi_swf_rad"],
                                                df_c["theta_swf_rad"], df_c["phi_swf_rad"]),
            "chi2_ref": ref_c["chi2_swf_red"],
            "chi2_run": df_c["chi2_swf_red"],
        })
        table = table.sort_values("dXmax", key=abs, ascending=False)
        print(f"\n{label} vs {ref_label}: {len(table)} common events, largest Xmax shifts first")
        print(table.head(n_rows).round(2).to_string())


# ------------------------
# Main
# ------------------------
runs = {label: load_run(results_dir + run["file"], label) for label, run in RUNS.items()}
print(f"Loaded {len(runs)} runs: {runs}")

print_summary(runs)
plot_distributions(runs)
plot_event_by_event(runs)
plot_xmax_vs_zenith(runs)
plot_prior_effect(runs)
plot_swf_vs_pwf(runs)
plt.show()
