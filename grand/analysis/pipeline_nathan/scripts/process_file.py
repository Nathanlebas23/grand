"""
Process a file and generate the corresponding output file.
This script reads a ROOT file, processes the events, and writes the results to an output ROOT file. It also generates a log file if specified in the configuration.
The script uses the configuration file to get the paths and parameters for processing.
"""
import sys  
import logging
import re
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------
# Load GRANDlib
#-------------------------------
from grand.analysis.pipeline_nathan.scripts.loading import load_config, log_memory

config_path = Path(__file__).parent.parent / "config.yaml"
config = load_config(config_path)

sys.path.append(config['paths']['grandlib_path'])

logger = logging.getLogger("grand.process")

from grand.aoi import EventList
from grand.dataio import TRecons
import grand.analysis.signals.extraction as ext
from grand.analysis.pipeline_nathan.scripts.reconstruction import reconstruct_event
from grand.analysis.pipeline_nathan.scripts.sanity_plot import plot_traces
from grand.analysis.pipeline_nathan.scripts.cuts.apply_nutrig_cut import (
    compute_nutrig_event,
    passes_nutrig_cut,
)
from grand.analysis.pipeline_nathan.scripts.loading import get_run_number
from grand.analysis.pipeline_nathan.scripts.sims_utils import get_antenna_positions_from_run


def process_file(
    rootfile_path: Path,
    output_dir: Path,
    antenna_position: pd.DataFrame,
    n_ant_cut: int,
    nutrig_templates_txt,
    templates_npz_path,
    nutrig_src_path,
    rho_min_threshold: float,
    rho_mean_threshold: float,
    chi2_adf_threshold: float,
    theta_adf_threshold: float,
    omega_min: float,
    omega_max: float,
    omega_excess_threshold: float,
    amplitude_excess_threshold: float,
    limit_events=None,
    do_plot=False,
    is_simulation=False,
    ) -> tuple[int, int, int, int, int, int, int, int]:
    """Process a single ROOT file and write the output to the specified directory."""

    logger.info(f"Opening ROOT file: {rootfile_path}")


    # -----------------------------------
    # Open EventList
    # -----------------------------------
    if is_simulation:
        # En simulation, rootfile_path EST le dossier sim_*_<index> : il contient
        # adc_*/efield_*/run_*, que DataDirectory associe entre eux.
        el = EventList(str(rootfile_path))
    else:
        el = EventList(
            str(rootfile_path),
            use_trawvoltage=True,
            trawvoltage_channels=[1, 2, 3],
        )
 
    # Determine the list of events to process
    targets = list(el.event_list)
    if limit_events is not None:
        targets = targets[:limit_events]
    logger.info(f"{len(targets)} events to process in {rootfile_path.name}")

    trecons = TRecons()

    files_creation_time = None
    n_pass = 0
    n_cut_nant = 0
    n_cut_nutrig = 0
    n_cut_chi2_adf = 0
    n_cut_theta_adf = 0
    n_cut_omega_band = 0
    n_cut_hight_excess_omega = 0
    n_fail = 0

    # --------------------------------------
    # Iterate over events in the ROOT file
    # --------------------------------------
    for event_number, run_number in targets: # Iterate over the events in the ROOT file 0,1,..
        if event_number % 50 == 0:
            log_memory(f"event {event_number} start")

        e = el.get_event(event_number=event_number, run_number=run_number)

        if e is None:
            logger.warning(f"Event {event_number} (run {run_number}) not found in {rootfile_path.name}")
            n_fail += 1
            continue

        # Reset per event: t0 is only assigned on the simulation branch below, and
        # plot_traces() reads it back. Without this, a failed event could silently reuse
        # the previous event's t0 when building the peak-time overlay.
        t0 = None

        if is_simulation:
            adc_run_number = 0
        else:
            run_number = get_run_number(rootfile_path, e)

        n_antennas = len(e.antennas)
        logger.debug(f"Event {event_number} (run {run_number}) has {n_antennas} antennas triggered")

        if n_antennas < n_ant_cut:
            logger.debug(
                f"Event {event_number} (run {run_number}) skipped: "
                f"only {n_antennas} antennas triggered "
                f"(cut threshold: {n_ant_cut})"
            )
            n_cut_nant += 1
            continue



        # -----------------------------------
        # Apply NUTRIG cut
        # -----------------------------------
        # nutrig_result = compute_nutrig_event(
        #     e,
        #     templates_npz_path=templates_npz_path,
        #     nutrig_src_path=nutrig_src_path,
        #     simulation=is_simulation
        # )
        if is_simulation:
            tadc = el.directory.tadc
            tadc.get_event(event_number, adc_run_number)

            ADC_traces = np.asarray(
                tadc.trace_ch,
                dtype=float,
            )
                       
            logger.debug(
                "Simulation event %s: Event run=%s, TADC run=%s, ADC shape=%s",
                event_number,
                run_number,
                adc_run_number,
                ADC_traces.shape,
            )

            nutrig_result = compute_nutrig_event(
                e,
                templates_npz_path=templates_npz_path,
                nutrig_src_path=nutrig_src_path,
                adc_traces=ADC_traces,
            )
        else:
            nutrig_result = compute_nutrig_event(
                e,
                templates_npz_path=templates_npz_path,
                nutrig_src_path=nutrig_src_path,
                adc_traces=None,
            )

        if not passes_nutrig_cut(
            nutrig_result,
            rho_min_threshold,
            rho_mean_threshold
        ):
            logger.debug(
                f"Event {event_number} (run {run_number}) skipped: "
                f"NUTRIG cut (rho_min={nutrig_result['rho_min']:.3f}, "
                f"rho_mean={nutrig_result['rho_mean']:.3f}, "
                f"n_valid={nutrig_result['n_valid']}/{len(nutrig_result['rho_max'])})"
            )
            n_cut_nutrig += 1
            continue

        #--------------------------------
        # Reconstruct the event
        #--------------------------------
        try:
            if is_simulation:

                tadc = el.directory.tadc
                tadc.get_event(event_number, adc_run_number)

                ADC_traces = np.asarray(tadc.trace_ch, dtype=float)

                t0 = ext.compute_t0_sims(tadc)

                trun = el.directory.trun
                du_indices = tadc.get_dus_indices_in_run(trun)
                Xants = get_antenna_positions_from_run(trun, du_indices)

                logger.debug(
                    "Simulation event %s: Event run=%s, TADC run=%s, ADC shape=%s",
                    event_number,
                    run_number,
                    adc_run_number,
                    ADC_traces.shape,
                )

                t0 = ext.compute_t0_sims(tadc)

                reconstruct_event(
                    e,
                    antenna_position,
                    trecons,
                    run_number,
                    nutrig_template=nutrig_templates_txt, # Single 1D template (load_nutrig_template already selects row 0)
                    ADC_traces=ADC_traces,
                    is_simulation=is_simulation,
                    t0=t0,
                    Xants=Xants
                )

            else:
                reconstruct_event(
                    e,
                    antenna_position,
                    trecons,
                    run_number,
                    nutrig_template=nutrig_templates_txt, # Single 1D template (load_nutrig_template already selects row 0)
                    ADC_traces=nutrig_result["ADC_traces"],
                    is_simulation=is_simulation,
                    t0=None,
                    Xants=None
                )

            # ------------------------------------------
            # Cuts on reconstructed event parameters
            # ------------------------------------------
            chi2_adf = trecons.chi2_adf
            if not np.isfinite(chi2_adf) or chi2_adf > chi2_adf_threshold:
                logger.debug(
                    f"Event {event_number} (run {run_number}) skipped: "
                    f"chi2_adf cut (chi2_adf={chi2_adf:.1f} > threshold={chi2_adf_threshold:.1f})"
                )
                n_cut_chi2_adf += 1
                continue

            theta_adf = trecons.zenith_adf
            if not np.isfinite(theta_adf) or theta_adf > theta_adf_threshold:
                logger.debug(
                    f"Event {event_number} (run {run_number}) skipped: "
                    f"theta_adf cut (theta_adf={theta_adf} > threshold={theta_adf_threshold:.3f})"
                )
                n_cut_theta_adf += 1
                continue

            omega = np.asarray(trecons.omega, dtype=float).reshape(-1)
            omega_deg = np.rad2deg(omega)
            logger.debug(
                f"Event {event_number} (run {run_number}) omega values: "
                f"{omega_deg} (min={np.nanmin(omega_deg):.2f}, max={np.nanmax(omega_deg):.2f})"
            )

            if (
                omega.size == 0
                or not np.all(np.isfinite(omega))
                or np.any((omega_deg < omega_min) | (omega_deg > omega_max))
            ):
                logger.debug(
                    f"Event {event_number} (run {run_number}) skipped: "
                    f"omega cut (range=[{np.nanmin(omega_deg):.2f}, {np.nanmax(omega_deg):.2f}], "
                    f"allowed=[{omega_min:.2f}, {omega_max:.2f}])"
                )
                n_cut_omega_band += 1
                continue

            peak_amps = np.asarray(trecons.peak_amps, dtype=float)

            if np.any(
                (np.rad2deg(omega) > omega_excess_threshold)
                & (peak_amps > amplitude_excess_threshold)
            ):
                logger.debug(
                    f"Event {event_number} (run {run_number}) skipped: "
                    f"high omega excess cut "
                    f"(omega_excess_threshold={omega_excess_threshold:.2f}, "
                    f"amplitude_excess_threshold={amplitude_excess_threshold:.2f})"
                )
                n_cut_hight_excess_omega += 1
                continue
                        
            trecons.rho_x = nutrig_result["rho_x"]
            trecons.rho_y = nutrig_result["rho_y"]
            trecons.rho_max = nutrig_result["rho_max"]
            trecons.rho_min = nutrig_result["rho_min"]
            trecons.rho_mean = nutrig_result["rho_mean"]

            # Captured now, before trecons.fill()/the next event's reconstruct_event() call
            # overwrites trecons's staged values - used below for the peak-time overlay.
            peak_times = np.asarray(trecons.peak_time).copy()

            shower = e.shower
            e.shower = None


            # This take the creation time of the first event 
            # and uses it for all subsequent events to ensure consistent output filenames and don't create other trees.

            #event 0
            #   files_creation_time = None
            #         ↓
            #   first value generated
            #         ↓
            #   e.write() can modify it
            #         ↓
            #   capture the value after e.write() and use it for all subsequent events
            #         ↓
            #   files_creation_time = 20260918_143047


            # event 1
            #   ↓
            # e.files_creation_time = 20260918_143047
            #   ↓
            # same shower ROOT


            # event 2
            #   ↓
            # e.files_creation_time = 20260918_143047
            #   ↓
            # same shower ROOT


            # event 3
            #   ↓
            # same shower ROOT


            try:
                e.files_creation_time = (
                    files_creation_time
                    or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                )
                e.write(out_dir=str(output_dir))
            finally:
                e.shower = shower

            files_creation_time = e.files_creation_time

            shower_path = (
                output_dir
                / f"shower_{files_creation_time}_0-0_L1_0000.root" # This is the default naming convention used by Event.write(). 
            )

            try:
                e.write_shower(str(shower_path))
            finally:
                if getattr(e, "tshower", None) is not None:
                    e.tshower.stop_using()

            trecons.fill()

        except Exception as exc:
            logger.warning(
                "Event processing failed for event %s (run %s): %s",
                event_number,
                run_number,
                exc,
            )
            logger.debug("Full traceback:", exc_info=True)
            n_fail += 1
            continue

        n_pass += 1
        logger.debug(f"Event {event_number} (run {run_number}) reconstructed OK")


        # -----------------------------------
        # Plotting Traces
        # -----------------------------------
        if do_plot:
            log_memory(f"event {event_number} before plot")
            try:
                plot_traces(
                    e,
                    output_dir=output_dir,
                    peak_times=peak_times,
                    ADC_traces=nutrig_result["ADC_traces"],
                    nutrig_result=nutrig_result,
                    t0_ns=t0,
                )
                logger.debug("Open matplotlib figures: %s", plt.get_fignums())
            except Exception as exc:
                logger.warning(
                    "Event plotting failed for event %s (run %s): %s",
                    event_number,
                    run_number,
                    exc,
                )
                logger.debug("Full traceback:", exc_info=True)


    # ---------------------------------------
    # Diagnostic and write TRecons output
    # ---------------------------------------
    if n_pass > 0:
        trecons_path = output_dir / f"{rootfile_path.stem}_trecons.root"
        trecons.write(str(trecons_path), overwrite=True)
        logger.info(
            f"Wrote TRecons output: {trecons_path} ({n_pass} events)"
        )
    else:
        logger.warning(
            f"No event reconstructed successfully in {rootfile_path.name} "
            f"(cut_nant={n_cut_nant}, cut_nutrig={n_cut_nutrig}, "
            f"cut_chi2_adf={n_cut_chi2_adf}, cut_theta_adf={n_cut_theta_adf}, cut_omega_band={n_cut_omega_band}, cut_hgih_excess_omega={n_cut_hight_excess_omega}, failed={n_fail});"
            f"TRecons output not written"
        )

    return n_pass, n_cut_nant, n_cut_nutrig, n_cut_chi2_adf, n_cut_theta_adf, n_cut_omega_band, n_cut_hight_excess_omega, n_fail
