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
        # In sim mode, rootfile_path is the folder sim_*_<index> containing the subfolders
        # adc_*/efield_*/run_*/shower_*.root, 
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
        logger.debug(
            "Event %s (run %s) contains %d DUs",
            event_number,
            run_number,
            n_antennas,
        )

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
            tadc.get_event(event_number, adc_run_number) # Load the TADC event for the current event number and run number

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

            trigger_mask_all = np.asarray(
                nutrig_result["trigg_ant_mask"],
                dtype=bool,
            )

            n_triggered = np.count_nonzero(trigger_mask_all)

            logger.debug(
                "Event %s (run %s): %d/%d antennas triggered",
                event_number,
                run_number,
                n_triggered,
                len(trigger_mask_all),
            )

            if n_triggered < n_ant_cut:
                logger.debug(
                    "Event %s (run %s) skipped: only %d triggered antennas "
                    "(cut threshold: %d)",
                    event_number,
                    run_number,
                    n_triggered,
                    n_ant_cut,
                )
                n_cut_nant += 1
                continue

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

                # Get the mask from NUTRIG results 
                trigg_ant_mask = np.asarray(
                    nutrig_result["trigg_ant_mask"],
                    dtype=bool,
                )

                ADC_traces = np.asarray(
                    nutrig_result["ADC_traces"],
                    dtype=float,
                )

                # Get all t0 and Xants for the simulation event...
                tadc = el.directory.tadc
                tadc.get_event(event_number, adc_run_number)

                t0_all = np.asarray(
                    ext.compute_t0_sims(tadc),
                    dtype=float,
                )

                # Information from trun
                trun = el.directory.trun
                du_indices = tadc.get_dus_indices_in_run(trun)

                Xants_all = np.asarray(
                    get_antenna_positions_from_run(trun, du_indices),
                    dtype=float,
                )
                # ...then select only the triggered antennas.
                t0 = t0_all[trigg_ant_mask]
                Xants = Xants_all[trigg_ant_mask]

                logger.debug(
                    "Simulation event %s: total=%d, triggered=%d, "
                    "ADC=%s, t0=%s, Xants=%s",
                    event_number,
                    len(trigg_ant_mask),
                    np.count_nonzero(trigg_ant_mask),
                    ADC_traces.shape,
                    t0.shape,
                    Xants.shape,
                )


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

                import time
                start_time = time.time()
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
                end_time = time.time()
                logger.debug(
                    "Event %s (run %s) reconstruction time: %.2f seconds",
                    event_number,
                    run_number,
                    end_time - start_time,
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


            # -------------------------------------------
            # Fill TRecons with additional information
            # -------------------------------------------
            # Captured now, before trecons.fill()/the next event's reconstruct_event() call
            # overwrites trecons's staged values - used below for the peak-time overlay.
            peak_times = np.asarray(trecons.peak_time).copy()

            shower = e.shower # Store the shower object before e.write() modifies it.
            e.shower = None # Set the shower object to None.

            # En simulation, e.tsimshower est le TShower L0 de la simulation et son
            # tree_name vaut "tshower" (event.py:451). Or e._event_trees le contient
            # (event.py:481-483) et la boucle de copie de Event.write() ne saute que
            # e.tshower (event.py:860), qui est None ici : la verite MC est donc recopiee
            # dans le tshower de SORTIE, et le e.write_shower() ci-dessous retrouve
            # (run,event) deja present -> NotUniqueEvent.
            # Prouve par le run du 25/09 : 4 "Writing tshower" pour exactement 4 echecs.
            # On retire tsimshower de la copie ; le tshower de sortie ne contient alors
            # que le shower reconstruit, comme en donnees reelles ou tsimshower est None.
            trees_backup = e._trees
            event_trees_backup = e._event_trees

            if is_simulation and getattr(e, "tsimshower", None) is not None:
                # 'is not' et non '!=' : les arbres sont des dataclasses, donc '=='
                # compare champ par champ et pourrait retirer un autre arbre.
                e._trees = [
                    t for t in e._trees
                    if t is not e.tsimshower
                ]
                e._event_trees = [
                    t for t in e._event_trees
                    if t is not e.tsimshower
                ]


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
                ) # First vaue generated (to reuse for all subsequent events)

                # Write all TTrees connected to the event e in a ROOT file. 
                # This will create a new ROOT file for each event, but with the same files_creation_time, 
                # so they will all have the same name and be written to the same file.
                # No valid tshower TTree in .../shower_..._L0_0000.root
                # Creating a new one.
                # Writing tshower
                # Comes frome e.write() which creates a new TShower object and assigns it to e.tshower.
                # For the sims it is already filled with the correct values, but for the real data 
                # it is empty and needs to be filled with the correct values.

                # Write everything but the shower which is set as None to avoid creating a new shower ROOT file for each event.
                e.write(out_dir=str(output_dir))
            finally:
                # Get the shower stored before.
                e.shower = shower
                # EventList reutilise le meme objet Event d'un evenement a l'autre
                # (event_list.py:70), donc ne pas restaurer corromprait les suivants.
                e._trees = trees_backup
                e._event_trees = event_trees_backup

            # Assign the files_creation_time after e.write() to ensure it is consistent for all events.
            files_creation_time = e.files_creation_time

            shower_path = (
                output_dir
                / f"shower_{files_creation_time}_0-0_L1_0000.root" # This is the default naming convention used by Event.write(). 
            )

            try:
                # Fill the shower tree and write it the the good rootfile. 
                e.write_shower(str(shower_path))
            finally:
                if getattr(e, "tshower", None) is not None:
                    e.tshower.stop_using()

            # Beacause the recons is excluded from the e.write() call
            # We need to fill the TRecons tree manually here.
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
                    n_antennas=nutrig_result["n_antennas"],
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
