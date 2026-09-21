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


#-------------------------------
# Load GRANDlib
#-------------------------------
from grand.analysis.pipeline_nathan.scripts.loading import load_config

config_path = Path(__file__).parent.parent / "config.yaml"
config = load_config(config_path)

sys.path.append(config['paths']['grandlib_path'])

logger = logging.getLogger("grand.process")

from grand.aoi import EventList
from grand.dataio import TRecons
from grand.analysis.pipeline_nathan.scripts.reconstruction import reconstruct_event
from grand.analysis.pipeline_nathan.scripts.sanity_plot import plot_traces
from grand.analysis.pipeline_nathan.scripts.cuts.apply_nutrig_cut import (
    compute_nutrig_event,
    passes_nutrig_cut,
)


def get_run_number(rootfile_path, e):
    match = re.search(r'GP80_(\d{4})(\d{2})\d{2}_\d+_RUN(\d+)_', Path(rootfile_path).name)
    if match:
        return int(match.group(3))
    return e.run_number

def process_file(
    rootfile_path: Path,
    output_dir: Path,
    antenna_position: pd.DataFrame,
    n_ant_cut: int,
    nutrig_src_path,
    templates_npz_path,
    rho_min_threshold: float,
    rho_mean_threshold: float,
    limit_events=None,
    do_plot=False
    ) -> tuple[int, int, int, int]:
    """Process a single ROOT file and write the output to the specified directory."""

    logger.info(f"Opening ROOT file: {rootfile_path}")

    # Check EventList
    el = EventList(str(rootfile_path), use_trawvoltage=True, trawvoltage_channels=[1, 2, 3])

    # Determine the list of events to process
    targets = list(el.event_list)
    if limit_events is not None:
        targets = targets[:limit_events]
    logger.info(f"{len(targets)} events to process in {rootfile_path.name}")

    trecons = TRecons()
    # Event.write() derives the shower output filename from e.files_creation_time, and
    # grand.aoi.Event.fill_shower_tree() only reuses/appends to an existing tshower tree when the
    # filename matches exactly (grand_tree_list lookup keyed on the exact string). On the very
    # first write to a brand-new output directory, Event.write() itself overwrites whatever value
    # we pre-set (its internal trun/voltage/efield file-creation bookkeeping assigns
    # self.files_creation_time = target_dir.cur_time_string the first time those trees need
    # creating), so a fixed value picked before the loop can still get silently replaced on event 1.
    # Instead: let the first successful write establish the value, capture whatever it ends up
    # being, and pin every subsequent event to that exact value.
    
    files_creation_time = None
    n_pass = 0
    n_cut_nant = 0
    n_cut_nutrig = 0
    n_fail = 0

    for event_number, run_number in targets: # Iterate over the events in the ROOT file 0,1,..
        e = el.get_event(event_number=event_number, run_number=run_number)

        if e is None:
            logger.warning(f"Event {event_number} (run {run_number}) not found in {rootfile_path.name}")
            n_fail += 1
            continue

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

        # A global NUTRIG failure (package unavailable, bad templates path, ...) is a
        # configuration problem, not a physically-bad event - it must raise here and
        # propagate out of process_file()/main(), not be swallowed by the try/except
        # below. Only per-channel local failures (caught inside compute_nutrig_event)
        # turn into a per-event NUTRIG cut.
        nutrig_result = compute_nutrig_event(
            e,
            nutrig_src_path=nutrig_src_path,
            templates_npz_path=templates_npz_path,
        )

        if not passes_nutrig_cut(
            nutrig_result,
            rho_min_threshold,
            rho_mean_threshold,
        ):
            logger.debug(
                f"Event {event_number} (run {run_number}) skipped: "
                f"NUTRIG cut (rho_min={nutrig_result['rho_min']:.3f}, "
                f"rho_mean={nutrig_result['rho_mean']:.3f}, "
                f"n_valid={nutrig_result['n_valid']}/{len(nutrig_result['rho_max'])})"
            )
            n_cut_nutrig += 1
            continue

        try:
            reconstruct_event(
                e,
                antenna_position,
                trecons,
                run_number,
                ADC_traces=nutrig_result["ADC_traces"],
            )

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

        if do_plot:
            try:
                plot_traces(
                    e,
                    output_dir=output_dir,
                    peak_times=peak_times,
                    ADC_traces=nutrig_result["ADC_traces"],
                    nutrig_result=nutrig_result,
                )
            except Exception as exc:
                logger.warning(
                    "Event plotting failed for event %s (run %s): %s",
                    event_number,
                    run_number,
                    exc,
                )
                logger.debug("Full traceback:", exc_info=True)

    if n_pass > 0:
        trecons_path = output_dir / f"{rootfile_path.stem}_trecons.root"
        trecons.write(str(trecons_path), overwrite=True)
        logger.info(
            f"Wrote TRecons output: {trecons_path} ({n_pass} events)"
        )
    else:
        logger.warning(
            f"No event reconstructed successfully in {rootfile_path.name} "
            f"(cut_nant={n_cut_nant}, cut_nutrig={n_cut_nutrig}, failed={n_fail}); "
            f"TRecons output not written"
        )

    return n_pass, n_cut_nant, n_cut_nutrig, n_fail
