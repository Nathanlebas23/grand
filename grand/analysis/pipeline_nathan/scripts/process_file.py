"""
Process a file and generate the corresponding output file.
This script reads a ROOT file, processes the events, and writes the results to an output ROOT file. It also generates a log file if specified in the configuration.
The script uses the configuration file to get the paths and parameters for processing.
"""
import sys  
sys.path.append("/home/lpnhe/grand")

import logging
import re
from datetime import datetime
from pathlib import Path

from grand.aoi import EventList
from grand.dataio import TRecons
from grand.analysis.pipeline_nathan.scripts.reconstruction import reconstruct_event

logger = logging.getLogger("grand.process")

def get_run_number(rootfile_path, e):
    match = re.search(r'GP80_(\d{4})(\d{2})\d{2}_\d+_RUN(\d+)_', Path(rootfile_path).name)
    if match:
        return int(match.group(3))
    return e.run_number


def process_file(rootfile_path: Path, output_dir: Path, antenna_position, nutrig_template, limit_events=None) -> tuple:
    """Reconstruct every event of a single ROOT file and write the enriched output."""
    logger.info(f"Opening ROOT file: {rootfile_path}")
    el = EventList(str(rootfile_path), use_trawvoltage=True, trawvoltage_channels=[1, 2, 3])

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
    n_ok = 0
    n_fail = 0

    for event_number, run_number in targets:
        e = el.get_event(event_number=event_number, run_number=run_number)
        if e is None:
            logger.warning(f"Event {event_number} (run {run_number}) not found in {rootfile_path.name}")
            n_fail += 1
            continue

        run_number = get_run_number(rootfile_path, e)

        try:
            reconstruct_event(e, antenna_position, nutrig_template, trecons, run_number)
            e.files_creation_time = files_creation_time or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            e.write(out_dir=str(output_dir))
            files_creation_time = e.files_creation_time
            # Only fill the TRecons entry once the standard ROOT write has succeeded, so
            # TRecons and the standard output never disagree about which events were saved.
            trecons.fill()
        except Exception as exc:
            logger.warning("Event processing failed for event %s (run %s): %s", event_number, run_number, exc)
            logger.debug("Full traceback:", exc_info=True)
            n_fail += 1
            continue

        n_ok += 1
        logger.debug(f"Event {event_number} (run {run_number}) reconstructed OK")

    if n_ok > 0:
        trecons_path = output_dir / f"{rootfile_path.stem}_trecons.root"
        trecons.write(str(trecons_path), overwrite=True)
        logger.info(f"Wrote TRecons output: {trecons_path} ({n_ok} events)")
    else:
        logger.warning(f"No event reconstructed successfully in {rootfile_path.name}; TRecons output not written")

    return n_ok, n_fail
