import sys  
sys.path.append("/home/lpnhe/grand")

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from grand.aoi import EventList, Shower
from grand.dataio import TRecons
import grand.analysis.signals.extraction as ext
import grand.analysis.fitting as fit
import grand.analysis.constants as cons
import grand.analysis.geom as geom
import grand.analysis.energy_reco as en
from setup_logger import setup_logger

logger = logging.getLogger("grand.process")

# Loadings 

# ---------------------------------------------------------------
# Determine the run number for an event
# Try to extract it from the ROOT file name; fall back to e.run_number
# ---------------------------------------------------------------

# Get run number from ROOT file name


# Reconstruct event

# Process file


def main():
    parser = argparse.ArgumentParser(
        description="Run AOI reconstruction on a single ROOT file selected by index from the configured input directory."
    )
    parser.add_argument(
        "file_number",
        type=int,
        help="Index of the ROOT file to process in the configured input directory.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to the config.yaml file (default: config.yaml next to this script).",
    )
    parser.add_argument(
        "--limit-events",
        type=int,
        default=None,
        help="Maximum number of events to process, for quick manual testing.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    log_cfg = config.get("logging", {})
    setup_logger(level=log_cfg.get("level", "INFO"), log_file=log_cfg.get("log_file"))
    logger.info(f"Config loaded from {args.config}")

    paths_cfg = config["paths"]

    input_dir = Path(paths_cfg["input_dir"])
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir does not exist or is not a directory: {input_dir}")

    rtk_path = Path(paths_cfg["rtk_antenna_positions_path"])
    if not rtk_path.is_file():
        raise FileNotFoundError(f"rtk_antenna_positions_path not found: {rtk_path}")

    template_path = Path(paths_cfg["nutrig_template_path"])
    if not template_path.is_file():
        raise FileNotFoundError(f"nutrig_template_path not found: {template_path}")

    logger.info(f"Input directory: {input_dir}")
    rootfiles = sorted(input_dir.glob("*.root"), key=natural_sort_key)
    logger.info(f"Found {len(rootfiles)} ROOT files")
    if not rootfiles:
        raise RuntimeError(f"No ROOT files found in {input_dir}")

    if args.file_number < 0 or args.file_number >= len(rootfiles):
        raise IndexError(
            f"ROOT file number {args.file_number} is out of range. "
            f"Found {len(rootfiles)} ROOT files in {input_dir}."
        )

    rootfile_path = rootfiles[args.file_number]
    logger.info(f"Selected ROOT file [{args.file_number}/{len(rootfiles) - 1}]: {rootfile_path.name}")

    antenna_position = load_antenna_positions(str(rtk_path))
    nutrig_template = load_nutrig_template(str(template_path))

    output_dir = Path(paths_cfg["output_dir"]) / rootfile_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing ROOT file: {rootfile_path}")
    n_ok, n_fail = process_file(rootfile_path, output_dir, antenna_position, nutrig_template, limit_events=args.limit_events)

    logger.info(
        f"Done. file={rootfile_path.name}, events_total={n_ok + n_fail}, "
        f"reconstructed_ok={n_ok}, failed={n_fail}"
    )


if __name__ == "__main__":
    main()
