"""
Loading functions for the GRAND pipeline.
This module contains functions to load configuration, antenna positions, and NUTRIG templates.
"""
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("grand.process")

#----------------------------------------------
# Loading functions for pipeline
#----------------------------------------------

# #-------------------------------
# # Load GRANDlib and nl_style
# #-------------------------------
# from grand.analysis.pipeline_nathan.scripts.loading import load_config

# config_path = Path(__file__).parent.parent / "config.yaml"
# config = load_config(config_path)

# sys.path.append(config['paths']['grandlib_path'])
# sys.path.append(config['paths']['nl_style_path'])

# from nl_style import set_style , NL_COLORS

# set_style()


def load_config(config_path: Path) -> dict:
    """Load the pipeline configuration from a YAML file."""
    return yaml.safe_load(config_path.read_text())


def natural_sort_key(path: Path):
    """Sort key that orders numeric chunks in a filename numerically (e.g. '...-2' before '...-10')."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r'(\d+)', path.name)]


def load_antenna_positions(rtk_path: str) -> pd.DataFrame:
    """Load RTK antenna positions (DU_id, x, y, z) from a text file."""
    column_names = ['DU_id', 'x', 'y', 'z']
    antenna_position = pd.read_csv(rtk_path, sep=r'\s+', names=column_names, header=0)
    antenna_position['DU_id'] = antenna_position['DU_id'].astype(int)
    return antenna_position


def load_nutrig_template(template_path: str) -> np.ndarray:
    """Load the NUTRIG templates file and return a single 1D template (row 0) for peak-time estimation."""
    templates = np.loadtxt(template_path, comments="#")
    template = templates[0]
    return template


def get_trecons_path(paths_cfg, file_number: int) -> tuple[Path, Path]:
    input_dir = Path(paths_cfg["input_dir"])
    rootfiles = sorted(input_dir.glob("*.root"), key=natural_sort_key)
    if not rootfiles:
        raise RuntimeError(f"No ROOT files found in {input_dir}")
    if file_number < 0 or file_number >= len(rootfiles):
        raise IndexError(
            f"ROOT file number {file_number} is out of range. "
            f"Found {len(rootfiles)} ROOT files."
        )
    rootfile_path = rootfiles[file_number]
    output_dir = Path(paths_cfg["output_dir"]) / rootfile_path.stem
    trecons_path = output_dir / f"{rootfile_path.stem}_trecons.root"
    if not trecons_path.is_file():
        raise FileNotFoundError(
            f"{trecons_path} not found - run scripts/main.py {file_number} first."
        )
    return trecons_path, rootfile_path
