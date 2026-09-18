"""
Loading functions for the GRAND pipeline.
This module contains functions to load configuration, antenna positions, and NUTRIG templates.
"""
import sys  
sys.path.append("/home/lpnhe/grand")

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# from grand.aoi import EventList, Shower
# from grand.dataio import TRecons
# import grand.analysis.signals.extraction as ext
# import grand.analysis.fitting as fit
# import grand.analysis.constants as cons
# import grand.analysis.geom as geom
# import grand.analysis.energy_reco as en
# from setup_logger import setup_logger

logger = logging.getLogger("grand.process")



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
