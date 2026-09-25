import sys  
from pathlib import Path

# GRAND repository root
GRAND_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(GRAND_ROOT))

import argparse
import logging
from setup_logger import setup_logger
from loading import load_config, load_antenna_positions, load_nutrig_template, natural_sort_key, get_sims_rootfiles_path
from process_file import process_file
from cuts.compute_nutrig import is_flt_available

logger = logging.getLogger("grand.process")

def main():
    parser = argparse.ArgumentParser(
        description="Run reconstruction on a single ROOT file selected by index from the configured input directory."
    )
    parser.add_argument(
        "file_number",
        type=int,
        help="Index of the ROOT file to process in the configured input directory.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "config.yaml",
        help="Path to the config.yaml file (default: config.yaml next to this script).",
    )
    parser.add_argument(
        "--limit-events",
        type=int,
        default=None,
        help="Maximum number of events to process, for quick manual testing.",
    )
    parser.add_argument(
        "--do-plot",
        action="store_true",
        help="Whether to generate plots for each event (default: False).",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Whether the input ROOT files are from simulation (default: False).",
    )

    args = parser.parse_args()

    logger.info("------------------------------------------------------------------------")
    logger.info("------------------- STARTING RECONSTRUCTION PIPELINE -------------------")
    logger.info("------------------------------------------------------------------------")

    ######################
    # Load configuration #
    ######################
    config = load_config(args.config)

    log_cfg = config.get("logging", {})
    setup_logger(level=log_cfg.get("level", "INFO"), log_file=log_cfg.get("log_file"))
    logger.info(f"Config loaded from {args.config}")


    #############
    # Load cuts #
    #############        
    cuts_cfg = config["cuts"]
    n_ant_cut = cuts_cfg["nant_cut"]["threshold"]
    logger.info(f"Antenna multiplicity cut: n_antennas >= {n_ant_cut}")

    nutrig_cut_cfg = cuts_cfg["nutrig_cut"]
    rho_min_threshold = nutrig_cut_cfg["rho_min_threshold"]
    rho_mean_threshold = nutrig_cut_cfg["rho_mean_threshold"]
    logger.info(
        f"NUTRIG cut: rho_min >= {rho_min_threshold}, rho_mean >= {rho_mean_threshold}"
    )

    chi2_adf_threshold = cuts_cfg["chi2_adf_cut"]["threshold"]
    logger.info(f"ADF chi2 cut: chi2_adf <= {chi2_adf_threshold}")

    theta_adf_threshold = cuts_cfg["theta_adf_cut"]["threshold"]
    logger.info(f"ADF theta cut: theta_adf <= {theta_adf_threshold}")

    omega_min = cuts_cfg["omega_cut"]["omega_min"]
    omega_max = cuts_cfg["omega_cut"]["omega_max"]
    logger.info(f"Omega cut: {omega_min} <= omega <= {omega_max}")

    omega_excess_threshold = cuts_cfg["high_omega_excess"]["omega_excess_threshold"]
    amplitude_excess_threshold = cuts_cfg["high_omega_excess"]["amlpitude_excess_threshold"]
    logger.info(
        f"High omega excess cut: omega_excess <= {omega_excess_threshold}, amplitude_excess <= {amplitude_excess_threshold}"
    )

    ##############
    # Load paths #
    ##############     
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

    nutrig_templates_npz_path = Path(paths_cfg["nutrig_templates_npz_path"])
    if not nutrig_templates_npz_path.is_file():
        raise FileNotFoundError(f"nutrig_templates_npz_path not found: {nutrig_templates_npz_path}")

    nutrig_src_path = Path(paths_cfg["nutrig_src_path"])
    if not nutrig_src_path.is_dir():
        raise NotADirectoryError(f"nutrig_src_path does not exist or is not a directory: {nutrig_src_path}")
    if not is_flt_available(nutrig_path=nutrig_src_path):
        raise ImportError(f"nutrig.flt package not importable from nutrig_src_path={nutrig_src_path}")

    if args.simulation:
        logger.info("Running in simulation mode.")
        logger.info(f"Input directory: {input_dir}")

        rootfile_path = get_sims_rootfiles_path(input_dir, args.file_number)
        logger.info(f"Selected simulation directory: {rootfile_path.name}")
    else:
        logger.info("Running in real data mode.")
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

    ###########################
    # Load antenna positions  #
    ###########################
    antenna_position = load_antenna_positions(str(rtk_path))
    logger.info(f"Loaded antenna positions from {rtk_path}")

    ##########################
    # Load NUTRIG template   #
    ##########################
    nutrig_template = load_nutrig_template(str(template_path))
    logger.info(f"Loaded NUTRIG template from {template_path}")

    output_dir = Path(paths_cfg["output_dir"]) / rootfile_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    ##################################
    # Process the selected ROOT file #
    ##################################
    logger.info("------------------------------------------------------------------------")
    logger.info(f"Processing ROOT file: {rootfile_path.stem}")
    logger.info("------------------------------------------------------------------------")
    n_pass, n_cut_nant, n_cut_nutrig, n_cut_chi2_adf, n_cut_theta_adf, n_cut_omega_band, n_cut_high_omega_excess, n_fail = process_file(
        rootfile_path,
        output_dir,
        antenna_position,
        n_ant_cut=n_ant_cut,
        nutrig_templates_txt = nutrig_template,
        templates_npz_path=nutrig_templates_npz_path,
        nutrig_src_path=nutrig_src_path,
        rho_min_threshold=rho_min_threshold,
        rho_mean_threshold=rho_mean_threshold,
        chi2_adf_threshold=chi2_adf_threshold,
        theta_adf_threshold=theta_adf_threshold,
        omega_min=omega_min,
        omega_max=omega_max,
        omega_excess_threshold=omega_excess_threshold,
        amplitude_excess_threshold=amplitude_excess_threshold,
        limit_events=args.limit_events,
        do_plot=args.do_plot,
        is_simulation=args.simulation
    )

    logger.info(
        f"Done. file={rootfile_path.name}, "
        f"events_total={n_pass + n_cut_nant + n_cut_nutrig + n_cut_chi2_adf + n_cut_theta_adf + n_cut_omega_band + n_cut_high_omega_excess, n_fail}, "
        f"passed={n_pass}, cut_nant={n_cut_nant}, cut_nutrig={n_cut_nutrig}, "
        f"cut_chi2_adf={n_cut_chi2_adf}, cut_theta_adf={n_cut_theta_adf}, cut_omega_band={n_cut_omega_band}, cut_high_omega_excess={n_cut_high_omega_excess}, failed={n_fail}"
    )

if __name__ == "__main__":
    main()
