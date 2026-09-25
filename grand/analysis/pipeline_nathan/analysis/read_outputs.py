import sys  
from pathlib import Path
import argparse
import logging
import numpy as np

# GRAND repository root
GRAND_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(GRAND_ROOT))

from grand.dataio import TRecons, DataDirectory
from grand.analysis.pipeline_nathan.scripts.loading import load_config, load_antenna_positions, get_trecons_path
from grand.analysis.pipeline_nathan.analysis.plotting import plot_event, plot_timing_residuals
from grand.analysis.pipeline_nathan.analysis.event_analysis import analyze_event, compute_timing_residuals, get_simulation_truth

logger = logging.getLogger("grand.process")


def main():
    parser = argparse.ArgumentParser(
        description="Run basic analysis on the reconstructed events from a TRecons file."
    )
    parser.add_argument(
        "file_number",
        type=int,
        help="Index of the TRecons file to analyze in the configured output directory.",
    )
    parser.add_argument(
        "--event-number",
        type=int,
        default=None,
        help="Specific event number to analyze (default: analyze all events in the TRecons file).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "config.yaml",
        help="Path to the config.yaml file (default: config.yaml next to this script).",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Whether the input TRecons files are from simulation (default: False).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg = config["paths"]

    ##################################################
    # Select the same input ROOT file as scripts/main.py
    ##################################################
    # Get the TRecons file path and the corresponding ROOT file path based on the file_number argument

    trecons_path, rootfile_path = get_trecons_path(
        paths_cfg, args.file_number, is_simulation=args.simulation,
    )

    # Communs aux deux modes : t_sim est lu plus bas quel que soit le mode, et
    # antenna_position est passe a plot_event dans les deux branches.
    t_sim = None
    t_recons = TRecons(str(trecons_path))
    antenna_position = load_antenna_positions(paths_cfg["rtk_antenna_positions_path"])

    if args.simulation:
        logger.info(f"Analyzing simulation TRecons file: {trecons_path}")

        # DataDirectory agrege les shower_*_L0_*.root du dossier en un TChain AVEC
        # BuildIndex (data_handling.py:276-285), donc get_event(ev, run) couvre tous
        # les evenements. NE PAS utiliser TShower(glob) : ce chemin construit un TChain
        # sans index (data_tree.py:366-372) et get_event() y echoue systematiquement.
        sim_dir = DataDirectory(str(rootfile_path))
        t_sim = sim_dir.tshower_l0
        if t_sim is None:
            raise FileNotFoundError(
                f"No simulation shower_*_L0_*.root found in {rootfile_path}"
            )

        logger.info(
            "Loaded simulation truth from %s (%d events)",
            rootfile_path.name,
            len(t_sim.get_list_of_events()),
        )
    else:
        logger.info(f"Analyzing real data TRecons file: {trecons_path}")

    plot_dir = Path(paths_cfg["plot_dir"]) / rootfile_path.stem
    plot_dir.mkdir(parents=True, exist_ok=True)

    ##################################################
    # Analyze and plot the selected event(s)
    ##################################################
    if args.event_number is not None:
        logger.info(f"Analyzing event {args.event_number} in TRecons file {t_recons.file_name}")
        events = t_recons.get_list_of_events()
        matches = [(ev_no, run_no) for ev_no, run_no in events if ev_no == args.event_number]
        if not matches:
            raise ValueError(f"Event {args.event_number} not found in {trecons_path}")
        if len(matches) > 1:
            raise ValueError(
                f"Event {args.event_number} exists in multiple runs: "
                f"{[run_no for _, run_no in matches]}"
            )
        ev_no, run_no = matches[0]
        t_recons.get_event(ev_no, run_no)

        sim_truth = None

        if t_sim is not None:
            sim_truth = get_simulation_truth(
                t_sim,
                ev_no,
                run_no,
            )

        results = analyze_event(
            t_recons,
            ev_no,
            run_no,
            config,
            sim_truth=sim_truth,
        )

        # results = analyze_event(t_recons, ev_no, run_no, config, is_simulation=args.simulation)
        plot_event(t_recons, results, antenna_position, plot_dir, is_simulation=args.simulation)

        # --------------------
        # Timing residuals
        #---------------------
        # timing = compute_timing_residuals(t_recons)
        # plot_timing_residuals(t_recons, timing, plot_dir)
        # logger.info(f"Plotted event {ev_no} run {run_no}")
    else:
        logger.info(f"Analyzing all events in TRecons file {t_recons.file_name}")
        for ev_no, run_no in t_recons.get_list_of_events():
            t_recons.get_event(ev_no, run_no)

            sim_truth = None

            if t_sim is not None:
                sim_truth = get_simulation_truth(
                    t_sim,
                    ev_no,
                    run_no,
                )

            results = analyze_event(
                t_recons,
                ev_no,
                run_no,
                config,
                sim_truth=sim_truth,
            )



            # results = analyze_event(t_recons, ev_no, run_no, config)
            plot_event(t_recons, results, antenna_position, plot_dir, is_simulation=args.simulation)

            # --------------------
            # Timing residuals
            #---------------------            
            # timing = compute_timing_residuals(t_recons)
            # plot_timing_residuals(t_recons, timing, plot_dir)
            # logger.info(f"Plotted event {ev_no} run {run_no}")

if __name__ == "__main__":
    main()