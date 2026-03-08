#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from train_dimer_backend import (
    COMPONENTS,
    DEFAULT_PAIRS,
    Hamiltonian,
    PairKernel,
    ParamSet,
    REPO_ROOT,
    calculate_weights,
    load_training_data,
    merge_shared_params_into_paramset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check trained shared-parameter dimer scans and render the corresponding XML.")
    parser.add_argument("--ff", type=Path, default=REPO_ROOT / "ff.xml")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "2_PES" / "data_sr.pickle")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS))
    parser.add_argument("--cutoff-angstrom", type=float, default=25.0)
    parser.add_argument("--box-angstrom", type=float, default=60.0)
    parser.add_argument("--weight-threshold", type=float, default=20.0)
    parser.add_argument("--weight-kt", type=float, default=2.494)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot-sids", nargs="*", default=None)
    return parser.parse_args()


def load_checkpoint(path: Path):
    with open(path, "rb") as ifile:
        checkpoint = pickle.load(ifile)
    if isinstance(checkpoint, ParamSet):
        return copy.deepcopy(checkpoint.parameters)
    return copy.deepcopy(checkpoint)


def build_args_like_training(args: argparse.Namespace):
    class Obj:
        pass

    obj = Obj()
    obj.data = args.data
    obj.pairs = args.pairs
    obj.weight_threshold = args.weight_threshold
    obj.weight_kt = args.weight_kt
    return obj


def render_xml(ff_path: Path, shared_params, output_xml: Path) -> None:
    hamiltonian = Hamiltonian(str(ff_path))
    base_paramset = hamiltonian.getParameters()
    hamiltonian.paramset = merge_shared_params_into_paramset(base_paramset, shared_params)
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    hamiltonian.renderXML(str(output_xml))


def calculate_com_distance(posA: np.ndarray, posB: np.ndarray) -> np.ndarray:
    centers_A = np.mean(posA, axis=1)
    centers_B = np.mean(posB, axis=1)
    return np.linalg.norm(centers_A - centers_B, axis=1)


def predict_pair_scans(pair_name: str, pair_data: dict, shared_params, args: argparse.Namespace):
    kernel = PairKernel(args.ff, pair_name, args.cutoff_angstrom, args.box_angstrom)
    predictions = {}
    for sid, scan in pair_data.items():
        E_ex, E_es, E_pol, E_disp, E_dhf, E_tot = kernel.predict_scan(shared_params, scan["posA"], scan["posB"])
        predictions[sid] = {
            "distance": calculate_com_distance(np.asarray(scan["posA"]), np.asarray(scan["posB"])),
            "ex": np.asarray(E_ex),
            "es": np.asarray(E_es),
            "pol": np.asarray(E_pol),
            "disp": np.asarray(E_disp),
            "dhf": np.asarray(E_dhf),
            "tot": np.asarray(E_tot),
        }
    return predictions


def calculate_rmsd(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - ref) ** 2)))


def summarize_rmsd(data: dict, predictions: dict):
    summary = {}
    for pair_name in data:
        pair_summary = {"all": {}, "weighted": {}}
        for component in COMPONENTS:
            pred_all = []
            ref_all = []
            pred_weighted = []
            ref_weighted = []
            for sid, scan in data[pair_name].items():
                pred = predictions[pair_name][sid][component]
                ref = np.asarray(scan[component])
                pred_all.append(pred)
                ref_all.append(ref)
                mask = np.asarray(scan["wts"]) > 1e-2
                pred_weighted.append(pred[mask])
                ref_weighted.append(ref[mask])
            pair_summary["all"][component] = calculate_rmsd(np.concatenate(pred_all), np.concatenate(ref_all))
            pair_summary["weighted"][component] = calculate_rmsd(
                np.concatenate(pred_weighted), np.concatenate(ref_weighted)
            )
        summary[pair_name] = pair_summary
    return summary


def choose_plot_sids(pair_data: dict, requested: list[str] | None) -> list[str]:
    sids = sorted(pair_data.keys())
    if requested:
        return [sid for sid in requested if sid in pair_data]
    return list(dict.fromkeys([sids[0], sids[len(sids) // 2], sids[-1]]))


def plot_total_scan_grid(pair_name: str, pair_data: dict, pred_data: dict, output_path: Path) -> None:
    sids = sorted(pair_data.keys())
    ncols = 5
    nrows = math.ceil(len(sids) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.0 * nrows), squeeze=False)
    axes = axes.flatten()

    for idx, sid in enumerate(sids):
        ax = axes[idx]
        ax.plot(pred_data[sid]["distance"], np.asarray(pair_data[sid]["tot"]), "--", color="black", linewidth=1.2, label="ref")
        ax.plot(pred_data[sid]["distance"], pred_data[sid]["tot"], "-", color="tab:red", linewidth=1.2, label="pred")
        ax.set_title(sid, fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
        if idx % ncols == 0:
            ax.set_ylabel("Tot (kJ/mol)")
        if idx >= len(sids) - ncols:
            ax.set_xlabel("COM distance (A)")
    for idx in range(len(sids), len(axes)):
        axes[idx].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(f"{pair_name} total scan comparison", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_component_scan(pair_name: str, sid: str, scan: dict, pred: dict, rmsd_summary: dict, output_path: Path) -> None:
    labels = {
        "tot": "Total",
        "ex": "Exchange",
        "es": "Electrostatics",
        "pol": "Polarization",
        "disp": "Dispersion",
        "dhf": "DHF",
    }
    colors = {
        "tot": "tab:blue",
        "ex": "tab:orange",
        "es": "tab:brown",
        "pol": "tab:green",
        "disp": "tab:red",
        "dhf": "tab:purple",
    }

    x = pred["distance"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), height_ratios=[2.0, 1.2])
    for component in COMPONENTS:
        axes[0].plot(x, np.asarray(scan[component]), "--", color=colors[component], linewidth=1.5, label=f"{labels[component]} ref")
        axes[0].plot(x, pred[component], "-", color=colors[component], linewidth=1.5, label=f"{labels[component]} pred")
    axes[0].set_title(f"{pair_name} {sid} scan", fontsize=14)
    axes[0].set_ylabel("Energy (kJ/mol)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=2, fontsize=9)

    total_error = pred["tot"] - np.asarray(scan["tot"])
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].scatter(x, total_error, s=18, color="tab:blue")
    axes[1].set_xlabel("COM distance (A)")
    axes[1].set_ylabel("Total error")
    axes[1].grid(alpha=0.25)

    lines = [f"{component}: {rmsd_summary['all'][component]:.2f} ({rmsd_summary['weighted'][component]:.2f})" for component in COMPONENTS]
    axes[1].text(
        0.98,
        0.98,
        "RMSD all(weighted)\n" + "\n".join(lines),
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_global_error_summary(data: dict, predictions: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    pair_colors = {
        "Pairs_EA_EA": "tab:blue",
        "Pairs_EA_H2O": "tab:orange",
        "Pairs_H2O_H2O": "tab:green",
    }
    for pair_name in data:
        distances = []
        errors = []
        for sid, scan in data[pair_name].items():
            distances.append(predictions[pair_name][sid]["distance"])
            errors.append(predictions[pair_name][sid]["tot"] - np.asarray(scan["tot"]))
        ax.scatter(
            np.concatenate(distances),
            np.concatenate(errors),
            s=10,
            alpha=0.6,
            color=pair_colors.get(pair_name),
            label=pair_name,
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("COM distance (A)")
    ax.set_ylabel("Total energy error (pred - ref)")
    ax.set_title("Global dimer scan error summary")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shared_params = load_checkpoint(args.checkpoint)
    data = load_training_data(build_args_like_training(args))

    xml_path = args.output_dir / "trained_params.xml"
    render_xml(args.ff, shared_params, xml_path)

    predictions = {pair_name: predict_pair_scans(pair_name, data[pair_name], shared_params, args) for pair_name in args.pairs}
    rmsd_summary = summarize_rmsd(data, predictions)

    with open(args.output_dir / "rmsd_summary.json", "w") as ofile:
        json.dump(rmsd_summary, ofile, indent=2)

    plot_global_error_summary(data, predictions, args.output_dir / "global_total_error.png")
    for pair_name in args.pairs:
        plot_total_scan_grid(
            pair_name=pair_name,
            pair_data=data[pair_name],
            pred_data=predictions[pair_name],
            output_path=args.output_dir / f"total_scan_grid_{pair_name}.png",
        )
        for sid in choose_plot_sids(data[pair_name], args.plot_sids):
            plot_component_scan(
                pair_name=pair_name,
                sid=sid,
                scan=data[pair_name][sid],
                pred=predictions[pair_name][sid],
                rmsd_summary=rmsd_summary[pair_name],
                output_path=args.output_dir / f"scan_components_{pair_name}_{sid}.png",
            )

    print(f"checkpoint: {args.checkpoint}")
    print(f"xml: {xml_path}")
    print(f"results: {args.output_dir}")


if __name__ == "__main__":
    main()
