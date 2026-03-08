#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scale a cubic PDB box and atom coordinates about the box center."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scale", type=float, required=True)
    return parser.parse_args()


def scale_cryst1(line: str, scale: float) -> str:
    a = float(line[6:15]) * scale
    b = float(line[15:24]) * scale
    c = float(line[24:33]) * scale
    alpha = float(line[33:40])
    beta = float(line[40:47])
    gamma = float(line[47:54])
    return (
        f"{line[:6]}"
        f"{a:9.3f}{b:9.3f}{c:9.3f}"
        f"{alpha:7.2f}{beta:7.2f}{gamma:7.2f}"
        f"{line[54:]}"
    )


def scale_atom(line: str, old_box: float, new_box: float) -> str:
    center_old = old_box / 2.0
    center_new = new_box / 2.0
    x = float(line[30:38])
    y = float(line[38:46])
    z = float(line[46:54])
    x = (x - center_old) * (new_box / old_box) + center_new
    y = (y - center_old) * (new_box / old_box) + center_new
    z = (z - center_old) * (new_box / old_box) + center_new
    return f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"


def main() -> None:
    args = parse_args()
    lines = args.input.read_text().splitlines()
    old_box = None
    new_lines = []
    for line in lines:
        if line.startswith("CRYST1") or line.startswith("CRYST "):
            old_box = float(line[6:15])
            new_box = old_box * args.scale
            new_lines.append(scale_cryst1(line, args.scale))
        elif line.startswith(("ATOM", "HETATM")):
            if old_box is None:
                raise ValueError("CRYST line must appear before atom records")
            new_lines.append(scale_atom(line, old_box, new_box))
        else:
            new_lines.append(line)
    args.output.write_text("\n".join(new_lines) + "\n")


if __name__ == "__main__":
    main()
