#!/usr/bin/env python3
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scale non-zero thole values in a DMFF/OpenMM XML force field."
    )
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parent / "ff.backend_amoeba_total1000_classical_intra.xml"),
        help="Input XML path.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output XML path.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        required=True,
        help="Multiplicative scale applied to each non-zero thole value.",
    )
    parser.add_argument(
        "--keep-zero",
        action="store_true",
        default=True,
        help="Keep zero thole entries unchanged.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    tree = ET.parse(input_path)
    root = tree.getroot()

    count = 0
    for node in root.iter("Polarize"):
        if "thole" not in node.attrib:
            continue
        old = float(node.attrib["thole"])
        if args.keep_zero and abs(old) < 1e-16:
            continue
        node.attrib["thole"] = f"{old * args.scale:.12g}"
        count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=False)

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"scale={args.scale}")
    print(f"updated_polarize_entries={count}")


if __name__ == "__main__":
    main()
