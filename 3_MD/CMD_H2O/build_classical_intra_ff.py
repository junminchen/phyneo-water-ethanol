#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inject classical intra-molecular force terms into a trained DMFF XML.")
    parser.add_argument("--source", type=Path, required=True, help="Trained XML without classical intra terms.")
    parser.add_argument("--template", type=Path, required=True, help="Template XML that already contains HarmonicBondForce/HarmonicAngleForce.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = ET.parse(args.source).getroot()
    template_root = ET.parse(args.template).getroot()

    source_tags = [child.tag for child in source_root]
    insert_after = source_tags.index("Residues") + 1 if "Residues" in source_tags else len(source_root)

    for force_tag in ("HarmonicBondForce", "HarmonicAngleForce"):
        source_root[:] = [child for child in source_root if child.tag != force_tag]
        template_force = template_root.find(force_tag)
        if template_force is None:
            continue
        source_root.insert(insert_after, copy.deepcopy(template_force))
        insert_after += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(source_root)
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    print(args.output)


if __name__ == "__main__":
    main()
