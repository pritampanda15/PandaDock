import os
from pathlib import Path
import numpy as np

def compute_centroid(pdb_file):
    coords = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')) and len(line) >= 54:
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except ValueError:
                    continue
    if coords:
        return np.mean(coords, axis=0)
    else:
        return None

def extract_all_grid_centers(root_dir, output_file="grid_centers.csv"):
    root_path = Path(root_dir)
    result_lines = []

    for subdir in sorted(root_path.iterdir()):
        if subdir.is_dir():
            pocket_files = list(subdir.glob("*pocket.pdb"))
            if not pocket_files:
                continue
            pocket_file = pocket_files[0]
            center = compute_centroid(pocket_file)
            if center is not None:
                result_lines.append(f"{subdir.name},{center[0]:.3f},{center[1]:.3f},{center[2]:.3f}")

    with open(output_file, "w") as out:
        out.write("ProteinID,X,Y,Z\n")
        out.write("\n".join(result_lines))
    print(f"Grid centers saved to {output_file}")

# Usage
if __name__ == "__main__":
    extract_all_grid_centers("PDbind-refined")

