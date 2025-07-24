import os
import json
import numpy as np

def compute_grasp_score_mask(data, threshold=0.15):
    """
    Compute a 5D mask vector based on grasp_score_wox values in the JSON structure.

    Returns:
        score_mask (np.ndarray): array of shape (5,)
    """
    score_mask = np.zeros(5, dtype=np.float32)
    score_mask[0] = 1.0  # Always set first element to 1.0

    checks = [
        ("situation_wo2", "grasp_score_wo2", 1),
        ("situation_wo3", "grasp_score_wo3", 2),
        ("situation_wo4", "grasp_score_wo4", 3),
        ("situation_wo5", "grasp_score_wo5", 4),
    ]

    for section, key, idx in checks:
        if section in data and key in data[section]:
            val = data[section][key]
            if isinstance(val, (float, int)) and val > threshold:
                score_mask[idx] = 1.0

    return score_mask


def update_json_with_score_mask(json_root_dir, threshold=0.15):
    """
    Batch process JSON files and add 'necessary_points_0.15' based on grasp_score thresholds.
    """
    for root, _, files in os.walk(json_root_dir):
        for file in sorted(files):
            if not file.endswith(".json"):
                continue

            file_path = os.path.join(root, file)

            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                score_mask = compute_grasp_score_mask(data, threshold)
                data[f"necessary_points_{threshold}"] = score_mask.tolist()

                with open(file_path, "w") as f:
                    json.dump(data, f, indent=2)

                print(f"Updated {file} | mask = {score_mask.tolist()}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")