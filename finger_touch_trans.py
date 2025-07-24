import os
import json
import numpy as np


def compute_shifted_finger_mask(finger_touch, rule='wo5'):
    finger_touch = np.array(finger_touch, dtype=np.float32).flatten()
    output_mask = np.zeros(5, dtype=np.float32)

    rule_mappings = {
        'wo2': {0: 2, 1: 3, 2: 4, 3: 0, 4: 1},
        'wo3': {0: 1, 1: 3, 2: 4, 3: 0, 4: 2},
        'wo4': {0: 1, 1: 2, 2: 4, 3: 0, 4: 3},
        'wo5': {0: 1, 1: 2, 2: 3, 3: 0, 4: 4},
        'rotate': {i: (i + 1) % 5 for i in range(5)}  # original rotate rule
    }

    if rule not in rule_mappings:
        raise ValueError(f"Invalid rule: {rule}. Must be one of {list(rule_mappings.keys())}")

    mapping = rule_mappings[rule]
    for i in range(5):
        if finger_touch[i] == 1.0:
            output_mask[mapping[i]] = 1.0

    return output_mask.tolist()


def update_finger_masks_in_json(json_root_dir, rule='wo5'):
    """
    Update finger_touch fields in JSON files under json_root_dir using specified rule.
    Args:
        json_root_dir (str): Directory containing JSON files.
        rule (str): Mapping rule to apply. Options: 'wo2', 'wo3', 'wo4', 'wo5', 'rotate'
    """
    for root, _, files in os.walk(json_root_dir):
        for file in sorted(files):
            if not file.endswith('.json'):
                continue

            file_path = os.path.join(root, file)

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                situation_key = f"situation_{rule}"
                finger_key = f"finger_touch_{rule}"

                finger_touch = data.get(situation_key, {}).get(finger_key, None)
                if finger_touch is None or len(finger_touch) != 5:
                    print(f"Skipped (invalid or missing): {file}")
                    continue

                new_mask = compute_shifted_finger_mask(finger_touch, rule)
                data[situation_key][finger_key] = new_mask

                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)

                print(f"Updated: {file} | from {finger_touch} to {new_mask}")

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")