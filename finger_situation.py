import os
import json
import time
import random
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R
from utils import grasp_score

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def process_grasp_scores(json_root_dir, object_model_root, sample_num=3000, seed=42, finger_idx=5):
    """
    Process a batch of JSON files and compute grasp scores using hand-object contact.

    Args:
        json_root_dir (str): Root directory containing JSON files.
        object_model_root (str): Root directory of object .obj models.
        sample_num (int): Number of surface points to sample from each object mesh.
        seed (int): Random seed.
        finger_idx (int): The digit to use in the 'situation_woX' field name.
    """
    set_seed(seed)

    total_inference_time = 0.0
    processed_files = 0
    global_start_time = time.perf_counter()

    field_name = f"situation_wo{finger_idx}"

    for root, dirs, files in os.walk(json_root_dir):
        for file in sorted(files):
            if not file.endswith('.json'):
                continue

            file_path = os.path.join(root, file)

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Parse rhand_mesh
                mesh_lines = data.get("rhand_mesh", [])
                verts, faces = [], []
                for line in mesh_lines:
                    if line.startswith("v "):
                        parts = line.strip().split()
                        if len(parts) == 4:
                            verts.append([float(x) for x in parts[1:]])
                    elif line.startswith("f "):
                        parts = line.strip().split()
                        if len(parts) == 4:
                            faces.append([int(x) - 1 for x in parts[1:]])  # OBJ index starts from 1

                if not verts or not faces:
                    continue

                verts = np.array(verts, dtype=np.float32)
                faces = np.array(faces, dtype=np.int32)

                # Parse object info
                obj_name = data.get("obj_name", None)
                obj_transl = np.array(data.get("obj_transl", []))
                obj_orient = np.array(data.get("obj_orient", []))

                if obj_name is None or obj_transl.shape != (3,) or obj_orient.shape != (3,):
                    continue

                # Load object mesh
                obj_path = os.path.join(object_model_root, obj_name, f"{obj_name}.obj")
                if not os.path.exists(obj_path):
                    continue

                obj_mesh = trimesh.load(obj_path, process=False)
                rot_mat = R.from_rotvec(obj_orient).as_matrix()
                obj_mesh.vertices = (rot_mat @ obj_mesh.vertices.T).T + obj_transl

                np.random.seed(seed)
                obj_xyz_resampled, _ = trimesh.sample.sample_surface(obj_mesh, sample_num)

                # Inference
                start_time = time.perf_counter()
                forces, torques, normals, finger_touch = grasp_score.get_contact_points_wo5(
                    verts, faces, obj_xyz_resampled
                )
                new_score = grasp_score.graspit_measure(forces, torques, normals)
                inference_time = time.perf_counter() - start_time

                # Update JSON
                data[field_name] = {
                    f"grasp_score_wo{finger_idx}": float(new_score),
                }

                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)

                total_inference_time += inference_time
                processed_files += 1

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    global_elapsed = time.perf_counter() - global_start_time
    average_time = total_inference_time / processed_files if processed_files > 0 else 0

    print("Processing complete.")
    print(f"Total runtime: {global_elapsed:.2f} seconds")
    print(f"Average inference time per sample: {average_time:.6f} seconds")
    print(f"Total samples processed: {processed_files}")