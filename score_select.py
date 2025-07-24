import os
import json
import numpy as np
import torch
import trimesh
import pickle
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from utils import grasp_score
import mano
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def verts_faces_to_obj_lines(verts, faces):
    lines = []
    for v in verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for f in faces:
        lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")  # OBJ index starts from 1
    return lines


def extract_low_score_grasps(
    meta_dir,
    mano_model_path,
    obj_path,
    output_json_dir,
    output_txt_path,
    score_threshold=0.1,
    min_score=0.0,
    sample_points=3000,
    seed=42
):
    set_seed(seed)
    os.makedirs(output_json_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mano_layer = mano.load(
        model_path=mano_model_path,
        model_type='mano',
        use_pca=False,
        num_pca_comps=45,
        batch_size=1,
        flat_hand_mean=True
    ).to(device)
    faces = mano_layer.faces

    obj_mesh = trimesh.load(obj_path, process=False)

    json_index = 0

    with open(output_txt_path, "w") as out_file:
        for filename in tqdm(sorted(os.listdir(meta_dir))):
            if not filename.endswith('.pkl'):
                continue
            file_path = os.path.join(meta_dir, filename)

            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)

                handpose = torch.from_numpy(data['handPose']).float().unsqueeze(0).to(device)
                handbetas = torch.from_numpy(data['handBeta']).float().unsqueeze(0).to(device)
                handtransl = torch.from_numpy(data['handTrans']).float().unsqueeze(0).to(device)

                output = mano_layer(
                    betas=torch.zeros(1, 10).to(device),
                    global_orient=handpose[:, :3],
                    hand_pose=handpose[:, 3:48],
                    transl=handtransl
                )
                verts = output.vertices.squeeze(0).detach().cpu().numpy()

                obj_rot = data['objRot']
                obj_trans = data['objTrans']

                obj_mesh_instance = obj_mesh.copy()
                rot_mat = R.from_rotvec(obj_rot.reshape(3)).as_matrix()
                transformed_verts = (rot_mat @ obj_mesh_instance.vertices.T).T + obj_trans
                obj_mesh_instance.vertices = transformed_verts

                np.random.seed(seed)
                obj_xyz_resampled, _ = trimesh.sample.sample_surface(obj_mesh_instance, sample_points)

                forces, torques, normals, finger_touch = grasp_score.get_contact_points(
                    verts, faces, obj_xyz_resampled
                )
                score = grasp_score.graspit_measure(forces, torques, normals)

                if min_score < score < score_threshold and finger_touch[-1] == 1:
                    print(f"{filename}: grasp score = {score:.4f}")
                    out_file.write(f"{filename} {score:.4f} {finger_touch}\n")

                    rhand_mesh_lines = verts_faces_to_obj_lines(verts, faces)

                    result_dict = {
                        "rhand_mesh": rhand_mesh_lines,
                        "handBeta": np.array(torch.zeros(1, 10)).flatten().tolist(),
                        "handOrient": np.array(handpose[:, :3].detach().cpu()).flatten().tolist(),
                        "handPose": np.array(handpose[:, 3:48].detach().cpu()).flatten().tolist(),
                        "handTransl": np.array(handtransl.detach().cpu()).flatten().tolist(),
                        "obj_name": data.get('objName', "unknown"),
                        "obj_transl": np.array(obj_trans).flatten().tolist(),
                        "obj_orient": np.array(obj_rot).flatten().tolist(),
                        "grasp_score": float(score),
                        "finger_touch": finger_touch.tolist()
                    }

                    json_filename = f"{json_index:04d}.json"
                    json_output_path = os.path.join(output_json_dir, json_filename)
                    with open(json_output_path, 'w') as jf:
                        json.dump(result_dict, jf, indent=2)

                    json_index += 1

            except Exception as e:
                print(f"Error processing file {filename}: {e}")