import os
import torch
import json
import numpy as np
import trimesh
import mano
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
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

class JSONHandObjectDataset(Dataset):
    def __init__(self, 
                 json_root='/root/autodl-tmp/try/dataset_building/output_0.10_less_obj',
                 obj_model_root='/root/autodl-tmp/try/dataset_building/obj_model',
                 mano_model_path='/root/autodl-tmp/fingerless_gen/MANO_RIGHT.pkl', 
                 sample_n=3000):
        set_seed(42)
        self.json_root = json_root
        self.obj_model_root = obj_model_root
        self.sample_n = sample_n
        self.extra_indices = [744, 320, 443, 554, 671]

        # collest wo2_finger_position == 10234
        self.json_files = []
        for root, _, files in os.walk(json_root):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            if 'wo2_finger_position' in data and data['wo2_finger_position'] == [1, 0, 2, 3, 4]:
                                self.json_files.append(file_path)
                    except Exception as e:
                        print(f"Cannot Read: {file_path}, ERROR: {e}")
        self.json_files.sort()

        self.mano_layer = mano.load(
            model_path=mano_model_path,
            model_type='mano',
            use_pca=False,
            num_pca_comps=45,
            batch_size=1,
            flat_hand_mean=True
        )

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_file = self.json_files[idx]
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Hand Param
        hand_beta = torch.tensor(data['handBeta'], dtype=torch.float32).unsqueeze(0)
        hand_orient = torch.tensor(data['handOrient'], dtype=torch.float32).unsqueeze(0)
        hand_pose = torch.tensor(data['handPose'], dtype=torch.float32).unsqueeze(0)
        hand_transl = torch.tensor(data['handTransl'], dtype=torch.float32).unsqueeze(0)
        hand_param = torch.cat([hand_beta, hand_orient, hand_pose, hand_transl], dim=1).squeeze(0)

        output = self.mano_layer(betas=hand_beta, global_orient=hand_orient,
                                 hand_pose=hand_pose, transl=hand_transl)
        hand_verts = output.vertices.squeeze(0)
        hand_joints = output.joints.view(-1)
        hand_fingertip = hand_verts[self.extra_indices, :]

        # Object pc
        obj_name = data['obj_name']
        obj_rot = np.array(data['obj_orient'])
        obj_trans = np.array(data['obj_transl'])
        obj_path = os.path.join(self.obj_model_root, obj_name, f"{obj_name}.obj")

        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"No Object Model: {obj_path}")

        obj_mesh = trimesh.load(obj_path, process=False)
        obj_points, _ = trimesh.sample.sample_surface(obj_mesh, self.sample_n)
        rot_mat = R.from_rotvec(obj_rot).as_matrix()
        obj_points = (rot_mat @ obj_points.T).T + obj_trans
        obj_pc = torch.tensor(obj_points, dtype=torch.float32)

        return obj_pc.detach(), hand_param.detach(), hand_fingertip.detach(), hand_joints.detach(), json_file

if __name__ == '__main__':
    dataset = JSONHandObjectDataset()
    print(f"Valid sample count: {len(dataset)}")
    sample = dataset[1]

    obj_pc, hand_param, hand_fingertip, hand_joints, json_file = sample
    print("obj_pc shape:", obj_pc.shape)             # [3000, 3]
    print("hand_param shape:", hand_param.shape)     # [61]
    print("hand_fingertip shape:", hand_fingertip.shape)  # [5, 3]
    print("hand_joints shape:", hand_joints.shape)   # [48]
    print("json file path:", json_file)