import os
import sys
import cv2
import numpy as np
import trimesh

sys.path.append("/root/autodl-tmp/fingerless_gen/affordance-CVAE/utils")
from utils import utils_1


def load_objects_HO3D(obj_root):
    object_names = [
        '011_banana', '021_bleach_cleanser', '003_cracker_box',
        '035_power_drill', '025_mug', '006_mustard_bottle',
        '019_pitcher_base', '010_potted_meat_can',
        '037_scissors', '004_sugar_box'
    ]

    obj_pc = {}
    obj_face = {}
    obj_scale = {}
    obj_pc_resampled = {}
    obj_resampled_faceid = {}

    for obj_name in object_names:
        obj_dir = os.path.join(obj_root, obj_name)
        obj_file = os.path.join(obj_dir, 'textured_simple.obj')
        mesh_data = utils_1.fast_load_obj(open(obj_file))[0]

        verts = mesh_data['vertices']
        faces = mesh_data['faces']
        obj_pc[obj_name] = verts
        obj_face[obj_name] = faces
        obj_scale[obj_name] = get_diameter(verts)

        resampled_path = obj_file.replace('textured_simple.obj', 'resampled.npy')
        faceid_path = obj_file.replace('textured_simple.obj', 'resample_face_id.npy')

        obj_pc_resampled[obj_name] = np.load(resampled_path)
        obj_resampled_faceid[obj_name] = np.load(faceid_path)

    return obj_pc, obj_face, obj_scale, obj_pc_resampled, obj_resampled_faceid


def resample_obj_xyz(verts, faces, path):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    sampled_points, face_ids = trimesh.sample.sample_surface(mesh, 3000)

    np.save(path.replace('textured_simple.obj', 'resampled.npy'), sampled_points)
    np.save(path.replace('textured_simple.obj', 'resample_face_id.npy'), face_ids)


def get_diameter(vertices):
    max_vals = np.max(vertices, axis=0)
    min_vals = np.min(vertices, axis=0)
    diff = max_vals - min_vals
    return np.linalg.norm(diff)


def readTxt(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def pose_from_RT_HO3D(R, T):
    pose = np.eye(4)
    pose[:3, 3] = T
    R33, _ = cv2.Rodrigues(R)
    pose[:3, :3] = R33
    return pose


if __name__ == '__main__':
    obj_root = '/root/autodl-tmp/try/dataset/models'
    obj_pc, obj_face, obj_scale, obj_pc_resampled, obj_resampled_faceid = load_objects_HO3D(obj_root)