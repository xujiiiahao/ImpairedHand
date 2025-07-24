import os
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from utils import utils_loss, loss, finger_repulsion_loss
import mano
from json_read_10234 import JSONHandObjectDataset
import numpy as np
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

def save_hand_param_to_json(json_path, hand_param_tensor):
    hand_param = hand_param_tensor.squeeze(0).detach().cpu().numpy().tolist()
    data = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
    data["wo2_finger_position_param"] = hand_param
  
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

def optimize_mano_params_in_batch(
    dataset,
    mano_model_path: str,
    save_json: bool = True,
    json_save_dir: str = None,
    batch_size: int = 128,
    num_iters: int = 6000,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    set_seed(42)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    extra_indices = [744, 320, 443, 554, 671]
    
    thumb_idx = finger_repulsion_loss.get_finger_indices('thumb').to(device)
    index_idx = finger_repulsion_loss.get_finger_indices('index').to(device)
    middle_idx = finger_repulsion_loss.get_finger_indices('middle').to(device)
    ring_idx = finger_repulsion_loss.get_finger_indices('ring').to(device)
    pinky_idx = finger_repulsion_loss.get_finger_indices('pinky').to(device)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Optimizing")):
        obj_pcs, hand_params, hand_fingertips, hand_joints, json_paths = batch

        B = obj_pcs.shape[0]
        mano_layer = mano.load(model_path=mano_model_path,
                           model_type='mano',
                           use_pca=False,
                           num_pca_comps=45,
                           batch_size=B,
                           flat_hand_mean=True).to(device)
        obj_pcs = obj_pcs.to(device)
        hand_params = hand_params.to(device)
        hand_fingertips = hand_fingertips.to(device)
        hand_joints = hand_joints.to(device)

        target_points = hand_fingertips[:, :4, :].to(device)
        target_joints = torch.cat([hand_joints[:, :3], hand_joints[:, 39:48], hand_joints[:, 3:12], hand_joints[:, 12:21], hand_joints[:, 30:39]], dim=1)
        target_joints = target_joints.view(B, -1).to(device)
        hand_params_copy = hand_params.clone().detach()

        fixed_betas = hand_params_copy[:, :10].to(device)
        hand_transl = hand_params_copy[:, 58:61].clone().to(device).requires_grad_(True)
        hand_orient = hand_params_copy[:, 10:13].clone().to(device).requires_grad_(True)
        fixed_finger_pose_1 = torch.zeros((B, 9), device=device) # index [13:22]
        fixed_finger_pose_1[:, 2] = -3.14
        finger_pose_2 = hand_params_copy[:, 13:22].clone().to(device).requires_grad_(True) # middle [22:31]
        finger_pose_3 = hand_params_copy[:, 40:49].clone().to(device).requires_grad_(True) # pinky [31:40]
        finger_pose_4 = hand_params_copy[:, 22:31].clone().to(device).requires_grad_(True) # ring [40:49]
        finger_pose_5 = hand_params_copy[:, 49:58].clone().to(device).requires_grad_(True) # thumb [49:58]
        faces = torch.from_numpy(mano_layer.faces.astype(np.int32)).view(1, -1, 3).repeat(B, 1, 1).to(device)

        optimizer = torch.optim.Adam([hand_orient, finger_pose_2, finger_pose_3, 
                                finger_pose_4, finger_pose_5, hand_transl], lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

        for iter in range(num_iters):
            optimizer.zero_grad()

            finger_pose_all = torch.cat([fixed_finger_pose_1, finger_pose_2, finger_pose_3, 
                                        finger_pose_4, finger_pose_5], dim=1)
            output = mano_layer(betas=fixed_betas,global_orient=hand_orient,
                                hand_pose=finger_pose_all,transl=hand_transl)
            pred_verts = output.vertices
            pred_joints = output.joints.view(B, -1)
            pred_joints_selected = torch.cat([pred_joints[:, :3], pred_joints[:, 39:48], pred_joints[:, 12:21],
                                        pred_joints[:, 30:39], pred_joints[:, 21:30]], dim=1)
            new_extra_indices = extra_indices[:1] + extra_indices[2:]
            pred_points = pred_verts[:, new_extra_indices, :]
            obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(obj_pcs, pred_verts)
            finger_verts = {"index": pred_verts[:, index_idx, :],
                            "middle": pred_verts[:, middle_idx, :],
                            "ring": pred_verts[:, ring_idx, :],
                            "pinky": pred_verts[:, pinky_idx, :],
                            "thumb": pred_verts[:, thumb_idx, :]}
            
            point_loss = torch.nn.functional.mse_loss(pred_points, target_points) 
            penetr_loss = loss.inter_penetr_loss(pred_verts, faces, obj_pcs,
                                                obj_nn_dist_recon, obj_nn_idx_recon)
            joint_loss = torch.nn.functional.mse_loss(pred_joints_selected, target_joints) 
            pose_l2_loss = torch.mean(finger_pose_all ** 2)
            repulse_loss = finger_repulsion_loss.repulsion_loss(finger_verts, threshold=0.0005)

            total_loss = point_loss * 1e10 + penetr_loss * 1e7 + pose_l2_loss * 1e-3 + abs( joint_loss - 1e-4 ) * 1e5 + repulse_loss * 1e9
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            tl = total_loss.item()
            pl = point_loss.item()
            pnl = penetr_loss.item()
            jl = joint_loss.item()
            ll = pose_l2_loss.item()
            rl = repulse_loss.item()

            if iter % 1000 == 0:
                print(f"Iteration {iter}/{num_iters}, "
                        f"Total Loss: {tl:.8f} | "
                        f"Point Loss: {pl:.8f} | "
                        f"Penetration Loss: {pnl:.8f} | "
                        f"Joint Loss: {jl:.8f} | "
                        f"L2 Loss: {ll:.8f} | "
                        f"Repulse Loss: {rl:.8f} | ")
                
        final_pose = torch.cat([fixed_finger_pose_1, finger_pose_2.detach(), 
                                finger_pose_3.detach(), finger_pose_4.detach(), finger_pose_5.detach()], dim=1)
        full_hand_param = torch.cat([
                fixed_betas,
                hand_orient.detach(),
                final_pose,
                hand_transl.detach()
            ], dim=1)  # [B, 61]
        if save_json:
                for i in range(B):
                    if json_save_dir:
                        os.makedirs(json_save_dir, exist_ok=True)
                        new_json_path = os.path.join(json_save_dir, os.path.basename(json_paths[i]))
                    else:
                        new_json_path = json_paths[i]
                    save_hand_param_to_json(new_json_path, full_hand_param[i, :])



        


if __name__ == "__main__":
    set_seed(42)
    mano_model_path = "/ImpairedHand/dataset/mano_model/MANO_RIGHT.pkl"
    json_dir = "/root/autodl-tmp/ImpairedHand/dataset/ImpairedHand_dataset"
    obj_model_root = "/root/autodl-tmp/ImpairedHand/dataset/obj_model"
    dataset = JSONHandObjectDataset(json_dir, obj_model_root, mano_model_path)
    optimize_mano_params_in_batch(
        dataset=dataset,
        mano_model_path=mano_model_path,
        batch_size=128,
        num_iters=6000,
        save_json=True,
        json_save_dir=None  
    )