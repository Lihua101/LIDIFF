import os
import numpy as np
import open3d as o3d
import sys
import os
sys.path.append(os.getcwd())
# print(sys.path)
from lidiff.utils.metrics import ChamferDistance, PrecisionRecall, CompletionIoU, RMSE 
import tqdm
from natsort import natsorted
from lidiff.tools.diff_completion_pipeline import DiffCompletion
from lidiff.utils.histogram_metrics import compute_hist_metrics 
import click
import json
import random
import yaml
import torch

PATH_DATA = './Datasets/SemanticKITTI/dataset/sequences/001'
label_mapping="./lidiff/config/semantic-kitti.yaml"


completion_iou = CompletionIoU()
rmse = RMSE()
chamfer_distance = ChamferDistance()
precision_recall = PrecisionRecall(0.05,2*0.05,100)

def parse_calibration(filename):
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib

def load_poses(calib_fname, poses_fname):
    if os.path.exists(calib_fname):
        calibration = parse_calibration(calib_fname)
        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

    poses_file = open(poses_fname)
    poses = []

    for line in poses_file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        if os.path.exists(calib_fname):
            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
        else:
            poses.append(pose)

    return poses


def get_scan_completion(scan_path, path, diff_completion, max_range):
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
        learning_map = semkittiyaml['learning_map']
    pcd_file = os.path.join(PATH_DATA, 'velodyne', scan_path)
    points = np.fromfile(pcd_file, dtype=np.float32)
    points = points.reshape(-1,4)

    lab_file = pcd_file.replace("velodyne", "labels").replace(".bin", ".label")
    l_part = np.fromfile(lab_file, dtype=np.uint32).reshape((-1, 1))
    l_part = l_part & 0xFFFF
    annotated_data = np.vectorize(learning_map.__getitem__)(l_part)
    l_set = l_part.reshape((-1))
    static_idx = (l_set < 252) & (l_set > 1)
    points = points[static_idx]
    annotated_data = annotated_data[static_idx]
    dist = np.sqrt(np.sum(points[:,:3]**2, axis=-1))
    a = dist < max_range
    points = points[a]
    annotated_data = annotated_data[a]
    one_hot_part = np.zeros((annotated_data.shape[0], 20))
    one_hot_part[np.arange(annotated_data.shape[0]), annotated_data.flatten()] = 1
    input_points = np.concatenate((points[:,:3],one_hot_part),axis=-1)
    
    
    if diff_completion is None:
        pred_path = f'{scan_path.split(".")[0]}.ply'
        pcd_pred = o3d.io.read_point_cloud(os.path.join(path, pred_path))
        points = np.array(pcd_pred.points)
        dist = np.sqrt(np.sum(points**2, axis=-1))
        pcd_pred.points = o3d.utility.Vector3dVector(points[dist < max_range])
    else:
        diff_complete_scan= diff_completion.complete_scan(input_points)
        # pcd_pred = complete_scan

        # diff_pcd_pred = o3d.geometry.PointCloud()
        # diff_pcd_pred.points = o3d.utility.Vector3dVector(diff_complete_scan)

        # pcd_pred = o3d.geometry.PointCloud()
        # pcd_pred.points = o3d.utility.Vector3dVector(complete_scan)

    return diff_complete_scan, input_points

def get_ground_truth(pose, cur_scan, seq_map, max_range):
    trans = pose[:-1,-1]
    dist_gt = np.sum((seq_map[:,:3] - trans)**2, axis=-1)**.5
    scan_gt = seq_map[dist_gt < max_range]
    scan_gt_2 = np.concatenate((scan_gt[:,:3], np.ones((len(scan_gt),1))), axis=-1)
    scan_gt_2 = (scan_gt_2 @ np.linalg.inv(pose).T)[:,:3]  #pose是雷达到全局
    scan_gt = scan_gt[(scan_gt_2[:,2] > -4.) & (scan_gt_2[:,2] < 4.4)]
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(scan_gt[:,:3])

    # filter only over the view point
    cur_pcd = o3d.geometry.PointCloud()
    cur_pcd.points = o3d.utility.Vector3dVector(cur_scan[:,:3])
    viewpoint_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cur_pcd, voxel_size=10.)
    in_viewpoint = viewpoint_grid.check_if_included(pcd_gt.points)


    pcd_gt_l = scan_gt[in_viewpoint]
    # points_gt = np.array(pcd_gt.points)
    # pcd_gt.points = o3d.utility.Vector3dVector(points_gt[in_viewpoint])

    return pcd_gt_l


@click.command()
@click.option('--path', '-p', type=str, default='results/diff_net_T50_s6.0', help='path to the scan sequence')
@click.option('--voxel_size', '-v', type=float, default=0.05, help='voxel size')
@click.option('--max_range', '-m', type=float, default=50, help='max range')
@click.option('--denoising_steps', '-t', type=int, default=10, help='number of denoising steps')
@click.option('--cond_weight', '-s', type=float, default=6.0, help='conditioning weights')
@click.option('--diff', '-d', type=str, help='run diffusion pipeline')
@click.option('--refine', '-r', type=str, help='path to the checkpoint for refinement net')
def main(path, voxel_size, max_range, denoising_steps, cond_weight, diff, refine): 
    diff_completion = DiffCompletion(diff, refine, denoising_steps, cond_weight)

    poses = load_poses(os.path.join(PATH_DATA, 'calib.txt'), os.path.join(PATH_DATA, 'poses.txt'))
    seq_map = np.load(f'{PATH_DATA}/map_clean1.npy')

    jsd_3d = []
    jsd_bev = []

    # for pose, scan_path in tqdm.tqdm(random.sample(list(zip(poses, natsorted(os.listdir(f'{PATH_DATA}/velodyne')))),100)):
    for pose, scan_path in tqdm.tqdm(list(zip(poses, natsorted(os.listdir(f'{PATH_DATA}/velodyne'))))):
        # diff_pcd_pred, pcd_pred, cur_scan = get_scan_completion(scan_path, path, diff_completion, max_range)
        diff_complete_scan, input_points = get_scan_completion(scan_path, path, diff_completion, max_range)
        pcd_gt_l = get_ground_truth(pose, input_points, seq_map, max_range)

        for class_id in range(20):
            indices_1 = np.where(diff_complete_scan[:, 3] == class_id)[0]
            points_1 = diff_complete_scan[indices_1, :3]
            diff_pcd_pred = o3d.geometry.PointCloud()
            diff_pcd_pred.points = o3d.utility.Vector3dVector(points_1)

            indices_2 = np.where(pcd_gt_l[:, 3] == class_id)[0]
            points_2 = pcd_gt_l[indices_2, :3]
            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(points_2)

            completion_iou.update(pcd_gt, diff_pcd_pred)
            thr_ious = completion_iou.compute()
            for v_size in thr_ious.keys():
                print(f'Voxel {v_size}cm IOU: {thr_ious[v_size]}')
            # print(1)

        # jsd_3d.append(compute_hist_metrics(pcd_gt, diff_pcd_pred, bev=False))
        # jsd_bev.append(compute_hist_metrics(pcd_gt, diff_pcd_pred, bev=True))
        # print(f'JSD 3D: {jsd_3d[-1]}')
        # print(f'JSD BEV: {jsd_bev[-1]}')

        # rmse.update(pcd_gt, diff_pcd_pred)
        # completion_iou.update(pcd_gt, diff_pcd_pred)
        # chamfer_distance.update(pcd_gt, diff_pcd_pred)
        # precision_recall.update(pcd_gt, diff_pcd_pred)

        # rmse_mean, rmse_std = rmse.compute()
        # print(f'RMSE Mean: {rmse_mean}\tRMSE Std: {rmse_std}')
        # thr_ious = completion_iou.compute()
        # for v_size in thr_ious.keys():
        #     print(f'Voxel {v_size}cm IOU: {thr_ious[v_size]}')
        # cd_mean, cd_std = chamfer_distance.compute()
        # print(f'CD Mean: {cd_mean}\tCD Std: {cd_std}')
        # pr, re, f1 = precision_recall.compute_auc()
        # print(f'Precision: {pr}\tRecall: {re}\tF-Score: {f1}')


    print('\n\n=================== FINAL RESULTS ===================\n\n')
    # print(f'JSD 3D: {np.array(jsd_3d).mean()}')
    # print(f'JSD BEV: {np.array(jsd_bev).mean()}')
    # print(f'RMSE Mean: {rmse_mean}\tRMSE Std: {rmse_std}')
    # thr_ious = completion_iou.compute()
    # for v_size in thr_ious.keys():
    #     print(f'Voxel {v_size}cm IOU: {thr_ious[v_size]}')
    # cd_mean, cd_std = chamfer_distance.compute()
    # print(f'CD Mean: {cd_mean}\tCD Std: {cd_std}')
    # pr, re, f1 = precision_recall.compute_auc()
    # print(f'Precision: {pr}\tRecall: {re}\tF-Score: {f1}')
    
    res_dict = {
        # 'jsd': np.array(jsd_bev).mean(),
        # 'jsd_noclip_3d': np.array(jsd_3d).mean(),
        # 'rmse_mean': rmse_mean, 'rmse_std': rmse_std,
        'ious': thr_ious,
        # 'cd_mean': cd_mean, 'cd_std': cd_std,
        # 'pr': pr, 're': re, 'f1': f1,
    }

    log_path = os.path.join(*path.split('/')[:-1])
    with open(f'{log_path}/res_log.yaml', 'w+') as log_res:
        json.dump(res_dict, log_res)

if __name__ == '__main__':
    main()
