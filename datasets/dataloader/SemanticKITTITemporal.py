import torch
from torch.utils.data import Dataset
from lidiff.utils.pcd_preprocess import point_set_to_coord_feats, aggregate_pcds, load_poses
from lidiff.utils.pcd_transforms import *
from lidiff.utils.data_map import learning_map
from lidiff.utils.collations import point_set_to_sparse
from natsort import natsorted
import os
import numpy as np
import yaml
import numba as nb

import warnings

warnings.filterwarnings('ignore')

#################################################
################## Data loader ##################
#################################################



def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)

def polar2cat(input_xyz_polar):
    # print(input_xyz_polar.shape)
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label

class TemporalKITTISet(Dataset):
    def __init__(self, data_dir, seqs, split, resolution, num_points, max_range, dataset_norm=False, std_axis_norm=False, label_mapping="./lidiff/config/semantic-kitti.yaml"):
        super().__init__()

        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']

        self.data_dir = data_dir
        self.n_clusters = 50
        self.resolution = resolution
        self.num_points = num_points
        self.max_range = max_range

        self.split = split
        self.seqs = seqs
        self.cache_maps = {}


        self.fixed_volume_space = True
        self.max_volume_space = [50.0,3.1415926,2.0]
        self.min_volume_space = [0.0,-3.1415926,-4.0]
        self.grid_size = np.asarray([480,360,32])
        self.ignore_label = 0


        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath_list()
        self.data_stats = {'mean': None, 'std': None}

        if os.path.isfile(f'utils/data_stats_range_{int(self.max_range)}m.yml') and dataset_norm:
            stats = yaml.safe_load(open(f'utils/data_stats_range_{int(self.max_range)}m.yml'))
            data_mean = np.array([stats['mean_axis']['x'], stats['mean_axis']['y'], stats['mean_axis']['z']])
            if std_axis_norm:
                data_std = np.array([stats['std_axis']['x'], stats['std_axis']['y'], stats['std_axis']['z']])
            else:
                data_std = np.array([stats['std'], stats['std'], stats['std']])
            self.data_stats = {
                'mean': torch.tensor(data_mean),
                'std': torch.tensor(data_std)
            }

        self.nr_data = len(self.points_datapath)

        print('The size of %s data is %d'%(self.split,len(self.points_datapath)))





    def datapath_list(self):
        self.points_datapath = []
        self.seq_poses = []
        self.labels_datapath = []

        for seq in self.seqs:
            point_seq_path = os.path.join(self.data_dir, 'dataset', 'sequences', seq)
            point_seq_bin = natsorted(os.listdir(os.path.join(point_seq_path, 'velodyne')))
            poses = load_poses(os.path.join(point_seq_path, 'calib.txt'), os.path.join(point_seq_path, 'poses.txt'))
            p_full = np.load(f'{point_seq_path}/map_clean1.npy') if self.split != 'test' else np.array([[1,0,0],[0,1,0],[0,0,1]])
            self.cache_maps[seq] = p_full
 
            for file_num in range(0, len(point_seq_bin)):
                self.points_datapath.append(os.path.join(point_seq_path, 'velodyne', point_seq_bin[file_num]))
                self.labels_datapath.append(os.path.join(point_seq_path, 'labels', (point_seq_bin[file_num].split('.')[0] + '.label')))
                self.seq_poses.append(poses[file_num])

    def transforms(self, points):
        points = np.expand_dims(points, axis=0)
        points[:,:,:3] = rotate_point_cloud(points[:,:,:3])
        points[:,:,:3] = rotate_perturbation_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_scale_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_flip_point_cloud(points[:,:,:3])

        return np.squeeze(points, axis=0)

    def __getitem__(self, index):
        seq_num = self.points_datapath[index].split('/')[-3]
        fname = self.points_datapath[index].split('/')[-1].split('.')[0]

        p_part = np.fromfile(self.points_datapath[index], dtype=np.float32)
        p_part = p_part.reshape((-1,4))
        annotated_data = np.zeros((p_part.shape[0],20)) 
        
        if self.split != 'test':
            l_part = np.fromfile(self.labels_datapath[index], dtype=np.uint32).reshape((-1, 1))
            l_part = l_part & 0xFFFF
            annotated_data = np.vectorize(self.learning_map.__getitem__)(l_part)
            l_set = l_part.reshape((-1))
            static_idx = (l_set < 252) & (l_set > 1)
            p_part = p_part[static_idx]
            annotated_data = annotated_data[static_idx]
        dist_part = np.sum(p_part**2, -1)**.5
        a = (dist_part < self.max_range) & (dist_part > 3.5) & (p_part[:,2] > -4.)
        p_part = p_part[a]
        annotated_data = annotated_data[a]
        one_hot_part = np.zeros((annotated_data.shape[0], 20))
        one_hot_part[np.arange(annotated_data.shape[0]), annotated_data.flatten()] = 1
        p_part = np.concatenate((p_part[:,:3],one_hot_part),axis=-1)



        # data_tuple = (p_part[:, :3], annotated_data.astype(np.uint8))
        # data_tuple += (p_part[:, 3],)
        # xyz, labels, sig = data_tuple

        # xyz_pol = cart2polar(xyz)

        # if self.fixed_volume_space:
        #     max_bound = np.asarray(self.max_volume_space)
        #     min_bound = np.asarray(self.min_volume_space)
        # # get grid index
        # crop_range = max_bound - min_bound
        # cur_grid_size = self.grid_size
        # intervals = crop_range / (cur_grid_size - 1)

        # if (intervals == 0).any(): print("Zero interval!")
        # grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(int)

        # voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        # dim_array = np.ones(len(self.grid_size) + 1, int)
        # dim_array[0] = -1
        # voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        # voxel_position = polar2cat(voxel_position)

        # processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label

        # label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        # label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        # processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        # data_tuple = (voxel_position, processed_label)

        # voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        # return_xyz = xyz_pol - voxel_centers
        # return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)
        # return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)

        # data_tuple += (grid_ind, labels, return_fea)
        # pytorch_device = torch.device('cuda:0')
        # demo_pt_fea = [data_tuple[4]]
        # demo_grid = [data_tuple[2]]



        ############################################################
        # demo_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in demo_pt_fea]
        # demo_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in demo_grid]
        ############################################################




        demo_batch_size = 1
        pose = self.seq_poses[index]


        if self.split != 'test':
            p_map = self.cache_maps[seq_num]
            p_map_dis = p_map[:,:3]
            p_map_lab = p_map[:,3].reshape(-1,1).astype(np.int64)
            trans = pose[:-1,-1]
            dist_full = np.sum((p_map_dis - trans)**2, -1)**.5
            a = dist_full < self.max_range
            p_full = p_map_dis[a]#  问题在这儿！
            p_map_lab = p_map_lab[a]
            # p_full = p_map
            p_full = np.concatenate((p_full, np.ones((len(p_full),1))), axis=-1)
            p_full = (p_full @ np.linalg.inv(pose).T)[:,:3]
            b = p_full[:,2] > -4.
            p_full = p_full[b]
            p_map_lab = p_map_lab[b]
            one_hot_full = np.zeros((p_map_lab.shape[0], 20))
            one_hot_full[np.arange(p_map_lab.shape[0]), p_map_lab.flatten()] = 1
            p_full = np.concatenate((p_full,one_hot_full),axis=-1)
            # print(1)
        else:
            p_full = p_part

        if self.split == 'train':
            p_concat = np.concatenate((p_full[:,:3], p_part[:,:3]), axis=0)
            p_concat = self.transforms(p_concat)

            p_full = np.concatenate((p_concat[:-len(p_part)],p_full[:,3:]),axis=-1)
            p_part = np.concatenate((p_concat[-len(p_part):],p_part[:,3:]),axis=-1)

        # patial pcd has 1/10 of the complete pcd size
        n_part = int(self.num_points / 10.)

        return point_set_to_sparse(
            p_full,
            p_part,
            self.num_points,
            n_part,
            self.resolution,
            self.points_datapath[index],
            p_mean=self.data_stats['mean'],
            p_std=self.data_stats['std'],
        )

    def __len__(self):
        return self.nr_data

##################################################################################################
