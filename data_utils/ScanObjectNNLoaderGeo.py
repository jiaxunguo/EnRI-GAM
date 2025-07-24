import torch
import numpy as np
import pickle
import os, sys, h5py
from tqdm import tqdm
from torch.utils.data import Dataset
import torch_geometric.transforms as T
from torch_geometric.data import Data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from data_utils import utils

from pointops.functions import pointops

def shuffle_points(data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(data.shape[-2])
    np.random.shuffle(idx)
    return data[idx,:]

def compute_LRA(xyz, weighting=True, nsample = 32):
    dists = torch.cdist(xyz, xyz)

    dists, idx = torch.topk(dists, nsample, dim=-1, largest=False, sorted=False)
    dists = dists.unsqueeze(-1)

    group_xyz = index_points(xyz, idx)
    group_xyz = group_xyz - xyz.unsqueeze(2)

    if weighting:
        dists_max, _ = dists.max(dim=2, keepdim=True)
        dists = dists_max - dists
        dists_sum = dists.sum(dim=2, keepdim=True)
        weights = dists / dists_sum
        weights[weights != weights] = 1.0
        M = torch.matmul(group_xyz.transpose(3,2), weights*group_xyz)
    else:
        M = torch.matmul(group_xyz.transpose(3,2), group_xyz)

    # eigen_values, vec = M.symeig(eigenvectors=True)
    eigen_values, vec = torch.linalg.eigh(M, UPLO='U')

    LRA = vec[:,:,:,0]
    LRA_length = torch.norm(LRA, dim=-1, keepdim=True)
    LRA = LRA / LRA_length
    return LRA # B N 3

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def pcs_normalize(pcs):
    pcs_normalized = np.stack([pc_normalize(pcs[k]) for k in range(len(pcs))], axis=0)
    return pcs_normalized

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    new_points = points[batch_indices, idx, :]      
    return new_points


   
class ScanObjectNNGeo(Dataset):
    def __init__(self, root, args, split='train', process_data=True, normalize=True, shuffle=True, transform='org'):
        super().__init__()
        assert (split == 'train' or split == 'test')
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.data_type = args.data_type
        
        self.normalize = normalize
        self.transform = transform
        self.shuffle = shuffle
        self.split = split
        
        if split == 'train':
            self.train = True
            if self.data_type == 'hardest':
                h5 = h5py.File(self.root + 'training_objectdataset_augmentedrot_scale75.h5', 'r')
            else:
                h5 = h5py.File(self.root + 'training_objectdataset.h5', 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            
            h5.close()
        elif split == 'test':
            self.train = False
            if self.data_type == 'hardest':
                h5 = h5py.File(self.root + 'test_objectdataset_augmentedrot_scale75.h5', 'r')
            else:
                h5 = h5py.File(self.root + 'test_objectdataset.h5', 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            
            h5.close()
        else:
            raise NotImplementedError()
        
        
        
        if self.uniform:
            self.points_ = torch.tensor(self.points).cuda()
            fps_idx = pointops.furthestsampling(self.points_, self.npoints)
            self.points_ = index_points(self.points_, fps_idx.long())
            self.points = self.points_.cpu().numpy()
            del self.points_
            del fps_idx
            torch.cuda.empty_cache()
            
            self.save_path = os.path.join(root, 'ScanObj_%s_%s_%dpts_fps_pca.dat' % (self.data_type, split, self.npoints))
        else:
            self.points = self.points[:, :self.npoints, :]
            
            self.save_path = os.path.join(root, 'ScanObj_%s_%s_%dpts_pca.dat' % (self.data_type, split, self.npoints))
            
        
        if self.process_data and os.path.exists(self.save_path):
            print('Load processed data from %s...' % self.save_path)
            with open(self.save_path, 'rb') as f:
                self.list_of_points, self.list_of_norm, self.list_of_labels = pickle.load(f)          
        else:
            if self.process_data and not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
            elif not self.process_data:
                print('Reprocessing data %s ...' % self.save_path)
                print('Make sure that there is no files in this path(%s)' % self.save_path)
                if os.path.exists(self.save_path):
                    raise ValueError("There are files in the offline save path")
                    
            self.list_of_points = [None] * self.points.shape[0]
            self.list_of_norm = [None] * self.points.shape[0]
            self.list_of_labels = [None] * self.points.shape[0]
            
            for index in tqdm(range(self.points.shape[0]), total=self.points.shape[0]):
                cls = self.labels[index]
                cls = np.array([cls]).astype(np.int32)              
                point_set = self.points[index] # shape = (n, 3)

                points_cuda = torch.from_numpy(point_set).float().cuda().unsqueeze(0)
                norm = compute_LRA(points_cuda, nsample=16).squeeze().cpu()
                    
                self.list_of_points[index] = point_set
                self.list_of_norm[index] = norm
                self.list_of_labels[index] = cls
            with open(self.save_path, 'wb') as f:
                pickle.dump([self.list_of_points, self.list_of_norm, self.list_of_labels], f)  
    
    def _augment_data(self, rotated_data, rotate=False, shiver=False, translate=False, jitter=False,
                      shuffle=True):

        if shuffle:
            return shuffle_points(rotated_data)
        else:
            return rotated_data
        
    def __getitem__(self, index):
       
        point_set, norm, cls = self.list_of_points[index], self.list_of_norm[index], self.list_of_labels[index]  

        point_set = np.hstack((point_set, norm))
        
        points = self._augment_data(point_set, shuffle=self.shuffle)              # only shuffle by default  
        
        
        data = Data(pos=torch.from_numpy(points[:, :3]).float(), y=torch.from_numpy(cls).long(),
                norm=torch.from_numpy(points[:, 3:]).float())
        
        if self.transform != 'org':
            o3d_normal = False
            o3d_normal_orien = False
            axes = None
            init_attr = 'default'
            
            if self.split=='train' and self.transform == 'so3':
                transform_list = T.Compose([utils.RandRotSO3([[360, 0], [360, 1], [360, 2]]),
                                                utils.RandomShiver(),
                                                utils.RandomTranslate(),
                                             utils.LRF(o3d_normal,
                                                       o3d_normal_orien,
                                                       axes=axes),
                                             utils.InitialAttributes(init_attr)])
            elif self.split=='train' and self.transform == 'z':
                transform_list = T.Compose([utils.RandRotSO3([[360, 2]]),
                                                utils.RandomShiver(),
                                                utils.RandomTranslate(),           # comment this when using --glob bary for a
                                                                                # better performance
                                             utils.LRF(o3d_normal,
                                                       o3d_normal_orien,
                                                       axes=axes),
                                             utils.InitialAttributes(init_attr)])
            if self.split=='test' and self.transform == 'so3':
                transform_list = T.Compose([utils.RandRotSO3([[360, 0], [360, 1], [360, 2]]),
                                            utils.LRF(o3d_normal,
                                                      o3d_normal_orien,
                                                      axes=axes),
                                            utils.InitialAttributes(init_attr)])
            elif self.split=='test' and self.transform == 'z':
                transform_list = T.Compose([
                                            utils.RandRotSO3([[360, 2]]),
                                            utils.LRF(o3d_normal,
                                                      o3d_normal_orien,
                                                      axes=axes),
                                            utils.InitialAttributes(init_attr)])
            data = transform_list(data)
        

        return data

    def __len__(self):
        return self.points.shape[0]