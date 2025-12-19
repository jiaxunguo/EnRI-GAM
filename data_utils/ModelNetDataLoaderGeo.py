import os, sys
import numpy as np
import warnings
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data


warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from data_utils import utils


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: Nx3 '''
    dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

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

    
class ModelNetDataLoaderGeo(Dataset):
    def __init__(self, root, npoints=1024, split='train', normalize=True, normal_channel=False,
                 modelnet10=False, cache_size=15000, shuffle=True, transform='org', drop_out=False):
        self.root = root
        self.split = split
        # self.batch_size = batch_size
        self.npoints = npoints
        self.normalize = normalize
        self.transform = transform
        self.shuffle = shuffle
        self.drop_out = drop_out
        if modelnet10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        if modelnet10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

        self.class_num = 40

    def _augment_data(self, rotated_data, rotate=False, shiver=False, translate=False, jitter=False,
                      shuffle=True):

        if shuffle:
            return shuffle_points(rotated_data)
        else:
            return rotated_data


    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
            point_set = point_set[0:self.npoints, :]

        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            # Take the first npoints
            point_set = point_set[0:self.npoints, :]
            if self.normalize:
                point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
            # if not self.normal_channel:
            #     point_set = point_set[:, 0:3]
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        if self.drop_out:
            point_set = random_point_dropout(point_set)

        points = self._augment_data(point_set, shuffle=self.shuffle)              # only shuffle by default
        
            
        if self.normal_channel:
            data = Data(pos=torch.from_numpy(points[:, :3]).float(), y=torch.from_numpy(cls).long(),
                    norm=torch.from_numpy(points[:, 3:]).float())
            
            o3d_normal = False
            o3d_normal_orien = False
            axes = None
            init_attr = 'default'
            
        else:
            data = Data(pos=torch.from_numpy(points[:, :3]).float(), y=torch.from_numpy(cls).long())
            data.id = index
 
            ugb = utils.GetBarycenter(48)
            data.norm = F.normalize(ugb(data))
            
            
            o3d_normal = False
            o3d_normal_orien = False
            axes = ["normal", "bary"]
            init_attr = 'default'

        if self.transform != 'org':
            
            if self.split=='train' and self.transform == 'so3':
                transform_list = T.Compose([
                                             utils.RandRotSO3([[360, 0], [360, 1], [360, 2]]),
                                                utils.RandomShiver(),
                                                 utils.RandomTranslate(),
                                              utils.LRF(o3d_normal,
                                                        o3d_normal_orien,
                                                        axes=axes),
                                              utils.InitialAttributes(init_attr)])
            elif self.split=='train' and self.transform == 'z':
                transform_list = T.Compose([
                                                utils.RandRotSO3([[360, 2]]),
                                                 utils.RandomShiver(),
                                                  utils.RandomTranslate(),           # comment this when using --glob bary for a
                                                                                # better performance
                                              utils.LRF(o3d_normal,
                                                        o3d_normal_orien,
                                                        axes=axes),
                                              utils.InitialAttributes(init_attr)])
            if self.split=='test' and self.transform == 'so3':
                transform_list = T.Compose([
                                            utils.RandRotSO3([[360, 0], [360, 1], [360, 2]]),
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
        return len(self.datapath)