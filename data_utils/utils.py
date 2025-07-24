import torch
import open3d as o3d
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.nn import knn
import torch.nn.functional as F
from termcolor import colored

import math
import random
from typing import Tuple, Union

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform, LinearTransformation


class RandomShuffle(object):
    def __init__(self, keys=None):
        self.keys = keys

    def __call__(self, data):
        indices = list(range(data.pos.size(0)))
        np.random.shuffle(indices)

        if self.keys is not None:
            for key in self.keys:
                data[key] = data[key][indices]

        else:
            # print(data.__dict__)
            for key in data.__dict__:
                # print(key)
                if data[key] is not None:
                    if data[key].size(0)==data.pos.size(0):
                        data[key] = data[key][indices]
        return data


class RandomTranslate(object):
    def __init__(self):
        pass

    def __call__(self, data):
        # xyz1 = torch.zeros(3).float().uniform_(2. / 3., 3. / 2.)
        xyz2 = torch.zeros(3).float().uniform_(-0.2, 0.2)

        data.pos = data.pos + xyz2.unsqueeze(0)

        return data

class RandomShiver(object):
    def __init__(self):
        pass

    def __call__(self, data):
        xyz1 = torch.zeros(3).float().uniform_(2. / 3., 3. / 2.)

        data.pos = data.pos*xyz1.unsqueeze(0)
        data.norm = data.norm / xyz1.unsqueeze(0)

        return data

class Jitter(object):
    def __init__(self, sigma=0.01, clip=0.02):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data):
        if self.sigma>0.001:
            N, C = data.pos.size()
            data.pos += np.clip(self.sigma * torch.randn(N, C), -1 * self.clip, self.clip)
            return data
        else:
            return data


class InitialAttributes(object):
    def __init__(self, init_attr):
        self.init_attr = init_attr
        pass

    def __call__(self, data):
        if self.init_attr=='default':
            center = data.pos.mean(keepdim=True, dim=0) # 1,3
            ray = data.pos - center  # 1024, 3
            dist = torch.norm(ray, dim=-1) # 1024
            # ray = F.normalize(ray)                         # key here!!!!!!!!!!!!!!!!!!!!!!!

            # angel between axis 0 and center vector
            normalized_ray = ray / dist.unsqueeze(-1) # 1024, 3

            cos_l0_ray = torch.einsum("ij,ij->i", data.l0, normalized_ray)  # 1024
            sin_l0_ray = torch.norm(data.l0.cross(normalized_ray, dim=1), dim=-1) # 1024

            # cos_l0_ray = torch.einsum("ij,ij->i", data.l1, normalized_ray)
            # sin_l0_ray = torch.norm(data.l1.cross(normalized_ray, dim=1), dim=-1)

            # angel between normal and barycenter
            data.x = torch.stack((dist, cos_l0_ray, sin_l0_ray), dim=1) # 1024, 3
        elif self.init_attr=='dist':
            center = data.pos.mean(keepdim=True, dim=0)
            ray = data.pos - center
            dist = torch.norm(ray, dim=-1)
            data.x = torch.stack((dist, dist, dist), dim=1)
        else:
            data.x = torch.ones_like(data.pos)

        assert data.x.size(1)==3

        return data

class GetGlobcenter(object):
    def __init__(self,):
        pass

    def __call__(self, data):
        return data.pos - torch.mean(data.pos, dim=0)
    
class GetBarycenter(object):
    def __init__(self, k):
        self.k = k
        pass

    def __call__(self, data):
        row, col = knn(data.pos, data.pos, self.k)
        return data.pos[col].view(-1, self.k, 3).mean(dim=1) - data.pos

class GetO3dNormal(object):
    def __init__(self, k, orien=False):
        self.k = k
        self.orien = orien
        # print('************************Using Open3D Normal************************')

    def __call__(self, data):
        # row, col = knn(data.pos, data.pos, self.k)
        # return data.pos[col].view(-1, self.k, 3).mean(dim=1) - data.pos

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data.pos.numpy())
        # pcd.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(radius=0.15))
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=self.k))

        # norm = torch.from_numpy(np.asarray(pcd.normals)).float()
        # mean_norm = []
        #
        # kd = o3d.geometry.KDTreeFlann(pcd)
        # for i in range(len(pcd.points)):
        #     k, idx, _ = kd.search_radius_vector_3d(pcd.points[i], radius=0.1)
        #     mean_norm.append(norm[idx].mean(dim=0))
        #
        # norm = torch.stack(mean_norm, dim=0)
        # assert list(norm.size())==[1024, 3]
        
        # o3d.visualization.draw_geometries([pcd])
        
        # pcd.orient_normals_consistent_tangent_plane(k * 2)
        
        if not self.orien:      
            pcd.orient_normals_towards_camera_location()
            # pcd.orient_normals_towards_camera_location(data.pos.mean(0) + np.array([0, 0, 1e6]))
            # pcd.normals = o3d.utility.Vector3dVector(-1. * np.asarray(pcd.normals))
            
            pcd.orient_normals_consistent_tangent_plane(self.k*2)
            
        center = np.asarray(pcd.get_center())
        pts    = np.asarray(pcd.points)
        nrm    = np.asarray(pcd.normals)
        inward = (np.einsum('ij,ij->i', nrm, pts-center) < 0)
        nrm[inward] *= -1.0
        
        # center = np.asarray(pcd.get_center())
        # vec2c = np.asarray(pcd.points) -center
        # dot = (vec2c * np.asarray(pcd.normals)).sum(1)
        # pcd.normals[dot < 0] *= -1
        
        pcd.normals = o3d.utility.Vector3dVector(nrm)

        
        # norm = torch.from_numpy(np.asarray(pcd.normals)).float()

        norm = torch.from_numpy(np.asarray(pcd.normals)).float()
        # print(f'norm:{torch.norm(data.norm, dim=1)}')
        norm = norm / torch.norm(norm, keepdim=True, dim=1)

        # if self.orien:
        #     mask = (norm*data.norm).sum(dim=1, keepdim=False) < 0
        #     norm[mask] = -norm[mask]

        pcd.normals = o3d.utility.Vector3dVector(norm.numpy())
        # print("hello")
        # o3d.visualization.draw_geometries([pcd])
        # print("hello end")
        data.norm = norm

        return norm

class LRF(object):
    def __init__(self, o3d_normal=False, orien=False, axes=None):
        if axes==None:
            axes = ["normal", "bary"]
        else:
            # print(axes)
            assert isinstance(axes, list) and len(axes)==2

        get_lrf=[]
        for i, name in enumerate(axes):
            if name=="normal":
                if not o3d_normal:
                    get_lrf.append(lambda data: data.norm)
                else:
                    get_lrf.append(GetO3dNormal(48, orien))

            elif name=="bary":
                get_lrf.append(GetBarycenter(48))
            elif name=="glob":
                get_lrf.append(lambda data: data.pos - torch.mean(data.pos, dim=0))
        self.get_lrf = get_lrf


    def __call__(self, data):
        data.l0 = F.normalize(self.get_lrf[0](data))
        data.l1 = F.normalize(self.get_lrf[1](data))

        return data

class RandomRotate(BaseTransform):
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval (functional name: :obj:`random_rotate`).

    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If :obj:`degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axis (int, optional): The rotation axis. (default: :obj:`0`)
    """
    def __init__(
        self,
        degrees: Union[Tuple[float, float], float],
        axis: int = 0,
    ) -> None:
        if isinstance(degrees, (int, float)):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axis = axis
        self.last_matrix = None

    def forward(self, data: Data) -> Data:
        assert data.pos is not None

        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if data.pos.size(-1) == 2:
            matrix = [[cos, sin], [-sin, cos]]
        else:
            if self.axis == 0:
                matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
            elif self.axis == 1:
                matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
            else:
                matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        
        self.last_matrix = torch.tensor(matrix, dtype=torch.float32)
        
        return LinearTransformation(torch.tensor(matrix))(data)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.degrees}, '
                f'axis={self.axis})')

def compute_total_rotation_matrix(rotators):
    R_total = torch.eye(3)
    for r in rotators:
        R_total = r @ R_total
    return R_total

class RandRotSO3(object):
    def __init__(self, rotates):
        rotates_list = []
        self.rotators = []
        for rot in rotates:
            rr = RandomRotate(*rot)
            rotates_list.append(rr)
            self.rotators.append(rr)
        self.rand_rot_so3 = T.Compose(rotates_list)

    def __call__(self, data):     
        data.pos_org = data.pos
        data.norm_org = data.norm
        
        if hasattr(data, 'norm'):
            key = 'norm'
            data.pos = torch.stack((data.pos, data[key]), dim=0)
        elif hasattr(data, 'normal'):
            key = 'normal'
            data.pos = torch.stack((data.pos, data[key]), dim=0)
        else:
            print('norm key error')

        data.pos = self.rand_rot_so3(data).pos

        # try:
        data[key] = data.pos[1]
        data.pos = data.pos[0]
        # except:
        #     pass
    
        rotation_matrices = torch.stack([r.last_matrix for r in self.rotators], dim=0)
        data.rotmat = compute_total_rotation_matrix(rotation_matrices)
        
        data.pos_rot = data.pos   
        return data

def grey_print(x):
    print(colored(x, "grey"))


def red_print(x):
    print(colored(x, "red"))


def green_print(x):
    print(colored(x, "green"))


def yellow_print(x):
    print(colored(x, "yellow"))


def blue_print(x):
    print(colored(x, "blue"))


def magenta_print(x):
    print(colored(x, "magenta"))


def cyan_print(x):
    print(colored(x, "cyan"))


def white_print(x):
    print(colored(x, "white"))


def print_arg(opt):
    cyan_print("PARAMETER: ")
    for a in opt.__dict__:
        print(
            "         "
            + colored(a, "yellow")
            + " : "
            + colored(str(opt.__dict__[a]), "cyan")
        )