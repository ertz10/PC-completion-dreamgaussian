import os
import math
import numpy as np
from typing import NamedTuple
from plyfile import PlyData, PlyElement

import torch
from torch import nn

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from simple_knn._C import distCUDA2

from sh_utils import eval_sh, SH2RGB, RGB2SH
from mesh import Mesh
from mesh_utils import decimate_mesh, clean_mesh

import kiui

import open3d as o3d

from scipy.spatial.transform import Rotation as R
from gaussianSplatsTransform import GaussianTransform

import torchvision
import torchvision.transforms as T 
from PIL import Image

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    
    def helper(step):
        if lr_init == lr_final:
            # constant lr, ignore other params
            return lr_init
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def gaussian_3d_coeff(xyzs, covs):
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = covs[:, 0], covs[:, 1], covs[:, 2], covs[:, 3], covs[:, 4], covs[:, 5]

    # eps must be small enough !!!
    inv_det = 1 / (a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24)
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det

    power = -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f) - x * y * inv_b - x * z * inv_c - y * z * inv_e

    power[power > 0] = -1e10 # abnormal values... make weights 0
        
    return torch.exp(power)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, opt_object, opt_alignment=None):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        # CUSTOM
        self.variable_points_length = 0
        self.static_points_length = 0
        self.original_points = None
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.static_points_mask = None
        self.original_xyz = None
        self.original_featuresdc = None
        self.original_features_rest = None
        self.original_opacity = None
        self.original_scaling = None
        self.original_rotation = None
        self.customGSTransform = GaussianTransform()
        self.opt_object = opt_object
        self.opt_alignment = opt_alignment
        self.only_dynamic_splats = opt_object.only_dynamic_splats

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        #CUSTOM
        #static_points_mask = self.static_points_mask
        #scaling = torch.vstack((self._scaling, self.original_scaling[static_points_mask == 0]))
        #return self.scaling_activation(scaling)
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        #CUSTOM
        #static_points_mask = self.static_points_mask
        #rotation = torch.vstack((self._rotation, self.original_rotation[static_points_mask == 0]))
        #return self.rotation_activation(rotation)
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        # CUSTOM
        #features_dc = self._features_dc
        #features_rest = self._features_rest
        #static_points_mask = self.static_points_mask
        #features_dc = torch.vstack((self._features_dc, self.original_featuresdc[static_points_mask == 0]))
        if (self.only_dynamic_splats):
            features_dc = self._features_dc
            features_rest = self._features_rest
        else:
            features_dc = torch.vstack((self._features_dc, self.original_featuresdc))
            features_rest = torch.vstack((self._features_rest, self.original_features_rest))
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @torch.no_grad()
    def extract_fields(self, resolution=128, num_blocks=16, relax_ratio=1.5):
        # resolution: resolution of field
        
        block_size = 2 / num_blocks

        assert resolution % block_size == 0
        split_size = resolution // num_blocks
        
        # TODO include original gaussian splats as well
        opacities = self.get_opacity
        #opacities = self.original_opacity
        if (self.only_dynamic_splats == False):
            opacities = torch.vstack((self.get_opacity, self.opacity_activation(self.original_opacity)))
        else:
            opacities = self.get_opacity

        # pre-filter low opacity gaussians to save computation
        mask = (opacities > 0.005).squeeze(1)
        # CUSTOM
        #mask = torch.logical_and(mask, ~torch.isinf(mask))

        #opacities = opacities[mask]
        #xyzs = self.get_xyz[mask]
        #stds = self.get_scaling[mask]
        #opacities = opacities[mask]
        #xyzs = self.original_xyz[mask]
        #stds = self.original_scaling[mask]
        opacities = opacities[mask]
        if (self.only_dynamic_splats == False):
            xyzs = torch.vstack((self.get_xyz, self.original_xyz))[mask]
            stds = torch.vstack((self.get_scaling, self.scaling_activation(self.original_scaling)))[mask]
        else:
            xyzs = self.get_xyz[mask]
            stds = self.get_scaling[mask]
        
        # normalize to ~ [-1, 1]
        mn, mx = xyzs.amin(0), xyzs.amax(0)
        self.center = (mn + mx) / 2
        self.scale = 1.8 / (mx - mn).amax().item()

        xyzs = (xyzs - self.center) * self.scale
        stds = stds * self.scale

        #covs = self.covariance_activation(stds, 1, self._rotation[mask])
        #covs = self.covariance_activation(stds, 1, self.original_rotation[mask])
        if (self.only_dynamic_splats == False):
            covs = self.covariance_activation(stds, 1, torch.vstack((self._rotation, self.original_rotation))[mask])
        else:
            covs = self.covariance_activation(stds, 1, self._rotation[mask])
        #covs = self.covariance_activation(stds, 1, self.original_rotation[mask])

        # tile
        device = opacities.device
        occ = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

        X = torch.linspace(-1, 1, resolution).split(split_size)
        Y = torch.linspace(-1, 1, resolution).split(split_size)
        Z = torch.linspace(-1, 1, resolution).split(split_size)


        # loop blocks (assume max size of gaussian is small than relax_ratio * block_size !!!)
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    # sample points [M, 3]
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)
                    # in-tile gaussians mask
                    vmin, vmax = pts.amin(0), pts.amax(0)
                    vmin -= block_size * relax_ratio
                    vmax += block_size * relax_ratio
                    mask = (xyzs < vmax).all(-1) & (xyzs > vmin).all(-1)
                    # if hit no gaussian, continue to next block
                    if not mask.any():
                        continue
                    mask_xyzs = xyzs[mask] # [L, 3]
                    mask_covs = covs[mask] # [L, 6]
                    mask_opas = opacities[mask].view(1, -1) # [L, 1] --> [1, L]

                    # query per point-gaussian pair.
                    g_pts = pts.unsqueeze(1).repeat(1, mask_covs.shape[0], 1) - mask_xyzs.unsqueeze(0) # [M, L, 3]
                    g_covs = mask_covs.unsqueeze(0).repeat(pts.shape[0], 1, 1) # [M, L, 6]

                    # batch on gaussian to avoid OOM
                    batch_g = 1024
                    val = 0
                    for start in range(0, g_covs.shape[1], batch_g):
                        end = min(start + batch_g, g_covs.shape[1])
                        w = gaussian_3d_coeff(g_pts[:, start:end].reshape(-1, 3), g_covs[:, start:end].reshape(-1, 6)).reshape(pts.shape[0], -1) # [M, l]
                        val += (mask_opas[:, start:end] * w).sum(-1)
                    
                    # kiui.lo(val, mask_opas, w)
                
                    occ[xi * split_size: xi * split_size + len(xs), 
                        yi * split_size: yi * split_size + len(ys), 
                        zi * split_size: zi * split_size + len(zs)] = val.reshape(len(xs), len(ys), len(zs)) 
        
        kiui.lo(occ, verbose=1)

        return occ
    
    def extract_mesh(self, path, density_thresh=1, resolution=128, decimate_target=1e5):

        os.makedirs(os.path.dirname(path), exist_ok=True)

        occ = self.extract_fields(resolution).detach().cpu().numpy()

        import mcubes
        vertices, triangles = mcubes.marching_cubes(occ, density_thresh)
        vertices = vertices / (resolution - 1.0) * 2 - 1

        # transform back to the original space
        vertices = vertices / self.scale + self.center.detach().cpu().numpy()

        vertices, triangles = clean_mesh(vertices, triangles, remesh=True, remesh_size=0.015)
        if decimate_target > 0 and triangles.shape[0] > decimate_target:
            vertices, triangles = decimate_mesh(vertices, triangles, decimate_target)

        v = torch.from_numpy(vertices.astype(np.float32)).contiguous().cuda()
        f = torch.from_numpy(triangles.astype(np.int32)).contiguous().cuda()

        print(
            f"[INFO] marching cubes result: {v.shape} ({v.min().item()}-{v.max().item()}), {f.shape}"
        )

        mesh = Mesh(v=v, f=f, device='cuda')

        return mesh
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float = 1):

        #### 
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # CUSTOM static_points_mask
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        # CUSTOM
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        # additional learning rate parameters
        self.rotation_scheduler_args = get_expon_lr_func(lr_init=training_args.rotation_lr,
                                                    lr_final=training_args.rotation_lr_final,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        self.scaling_scheduler_args = get_expon_lr_func(lr_init=training_args.scaling_lr,
                                                    lr_final=training_args.scaling_lr_final,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        self.feature_scheduler_args = get_expon_lr_func(lr_init=training_args.feature_lr,
                                                    lr_final=training_args.feature_lr_final,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
    
    def update_feature_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "f_dc":
                lr = self.feature_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def update_rotation_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "rotation":
                lr = self.rotation_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def update_scaling_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "scaling":
                lr = self.scaling_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        #xyz = self._xyz.detach().cpu().numpy()
        #normals = np.zeros_like(xyz)
        #f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        #f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        #opacities = self._opacity.detach().cpu().numpy()
        #scale = self._scaling.detach().cpu().numpy()
        #rotation = self._rotation.detach().cpu().numpy()

        if (self.only_dynamic_splats == False):
            #CUSTOM
            xyz = np.vstack((self._xyz.detach().cpu().numpy(), self.original_xyz.detach().cpu().numpy()))
            normals = np.zeros_like(xyz)
            f_dc = np.vstack((self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(), self.original_featuresdc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()))
            f_rest = np.vstack((self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(), self.original_features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()))
            opacities = np.vstack((self._opacity.detach().cpu().numpy(), self.original_opacity.detach().cpu().numpy()))
            scale = np.vstack((self._scaling.detach().cpu().numpy(), self.original_scaling.detach().cpu().numpy()))
            rotation = np.vstack((self._rotation.detach().cpu().numpy(), self.original_rotation.detach().cpu().numpy()))
        else:
            xyz = self._xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity.detach().cpu().numpy()
            scale = self._scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01)) #modified
        #opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.1)) #original
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # CUSTOM
    def AABB_prune_points(self, mask):
        # negate mask for correct usage for _prune_optimizer
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    # CUSTOM
    def removeElements(self, index_mask: np.ndarray):

        def torch_delete(tensor, indices):
            #mask = torch.ones(tensor.shape[0], dtype=torch.bool)
            #mask[indices] = False
            return tensor[indices.cpu().numpy().astype(bool)]
        
        #new_xyz = torch_delete(self._xyz, index_mask)
        #new_features_dc = torch_delete(self._features_dc, index_mask)
        #new_features_rest = torch_delete(self._features_rest, index_mask)
        #new_opacities = torch_delete(self._opacity, index_mask)
        #new_scaling = torch_delete(self._scaling, index_mask)
        #new_rotation = torch_delete(self._rotation, index_mask)

        #self.xyz_gradient_accum = torch_delete(self.xyz_gradient_accum, index_mask)
        #self.denom = torch_delete(self.denom, index_mask)
        #self.max_radii2D = torch_delete(self.max_radii2D, index_mask)
        
        #self.AABB_cropping_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        self.AABB_prune_points(index_mask)
        #ext_length = len(self.static_points_mask) - len(index_mask)
        #ext_mask = torch.hstack((index_mask, torch.ones(ext_length, device=index_mask.device)))
        # 1 == Dynamic points, 0 == static points
        #self.static_points_mask = torch_delete(self.static_points_mask, ext_mask)

        # also change original stuff
        #self.original_xyz = torch_delete(self.original_xyz, ext_mask)
        #self.original_featuresdc = torch_delete(self.original_featuresdc, ext_mask)
        #self.original_features_rest = torch_delete(self.original_features_rest, ext_mask)
        #self.original_opacity = torch_delete(self.original_opacity, ext_mask)
        #self.original_scaling = torch_delete(self.original_scaling, ext_mask)
        #self.original_rotation = torch_delete(self.original_rotation, ext_mask)
    
    # CUSTOM
    def TestAgainstBB(self, AABB: np.ndarray, removePoints: bool):

        #pcd_debug = o3d.geometry.PointCloud()
        #pcd_debug.points = o3d.utility.Vector3dVector(self._xyz.cpu().detach().numpy())
        #o3d.io.write_point_cloud("debug/pcd_xyz_AABB_debug.ply", pcd_debug, write_ascii=True, compressed=False)
        bools = (self._xyz[:, 0] < AABB[0]) | (self._xyz[:, 0] > AABB[1])
        bools2 = (self._xyz[:, 1] < AABB[2]) | (self._xyz[:, 1] > AABB[3])
        bools3 = (self._xyz[:, 2] < AABB[4]) | (self._xyz[:, 2] > AABB[5])

        bools_final = bools | bools2 | bools3

        if(removePoints):
            print("Points to be removed: ", bools_final.cpu().numpy().astype(int).sum())
            self.removeElements(bools_final)
        else:
            return bools_final
        
    def load_static_ply(self, path, spatial_lr_scale):
        import trimesh
        #@ERLER
        def transform_points_around_pivot(pts: np.ndarray, trans_matrix: np.ndarray, pivot: np.ndarray):
            """
            rotate_points_around_pivot
            :param pts: np.ndarray[n, dims=3]
            :param rotation_mat: np.ndarray[4, 4]
            :param pivot: np.ndarray[dims=3]
            :return:
            """
            
            #pivot_bc = np.broadcast_to(pivot[np.newaxis, :], pts.shape)

            #pts -= pivot_bc
            pts = trimesh.transformations.transform_points(pts, trans_matrix)
            #pts += pivot_bc

            return pts
        
        #CUSTOM
        

        #CUSTOM
        def calculateCenterOfMass(xyz: np.ndarray):
            return xyz.mean(axis=0)

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        print("Number of points at loading : ", xyz.shape[0])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        self.max_sh_degree = 3
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        #self.max_sh_degree = 0
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        #CUSTOM apply rigid transformation to xyz
        #rotation = R.from_euler('zyx', [-92.5, 0, -110.5], degrees=True)
        #x
        centroid = calculateCenterOfMass(xyz)
        # first center point cloud
        translated_pts = xyz - centroid

        xyz, norm_factor = self.customGSTransform.normalize_PC(translated_pts) # TODO also normalize scale!! maybe don't normalize ???

        print("Point cloud normalization factor: ", norm_factor)
        if (norm_factor >= 1.0):
            scales = scales - np.log(norm_factor)
        else:
            norm_factor += 1.0
            scales = scales + np.log(norm_factor)
        scales = scales + np.log(self.opt_object.scale)

        # remove big splats
        xyz, rots, scales, opacities, features_dc, features_extra = self.PreprocessCloud(xyz, rots, scales, opacities, features_dc, features_extra)

        mask_subsampled = np.zeros(len(xyz))

        fused_point_cloud = torch.tensor(np.asarray(xyz)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(xyz)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        #features[:, :3, 0 ] = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float, device="cuda")
        #features[:, 3:, 1:] = 0.0
        # Average of existing colors
        avg_col = np.mean(features_dc, axis=0).squeeze(1)#torch.mean(SH2RGB(fused_color), dim=0)
        #avg_col = RGB2SH(avg_col) * 255.0
        features[:, :3, 0] = torch.tensor(avg_col, dtype=torch.float, device="cuda")
        #features[:, :3, 0] = torch.tensor([0.3, 0.3, 0.3], dtype=torch.float, device="cuda")
        features[:, 3:, 1:] = 0.0


        pcd_debug = o3d.geometry.PointCloud()
        pcd_debug.points = o3d.utility.Vector3dVector(xyz)
        #pcd_debug.points = o3d.utility.Vector3dVector(xyz_full)
        #o3d.io.write_point_cloud("debug/pcd_debug_transformed.ply", pcd_debug)

        '''
        self._xyz = nn.Parameter(torch.tensor(xyz[mask_subsampled == 1], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc[mask_subsampled == 1], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        # TODO try to scrap features_rest
        self._features_rest = nn.Parameter(torch.tensor(features_extra[mask_subsampled == 1], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities[mask_subsampled == 1], dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales[mask_subsampled == 1], dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots[mask_subsampled == 1], dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        self.spatial_lr_scale = 10
        self.static_points_mask = mask_subsampled.copy()
        #self.max_radii2D = torch.zeros((self.get_xyz.shape[0] + len(mask_subsampled[mask_subsampled == 0])), device="cuda")
        self.max_radii2D = torch.zeros((len(self.static_points_mask[self.static_points_mask == 1])), device="cuda")
        #self.max_radii2D = torch.zeros((len(self.static_points_mask)), device="cuda")
        '''

        self._xyz = torch.tensor(xyz[mask_subsampled == 0], dtype=torch.float, device="cuda").requires_grad_(False)
        self._features_dc = torch.tensor(features_dc[mask_subsampled == 0], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False)
        self._features_rest = torch.tensor(features_extra[mask_subsampled == 0], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False)
        self._opacity = torch.tensor(opacities[mask_subsampled == 0], dtype=torch.float, device="cuda").requires_grad_(False)
        self._scaling = torch.tensor(scales[mask_subsampled == 0], dtype=torch.float, device="cuda").requires_grad_(False)
        self._rotation = torch.tensor(rots[mask_subsampled == 0], dtype=torch.float, device="cuda").requires_grad_(False)

        self.active_sh_degree = self.max_sh_degree
        self.spatial_lr_scale = spatial_lr_scale
        self.static_points_mask = mask_subsampled.copy()
        #self.max_radii2D = torch.zeros((self.get_xyz.shape[0] + len(mask_subsampled[mask_subsampled == 0])), device="cuda")
        self.max_radii2D = torch.zeros((len(self.static_points_mask[self.static_points_mask == 1])), device="cuda")
        #self.max_radii2D = torch.zeros((len(self.static_points_mask)), device="cuda")

        self.original_xyz = torch.tensor(xyz[mask_subsampled == 0], dtype=torch.float, device="cuda").requires_grad_(False)
        self.original_featuresdc = torch.tensor(features_dc[mask_subsampled == 0], dtype=torch.float, device="cuda").transpose(1,2).contiguous().requires_grad_(False)
        self.original_features_rest = torch.tensor(features_extra[mask_subsampled == 0], dtype=torch.float, device="cuda").transpose(1,2).contiguous().requires_grad_(False)
        self.original_opacity = torch.tensor(opacities[mask_subsampled == 0], dtype=torch.float, device="cuda").requires_grad_(False)
        self.original_scaling = torch.tensor(scales[mask_subsampled == 0], dtype=torch.float, device="cuda").requires_grad_(False)
        self.original_rotation = torch.tensor(rots[mask_subsampled == 0], dtype=torch.float, device="cuda").requires_grad_(False)

    #@ERLER
    def transform_points_around_pivot(self, pts: np.ndarray, trans_matrix: np.ndarray, pivot: np.ndarray):
        """
        rotate_points_around_pivot
        :param pts: np.ndarray[n, dims=3]
        :param rotation_mat: np.ndarray[4, 4]
        :param pivot: np.ndarray[dims=3]
        :return:
        """
        import trimesh
        
        #pivot_bc = np.broadcast_to(pivot[np.newaxis, :], pts.shape)

        #pts -= pivot_bc
        pts = trimesh.transformations.transform_points(pts, trans_matrix)
        #pts += pivot_bc

        return pts
    
    #CUSTOM
    def calculateCenterOfMass(self, xyz: np.ndarray):
            return xyz.mean(axis=0)
        
    def load_ply(self, path, AABB, spatial_lr_scale, no_transform=False, no_rotation=False, blob_init_size=None, num_pts_init=None, flip_z=False, normalize=True, transform_splats_only=False):
        import trimesh
        
        #CUSTOM
    
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        print("Number of points at loading : ", xyz.shape[0])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        has_extras = True
        if (len(extra_f_names) == 0):
            extra_f_names = ["f_rest_" + str(i) for i in range(0,45)]
            has_extras = False
        self.max_sh_degree = 3
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        #self.max_sh_degree = 0

        #'''
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        if has_extras:
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        #'''

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        #CUSTOM apply rigid transformation to xyz
        #rotation = R.from_euler('zyx', [-92.5, 0, -110.5], degrees=True)
        #x
        centroid = self.calculateCenterOfMass(xyz)
        # first center point cloud
        translated_pts = xyz - centroid
        #rotation = trimesh.transformations.euler_matrix(0, 0, 180, 'sxyz')#trimesh.transformations.rotation_matrix(-110.5, [1, 0, 0], [0, 0, 0])

        
        if (no_transform == False):
            if normalize:
                xyz, norm_factor = self.customGSTransform.normalize_PC(translated_pts)

            opt_rots = self.opt_object.rotation if self.opt_alignment == None else self.opt_alignment.rotation # used for trellis
            
            if no_rotation == False:
                rotation1 = trimesh.transformations.rotation_matrix(opt_rots[0], [1, 0, 0], [0, 0, 0])
                rotation2 = trimesh.transformations.rotation_matrix(opt_rots[1], [0, 0, 1], [0, 0, 0])
                rotation3 = trimesh.transformations.rotation_matrix(opt_rots[2], [0, 1, 0], [0, 0, 0])
                xyz = self.transform_points_around_pivot(pts=xyz, trans_matrix=rotation1, pivot=[0, 0, 0])
                xyz = self.transform_points_around_pivot(pts=xyz, trans_matrix=rotation2, pivot=[0, 0, 0])
                xyz = self.transform_points_around_pivot(pts=xyz, trans_matrix=rotation3, pivot=[0, 0, 0])
            
            if flip_z:
                flip_rotation = trimesh.transformations.rotation_matrix(3.14159, [0, 0, 1], [0, 0, 0])
                xyz = self.transform_points_around_pivot(pts=xyz, trans_matrix=flip_rotation, pivot=[0, 0, 0])
           
            #### SCALING
            opt_scale = self.opt_object.scale if self.opt_alignment == None else self.opt_alignment.scale # used for trellis
            scale_mat = trimesh.transformations.scale_matrix(opt_scale, [0, 0, 0])
            xyz = self.transform_points_around_pivot(pts=xyz, trans_matrix=scale_mat, pivot=[0, 0, 0])
            #scales *= opt_scale
            # scale AABB as well
            if self.opt_object.AABB is not None:
                AABB = np.array(AABB) * opt_scale
            #### TRANSLATION
            xyz += np.array(self.opt_object.translation) if self.opt_alignment == None else self.opt_alignment.translation # used for trellis<
            
            rots = self.customGSTransform.rotate(rots, rotation1)
            rots = self.customGSTransform.rotate(rots, rotation2)
            rots = self.customGSTransform.rotate(rots, rotation3)
            if normalize:
                print("Point cloud normalization factor: ", norm_factor)
                if (norm_factor >= 1.0):
                    scales = scales - np.log(norm_factor)
                else:
                    norm_factor += 1.0
                    scales = scales + np.log(norm_factor)
            #scales = scales / np.log(norm_factor)
            # instead of adjusting the radius, adjust the object scale (and splat scale)
            scales = scales + np.log(opt_scale) # multiply logs = log() + logs()
        if blob_init_size is not None:
            gs_init_scales = torch.log(torch.tensor((blob_init_size, blob_init_size, blob_init_size)))
        else:
            gs_init_scales = torch.log(torch.tensor(self.opt_object.dyn_gs_init_scale)) # inverse of e function, convert to exp format

        opt_rots = self.opt_object.rotation if self.opt_alignment == None else self.opt_alignment.rotation # used for trellis
            
        if transform_splats_only:
            rotation1 = trimesh.transformations.rotation_matrix(opt_rots[0], [1, 0, 0], [0, 0, 0])
            rotation2 = trimesh.transformations.rotation_matrix(opt_rots[1], [0, 0, 1], [0, 0, 0])
            rotation3 = trimesh.transformations.rotation_matrix(opt_rots[2], [0, 1, 0], [0, 0, 0])
            rots = self.customGSTransform.rotate(rots, rotation1)
            rots = self.customGSTransform.rotate(rots, rotation2)
            rots = self.customGSTransform.rotate(rots, rotation3)
            opt_scale = self.opt_object.scale if self.opt_alignment == None else self.opt_alignment.scale # used for trellis
            scales = scales + np.log(opt_scale) # multiply logs = log() + logs()

        #############################################
        ########### Init gaussian blob ##############
        #############################################
        if num_pts_init is not None:
            num_pts = num_pts_init
        else:
            num_pts = self.opt_object.num_pts_init

        if self.opt_object.init_sphere_radius is not None:
            radius = self.opt_object.init_sphere_radius
        else:
            radius = 0.75       

        # remove big splats
        xyz, rots, scales, opacities, features_dc, features_extra = self.PreprocessCloud(xyz, rots, scales, opacities, features_dc, features_extra)
        #xyz, rots, scales, opacities, features_dc = self.PreprocessCloud(xyz, rots, scales, opacities, features_dc)

        pcd_debug = o3d.geometry.PointCloud()
        pcd_debug.points = o3d.utility.Vector3dVector(xyz)
        #pcd_debug.points = o3d.utility.Vector3dVector(xyz_full)
        #out_path = path.split('/')
        o3d.io.write_point_cloud(self.opt_object.data_path + "/" + self.opt_object.data_path.split('/')[-1] + "_transformed.ply", pcd_debug)

        
        if self.opt_object.AABB is None:
            # radius depending on bounding box of object
            delta_x = np.abs(np.max(xyz[0])) + np.abs(np.min(xyz[0]))
            delta_y = np.abs(np.max(xyz[1])) + np.abs(np.min(xyz[1]))
            delta_z = np.abs(np.max(xyz[2])) + np.abs(np.min(xyz[2]))
            radius = np.max(np.array([delta_x, delta_y, delta_z]))# / 2.0

            # init from random point cloud
            #'''
            phis = np.random.random((num_pts,)) * 2 * np.pi
            costheta = np.random.random((num_pts,)) * 2 - 1
            thetas = np.arccos(costheta)
            mu = np.random.random((num_pts,))
            radius = radius * np.cbrt(mu)
            x = radius * np.sin(thetas) * np.cos(phis)
            y = radius * np.sin(thetas) * np.sin(phis)
            z = radius * np.cos(thetas)
            '''
            # use bounding box of object
            x_min = np.min(xyz[0]) * 2.0
            x_max = np.max(xyz[0]) * 2.0
            y_min = np.min(xyz[1]) * 2.0
            y_max = np.max(xyz[1]) * 2.0
            z_min = np.min(xyz[2]) * 2.0
            z_max = np.max(xyz[2]) * 2.0
            x = np.random.uniform(low=x_min, high=x_max, size=num_pts)
            y = np.random.uniform(low=y_min, high=y_max, size=num_pts)
            z = np.random.uniform(low=z_min, high=z_max, size=num_pts)
            '''
        else:
            # TODO instead of sphere, do cube sampling
            #'''
            x = np.random.uniform(low=AABB[0], high=AABB[1], size=num_pts)
            y = np.random.uniform(low=AABB[2], high=AABB[3], size=num_pts)
            z = np.random.uniform(low=AABB[4], high=AABB[5], size=num_pts)
            #'''
        xyz_2 = np.stack((x, y, z), axis=1)

        #xyz[mask_subsampled.astype(int) == 1] = xyz_2
        xyz = np.vstack((xyz_2, xyz))
        #temp_opacities = inverse_sigmoid(0.1 * torch.ones((xyz_2.shape[0], 1), dtype=torch.float, device="cuda")).cpu().numpy()
        temp_opacities = inverse_sigmoid(torch.ones((xyz_2.shape[0], 1), dtype=torch.float, device="cuda") * 0.01).cpu().numpy()
        #temp_opacities = inverse_sigmoid(torch.ones((xyz_2.shape[0], 1), dtype=torch.float, device="cuda") * 0.1).cpu().numpy()
        #temp_opacities = inverse_sigmoid(torch.ones((xyz_2.shape[0], 1), dtype=torch.float, device="cuda") * 0.25).cpu().numpy()
        # TODO FILTER OUT INF VALUES, and NAN ?!!
        #opacities = inverse_sigmoid(torch.from_numpy(opacities)).cpu().numpy()
        #opacities = opacities[~np.isinf(opacities)]
        opacities = np.vstack((temp_opacities, opacities))
        temp_scales = np.repeat([gs_init_scales], len(xyz_2), axis = 0)
        scales = np.vstack((temp_scales, scales))
        temp_rots = np.repeat([[1, 0, 0, 0]], len(xyz_2), axis = 0)
        rots = np.vstack((temp_rots, rots))

        mask_subsampled = np.zeros(len(xyz))
        mask_subsampled[0:len(xyz_2)] = 1

        #############################################
        # End init half of object to gaussian blob ##
        #############################################

        fused_point_cloud = torch.tensor(np.asarray(xyz)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(xyz)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        #features[:, :3, 0 ] = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float, device="cuda")
        #features[:, 3:, 1:] = 0.0
        # Average of existing colors
        avg_col = np.mean(features_dc, axis=0).squeeze(1)#torch.mean(SH2RGB(fused_color), dim=0)
        #avg_col = RGB2SH(avg_col) * 255.0
        features[:, :3, 0] = torch.tensor(avg_col, dtype=torch.float, device="cuda")
        #features[:, :3, 0] = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float, device="cuda")
        features[:, 3:, 1:] = 0.0


        
        #features_dc[mask_subsampled == 1] = features[mask_subsampled == 1, :, 0:1].cpu().numpy()
        #features_extra[mask_subsampled == 1] = features[mask_subsampled == 1, :, 1:].cpu().numpy()
        features_dc = np.vstack((features[0:len(xyz_2), :, 0:1].cpu().numpy(), features_dc))
        #'''
        features_extra = features[:,:,1:].cpu().numpy() #np.vstack((features[0:len(xyz_2), :, 1:].cpu().numpy(), features_extra)) # TODO omit SH coefficients
        #'''
        # TODO TEST #
        #features_dc = features[:, :, 0:1]
        #features_extra = features[:, :, 1:]
        self._xyz = nn.Parameter(torch.tensor(xyz[mask_subsampled == 1], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc[mask_subsampled == 1], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        # TODO try to scrap features_rest
        self._features_rest = nn.Parameter(torch.tensor(features_extra[mask_subsampled == 1], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        #self._features_rest = nn.Parameter(torch.tensor(features_dc[mask_subsampled == 1], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities[mask_subsampled == 1], dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales[mask_subsampled == 1], dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots[mask_subsampled == 1], dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        self.spatial_lr_scale = spatial_lr_scale

        # filter inf and nans
        if (np.isinf(opacities).any()):
            print("Found inf value in opacities, filtering ...")
            mask = np.squeeze(np.isinf(opacities), 1)
            mask_subsampled = np.delete(mask_subsampled, mask) #np.concatenate((mask_subsampled[np.isinf(opacities) == False], mask_subsampled[mask_subsampled == 1]))
            xyz = np.delete(xyz, mask, 0)
            features_dc = np.delete(features_dc, mask, 0)
            features_extra = np.delete(features_extra, mask, 0)
            opacities = np.delete(opacities, mask, 0)
            scales = np.delete(scales, mask, 0)
            rots = np.delete(rots, mask, 0)

        self.static_points_mask = mask_subsampled.copy()
        self.max_radii2D = torch.zeros((len(self.static_points_mask[self.static_points_mask == 1])), device="cuda")

        self.original_xyz = torch.tensor(xyz[mask_subsampled == 0], dtype=torch.float, device="cuda").requires_grad_(False)
        self.original_featuresdc = torch.tensor(features_dc[mask_subsampled == 0], dtype=torch.float, device="cuda").transpose(1,2).contiguous().requires_grad_(False)
        self.original_features_rest = torch.tensor(features_extra[mask_subsampled == 0], dtype=torch.float, device="cuda").transpose(1,2).contiguous().requires_grad_(False)
        #self.original_features_rest = torch.tensor(features_dc[mask_subsampled == 0], dtype=torch.float, device="cuda").transpose(1,2).contiguous().requires_grad_(False)
        self.original_opacity = torch.tensor(opacities[mask_subsampled == 0], dtype=torch.float, device="cuda").requires_grad_(False)
        self.original_scaling = torch.tensor(scales[mask_subsampled == 0], dtype=torch.float, device="cuda").requires_grad_(False)
        self.original_rotation = torch.tensor(rots[mask_subsampled == 0], dtype=torch.float, device="cuda").requires_grad_(False)

    def load_static_and_dynamic_ply(self, path, original_file_path, spatial_lr_scale):        
    
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        print("Number of points at loading : ", xyz.shape[0])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        self.max_sh_degree = 3
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        #self.max_sh_degree = 0
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # Center
        centroid = self.calculateCenterOfMass(xyz)
        # first center point cloud
        #translated_pts = xyz - centroid
        translated_pts = xyz - centroid

        import trimesh

        xyz, norm_factor = self.customGSTransform.normalize_PC(translated_pts)
        #opt_rots = self.opt_object.rotation
        #rotation1 = trimesh.transformations.rotation_matrix(opt_rots[0], [1, 0, 0], [0, 0, 0])
        #rotation2 = trimesh.transformations.rotation_matrix(opt_rots[1], [0, 0, 1], [0, 0, 0])
        #rotation3 = trimesh.transformations.rotation_matrix(opt_rots[2], [0, 1, 0], [0, 0, 0])
        #xyz = self.transform_points_around_pivot(pts=xyz, trans_matrix=rotation1, pivot=[0, 0, 0])
        #xyz = self.transform_points_around_pivot(pts=xyz, trans_matrix=rotation2, pivot=[0, 0, 0])
        #xyz = self.transform_points_around_pivot(pts=xyz, trans_matrix=rotation3, pivot=[0, 0, 0])
        # TODO implement translation and scaling ?
        #### SCALING
        opt_scale = self.opt_object.scale
        scale_mat = trimesh.transformations.scale_matrix(opt_scale, [0, 0, 0])
        xyz = self.transform_points_around_pivot(pts=xyz, trans_matrix=scale_mat, pivot=[0, 0, 0])
        #scales *= opt_scale
        # scale AABB as well
        #AABB = np.array(AABB) * opt_scale
        #### TRANSLATION
        #xyz += np.array(self.opt_object.translation)
        
        #rots = self.customGSTransform.rotate(rots, rotation1)
        #rots = self.customGSTransform.rotate(rots, rotation2)
        #rots = self.customGSTransform.rotate(rots, rotation3)
        print("Point cloud normalization factor: ", norm_factor)
        if (norm_factor >= 1.0):
            scales = scales - np.log(norm_factor)
        else:
            norm_factor += 1.0
            scales = scales + np.log(norm_factor)
        #scales = scales / np.log(norm_factor)
        # instead of adjusting the radius, adjust the object scale (and splat scale)
        scales = scales + np.log(opt_scale) # multiply logs = log() + logs()

        #temp_opacities = inverse_sigmoid(0.1 * torch.ones((xyz_2.shape[0], 1), dtype=torch.float, device="cuda")).cpu().numpy()
        # TODO FILTER OUT INF VALUES, and NAN ?!!
        #opacities = inverse_sigmoid(torch.from_numpy(opacities)).cpu().numpy()
        #opacities = opacities[~np.isinf(opacities)]


        # Get original number of static points for mask
        plydata_original = PlyData.read(original_file_path)

        xyz_original = np.stack((np.asarray(plydata_original.elements[0]["x"]),
                        np.asarray(plydata_original.elements[0]["y"]),
                        np.asarray(plydata_original.elements[0]["z"])),  axis=1)
        mask_subsampled = np.ones(len(xyz))
        mask_subsampled[len(xyz) - len(xyz_original):] = 0 # static points

        #features_dc = features[:, :, 0:1]
        #features_extra = features[:, :, 1:]
        self._xyz = nn.Parameter(torch.tensor(xyz[mask_subsampled == 1], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc[mask_subsampled == 1], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        # TODO try to scrap features_rest
        self._features_rest = nn.Parameter(torch.tensor(features_extra[mask_subsampled == 1], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities[mask_subsampled == 1], dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales[mask_subsampled == 1], dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots[mask_subsampled == 1], dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        self.spatial_lr_scale = spatial_lr_scale
        self.static_points_mask = mask_subsampled.copy()
        #self.max_radii2D = torch.zeros((self.get_xyz.shape[0] + len(mask_subsampled[mask_subsampled == 0])), device="cuda")
        self.max_radii2D = torch.zeros((len(self.static_points_mask[self.static_points_mask == 1])), device="cuda")
        #self.max_radii2D = torch.zeros((len(self.static_points_mask)), device="cuda")

        self.original_xyz = torch.tensor(xyz[mask_subsampled == 0], dtype=torch.float, device="cuda").requires_grad_(False)
        self.original_featuresdc = torch.tensor(features_dc[mask_subsampled == 0], dtype=torch.float, device="cuda").transpose(1,2).contiguous().requires_grad_(False)
        self.original_features_rest = torch.tensor(features_extra[mask_subsampled == 0], dtype=torch.float, device="cuda").transpose(1,2).contiguous().requires_grad_(False)
        self.original_opacity = torch.tensor(opacities[mask_subsampled == 0], dtype=torch.float, device="cuda").requires_grad_(False)
        self.original_scaling = torch.tensor(scales[mask_subsampled == 0], dtype=torch.float, device="cuda").requires_grad_(False)
        self.original_rotation = torch.tensor(rots[mask_subsampled == 0], dtype=torch.float, device="cuda").requires_grad_(False)

    def PreprocessCloud(self, xyz, rots, scales, opacities, features_dc, features_extra):
        #avg_scale_y = np.mean(scales, axis=1)
        avg_scale_y = np.sum(np.exp(scales), axis=1) # convert log to exp
        avg_scale = np.mean(scales)
        max_scale = np.max(avg_scale_y)
        #bool_mask = np.mean(scales, axis=1) < avg_scale + 0.5 * abs(avg_scale)
        #bool_mask = np.mean(scales, axis=1) < max_scale * 0.1
        bool_mask = np.sum(np.exp(scales), axis=1) < max_scale * 0.65
        xyz = xyz[bool_mask]
        rots = rots[bool_mask]
        opacities = opacities[bool_mask]
        features_dc = features_dc[bool_mask]
        features_extra = features_extra[bool_mask]
        scales = scales[bool_mask]

        
        return xyz, rots, scales, opacities, features_dc, features_extra
        #todo

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        
        #CUSTOM
        #spm = self.static_points_mask
        #xyz = torch.vstack((self.get_xyz, self.original_xyz[self.static_points_mask == 0]))
        n_init_points = self.get_xyz.shape[0]
        #n_init_points = xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        #CUSTOM 
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        #rotation = torch.vstack((self._rotation, self.original_rotation[spm == 0]))
        #rots = build_rotation(rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        #xyz = torch.vstack((self.get_xyz, self.original_xyz[spm == 0]))
        #new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + xyz[selected_pts_mask].repeat(N, 1)

        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent
        )

        #CUSTOM
        selected_pts_mask = selected_pts_mask[0:self._xyz.shape[0]]
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        print("Average opacity: ", torch.mean(self.get_opacity))
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune(self, min_opacity, extent, max_screen_size): # only called at the last iteration

        # CUSTOM
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda") # otherwise will not include any points in the result

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


    def add_densification_stats(self, viewspace_point_tensor, update_filter, only_dynamic_splats):
        #CUSTOM
        if (only_dynamic_splats == False):
            length = len(viewspace_point_tensor.grad) - len(self.original_xyz)#len(self.static_points_mask[self.static_points_mask == 0])
        else:
            length = len(viewspace_point_tensor.grad)
        gradients = viewspace_point_tensor.grad
        gradients = gradients[0:length]
        update_filter = update_filter[0:length]
        self.xyz_gradient_accum[update_filter] += torch.norm(gradients[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        w2c = np.linalg.inv(c2w)

        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3]).cuda()


class Renderer:
    # CUSTOM
    def __init__(self, sh_degree=3, white_background=True, radius=1, opt_object=None, opt_alignment=None):
        
        self.sh_degree = sh_degree
        self.white_background = white_background
        self.radius = radius

        self.gaussians = GaussianModel(sh_degree, opt_object, opt_alignment=opt_alignment)

        self.bg_color = torch.tensor(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )
    
    def initialize(self, input=None, num_pts=5000, radius=0.5, AABB=np.array((0, 1, 0, 1, 0, 1)), spatial_lr_scale=1, no_transform=False, no_rotation=False, blob_init_size=None, num_pts_init=None, flip_z=False, normalize=True, transform_splats_only=False):
        # load checkpoint
        if input is None:
            # init from random point cloud
            
            phis = np.random.random((num_pts,)) * 2 * np.pi
            costheta = np.random.random((num_pts,)) * 2 - 1
            thetas = np.arccos(costheta)
            mu = np.random.random((num_pts,))
            radius = radius * np.cbrt(mu)
            x = radius * np.sin(thetas) * np.cos(phis)
            y = radius * np.sin(thetas) * np.sin(phis)
            z = radius * np.cos(thetas)
            xyz = np.stack((x, y, z), axis=1)
            # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3

            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(
                points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
            )
            self.gaussians.create_from_pcd(pcd, 10)
        elif isinstance(input, BasicPointCloud):
            # load from a provided pcd
            #self.gaussians.create_from_pcd(input, 1)
            # CUSTOM TODO: maybe learning rate was too low (1) try with 10?
            self.gaussians.create_from_pcd(input, 1)
        else:
            # load from saved ply
            self.gaussians.load_ply(input, AABB, spatial_lr_scale=spatial_lr_scale, no_transform=no_transform, no_rotation=no_rotation, blob_init_size=blob_init_size, num_pts_init=num_pts_init, flip_z=flip_z, normalize=normalize, transform_splats_only=transform_splats_only)

    def initialize_static_and_dynamic(self, input=None, original_file_path=None, spatial_lr_scale=1):
        self.gaussians.load_static_and_dynamic_ply(input, original_file_path, spatial_lr_scale=spatial_lr_scale)

    def render(
        self,
        viewpoint_camera,
        scaling_modifier=1.0,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
        static_color=None, # for visualizing overlapping areas
        dynamic_color=None,
        only_dynamic_splats=False,
        only_static_splats=False
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        #CUSTOM
        self.gaussians.only_dynamic_splats = only_dynamic_splats
        xyz = None
        if (only_dynamic_splats):
            xyz = self.gaussians.get_xyz
        elif (only_static_splats):
            xyz = self.gaussians.original_xyz
        else:
            xyz = torch.vstack((self.gaussians.get_xyz, self.gaussians.original_xyz))

        screenspace_points = (
            torch.zeros_like(
                xyz,#CUSTOM,
                #self.gaussians.get_xyz,
                dtype=self.gaussians.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )

        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if bg_color is None else bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        #means3D = self.gaussians.get_xyz
        #means2D = screenspace_points
        #opacity = self.gaussians.get_opacity
        # CUSTOM 
        opac = None
        if (only_dynamic_splats):
            opac = self.gaussians.get_opacity
        elif (only_static_splats):
            opac = self.gaussians.original_opacity
        else:
            opac = torch.vstack((self.gaussians.get_opacity, self.gaussians.opacity_activation(self.gaussians.original_opacity)))
            #if static_color is not None:
            #    opac[len(self.gaussians.get_xyz):] = 0.25
            #if dynamic_color is not None:
            #    opac[:len(self.gaussians.get_xyz)] = 0.1
        means3D = xyz
        means2D = screenspace_points
        opacity = opac

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
        else:
            scales = self.gaussians.get_scaling
            rotations = self.gaussians.get_rotation

        #CUSTOM
        sc = None
        rot = None
        if (only_dynamic_splats):
            sc = scales
            rot = rotations
        elif (only_static_splats):
            sc = self.gaussians.scaling_activation(self.gaussians.original_scaling)
            rot = self.gaussians.rotation_activation(self.gaussians.original_rotation)
        else:
            sc = torch.vstack((scales, self.gaussians.scaling_activation(self.gaussians.original_scaling)))
            rot = torch.vstack((rotations, self.gaussians.rotation_activation(self.gaussians.original_rotation)))
        scales = sc
        rotations = rot

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = self.gaussians.get_features.transpose(1, 2).view(
                    -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = self.gaussians.get_xyz - viewpoint_camera.camera_center.repeat(
                    self.gaussians.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                if (only_static_splats):
                    shs = torch.cat((self.gaussians.original_featuresdc, self.gaussians.original_features_rest), dim=1)
                    if static_color is not None:
                        shs[:,0] = static_color
                        shs[:,1:] = 0.0
                else:
                    shs = self.gaussians.get_features
                    # colorize gaussians if given a color
                    if static_color is not None:
                        shs[len(self.gaussians.get_xyz):, 0] = static_color
                        shs[len(self.gaussians.get_xyz):, 1:] = 0.0
                    if dynamic_color is not None:
                        shs[:len(self.gaussians.get_xyz), 0] = dynamic_color
                        shs[:len(self.gaussians.get_xyz), 1:] = 0.0
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        rendered_image = rendered_image.clamp(0, 1)

        #gaussian blur
        #rendered_image = torchvision.transforms.GaussianBlur(35, sigma=(32,64))(rendered_image)

        #CUSTOM
        # CUSTOME
        def write_image_to_drive(input_tensor, index=None):
                # transform torch tensor to rgb image and write to drive for DEBUG purposes
                transform = T.ToPILImage()
                img = transform(input_tensor[0])
                #img = img.save("debug/train_step_debug" + str(index) +".jpg")
                img = img.save("debug/gaussian_raster_depth_debug.jpg")

        #write_image_to_drive(rendered_depth)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }
