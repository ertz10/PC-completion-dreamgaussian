import torch
import numpy as np
import trimesh

class GaussianTransform:

    '''
    From gs_renderer.py
    '''
    def build_rotation(self, r):
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

    def __init__(self):
        pass
        
    def rotate(self, rotations: np.ndarray, rotation_matrix: np.ndarray): 


        #new_rots = trimesh.transformations.transform_points(rotations, rotation_matrix)
        # original rotations are in quaternions
        current_rot_mat = self.build_rotation(torch.tensor(rotations)).cpu().numpy() # builds rotation matrix
        idx = 0
        for mat in current_rot_mat:
            new_rot_mat = rotation_matrix[:3,:3] @ mat
            rotations[idx, :] = trimesh.transformations.quaternion_from_matrix(new_rot_mat)
            idx += 1

        # just multiply 2 quaternions ?
        #rotation_quaternion = trimesh.transformations.quaternion_from_matrix(rotation_matrix)
        #transformed_rotations = rotation_quaternion * rotations

        return rotations
    
    def normalize_PC(self, xyz: np.ndarray):
            xmin, xmax = xyz[:, 0].min(), xyz[:, 0].max()
            ymin, ymax = xyz[:, 1].min(), xyz[:, 1].max()
            zmin, zmax = xyz[:, 2].min(), xyz[:, 2].max()
            
            # find axis with longest extent
            x_ext = np.abs((xmax - xmin))
            y_ext = np.abs((ymax - ymin))
            z_ext = np.abs((zmax - zmin))
            max_extents = np.array((x_ext, y_ext, z_ext))
            max_extent_idx = np.argmax(max_extents)
            norm_factor = np.max(max_extents)
            xyz_norm = xyz / (max_extents[max_extent_idx])

            return xyz_norm, norm_factor # double norm factor since we want [-0.5, 0.5] range