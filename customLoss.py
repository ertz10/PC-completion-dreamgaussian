import torch
import numpy as np
from gs_renderer import Renderer, GaussianModel

class AABBLoss:

    def __init__(self):
        pass

    def AABBLoss(self, AABB: np.ndarray, gaussians: GaussianModel, removePoints: bool, step: int): 
        mask = gaussians.TestAgainstBB(AABB, removePoints=removePoints)

        loss = torch.sum(mask[mask == True].long()) # points outside = True

        # further away from center = higher loss, using euler function
        positions = gaussians._xyz.clone()
        positions = positions.cpu().detach().numpy()
        AABBCenter = np.array([[(AABB[1] - AABB[0]) / 2.0, (AABB[3] - AABB[2]) / 2.0, (AABB[5] - AABB[4]) / 2.0]])
        AABBCenter = np.repeat(AABBCenter, positions.shape[0], axis = 0)
        vecs = np.array(positions - AABBCenter)
        #norm = np.linalg.norm(vecs, axis=1)
        #vecs_normalized = vecs / np.stack((norm, norm, norm), axis=1)
        # TODO loss, 80 % of width between center and bbox outside, compute non linear func
        box_width = np.abs(AABB[1] - AABB[0])
        box_height = np.abs(AABB[3] - AABB[2])
        box_depth = np.abs(AABB[5] - AABB[4])
        threshold = 0.8/2.0
        #mask = np.zeros(positions.shape[0])
        mask_x = np.abs(vecs[:, 0]) > box_width * threshold # filter x that are above 80 % width threshold
        mask_y = np.abs(vecs[:, 1]) > box_height * threshold
        mask_z = np.abs(vecs[:, 2]) > box_depth * threshold
        mask_sum = mask_x | mask_y | mask_z
        loss = vecs[mask_sum]
        # now do normalization!
        norm = np.linalg.norm(loss, axis = 1)
        loss = loss / np.stack((norm, norm, norm), axis = 1)
        # compute magnitude of unit vector
        loss = np.linalg.norm(vecs, axis=1)
        # feed function
        loss = np.exp(5.0*loss) # e^5x
        loss = np.sum(loss)
        if step % 5 == 0:
            print("Loss from AABB: ", loss)
        return loss