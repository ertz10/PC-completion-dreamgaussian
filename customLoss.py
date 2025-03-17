import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from gs_renderer import Renderer, GaussianModel
import open3d as o3d
from gs_renderer import MiniCam
import PIL.Image
import trimesh
from gaussianSplatsRender import GaussianCustomRenderer

class AABBLoss:

    def __init__(self, AABB):
        self.AABB = AABB
        self.CustomGSRenderer = GaussianCustomRenderer()

    def GSRendererDepthBlending(self, gaussians: GaussianModel, cur_cam: MiniCam, bg_color: torch.tensor, only_dynamic_splats: bool):
        # render scene 2 times, first static points with depth map,
        # then dynamic points with depth map
        static_rendering = None
        if (only_dynamic_splats == False):
            static_rendering = self.CustomGSRenderer.render_custom(gaussians=gaussians, viewpoint_camera=cur_cam, bg_color=bg_color, render_type="static") # TODO add variable that tells whether to render static or dynamic points
        dynamic_pts_rendering = self.CustomGSRenderer.render_custom(gaussians=gaussians, viewpoint_camera=cur_cam, bg_color=bg_color, render_type="dynamic", only_dynamic_splats=only_dynamic_splats)
        
        if (static_rendering == None):
            static_rendering = dynamic_pts_rendering # TODO just for testing purposes, remove in final version
        
        return static_rendering["image"], dynamic_pts_rendering["image"], static_rendering["depth"], dynamic_pts_rendering["depth"], static_rendering["alpha"], dynamic_pts_rendering["alpha"]
        
    def GSRendererStaticRendering(self, gaussians: GaussianModel, cur_cam: MiniCam, bg_color: torch.tensor):
        # render scene 1 time, only static points with depth map
        static_rendering = self.CustomGSRenderer.render_static(gaussians=gaussians, viewpoint_camera=cur_cam, bg_color=bg_color, render_type="static") # TODO add variable that tells whether to render static or dynamic points
        
        return static_rendering["image"], static_rendering["depth"], static_rendering["alpha"]
    
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
    

    def AABBRender(self, AABB: np.ndarray, currentCam: MiniCam, radius: float):
        box_width = np.abs(AABB[1] - AABB[0])
        box_height = np.abs(AABB[3] - AABB[2])
        box_depth = np.abs(AABB[5] - AABB[4])
        #BBoxMesh = o3d.geometry.TriangleMesh.create_box(width=box_width, height=box_height, depth=box_depth) # returns triangle mesh
        BBoxMesh = trimesh.primitives.Box(extents=np.array([box_depth, box_height, box_width]), mutable=True) # returns triangle mesh

        AABBCenter = np.array([(AABB[1] - AABB[0]) / 2.0, (AABB[3] - AABB[2]) / 2.0, (AABB[5] - AABB[4]) / 2.0])
        mesh_center = BBoxMesh.center_mass

        translation_mat = trimesh.transformations.translation_matrix((0.0, 0.0, 0.0))
        BBoxMesh.apply_transform(translation_mat)

        import pyglet
        pyglet.options["headless"] = True

        # depth, height, width
        #box_mesh = trimesh.primitives.Box(extents=np.array([box_depth, box_height, box_width]), mutable=True)
        scene = trimesh.Scene()
        scene.add_geometry(BBoxMesh)
        #https://github.com/mikedh/trimesh/blob/main/examples/offscreen_render.py
        # TODO some transformations with trimesh trimesh.trnafomrations.rotation_matrix ....
        # https://github.com/cg-tuwien/ppsurf/blob/main/source/base/visualization.py
        img = None
        while img is None:
            try:
                transformations = trimesh.transformations.decompose_matrix(currentCam.world_view_transform.cpu().detach())
                rotation = transformations[2] # gets Euler angles around x,y,z axes
                import math
                camera = scene.set_camera(angles=(rotation[0], rotation[1], rotation[2]), distance=radius, center=(0.0, 0.0, 0.0), fov=(math.degrees(currentCam.FoVx), math.degrees(currentCam.FoVy)))
                
                #file_name = "debug/AABB_rendering_debug.png"
                img = scene.save_image(resolution=(currentCam.image_width, currentCam.image_height), visible=True)
                import io
                imageArr = PIL.Image.open(io.BytesIO(img))
                trans = T.Compose([T.ToTensor()]) # transform to tensor in [0,1] range
                imageArr = trans(imageArr)
                # reshape to match (1, 3, resx, resy)
                #imageArr = imageArr.swapaxes(0, 2)
                # also swap other axes since the image is now rotated 
                #imageArr = imageArr.swapaxes(1, 2)
                # remove alpha channel
                imageArr = imageArr[:3]
                # add extra dimension at index 0
                imageArr = torch.unsqueeze(imageArr, 0)
                return imageArr
                #self.write_capture_to_drive(img)
            except BaseException as E:
                print("Unable to save AABB rendering image!", str(E))
            
    def blend_images(self, images1: list, images2: list, depth_images_1: list, depth_images_2: list):

        # images1 = dynamic images
        # images2 = static images
        dynamic_depth = []#torch.less(dynamic_depth_images, static_depth_images) # 
        static_depth = []
        for idx, depth_img in enumerate(depth_images_1):
            # TODO make background full white, since the furthest away depth is 1, and nearest is 0
            
            zeroes_dynamic = depth_images_1[idx] == 0.0
            zeroes_static = depth_images_2[idx] == 0.0
            #zeroes_dynamic = torch.vstack((zeroes_dynamic, zeroes_dynamic, zeroes_dynamic))
            #zeroes_static = torch.vstack((zeroes_static, zeroes_static, zeroes_static))
            depth_images_1[idx][zeroes_dynamic] = 1.0
            depth_images_2[idx][zeroes_static] = 1.0
            
            dynamic_depth.append(torch.less(depth_img, depth_images_2[idx]).long()) # true for each pixel if is in front of static pixel
            static_depth.append(torch.less(depth_images_2[idx], depth_img).long())
            #dynamic_depth.append(depth_img)
            #static_depth.append(depth_images_2[idx])

        #self.write_capture_to_drive(depth_images_1, depth_images_1[0].shape[1], depth_images_1[0].shape[2], 4, "_dynamic_depth_image")            
        #self.write_capture_to_drive(depth_images_2, depth_images_2[0].shape[1], depth_images_2[0].shape[2], 4, "_static_depth_image")            
        #self.write_capture_to_drive(dynamic_depth, dynamic_depth[0].shape[1], dynamic_depth[0].shape[2], 4, "_dynamic_depth")
        #self.write_capture_to_drive(static_depth, static_depth[0].shape[1], static_depth[0].shape[2], 4, "_static_depth")

        imgs_blended = images1.copy()
        for i, img in enumerate(images1):
            img2 = images2[i][:3] # only rgb
            imgs_blended[i][:3] = (img[:3] * torch.vstack((static_depth[i], static_depth[i], static_depth[i]))
                                      + img2 * torch.vstack((dynamic_depth[i], dynamic_depth[i], dynamic_depth[i])))
        
            imgs_blended[i] = torch.clamp(imgs_blended[i], 0.0, 1.0)
        #return torch.stack((imgs_blended)) # convert list to tensor of tensors
        return dynamic_depth, static_depth

    def write_capture_to_drive(self, input_data, image_width, image_height, num_images, string=""):

            #img_width = input_tensor.shape[2]
            #img_height = input_tensor.shape[3]
            # create figure
            figure = PIL.Image.new('RGB', (image_width*num_images, image_height), color=(255, 255, 255))

            import io
            # add images
            for i, img in enumerate(input_data):
                #image = PIL.Image.open(io.BytesIO(img))
                #image = PIL.Image.new(img)
                transform = T.ToPILImage()
                img = img.float().clone()#torch.tensor(img, dtype=torch.float64)
                image = transform(img)
                figure.paste(image, (i * image_width, 0))

            figure.save(r"debug/DepthBlending_rendering_debug" + str(string), "JPEG")

    def guidance_weighting(self, step: int, guidance_type: str, xp: list, fp: list):
        
        #step_ratio =  np.interp(self.step, xp, fp)
        interp = 0.0
        if(guidance_type == "text"):
            interp = np.interp(step, xp, fp)
            interp = 1.0
            print("Text guidance weighting: ", interp)
        if(guidance_type == "image"):
            # apply more image guidance in earlier steps
            interp = 1.0 - np.interp(step, xp, fp)
            interp = 1.0
            print("Image guidance weighting: ", interp)
        return interp


