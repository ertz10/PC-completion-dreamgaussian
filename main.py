import os
import cv2
import time
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg

import torch
import torch.nn.functional as F

import torchvision
import torchvision.transforms as T
from PIL import Image

import rembg

from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam, BasicPointCloud, SH2RGB

from grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize

import matplotlib.pyplot as plt

import open3d as o3d

from customLoss import AABBLoss

class GUI:
    def __init__(self, opt, opt_object):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.opt_object = opt_object
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_sd = None
        self.guidance_zero123 = None

        self.enable_sd = False
        self.enable_zero123 = False

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree, opt_object=self.opt_object)
        self.gaussain_scale_factor = 1

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop

        # CUSTOM
        self.debug_step = 0
        self.static_points_mask = None
        self.variable_points_mask_length = 0
        self.static_points_length = 0
        self.all_steps = []

        #self.couch_AABB = np.array([-0.0, 0.8, -0.3, 0.3, -0.25, 0.25], dtype=np.float32)
        #self.couch_captured_angles_hor = [-180, 0] # hard coded for now
        #self.trashcan_AABB = np.array([-0.15, 0.15, -0.3, 0.3, -0.15, 0.15], dtype=np.float32)
        #self.elephant_AABB = np.array([-0.3, 0.3, -0.4, 0.4, 0.25, 0.6], dtype=np.float32)
        #self.elephant_captured_angles_hor = [100, 180] # hard coded for now
        #self.hocker_AABB = np.array([0.0, 0.4, -0.4, 0.0, -0.2, 0.2], dtype=np.float32)
        #self.vase_AABB = np.array([-0.1, 0.1, -0.2, 0.0, 0.1, 0.1], dtype=np.float32)
        #self.vase_captured_angles_hor = [90, 220]
        #self.chicken_AABB = np.array([-0.1, 0.1, -0.3, -0.2, -0.1, 0.1], dtype=np.float32)
        #self.chicken_captured_angles_hor = [60, 120]
        #self.shoe_AABB = np.array([-0.1, 0.1, -0.2, 0.0, -0.3, 0.1], dtype=np.float32)
        #self.shoe_AABB = np.array([-0.2, 0.2, -0.2, 0.2, -0.2, 0.2], dtype=np.float32)
        #self.shoe_captured_angles_hor = [0, 130] # hard coded for now, gives the approximate angles of the best views on the static part
        self.AABB = self.opt_object.AABB
        self.customLoss = AABBLoss(self.opt_object.AABB)
        self.captured_angles_hor = self.opt_object.visible_angles
        
        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input)

        # CUSTOM POINT_CLOUD LOADING
        # Use fixed positions for the input point cloud, downsample as well, everything missing
        # to 5000 points, fill with random initialized coordinates
        if self.opt.point_cloud is not None:
            ply = o3d.io.read_point_cloud(self.opt.point_cloud)
            pcd = o3d.io.write_point_cloud("pointcloud_test.pcd", ply)
            print(pcd)
            pcd_read = o3d.io.read_point_cloud("pointcloud_test.pcd")
            downpcd = pcd_read.voxel_down_sample(voxel_size = 0.05)
            downpcd = pcd_read

            num_pts = len(downpcd.points) # TODO hardcoded at the moment, not good
            radius = 0.5
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
            #xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3

            pts = downpcd.points
            # convert from open3d to numpy
            pts = np.asarray(pts)
            shs = np.random.random((num_pts, 3)) / 255.0
            xyz_full = xyz.copy()
            xyz = xyz[len(pts):]
            pts_merged = np.vstack((pts, xyz))
            #pts_merged = np.vstack((pts, pts[0:(5000-len(pts))]))
            pts_merged = pts.copy()
            pcd = BasicPointCloud(
                points=pts_merged, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
            )

            pcd_fulluniform = BasicPointCloud(
                points=xyz_full, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
            )
            
            self.renderer.initialize(input=pcd, AABB=self.AABB)

            #debug pcd
            pcd_debug = o3d.geometry.PointCloud()
            pcd_debug.points = o3d.utility.Vector3dVector(pts_merged)
            #pcd_debug.points = o3d.utility.Vector3dVector(xyz_full)
            o3d.io.write_point_cloud("debug/pcd_debug.ply", pcd_debug)
            #debug pcd end
        
        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None:
            self.negative_prompt = self.opt.negative_prompt

        # override if provide a checkpoint
        #if self.opt.load is not None:
        #    self.renderer.initialize(self.opt.load, AABB=self.AABB)   
        # CUSTOM load from object file
        if self.opt.load is not None:
            self.renderer.initialize(self.opt_object.load, AABB=self.AABB)         
        else:
            # CUSTOM CODE
            if self.opt.point_cloud is None:
                # initialize gaussians to a blob
                self.renderer.initialize(num_pts=self.opt.num_pts)

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def prepare_train(self):

        self.step = 0

        # setup training
        # CUSTOM with self.static_points_mask
        #self.renderer.gaussians.static_points_mask = self.static_points_mask # CUSTOM
        #self.renderer.gaussians.variable_points_length = self.variable_points_mask_length
        #self.renderer.gaussians.static_points_length = self.static_points_length
        #self.renderer.gaussians.original_points = self.original_points
        self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        # default camera
        if self.opt.mvdream or self.opt.imagedream:
            # the second view is the front view for mvdream/imagedream.
            pose = orbit_camera(self.opt.elevation, 90, self.opt.radius)
        else:
            pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.fixed_cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )

        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""
        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None

        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd:
            if self.opt.mvdream:
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream
                self.guidance_sd = MVDream(self.device)
                print(f"[INFO] loaded MVDream!")
            elif self.opt.imagedream:
                print(f"[INFO] loading ImageDream...")
                from guidance.imagedream_utils import ImageDream
                self.guidance_sd = ImageDream(self.device)
                print(f"[INFO] loaded ImageDream!")
            else:
                print(f"[INFO] loading SD...")
                from guidance.sd_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD!")

        if self.guidance_zero123 is None and self.enable_zero123:
            print(f"[INFO] loading zero123...")
            from guidance.zero123_utils import Zero123
            if self.opt.stable_zero123:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/stable-zero123-diffusers')
            else:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/zero123-xl-diffusers')
            print(f"[INFO] loaded zero123!")

        # input image
        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

        # prepare embeddings
        with torch.no_grad():

            if self.enable_sd:
                if self.opt.imagedream:
                    self.guidance_sd.get_image_text_embeds(self.input_img_torch, [self.prompt], [self.negative_prompt])
                else:
                    self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

            if self.enable_zero123:
                self.guidance_zero123.get_img_embeds(self.input_img_torch)

    

    def train_step(self):

        # CUSTOME
        def write_image_to_drive(input_tensor, index):
                # transform torch tensor to rgb image and write to drive for DEBUG purposes
                transform = T.ToPILImage()
                img = transform(input_tensor[0])
                img = img.save(r"debug/train_step_debug" + str(index) + r".jpg")
                #img = img.save("debug/train_step_debug.jpg")

        # CUSTOME
        def write_images_to_drive(input_tensor, index):

                import PIL.Image

                img_width = input_tensor[0].shape[2]
                img_height = input_tensor[0].shape[3]
                # create figure
                figure = PIL.Image.new('RGB', (img_width * 4, img_height), color=(255, 255, 255))

                # add images
                for i, img in enumerate(input_tensor):
                    transform = T.ToPILImage()
                    img = torch.tensor(img)
                    img = transform(img[0])
                    figure.paste(img, (i * img_width, 0))

                try:
                    figure.save("debug/train_step_debug.jpg")
                except OSError:
                    print("Cannot save image")

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        
        for _ in range(self.train_steps):
            # CUSTOM
            self.debug_step += 1

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)
            ############ CUSTOM ######################################
            ## Increase step ratio, this is used by mvdream as well ##
            ##########################################################
            #xp = [0,    100,  200,  300, 450, 500,  600]
            #fp = [0.02, 0.1, 0.3,  0.4, 0.5, 0.8, 0.99]
            xp = [0, 200, 400, 600, 800, 1000]
            fp = [0.02, 0.05, 0.2, 0.3, 0.8, 0.98]
            #step_ratio =  np.interp(self.step, xp, fp)
            self.all_steps = np.append(self.all_steps, step_ratio)
            ############# plot #####################
            #plt.plot(np.arange(self.step), np.ones(len(self.all_steps)) - self.all_steps)
            #plt.show()
            #plt.savefig(r"debug/graph_plot.png")
            ##########################################################

            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)

            loss = 0

            ### known view
            # CUSTOM Only for IMAGE INPUT
            if self.input_img_torch is not None and not self.opt.imagedream:
                cur_cam = self.fixed_cam
                out = self.renderer.render(cur_cam)

                # rgb loss
                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                # TODO remove comment or remove input image from text only run loss = loss + 10000 * (step_ratio if self.opt.warmup_rgb_loss else 1) * F.mse_loss(image, self.input_img_torch)                                                                                                          

                # mask loss
                mask = out["alpha"].unsqueeze(0) # [1, 1, H, W] in [0, 1]
                #loss = loss + 1000 * (step_ratio if self.opt.warmup_rgb_loss else 1) * F.mse_loss(mask, self.input_mask_torch)

            ### novel view (manual batch)
            render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
            images = []
            AABBimages = []
            static_images = []
            dynamic_images = []
            static_depth_images = []
            dynamic_depth_images = []
            poses = []
            vers, hors, radii = [], [], []
            # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
            min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
            max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)

            for _ in range(self.opt.batch_size):

                # render random view
                # CUSTOM maybe use not random but fixed angles per view ?
                ver = np.random.randint(min_ver, max_ver)
                #ver = -45.0
                #if(self.step % 9 == 0):
                #    hor = 75.0
                #else:
                #hor = np.random.randint(-180, 180)
                #hor = np.random.randint(0, 360)
                #CUSTOM
                # TODO sample known angles more often
                if (self.step % 3 == 0):
                    angle1 = self.opt_object.visible_angles[0]
                    angle2 = self.opt_object.visible_angles[1]
                    if (angle1 > angle2): # e.g. [280, 30] going around zero 
                        rand1 = np.random.randint(angle1, 360)
                        rand2 = np.random.randint(0, angle2)
                        decider = np.random.choice([0, 1])
                        hor = rand1 - 180 if decider == 0 else rand2 - 180
                    else:
                        hor = int(np.random.randint(angle1, angle2) - 180.0)
                else:
                    hor = int((360.0 / self.opt.iters) * self.step - 180.0)

                radius = 0.0#-1.25

                # CUSTOM
                pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                #pose = orbit_camera(self.opt.elevation, hor, self.opt.radius + radius)
                #
                poses.append(pose)

                vers.append(ver) 
                # convert hor to hor >= 0
                hor = 180 + (180 - abs(hor)) if hor < 0 else hor
                hors.append(hor)
                radii.append(radius)

                cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                # Custom
                out = None
                #if (self.step == 500):
                #    out = self.renderer.render(cur_cam, bg_color=bg_color)
                #else:
                    # ONLY FOR DEBUG PURPOSE TODO remove in final version
                out = self.renderer.render(cur_cam, bg_color=bg_color)

                #CUSTOM render Bounding Box to image
                #AABBimage = self.customLoss.AABBRender(self.AABB, cur_cam, self.opt.radius + radius)
                #AABBimage = self.customLoss.GSRendererDepthBlending(self.renderer.gaussians, cur_cam, bg_color=bg_color)
                #AABBimages.append(AABBimage)
                ######################################################################
                ########### dynamic, static point rendering ##########################
                ######################################################################
                static_points_image, dynamic_points_image, static_points_depth, dynamic_points_depth, static_points_alpha, dynamic_points_alpha = self.customLoss.GSRendererDepthBlending(self.renderer.gaussians, cur_cam, bg_color=bg_color)
                static_images.append(torch.vstack((static_points_image, static_points_alpha)))
                dynamic_images.append(torch.vstack((dynamic_points_image, dynamic_points_alpha)))
                static_depth_images.append(static_points_depth)
                dynamic_depth_images.append(dynamic_points_depth)
                ######################################################################
                ######################################################################

                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                images.append(image)

                # enable mvdream training
                if self.opt.mvdream or self.opt.imagedream:
                    for view_i in range(1, 4):
                        # Custom
                        #hor = np.random.randint(-180, 180)
                        #ver = np.random.randint(min_ver, max_ver)

                        # convert to (-180, 180) again
                        hor_i = hor + 90 * view_i
                        hor_i = hor_i if hor_i < 180 else -180 + (hor_i - 180)
                        # modulo operator to get the actual value
                        #hor_i = hor_i % 180
                        #pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i, self.opt.radius + radius)
                        pose_i = orbit_camera(self.opt.elevation + ver, hor_i, self.opt.radius + radius)
                        #hors.append(hor + 90 * view_i)

                        # convert hors to hors >= 0 for convenience when comparing visibility of camera angles
                        hor_i = 180 + (180 - abs(hor_i)) if hor_i < 0 else hor_i
                        hors.append(hor_i)
                        poses.append(pose_i)

                        cur_cam_i = MiniCam(pose_i, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                        # bg_color = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device="cuda")
                        #if (self.step == 500):
                        #    out_i = self.renderer.render(cur_cam_i, bg_color=bg_color)
                        #else:
                            # ONLY FOR DEBUG PURPOSE TODO remove in final version
                        out_i = self.renderer.render(cur_cam_i, bg_color=bg_color)

                        #CUSTOM render Bounding Box to image
                        #AABBimage = self.customLoss.AABBRender(self.AABB, cur_cam_i, self.opt.radius + radius)
                        ######################################################################
                        ########### dynamic, static point rendering ##########################
                        ######################################################################
                        static_points_image, dynamic_points_image, static_points_depth, dynamic_points_depth, static_points_alpha, dynamic_points_alpha = self.customLoss.GSRendererDepthBlending(self.renderer.gaussians, cur_cam_i, bg_color=bg_color)
                        static_images.append(torch.vstack((static_points_image, static_points_alpha)))
                        dynamic_images.append(torch.vstack((dynamic_points_image, dynamic_points_alpha)))
                        static_depth_images.append(static_points_depth)
                        dynamic_depth_images.append(dynamic_points_depth)
                        #AABBimage = self.customLoss.GSRendererDepthBlending(self.renderer.gaussians, cur_cam, bg_color=bg_color)
                        #AABBimages.append(AABBimage)
                        ######################################################################
                        ######################################################################      

                        image = out_i["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                        images.append(image)

                # CUSTOM
                #self.customLoss.write_capture_to_drive(AABBimages, cur_cam.image_width, cur_cam.image_height, len(AABBimages))
                # Debug: TODO maybe remove later
                #images_blended = self.customLoss.blend_images(AABBimages, images)

                if self.debug_step % 1 == 0:
                    write_images_to_drive(images, 0)
                    self.debug_step = 0

            images = torch.cat(images, dim=0)
            poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)

            #import kiui
            #print(hor, ver)
            #kiui.vis.plot_image(images)

            # guidance loss
            if self.enable_sd:
                if self.opt.mvdream or self.opt.imagedream:
                    #loss = loss + 1.0 * self.customLoss.guidance_weighting(step=self.step, guidance_type="text", xp=xp, fp=fp) * self.opt.lambda_sd * self.guidance_sd.train_step(images, poses, self.customLoss, step_ratio=step_ratio if self.opt.anneal_timestep else None, 
                    #                                                                                                                                                        dynamic_images=dynamic_images, static_images=static_images, 
                    #                                                                                                                                                        dynamic_depth_images=dynamic_depth_images, static_depth_images=static_depth_images, current_cam_hors=hors, captured_angles_hor=self.captured_angles_hor)
                    loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, poses, self.customLoss, step_ratio=step_ratio if self.opt.anneal_timestep else None, 
                                                                                                                                                                                dynamic_images=dynamic_images, static_images=static_images, 
                                                                                                                                                                                dynamic_depth_images=dynamic_depth_images, static_depth_images=static_depth_images, current_cam_hors=hors, captured_angles_hor=self.captured_angles_hor, object_params=self.opt_object)
                else:
                    loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, step_ratio=step_ratio if self.opt.anneal_timestep else None)

            #if self.enable_zero123:
                #loss = loss + 0.0 * self.customLoss.guidance_weighting(step=self.step, guidance_type="image", xp=xp, fp=fp) * self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, self.input_img_torch, vers, hors, radii, self.customLoss, step_ratio=step_ratio if self.opt.anneal_timestep else None, default_elevation=self.opt.elevation, 
                #                                                                                                                                                                   dynamic_images=dynamic_images, static_images=static_images, 
                #                                                                                                                                                                   dynamic_depth_images=dynamic_depth_images, static_depth_images=static_depth_images)
                #loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio=step_ratio if self.opt.anneal_timestep else None, default_elevation=self.opt.elevation)
            
            # TODO add loss for points inside/outside bounding box
            #loss = loss + 1000.0 * self.customLoss.AABBLoss(self.AABB, self.renderer.gaussians, removePoints=False, step=self.step)

            ################ REFERENCE IMAGE LOSS ########################
            #img_interp = F.interpolate(images, (256, 256), mode="bilinear", align_corners=False)
            #loss = loss + 10000.0 * F.mse_loss(img_interp[0].unsqueeze(0), self.input_img_torch)
            #write_image_to_drive(self.input_img_torch, 0)
            ##############################################################

            # optimize step

            loss.backward()
            # CUSTOM this is where gaussian pos change happens, optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # densify and prune
            if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
                #CUSTOM .to("cpu")
                viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"].to("cuda"), out["radii"]
                #CUSTOM
                length = len(viewspace_point_tensor) - len(self.renderer.gaussians.original_xyz)#len(self.renderer.gaussians.static_points_mask[self.renderer.gaussians.static_points_mask == 0])

                viewspace_point_tensor_temp = viewspace_point_tensor[0:length]
                visibility_filter_temp = visibility_filter[0:length]
                radii_temp = radii[0:length]
                self.renderer.gaussians.max_radii2D[visibility_filter_temp] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter_temp], radii_temp[visibility_filter_temp])
                self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.step % self.opt.densification_interval == 0:
                    self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=4, max_screen_size=1)
                
                if self.step % self.opt.opacity_reset_interval == 0:
                    self.renderer.gaussians.reset_opacity()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

        if self.gui:
            dpg.set_value("_log_train_time", f"{t:.4f}ms")
            dpg.set_value(
                "_log_train_log",
                f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f}",
            )

        # dynamic train steps (no need for now)
        # max allowed train time per-frame is 500 ms
        # full_t = t / self.train_steps * 16
        # train_steps = min(16, max(4, int(16 * 500 / full_t)))
        # if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
        #     self.train_steps = train_steps

    @torch.no_grad()
    def test_step(self):
        # ignore if no need to update
        if not self.need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:
            # render image

            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )

            out = self.renderer.render(cur_cam, self.gaussain_scale_factor)

            buffer_image = out[self.mode]  # [3, H, W]

            if self.mode in ['depth', 'alpha']:
                buffer_image = buffer_image.repeat(3, 1, 1)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

            buffer_image = F.interpolate(
                buffer_image.unsqueeze(0),
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            self.buffer_image = (
                buffer_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )

            # display input_image
            if self.overlay_input_img and self.input_img is not None:
                self.buffer_image = (
                    self.buffer_image * (1 - self.overlay_input_img_ratio)
                    + self.input_img * self.overlay_input_img_ratio
                )

            self.need_update = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_texture", self.buffer_image
            )  # buffer must be contiguous, else seg fault!

    def load_input(self, file):
        # load image
        print(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        self.input_mask = img[..., 3:]
        # white bg
        self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)
        # bgr to rgb
        self.input_img = self.input_img[..., ::-1].copy()

        # load prompt
        file_prompt = file.replace("_rgba.png", "_caption.txt")
        if os.path.exists(file_prompt):
            print(f'[INFO] load prompt from {file_prompt}...')
            with open(file_prompt, "r") as f:
                self.prompt = f.read().strip()

    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024, index=0):
        os.makedirs(self.opt.outdir, exist_ok=True)
        if mode == 'geo':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.ply')
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)
            mesh.write_ply(path)

        elif mode == 'geo+tex':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.' + self.opt.mesh_format)
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)

            # perform texture extraction
            print(f"[INFO] unwrap uv...")
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()

            albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
            cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

            # self.prepare_train() # tmp fix for not loading 0123
            # vers = [0]
            # hors = [0]
            vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

            render_resolution = 512

            import nvdiffrast.torch as dr

            if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
                glctx = dr.RasterizeGLContext()
            else:
                glctx = dr.RasterizeCudaContext()

            for ver, hor in zip(vers, hors):
                # render image
                pose = orbit_camera(ver, hor, self.cam.radius)

                cur_cam = MiniCam(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )
                
                cur_out = self.renderer.render(cur_cam)

                rgbs = cur_out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]

                # enhance texture quality with zero123 [not working well]
                # if self.opt.guidance_model == 'zero123':
                #     rgbs = self.guidance.refine(rgbs, [ver], [hor], [0])
                    # import kiui
                    # kiui.vis.plot_image(rgbs)
                    
                # get coordinate in texture image
                pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
                proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).to(self.device)

                v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
                v_clip = v_cam @ proj.T
                rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

                depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
                depth = depth.squeeze(0) # [H, W, 1]

                alpha = (rast[0, ..., 3:] > 0).float()

                uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

                # use normal to produce a back-project mask
                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal[0])

                # rotated normal (where [0, 0, 1] always faces camera)
                rot_normal = normal @ pose[:3, :3]
                viewcos = rot_normal[..., [2]]

                mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
                mask = mask.view(-1)

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
                
                # update texture image
                cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                    h, w,
                    uvs[..., [1, 0]] * 2 - 1,
                    rgbs,
                    min_resolution=256,
                    return_count=True,
                )
                
                # albedo += cur_albedo
                # cnt += cur_cnt
                mask = cnt.squeeze(-1) < 0.1
                albedo[mask] += cur_albedo[mask]
                cnt[mask] += cur_cnt[mask]

            mask = cnt.squeeze(-1) > 0
            albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

            mask = mask.view(h, w)

            albedo = albedo.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            # dilate texture
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            inpaint_region = binary_dilation(mask, iterations=32)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
                search_coords
            )
            _, indices = knn.kneighbors(inpaint_coords)

            albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

            mesh.albedo = torch.from_numpy(albedo).to(self.device)
            mesh.write(path)

        else:
            #path = os.path.join(self.opt.outdir, self.opt.save_path + '_model' + str(index) + '.ply')
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_model.ply')
            self.renderer.gaussians.save_ply(path)

        print(f"[INFO] save model to {path}.")

    def register_dpg(self):
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                # input stuff
                def callback_select_input(sender, app_data):
                    # only one item
                    for k, v in app_data["selections"].items():
                        dpg.set_value("_log_input", k)
                        self.load_input(v)

                    self.need_update = True

                with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=callback_select_input,
                    file_count=1,
                    tag="file_dialog_tag",
                    width=700,
                    height=400,
                ):
                    dpg.add_file_extension("Images{.jpg,.jpeg,.png}")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="input",
                        callback=lambda: dpg.show_item("file_dialog_tag"),
                    )
                    dpg.add_text("", tag="_log_input")
                
                # overlay stuff
                with dpg.group(horizontal=True):

                    def callback_toggle_overlay_input_img(sender, app_data):
                        self.overlay_input_img = not self.overlay_input_img
                        self.need_update = True

                    dpg.add_checkbox(
                        label="overlay image",
                        default_value=self.overlay_input_img,
                        callback=callback_toggle_overlay_input_img,
                    )

                    def callback_set_overlay_input_img_ratio(sender, app_data):
                        self.overlay_input_img_ratio = app_data
                        self.need_update = True

                    dpg.add_slider_float(
                        label="ratio",
                        min_value=0,
                        max_value=1,
                        format="%.1f",
                        default_value=self.overlay_input_img_ratio,
                        callback=callback_set_overlay_input_img_ratio,
                    )

                # prompt stuff
            
                dpg.add_input_text(
                    label="prompt",
                    default_value=self.prompt,
                    callback=callback_setattr,
                    user_data="prompt",
                )

                dpg.add_input_text(
                    label="negative",
                    default_value=self.negative_prompt,
                    callback=callback_setattr,
                    user_data="negative_prompt",
                )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    def callback_save(sender, app_data, user_data):
                        self.save_model(mode=user_data)

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=callback_save,
                        user_data='model',
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    dpg.add_button(
                        label="geo",
                        tag="_button_save_mesh",
                        callback=callback_save,
                        user_data='geo',
                    )
                    dpg.bind_item_theme("_button_save_mesh", theme_button)

                    dpg.add_button(
                        label="geo+tex",
                        tag="_button_save_mesh_with_tex",
                        callback=callback_save,
                        user_data='geo+tex',
                    )
                    dpg.bind_item_theme("_button_save_mesh_with_tex", theme_button)

                    dpg.add_input_text(
                        label="",
                        default_value=self.opt.save_path,
                        callback=callback_setattr,
                        user_data="save_path",
                    )

            # training stuff
            with dpg.collapsing_header(label="Train", default_open=True):
                # lr and train button
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            self.prepare_train()
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")

                    # dpg.add_button(
                    #     label="init", tag="_button_init", callback=self.prepare_train
                    # )
                    # dpg.bind_item_theme("_button_init", theme_button)

                    dpg.add_button(
                        label="start", tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_time")
                    dpg.add_text("", tag="_log_train_log")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("image", "depth", "alpha"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )

                def callback_set_gaussain_scale(sender, app_data):
                    self.gaussain_scale_factor = app_data
                    self.need_update = True

                dpg.add_slider_float(
                    label="gaussain scale",
                    min_value=0,
                    max_value=1,
                    format="%.2f",
                    default_value=self.gaussain_scale_factor,
                    callback=callback_set_gaussain_scale,
                )

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="Gaussian3D",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()
    
    # no gui mode
    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()

            # CUSTOM
            pc_index = 0
            for i in tqdm.trange(iters):
                self.train_step()
                #CUSTOM
                if(pc_index == 50):
                    self.save_model(mode="model", index=i)
                    pc_index = 0
                pc_index += 1
            # do a last prune
            self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
        # save
        self.save_model(mode='model')
        self.save_model(mode='geo+tex')
        

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    # https://pavolkutaj.medium.com/how-to-attach-debugger-to-python-script-called-from-terminal-in-visual-studio-code-ddd377d99456
    #input("Press enter to start ... (this prompt enables attaching the Python DEBUGGER!)")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    parser.add_argument("--object_conf", required=True, help="path to the object's config file")
    args, extras = parser.parse_known_args()
    opt_object = OmegaConf.merge(OmegaConf.load(args.object_conf), OmegaConf.from_cli(extras))

    gui = GUI(opt, opt_object)

    if opt.gui:
        gui.render()
    else:
        gui.train(opt.iters)
