import os, sys
sys.path.append('../')

import glob
import argparse
import numpy as np
import torch




from gs_renderer import Renderer, MiniCam, BasicPointCloud, SH2RGB

from cam_utils import orbit_camera, OrbitCamera

# Helper for loading gsply files (e.g. from trellis)
# Handle correct capture of point of view to compare to our method
'''
parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='logs', type=str, help='Directory where obj files are stored')
parser.add_argument('--out', default='videos', type=str, help='Directory where videos will be saved')
args = parser.parse_args()

out = args.out
os.makedirs(out, exist_ok=True)

files = glob.glob(f'{args.dir}/*.obj')
for f in files:
    name = os.path.basename(f)
    # first stage model, ignore
    if name.endswith('_mesh.obj'): 
        continue
    print(f'[INFO] process {name}')
    os.system(f"python -m kiui.render {f} --save_video {os.path.join(out, name.replace('.obj', '.mp4'))} ")

'''

class GSPLY_Handler:

    def __init__(self, path, object_name, opt_object, opt, opt_alignment=None):
        self.renderer = Renderer(sh_degree=3, opt_object=opt_object, opt_alignment=opt_alignment)
        self.path = path
        self.object_name = object_name
        self.opt_object = opt_object
        self.opt = opt

        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

    def train_step(self, image_path=None, azimuth=None, iter_angle=None, elevation=0.0):
    
            
        ### novel view (manual batch)
        #render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
        render_resolution = 512#128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512) #self.opt_object.mv_dream_render_res#
        images = []
        colored_images = []
        colored_images_static = []
        colored_images_alpha = []
        colored_images_static_alpha = []
        masks = []

        static_images = []
        dynamic_images = []
        static_depth_images = []
        dynamic_depth_images = []
        poses = []
        vers, hors, radii = [], [], []
        # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
        min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
        max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)


        # render random view
        # CUSTOM maybe use not random but fixed angles per view ?
        #ver = np.random.randint(min_ver, max_ver)
        #hor = np.random.randint(0, 360)

        radius = 0.0#-1.25

        # CUSTOM
        #if (self.step == self.opt_object.max_steps):
            #pose = orbit_camera(10, 0, self.opt.radius + radius)
        ver = -20 + elevation
        #hor = self.opt_object.reference_angle_hor #0
        hor = azimuth #0
        pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
        
        # TODO maybe change hor to -180 to 180 for pose only ?
        #pose = orbit_camera(self.opt.elevation + ver, hor - 180, self.opt.radius + radius)
        #pose = orbit_camera(self.opt.elevation, hor, self.opt.radius + radius)
        #
        poses.append(pose)

        vers.append(ver) 
        # convert hor to hor 0 <= hor <= 360
        #############hor = 180 + (180 - abs(hor)) if hor < 0 else hor
        #hor = hor - 360 if hor > 360 else hor
        hors.append(hor)
        radii.append(radius)

        cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

        #bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
        #if (self.step == self.opt_object.max_steps):
            # use white for the last iteration
        bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

        # Custom
        out = None
        #if (self.step == 500):
        #    out = self.renderer.render(cur_cam, bg_color=bg_color)
        #else:
            # ONLY FOR DEBUG PURPOSE TODO remove in final version
        out = self.renderer.render(cur_cam, bg_color=bg_color, only_dynamic_splats=self.opt_object.only_dynamic_splats)
        
        # DEBUG render
        ##############
        pose_debug = orbit_camera(self.opt_object.reference_angle_v, int((360.0 / self.opt.iters) * 1 * 2.0), self.opt.radius + radius)
        cur_cam_debug = MiniCam(pose_debug, 1024, 1024, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
        out_debug = self.renderer.render(cur_cam_debug, bg_color=torch.tensor([1,1,1], dtype=torch.float32, device='cuda'), only_dynamic_splats=self.opt_object.only_dynamic_splats)
        # colorize static and dynamic gaussians
        static_color = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        dynamic_color = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device="cuda")
        debug_color_image = self.renderer.render(cur_cam_debug, bg_color=torch.tensor([1,1,1], dtype=torch.float32, device='cuda'), static_color=static_color, dynamic_color=dynamic_color, only_dynamic_splats=self.opt_object.only_dynamic_splats)
        # render only static part
        out_debug_col = self.renderer.render(cur_cam, bg_color=bg_color, static_color=static_color, dynamic_color=dynamic_color)
        out_debug_col_static = self.renderer.render(cur_cam, bg_color=bg_color, static_color=static_color, dynamic_color=dynamic_color, only_static_splats=True)
        ##############
        # DEBUG render end

        out_alpha = self.renderer.render(cur_cam, bg_color=bg_color, only_dynamic_splats=False)
        out_static_alpha = self.renderer.render(cur_cam, bg_color=bg_color, only_static_splats=True)

        '''
        static_points_image, dynamic_points_image, static_points_depth, dynamic_points_depth, static_points_alpha, dynamic_points_alpha = self.customLoss.GSRendererDepthBlending(self.renderer.gaussians, cur_cam, bg_color=bg_color, only_dynamic_splats=self.opt_object.only_dynamic_splats)
        static_images.append(torch.vstack((static_points_image, static_points_alpha)))
        dynamic_images.append(torch.vstack((dynamic_points_image, dynamic_points_alpha)))
        static_depth_images.append(static_points_depth)
        dynamic_depth_images.append(dynamic_points_depth)
        '''

        image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
        images.append(image)
        image_alpha = out_alpha["depth"].unsqueeze(0)#.unsqueeze(0).repeat_interleave(3, 1)
        image_static_alpha = out_static_alpha["depth"].unsqueeze(0)
        colored_image = out_debug_col["image"].unsqueeze(0)
        colored_image_alpha = out_debug_col["alpha"].unsqueeze(0)
        colored_image_static = out_debug_col_static["image"].unsqueeze(0)
        colored_image_static_alpha = out_debug_col_static["alpha"].unsqueeze(0)
        colored_images.append(colored_image)
        colored_images_static.append(colored_image_static)
        #colored_images_alpha.append(colored_image_alpha)
        #colored_images_static_alpha.append(colored_image_static_alpha)
        colored_images_alpha.append(image_alpha)
        colored_images_static_alpha.append(image_static_alpha)
        masks.append(colored_image_alpha)



        # enable mvdream training
        if self.opt.mvdream or self.opt.imagedream:
            for view_i in range(1, 4):
                # Custom
                #hor = np.random.randint(-180, 180)
                #ver = np.random.randint(min_ver, max_ver)

                # convert to (-180, 180) again
                #if (self.step % 3 == 0):
                    # same angle as first image
                #    hor_i = hor
                #else:
                hor_i = hor + 90 * view_i
                #hor_i = hor_i if hor_i < 180 else -180 + (hor_i - 180)
                hor_i = hor_i - 360 if hor_i > 360 else hor_i
                # modulo operator to get the actual value
                #hor_i = hor_i % 180
                #pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i, self.opt.radius + radius)
                pose_i = orbit_camera(self.opt.elevation + ver, hor_i, self.opt.radius + radius)
                # TODO change hor_i to -180 180 for pose_i only!
                #pose_i = orbit_camera(self.opt.elevation + ver, hor_i - 180, self.opt.radius + radius)
                #hors.append(hor + 90 * view_i)

                #hor_i = hor_i - 360 if hor_i > 360 else hor_i
                hors.append(hor_i)
                poses.append(pose_i)

                cur_cam_i = MiniCam(pose_i, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                # bg_color = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device="cuda")
                #if (self.step == 500):
                #    out_i = self.renderer.render(cur_cam_i, bg_color=bg_color)
                #else:
                    # ONLY FOR DEBUG PURPOSE TODO remove in final version
                out_i = self.renderer.render(cur_cam_i, bg_color=bg_color, only_dynamic_splats=self.opt_object.only_dynamic_splats)

                out_alpha = self.renderer.render(cur_cam_i, bg_color=bg_color, only_dynamic_splats=False)
                out_static_alpha = self.renderer.render(cur_cam_i, bg_color=bg_color, only_static_splats=True)

                #CUSTOM render Bounding Box to image
                #AABBimage = self.customLoss.AABBRender(self.AABB, cur_cam_i, self.opt.radius + radius)
                ######################################################################
                ########### dynamic, static point rendering ##########################
                ######################################################################
                '''
                static_points_image, dynamic_points_image, static_points_depth, dynamic_points_depth, static_points_alpha, dynamic_points_alpha = self.customLoss.GSRendererDepthBlending(self.renderer.gaussians, cur_cam_i, bg_color=bg_color, only_dynamic_splats=self.opt_object.only_dynamic_splats)
                static_images.append(torch.vstack((static_points_image, static_points_alpha)))
                dynamic_images.append(torch.vstack((dynamic_points_image, dynamic_points_alpha)))
                static_depth_images.append(static_points_depth)
                dynamic_depth_images.append(dynamic_points_depth)
                #AABBimage = self.customLoss.GSRendererDepthBlending(self.renderer.gaussians, cur_cam, bg_color=bg_color)
                #AABBimages.append(AABBimage)
                ######################################################################
                ######################################################################    
                # 
                '''
                out_debug_col = self.renderer.render(cur_cam_i, bg_color=bg_color, static_color=static_color, dynamic_color=dynamic_color)
                out_debug_col_static = self.renderer.render(cur_cam_i, bg_color=bg_color, static_color=static_color, dynamic_color=dynamic_color, only_static_splats=True)
                    

                image = out_i["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                images.append(image)
                image_alpha = out_alpha["depth"].unsqueeze(0)#.unsqueeze(0).repeat_interleave(3, 1)
                #print(image.shape)
                #print(image_alpha.shape)
                image_static_alpha = out_static_alpha["depth"].unsqueeze(0)
                colored_image = out_debug_col["image"].unsqueeze(0)
                colored_image_alpha = out_debug_col["alpha"].unsqueeze(0)
                colored_images.append(colored_image)
                masks.append(colored_image_alpha)
                #colored_images_alpha.append(colored_image_alpha)
                colored_image_static = out_debug_col_static["image"].unsqueeze(0)
                colored_image_static_alpha = out_debug_col_static["alpha"].unsqueeze(0)
                colored_images_static.append(colored_image_static)
                #colored_images_static_alpha.append(colored_image_static_alpha)
                colored_images_alpha.append(image_alpha)
                colored_images_static_alpha.append(image_static_alpha)


        #images = torch.cat(images, dim=0)
        #poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)     

        # 
        ####################################
        ########## write images ############
        ####################################
        import PIL.Image
        import torch.nn.functional as F
        import torchvision.transforms as T 

        
        img_width = 256 #input1.shape[2]
        img_height = 256 #input1.shape[3]

        inputs = torch.vstack((images))
        inputs2 = torch.vstack((colored_images_alpha)) #depth
        #inputs2 = torch.repeat_interleave(inputs2.unsqueeze(1), 3, 1)
        print(inputs.shape)
        print(inputs2.shape)
        inputs3 = torch.vstack((colored_images))
        inputs4 = torch.vstack((masks))
        # create figure
        figure = PIL.Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
        figure2 = PIL.Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
        figure3 = PIL.Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
        figure4 = PIL.Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
        #figure = PIL.Image.new('RGB', (512 * 4, 512 * 5), color=(255, 255, 255))

        
        inputs = F.interpolate((inputs), (img_width, img_height), mode="bilinear", align_corners=False)
        inputs2 = F.interpolate((inputs2), (img_width, img_height), mode="bilinear", align_corners=False)
        inputs3 = F.interpolate((inputs3), (img_width, img_height), mode="bilinear", align_corners=False)
        inputs4 = F.interpolate((inputs4), (img_width, img_height), mode="bilinear", align_corners=False)

        # transform depth to uint8
        inputs2 = F.interpolate((inputs2), (img_width, img_height), mode="bilinear", align_corners=False)
        inputs2 = (inputs2 - inputs2.min()) / (inputs2.max() - inputs2.min()) * 255
        inputs2 = inputs2.to(torch.uint8)
        #
        
        
        # add images
        transform = T.ToPILImage()
        image = transform(inputs[0])
        image2 = transform(inputs2[0])
        image3 = transform(inputs3[0])
        image4 = transform(inputs4[0])
        figure.paste(image, (0 * img_width, 0 * img_height))
        figure2.paste(image2, (0 * img_width, 0 * img_height))
        figure3.paste(image3, (0 * img_width, 0 * img_height))
        figure4.paste(image4, (0 * img_width, 0 * img_height))

        #image = transform(inputs[2])
        #image = transform(inputs[1])
        #figure.paste(image, (1 * img_width, 0 * img_height))

        try:
            #figure.save(r"debug/diffModelDebug" + str(string) + r".jpg")
            #dp = str(self.path)
            #name = dp.split('/')[-1] # get last element as name
            #figure.save(self.path + '/' + self.object_name + '_trellis' + '.png')
            angle = str(int(iter_angle))
            print(image_path + "_" + angle + "_color.png")
            figure.save(image_path + "_" + angle + "_color.png")
            figure2.save(image_path + "_" + angle + "_depth.png")
            figure4.save(image_path + "_" + angle + "_alpha.png")
            #figure3.save(image_path + "_" + angle + "_vis.png")
            #figure.save(dp + '/' + name + string + '_sdsonly.png')
        except OSError:
            print("Cannot save image")
        
        ####################################
        ######### write images end #########
        ####################################
                                                                                                                                                        
class test_capsule:

    def __init__(self, opt_object_path_prefix, opt_path, path, image_folder_path, image_suffix, object_suffix, *renderArgs):
        self.opt_object_path_prefix = opt_object_path_prefix
        self.opt_path = opt_path
        self.opt_object = None#OmegaConf.merge(OmegaConf.load(opt_object_path))
        self.opt = OmegaConf.merge(OmegaConf.load(opt_path))
        self.path = path
        self.object_path = "" # will be filled later
        self.object_suffix = object_suffix
        self.image_path = "" # --,,--
        self.image_folder_path = image_folder_path
        self.image_suffix = image_suffix
        self.data_handler = GSPLY_Handler(path, "", None, opt=self.opt)
        self.data_handler.initialize(renderArgs)

    def loadObjectConf(self, object):
        self.opt_object = OmegaConf.merge(OmegaConf.load(self.opt_object_path_prefix + str(object) + "/conf.yaml"))
        self.data_handler.opt_object = self.opt_object


if __name__ == "__main__":
    
    #test_objects = ["shoe", "couch_blender", "vase", "elephant", "hocker", "banana_tuna", "chicken", "plant", "pumpkins", "knife_block", "rubiks_cube", "headset", "leather_book", "hat", "sponge", "coffee_mug", "bread", "fish"]
    #
    #test_objects = ["bear", "bicycle", "bonsai", "garden_desk", "train", "truck"]
    test_objects = [""]
    test_partial_meshes = ["diner_seats", "flip_flop", "orc_warrior", "pixel_cat", "trumpet"]
    '''# static inputs
    for object in test_objects:
        from omegaconf import OmegaConf
        opt_object_path = "../data/" + str(object) + "/conf.yaml"
        opt_path = "../configs/text_mv.yaml"
        opt_object = OmegaConf.merge(OmegaConf.load(opt_object_path))
        opt = OmegaConf.merge(OmegaConf.load(opt_path))
        #config_path = "../data/metrics/baselines/trellis/configs/" + str(object) + "_config.yaml"
        #opt_alignment = OmegaConf.merge(OmegaConf.load(config_path))

        # load gsply
        path = "../data/BACKUPS/full_pipe/" + str(object) + "/" + str(object) # + "/" + str(object) + ".ply"
        object_path = "../data/BACKUPS/full_pipe/" + str(object) + "/" + str(object) + "_cropped.ply"
       
        gsply_handler = GSPLY_Handler(path, str(object), opt_object=opt_object, opt=opt)
        gsply_handler.renderer.initialize(False, object_path, no_transform=False, no_rotation=False, blob_init_size=0.0001, num_pts_init=1, flip_z=False, normalize=True)
        image_path = "../data/test_images/input/" + str(object) + "/" + str(object) + "_input"
        for i in range(0,8):
            azimuth = opt_object.reference_angle_hor + 45.0 * i
            iter_angle = 45.0 * i
            gsply_handler.train_step(image_path, azimuth, iter_angle) 
    '''

    # [isRawPCD, object_path, no_transform, no_rotation, blob_init_size, num_pts_init, flip_z, normalize]
    args = [False, "", False, False, 0.0001, 1, False, True]
    test_capsule_input = test_capsule("../data/BACKUPS/full_pipe/", "../configs/text_mv.yaml", "../data/BACKUPS/full_pipe/",
                                    "../data/test_images/input/", "_input", "_cropped.ply", *args)
    
    #False, object_path, no_transform=True, no_rotation=True, blob_init_size=0.0001, num_pts_init=1, flip_z=False, normalize=False
    args = [False, "", True, True, 0.0001, 1, False, False]
    test_capsule_full = test_capsule("../data/BACKUPS/full_pipe/", "../configs/text_mv.yaml", "../data/BACKUPS/full_pipe/",
                                    "../data/test_images/full/", "_full", "_final.ply" *args)
    


    test_capsules = [test_capsule_input]

    for tc_item in test_capsules:
        for object in test_partial_meshes:
            from omegaconf import OmegaConf
            tc_item.loadObjectConf(str(object) + "/conf.yaml")
            opt_path = tc_item.opt_object
            opt_object = tc_item.opt_object
            opt = tc_item.opt

            # load gsply
            tc_item.path = tc_item.path + str(object) + "/" + str(object) # + "/" + str(object) + ".ply"
            tc_item.object_path = tc_item.path + tc_item.object_suffix
            tc_item.data_handler.object_name = str(object)
            tc_item.image_path = tc_item.image_folder_path + str(object) + "/" + str(object) + tc_item.image_suffix
            for i in range(0,8):
                azimuth = tc_item.opt_object.reference_angle_hor + 45.0 * i
                iter_angle = 45.0 * i
                tc_item.data_handler.train_step(tc_item.image_path, azimuth, iter_angle) 

    # our test objects and capture from different angles
    # load gsply's and capture reference image
    '''
    for object in test_objects:
        from omegaconf import OmegaConf
        opt_object_path = "../data/" + str(object) + "/conf.yaml"
        opt_path = "../configs/text_mv.yaml"
        opt_object = OmegaConf.merge(OmegaConf.load(opt_object_path))
        opt = OmegaConf.merge(OmegaConf.load(opt_path))
        #config_path = "../data/metrics/baselines/trellis/configs/" + str(object) + "_config.yaml"
        #opt_alignment = OmegaConf.merge(OmegaConf.load(config_path))

        # load gsply
        path = "../data/BACKUPS/full_pipe/" + str(object) + "/" + str(object) # + "/" + str(object) + ".ply"
        object_path = "../data/BACKUPS/full_pipe/" + str(object) + "/" + str(object) + "_final.ply"
       
        gsply_handler = GSPLY_Handler(path, str(object), opt_object=opt_object, opt=opt)
        gsply_handler.renderer.initialize()
        image_path = "../data/test_images/full/" + str(object) + "/" + str(object) + "_full"
        for i in range(0,8):
            azimuth = opt_object.reference_angle_hor + 45.0 * i
            iter_angle = 45.0 * i
            gsply_handler.train_step(image_path, azimuth, iter_angle) 
    '''


    '''# ABLATION: no preserve loss
    for object in test_objects:
        from omegaconf import OmegaConf
        opt_object_path = "../data/" + str(object) + "/conf.yaml"
        opt_path = "../configs/text_mv.yaml"
        opt_object = OmegaConf.merge(OmegaConf.load(opt_object_path))
        opt = OmegaConf.merge(OmegaConf.load(opt_path))
        #config_path = "../data/metrics/baselines/trellis/configs/" + str(object) + "_config.yaml"
        #opt_alignment = OmegaConf.merge(OmegaConf.load(config_path))

        # load gsply
        path = "../data/BACKUPS/no_preserve_loss/" + str(object) + "/" + str(object) # + "/" + str(object) + ".ply"
        object_path = "../data/BACKUPS/no_preserve_loss/" + str(object) + "/" + str(object) + "_final.ply"
       
        gsply_handler = GSPLY_Handler(path, str(object), opt_object=opt_object, opt=opt)
        gsply_handler.renderer.initialize(False, object_path, no_transform=True, no_rotation=True, blob_init_size=0.0001, num_pts_init=1, flip_z=False, normalize=False)
        image_path = "../data/test_images/no_preserve_loss/" + str(object) + "/" + str(object) + "_no_preserve_loss"
        for i in range(0,8):
            azimuth = opt_object.reference_angle_hor + 45.0 * i
            iter_angle = 45.0 * i
            gsply_handler.train_step(image_path, azimuth, iter_angle) 
    '''

    '''# ABLATION: no preserve loss no init no schedule
    for object in test_objects:
        from omegaconf import OmegaConf
        opt_object_path = "../data/" + str(object) + "/conf.yaml"
        opt_path = "../configs/text_mv.yaml"
        opt_object = OmegaConf.merge(OmegaConf.load(opt_object_path))
        opt = OmegaConf.merge(OmegaConf.load(opt_path))
        #config_path = "../data/metrics/baselines/trellis/configs/" + str(object) + "_config.yaml"
        #opt_alignment = OmegaConf.merge(OmegaConf.load(config_path))

        # load gsply
        path = "../data/BACKUPS/no_preserve_no_init_no_schedule/" + str(object) + "/" + str(object) # + "/" + str(object) + ".ply"
        object_path = "../data/BACKUPS/no_preserve_no_init_no_schedule/" + str(object) + "/" + str(object) + "_final.ply"
       
        gsply_handler = GSPLY_Handler(path, str(object), opt_object=opt_object, opt=opt)
        gsply_handler.renderer.initialize(False, object_path, no_transform=True, no_rotation=True, blob_init_size=0.0001, num_pts_init=1, flip_z=False, normalize=False)
        image_path = "../data/test_images/no_preserve_no_init_no_schedule/" + str(object) + "/" + str(object) + "_no_preserve_no_init_no_schedule"
        for i in range(0,8):
            azimuth = opt_object.reference_angle_hor + 45.0 * i
            iter_angle = 45.0 * i
            gsply_handler.train_step(image_path, azimuth, iter_angle) 
    '''

    '''# ABLATION: no schedule
    for object in test_objects:
        from omegaconf import OmegaConf
        opt_object_path = "../data/" + str(object) + "/conf.yaml"
        opt_path = "../configs/text_mv.yaml"
        opt_object = OmegaConf.merge(OmegaConf.load(opt_object_path))
        opt = OmegaConf.merge(OmegaConf.load(opt_path))
        #config_path = "../data/metrics/baselines/trellis/configs/" + str(object) + "_config.yaml"
        #opt_alignment = OmegaConf.merge(OmegaConf.load(config_path))

        # load gsply
        path = "../data/BACKUPS/no_schedule/" + str(object) + "/" + str(object) # + "/" + str(object) + ".ply"
        object_path = "../data/BACKUPS/no_schedule/" + str(object) + "/" + str(object) + "_final.ply"
       
        gsply_handler = GSPLY_Handler(path, str(object), opt_object=opt_object, opt=opt)
        gsply_handler.renderer.initialize(False, object_path, no_transform=True, no_rotation=True, blob_init_size=0.0001, num_pts_init=1, flip_z=False, normalize=False)
        image_path = "../data/test_images/no_schedule/" + str(object) + "/" + str(object) + "_no_schedule"
        for i in range(0,8):
            azimuth = opt_object.reference_angle_hor + 45.0 * i
            iter_angle = 45.0 * i
            gsply_handler.train_step(image_path, azimuth, iter_angle) 
    '''
    
    
    
    # TRELLIS
    # load gsply's and capture reference image
    '''
    for object in test_objects:
        from omegaconf import OmegaConf
        opt_object_path = "../data/" + str(object) + "/conf.yaml"
        opt_path = "../configs/text_mv.yaml"
        opt_object = OmegaConf.merge(OmegaConf.load(opt_object_path))
        opt = OmegaConf.merge(OmegaConf.load(opt_path))
        config_path = "../data/metrics/baselines/trellis/configs/" + str(object) + "_config.yaml"
        opt_alignment = OmegaConf.merge(OmegaConf.load(config_path))

        # load gsply
        path = "../data/metrics/baselines/trellis/" + str(object) # + "/" + str(object) + ".ply"
        object_path = "../data/metrics/baselines/trellis/" + str(object) + "/" + str(object) + "_aligned" + ".ply"
       
        gsply_handler = GSPLY_Handler(path, str(object), opt_object=opt_object, opt=opt, opt_alignment=opt_alignment)
        gsply_handler.renderer.initialize(False, object_path, no_transform=True, no_rotation=False, blob_init_size=0.0001, num_pts_init=1, flip_z=False, normalize=False, transform_splats_only=True)

        image_path = "../data/test_images/baselines/trellis/" + str(object) + "/" + str(object) + "_trellis"
        for i in range(0,8):
            azimuth = opt_object.reference_angle_hor + 45.0 * i
            iter_angle = 45.0 * i
            gsply_handler.train_step(image_path, azimuth, iter_angle) 
    '''

    # TRELLIS MV
    # load gsply's and capture reference image
    '''
    for object in test_objects:
        from omegaconf import OmegaConf
        opt_object_path = "../data/" + str(object) + "/conf.yaml"
        opt_path = "../configs/text_mv.yaml"
        opt_object = OmegaConf.merge(OmegaConf.load(opt_object_path))
        opt = OmegaConf.merge(OmegaConf.load(opt_path))
        config_path = "../data/metrics/baselines/trellis/multiview/" + str(object) + "_config.yaml"
        opt_alignment = OmegaConf.merge(OmegaConf.load(config_path))

        # load gsply
        path = "../data/metrics/baselines/trellis/multiview/" + str(object) # + "/" + str(object) + ".ply"
        object_path = "../data/metrics/baselines/trellis/multiview/" + str(object) + "/" + str(object) + "_mv_aligned" + ".ply"
       
        gsply_handler = GSPLY_Handler(path, str(object), opt_object=opt_object, opt=opt, opt_alignment=opt_alignment)
        gsply_handler.renderer.initialize(False, object_path, no_transform=True, no_rotation=False, blob_init_size=0.0001, num_pts_init=1, flip_z=False, normalize=False, transform_splats_only=True)

        image_path = "../data/test_images/baselines/trellis_mv/" + str(object) + "/" + str(object) + "_trellis_mv"
        for i in range(0,8):
            azimuth = opt_object.reference_angle_hor + 45.0 * i
            iter_angle = 45.0 * i
            gsply_handler.train_step(image_path, azimuth, iter_angle) 
    '''


    # InstantMesh
    # load gsply's and capture reference image
    '''
    for object in test_objects:
        from omegaconf import OmegaConf
        opt_object_path = "../data/" + str(object) + "/conf.yaml"
        opt_path = "../configs/text_mv.yaml"
        opt_object = OmegaConf.merge(OmegaConf.load(opt_object_path))
        opt = OmegaConf.merge(OmegaConf.load(opt_path))
        config_path = "../data/metrics/baselines/instantmesh/" + str(object) + "/" + str(object) + "_config.yaml"
        opt_alignment = OmegaConf.merge(OmegaConf.load(config_path))

        # load gsply
        path = "../data/metrics/baselines/instantmesh/" + str(object) # + "/" + str(object) + ".ply"
        object_path = "../data/metrics/baselines/instantmesh/" + str(object) + "/" + str(object) + "_aligned" + ".ply"
       
        gsply_handler = GSPLY_Handler(path, str(object), opt_object=opt_object, opt=opt, opt_alignment=opt_alignment)
        gsply_handler.renderer.initialize(False, object_path, no_transform=True, no_rotation=False, blob_init_size=0.0001, num_pts_init=1, flip_z=False, normalize=False, transform_splats_only=True)

        image_path = "../data/test_images/baselines/instantmesh/" + str(object) + "/" + str(object) + "_instantmesh"
        for i in range(0,8):
            azimuth = opt_object.reference_angle_hor + 45.0 * i
            iter_angle = 45.0 * i
            gsply_handler.train_step(image_path, azimuth, iter_angle) 
    '''

    ''' limitations
    #for object in test_objects:
    object = "rubiks_cube"
    from omegaconf import OmegaConf
    opt_object_path = "../data/" + str(object) + "/conf.yaml"
    opt_path = "../configs/text_mv.yaml"
    opt_object = OmegaConf.merge(OmegaConf.load(opt_object_path))
    opt = OmegaConf.merge(OmegaConf.load(opt_path))
    #config_path = "../data/metrics/baselines/trellis/configs/" + str(object) + "_config.yaml"
    #opt_alignment = OmegaConf.merge(OmegaConf.load(config_path))

    # load gsply
    path = "../data/BACKUPS/limitations/" + str(object) + "/" + str(object) # + "/" + str(object) + ".ply"
    object_path = "../data/BACKUPS/limitations/" + str(object) + "/" + str(object) + "_final.ply"

    gsply_handler = GSPLY_Handler(path, str(object), opt_object=opt_object, opt=opt)
    gsply_handler.renderer.initialize(False, object_path, no_transform=True, no_rotation=True, blob_init_size=0.0001, num_pts_init=1, flip_z=False, normalize=False)
    image_path = "../data/test_images/limitations/" + str(object) + "/" + str(object) + "_limitation"
    for i in range(0,8):
        azimuth = opt_object.reference_angle_hor + 45.0 * i
        iter_angle = 45.0 * i
        gsply_handler.train_step(image_path, azimuth, iter_angle) 
    '''

    ''' limitations input
    #for object in test_objects:
    object = "rubiks_cube"
    from omegaconf import OmegaConf
    opt_object_path = "../data/" + str(object) + "/conf.yaml"
    opt_path = "../configs/text_mv.yaml"
    opt_object = OmegaConf.merge(OmegaConf.load(opt_object_path))
    opt = OmegaConf.merge(OmegaConf.load(opt_path))
    #config_path = "../data/metrics/baselines/trellis/configs/" + str(object) + "_config.yaml"
    #opt_alignment = OmegaConf.merge(OmegaConf.load(config_path))

    # load gsply
    path = "../data/BACKUPS/limitations_input/" + str(object) + "/" + str(object) # + "/" + str(object) + ".ply"
    object_path = "../data/BACKUPS/limitations_input/" + str(object) + "/" + str(object) + "_cropped_much.ply"

    gsply_handler = GSPLY_Handler(path, str(object), opt_object=opt_object, opt=opt)
    gsply_handler.renderer.initialize(False, object_path, no_transform=False, no_rotation=False, blob_init_size=0.0001, num_pts_init=1, flip_z=False, normalize=True)
    image_path = "../data/test_images/limitations_input/" + str(object) + "/" + str(object) + "_limitation_input"
    for i in range(0,8):
        azimuth = opt_object.reference_angle_hor + 45.0 * i
        iter_angle = 45.0 * i
        gsply_handler.train_step(image_path, azimuth, iter_angle) 
'''



    test_objects = ["plant"]
    ''' # VARIANCE
    for object in test_objects:
        from omegaconf import OmegaConf
        opt_object_path = "../data/" + str(object) + "/conf.yaml"
        opt_path = "../configs/text_mv.yaml"
        opt_object = OmegaConf.merge(OmegaConf.load(opt_object_path))
        opt = OmegaConf.merge(OmegaConf.load(opt_path))
        #config_path = "../data/metrics/baselines/trellis/configs/" + str(object) + "_config.yaml"
        #opt_alignment = OmegaConf.merge(OmegaConf.load(config_path))

        # load gsply
        path = "../data/BACKUPS/variance/12/" + str(object) + "/" + str(object) # + "/" + str(object) + ".ply"
        object_path = "../data/BACKUPS/variance/12/" + str(object) + "/" + str(object) + "_final.ply"
       
        gsply_handler = GSPLY_Handler(path, str(object), opt_object=opt_object, opt=opt)
        gsply_handler.renderer.initialize(False, object_path, no_transform=True, no_rotation=True, blob_init_size=0.0001, num_pts_init=1, flip_z=False, normalize=False)
        image_path = "../data/test_images/variance/12/" + str(object) + "/" + str(object) + "_variance12"
        for i in range(0,8):
            azimuth = opt_object.reference_angle_hor + 45.0 * i
            iter_angle = 45.0 * i
            gsply_handler.train_step(image_path, azimuth, iter_angle) 
    '''
    test_objects = ["plant"]
    ''' # VARIANCE
    for object in test_objects:
        from omegaconf import OmegaConf
        opt_object_path = "../data/" + str(object) + "/conf.yaml"
        opt_path = "../configs/text_mv.yaml"
        opt_object = OmegaConf.merge(OmegaConf.load(opt_object_path))
        opt = OmegaConf.merge(OmegaConf.load(opt_path))
        #config_path = "../data/metrics/baselines/trellis/configs/" + str(object) + "_config.yaml"
        #opt_alignment = OmegaConf.merge(OmegaConf.load(config_path))

        # load gsply
        path = "../data/BACKUPS/variance/333/" + str(object) + "/" + str(object) # + "/" + str(object) + ".ply"
        object_path = "../data/BACKUPS/variance/333/" + str(object) + "/" + str(object) + "_final.ply"
    
        gsply_handler = GSPLY_Handler(path, str(object), opt_object=opt_object, opt=opt)
        gsply_handler.renderer.initialize(False, object_path, no_transform=True, no_rotation=True, blob_init_size=0.0001, num_pts_init=1, flip_z=False, normalize=False)
        image_path = "../data/test_images/variance/333/" + str(object) + "/" + str(object) + "_variance333"
        for i in range(0,8):
            azimuth = opt_object.reference_angle_hor + 45.0 * i
            iter_angle = 45.0 * i
            gsply_handler.train_step(image_path, azimuth, iter_angle) 
    '''
    '''
    test_objects = ["plant"]
    for object in test_objects:
        from omegaconf import OmegaConf
        opt_object_path = "../data/" + str(object) + "/conf.yaml"
        opt_path = "../configs/text_mv.yaml"
        opt_object = OmegaConf.merge(OmegaConf.load(opt_object_path))
        opt = OmegaConf.merge(OmegaConf.load(opt_path))
        #config_path = "../data/metrics/baselines/trellis/configs/" + str(object) + "_config.yaml"
        #opt_alignment = OmegaConf.merge(OmegaConf.load(config_path))

        # load gsply
        path = "../data/BACKUPS/variance/1024/" + str(object) + "/" + str(object) # + "/" + str(object) + ".ply"
        object_path = "../data/BACKUPS/variance/1024/" + str(object) + "/" + str(object) + "_final.ply"
    
        gsply_handler = GSPLY_Handler(path, str(object), opt_object=opt_object, opt=opt)
        gsply_handler.renderer.initialize(False, object_path, no_transform=True, no_rotation=True, blob_init_size=0.0001, num_pts_init=1, flip_z=False, normalize=False)
        image_path = "../data/test_images/variance/1024/" + str(object) + "/" + str(object) + "_variance1024"
        for i in range(0,8):
            azimuth = opt_object.reference_angle_hor + 45.0 * i
            iter_angle = 45.0 * i
            gsply_handler.train_step(image_path, azimuth, iter_angle) 
    '''


    '''
    test_objects = ["hat"]
    for object in test_objects:
        from omegaconf import OmegaConf
        opt_object_path = "../data/" + str(object) + "/conf.yaml"
        opt_path = "../configs/text_mv.yaml"
        opt_object = OmegaConf.merge(OmegaConf.load(opt_object_path))
        opt = OmegaConf.merge(OmegaConf.load(opt_path))
        #config_path = "../data/metrics/baselines/trellis/configs/" + str(object) + "_config.yaml"
        #opt_alignment = OmegaConf.merge(OmegaConf.load(config_path))

        # load gsply
        path = "../data/BACKUPS/variance/643/" + str(object) + "/" + str(object) # + "/" + str(object) + ".ply"
        object_path = "../data/BACKUPS/variance/643/" + str(object) + "/" + str(object) + "_final.ply"
    
        gsply_handler = GSPLY_Handler(path, str(object), opt_object=opt_object, opt=opt)
        gsply_handler.renderer.initialize(False, object_path, no_transform=True, no_rotation=True, blob_init_size=0.0001, num_pts_init=1, flip_z=False, normalize=False)
        image_path = "../data/test_images/variance/643/" + str(object) + "/" + str(object) + "_variance643"
        for i in range(0,8):
            azimuth = opt_object.reference_angle_hor + 45.0 * i
            iter_angle = 45.0 * i
            gsply_handler.train_step(image_path, azimuth, iter_angle) 
    '''

    '''
    test_objects = ["hat"]
    for object in test_objects:
        from omegaconf import OmegaConf
        opt_object_path = "../data/" + str(object) + "/conf.yaml"
        opt_path = "../configs/text_mv.yaml"
        opt_object = OmegaConf.merge(OmegaConf.load(opt_object_path))
        opt = OmegaConf.merge(OmegaConf.load(opt_path))
        #config_path = "../data/metrics/baselines/trellis/configs/" + str(object) + "_config.yaml"
        #opt_alignment = OmegaConf.merge(OmegaConf.load(config_path))

        # load gsply
        path = "../data/BACKUPS/variance/23/" + str(object) + "/" + str(object) # + "/" + str(object) + ".ply"
        object_path = "../data/BACKUPS/variance/23/" + str(object) + "/" + str(object) + "_final.ply"
    
        gsply_handler = GSPLY_Handler(path, str(object), opt_object=opt_object, opt=opt)
        gsply_handler.renderer.initialize(False, object_path, no_transform=True, no_rotation=True, blob_init_size=0.0001, num_pts_init=1, flip_z=False, normalize=False)
        image_path = "../data/test_images/variance/23/" + str(object) + "/" + str(object) + "_variance23"
        for i in range(0,8):
            azimuth = opt_object.reference_angle_hor + 45.0 * i
            iter_angle = 45.0 * i
            gsply_handler.train_step(image_path, azimuth, iter_angle) 
    '''

    '''
    test_objects = ["coffee_mug"]
    for object in test_objects:
        from omegaconf import OmegaConf
        opt_object_path = "../data/" + str(object) + "/conf.yaml"
        opt_path = "../configs/text_mv.yaml"
        opt_object = OmegaConf.merge(OmegaConf.load(opt_object_path))
        opt = OmegaConf.merge(OmegaConf.load(opt_path))
        #config_path = "../data/metrics/baselines/trellis/configs/" + str(object) + "_config.yaml"
        #opt_alignment = OmegaConf.merge(OmegaConf.load(config_path))

        # load gsply
        path = "../data/BACKUPS/variance/424/" + str(object) + "/" + str(object) # + "/" + str(object) + ".ply"
        object_path = "../data/BACKUPS/variance/424/" + str(object) + "/" + str(object) + "_final.ply"
    
        gsply_handler = GSPLY_Handler(path, str(object), opt_object=opt_object, opt=opt)
        gsply_handler.renderer.initialize(False, object_path, no_transform=True, no_rotation=True, blob_init_size=0.0001, num_pts_init=1, flip_z=False, normalize=False)
        image_path = "../data/test_images/variance/424/" + str(object) + "/" + str(object) + "_variance424"
        for i in range(0,8):
            azimuth = opt_object.reference_angle_hor + 45.0 * i
            iter_angle = 45.0 * i
            gsply_handler.train_step(image_path, azimuth, iter_angle) 
    '''

    '''
    test_objects = ["coffee_mug"]
    for object in test_objects:
        from omegaconf import OmegaConf
        opt_object_path = "../data/" + str(object) + "/conf.yaml"
        opt_path = "../configs/text_mv.yaml"
        opt_object = OmegaConf.merge(OmegaConf.load(opt_object_path))
        opt = OmegaConf.merge(OmegaConf.load(opt_path))
        #config_path = "../data/metrics/baselines/trellis/configs/" + str(object) + "_config.yaml"
        #opt_alignment = OmegaConf.merge(OmegaConf.load(config_path))

        # load gsply
        path = "../data/BACKUPS/variance/860/" + str(object) + "/" + str(object) # + "/" + str(object) + ".ply"
        object_path = "../data/BACKUPS/variance/860/" + str(object) + "/" + str(object) + "_final.ply"
    
        gsply_handler = GSPLY_Handler(path, str(object), opt_object=opt_object, opt=opt)
        gsply_handler.renderer.initialize(False, object_path, no_transform=True, no_rotation=True, blob_init_size=0.0001, num_pts_init=1, flip_z=False, normalize=False)
        image_path = "../data/test_images/variance/860/" + str(object) + "/" + str(object) + "_variance860"
        for i in range(0,8):
            azimuth = opt_object.reference_angle_hor + 45.0 * i
            iter_angle = 45.0 * i
            gsply_handler.train_step(image_path, azimuth, iter_angle) 
    '''





































    ''' high elevation of banana_tuna
    test_objects = ["banana_tuna"]
    for object in test_objects:
        from omegaconf import OmegaConf
        opt_object_path = "../data/" + str(object) + "/conf.yaml"
        opt_path = "../configs/text_mv.yaml"
        opt_object = OmegaConf.merge(OmegaConf.load(opt_object_path))
        opt = OmegaConf.merge(OmegaConf.load(opt_path))
        #config_path = "../data/metrics/baselines/trellis/configs/" + str(object) + "_config.yaml"
        #opt_alignment = OmegaConf.merge(OmegaConf.load(config_path))

        # load gsply
        path = "../data/BACKUPS/full_pipe/" + str(object) + "/" + str(object) # + "/" + str(object) + ".ply"
        object_path = "../data/BACKUPS/full_pipe/" + str(object) + "/" + str(object) + "_final.ply"
    
        gsply_handler = GSPLY_Handler(path, str(object), opt_object=opt_object, opt=opt)
        gsply_handler.renderer.initialize(object_path, no_transform=True, no_rotation=True, blob_init_size=0.0001, num_pts_init=1, flip_z=False, normalize=False)
        image_path = "../data/test_images/high_elevation/" + str(object) + "/" + str(object) + "_high_elev"
        for i in range(0,8):
            azimuth = opt_object.reference_angle_hor + 45.0 * i
            iter_angle = 45.0 * i
            gsply_handler.train_step(image_path, azimuth, iter_angle, -10.0) 
    '''
