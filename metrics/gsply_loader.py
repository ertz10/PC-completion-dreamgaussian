import os, sys
sys.path.append('../')

import glob
import argparse
import numpy as np
import torch




from gs_renderer import Renderer, MiniCam, BasicPointCloud, SH2RGB
from mesh_renderer import Renderer as MeshRenderer

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

    def __init__(self, path, object_name, opt_object, opt, opt_alignment=None, mesh_path=None):
        self.renderer = None #Renderer(sh_degree=3, opt_object=opt_object, opt_alignment=opt_alignment)
        #self.reference_renderer = None
        self.mesh_renderer = None
        self.path = path
        self.object_name = object_name
        self.opt_object = opt_object
        self.opt = opt

        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

    def loadRenderer(self, opt_object=None, opt_alignment=None, *renderArgs):
        self.renderer = Renderer(sh_degree=3, opt_object=opt_object, opt_alignment=opt_alignment)
        self.renderer.initialize(*renderArgs)
        #self.reference_renderer = Renderer(sh_degree=3, opt_object=opt_object, opt_alignment=None)

    def loadMesh(self, mesh_path):
        self.mesh_renderer = MeshRenderer(self.opt, opt_object=self.opt_object, loadMesh=mesh_path).to('cuda')

    def train_step(self, image_path=None, azimuth=None, iter_angle=None, elevation=0.0, ref_depth_min=None, ref_depth_norm_fac=None):
    
        print(ref_depth_min)
        print(ref_depth_norm_fac)
        ### novel view (manual batch)
        #render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
        render_resolution = 512#128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512) #self.opt_object.mv_dream_render_res#
        images = []
        images_depth = []
        images_alpha = []
        #colored_images = []
        #colored_images_static = []
        #colored_images_alpha = []
        #colored_images_static_alpha = []
        #masks = []
        MeshImagesAlpha = []
        MeshImagesDepth = []
        MeshImages = []

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
        if self.mesh_renderer == None: 
            out = self.renderer.render(cur_cam, bg_color=bg_color, only_dynamic_splats=self.opt_object.only_dynamic_splats)
             

        ###### MESH RENDER #########
        else:
            ssaa = 1.0#min(2.0, max(0.125, 2 * np.random.random()))
            out_mesh = self.mesh_renderer.render(pose, self.cam.perspective, render_resolution, render_resolution, ssaa=ssaa, background=torch.tensor([1.0, 1.0, 1.0]))
            out_mesh_albedo = out_mesh['image'] # shape [H, W, C]
            out_mesh_albedo = torch.swapaxes(out_mesh_albedo, 1, 2)
            out_mesh_albedo = torch.swapaxes(out_mesh_albedo, 0, 1).unsqueeze(0)

            out_mesh_depth = out_mesh['depth']
            out_mesh_depth = torch.swapaxes(out_mesh_depth, 1, 2)
            out_mesh_depth = torch.swapaxes(out_mesh_depth, 0, 1).unsqueeze(0)

            out_mesh_alpha = out_mesh['alpha']
            out_mesh_alpha = torch.swapaxes(out_mesh_alpha, 1, 2)
            out_mesh_alpha = torch.swapaxes(out_mesh_alpha, 0, 1).unsqueeze(0)

            MeshImagesAlpha.append(out_mesh_alpha)
            MeshImagesDepth.append(out_mesh_depth)
            MeshImages.append(out_mesh_albedo)
            ############################
        
        # DEBUG render
        ##############
        '''
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
        '''

        if self.mesh_renderer == None:
            #out = self.renderer.render(cur_cam, bg_color=bg_color, only_dynamic_splats=False)
            #out_static_alpha = self.renderer.render(cur_cam, bg_color=bg_color, only_static_splats=True)

            '''
            static_points_image, dynamic_points_image, static_points_depth, dynamic_points_depth, static_points_alpha, dynamic_points_alpha = self.customLoss.GSRendererDepthBlending(self.renderer.gaussians, cur_cam, bg_color=bg_color, only_dynamic_splats=self.opt_object.only_dynamic_splats)
            static_images.append(torch.vstack((static_points_image, static_points_alpha)))
            dynamic_images.append(torch.vstack((dynamic_points_image, dynamic_points_alpha)))
            static_depth_images.append(static_points_depth)
            dynamic_depth_images.append(dynamic_points_depth)
            '''

            image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
            images.append(image)
            image_depth = out["depth"].unsqueeze(0)#.unsqueeze(0).repeat_interleave(3, 1)
            images_depth.append(image_depth)
            image_alpha = out["alpha"].unsqueeze(0)
            images_alpha.append(image_alpha)
            #image_static_alpha = out_static_alpha["depth"].unsqueeze(0)
            #colored_image = out_debug_col["image"].unsqueeze(0)
            #colored_image_alpha = out_debug_col["alpha"].unsqueeze(0)
            #colored_image_static = out_debug_col_static["image"].unsqueeze(0)
            #colored_image_static_alpha = out_debug_col_static["alpha"].unsqueeze(0)
            #colored_images.append(colored_image)
            #colored_images_static.append(colored_image_static)
            #colored_images_alpha.append(colored_image_alpha)
            #colored_images_static_alpha.append(colored_image_static_alpha)
            
            #colored_images_static_alpha.append(image_static_alpha)
            #masks.append(colored_image_alpha)



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
                out_i = None
                if self.mesh_renderer == None:
                    out_i = self.renderer.render(cur_cam_i, bg_color=bg_color, only_dynamic_splats=self.opt_object.only_dynamic_splats)

                    #out_alpha = self.renderer.render(cur_cam_i, bg_color=bg_color, only_dynamic_splats=False)
                    #out_static_alpha = self.renderer.render(cur_cam_i, bg_color=bg_color, only_static_splats=True)

                else:
                    ############## MESH images ##############
                    out_mesh_i = self.mesh_renderer.render(pose_i, self.cam.perspective, render_resolution, render_resolution, ssaa=ssaa, background=torch.tensor([1.0, 1.0, 1.0]))
                    out_mesh_i_albedo = out_mesh_i['image']
                    out_mesh_i_albedo = torch.swapaxes(out_mesh_i_albedo, 1, 2)
                    out_mesh_i_albedo = torch.swapaxes(out_mesh_i_albedo, 0, 1).unsqueeze(0)

                    out_mesh_i_depth = out_mesh_i['depth']
                    out_mesh_i_depth = torch.swapaxes(out_mesh_i_depth, 1, 2)
                    out_mesh_i_depth = torch.swapaxes(out_mesh_i_depth, 0, 1).unsqueeze(0)

                    out_mesh_i_alpha = out_mesh_i['alpha']
                    out_mesh_i_alpha = torch.swapaxes(out_mesh_i_alpha, 1, 2)
                    out_mesh_i_alpha = torch.swapaxes(out_mesh_i_alpha, 0, 1).unsqueeze(0)

                    MeshImagesAlpha.append(out_mesh_i_alpha)
                    MeshImagesDepth.append(out_mesh_i_depth)
                    MeshImages.append(out_mesh_i_albedo)
                    #########################################

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

                if self.mesh_renderer == None:
                    #out_debug_col = self.renderer.render(cur_cam_i, bg_color=bg_color, static_color=static_color, dynamic_color=dynamic_color)
                    #out_debug_col_static = self.renderer.render(cur_cam_i, bg_color=bg_color, static_color=static_color, dynamic_color=dynamic_color, only_static_splats=True)

                    image = out_i["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                    images.append(image)
                    image_depth = out_i["depth"].unsqueeze(0)#.unsqueeze(0).repeat_interleave(3, 1)
                    images_depth.append(image_depth)
                    image_alpha = out_i["alpha"].unsqueeze(0)
                    images_alpha.append(image_alpha)

                    #print(image.shape)
                    #print(image_alpha.shape)
                    #image_static_alpha = out_static_alpha["depth"].unsqueeze(0)
                    #colored_image = out_debug_col["image"].unsqueeze(0)
                    #colored_image_alpha = out_debug_col["alpha"].unsqueeze(0)
                    #colored_images.append(colored_image)
                    #masks.append(colored_image_alpha)
                    #colored_images_alpha.append(colored_image_alpha)
                    #colored_image_static = out_debug_col_static["image"].unsqueeze(0)
                    #colored_image_static_alpha = out_debug_col_static["alpha"].unsqueeze(0)
                    #colored_images_static.append(colored_image_static)
                    #colored_images_static_alpha.append(colored_image_static_alpha)
                    #colored_images_static_alpha.append(image_static_alpha)


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

        inputs = torch.vstack((images)) if self.mesh_renderer == None else torch.vstack((MeshImages))
        inputs2 = torch.vstack((images_depth)) if self.mesh_renderer == None else torch.vstack((MeshImagesDepth)) #depth
        #inputs2 = torch.repeat_interleave(inputs2.unsqueeze(1), 3, 1)
        inputs3 = torch.vstack((images_alpha)) if self.mesh_renderer == None else torch.vstack((MeshImagesAlpha))

        # create figure
        figure = PIL.Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
        figure2 = PIL.Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
        figure3 = PIL.Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
        
        inputs = F.interpolate((inputs), (img_width, img_height), mode="bilinear", align_corners=False)
        inputs2 = F.interpolate((inputs2), (img_width, img_height), mode="bilinear", align_corners=False)
        inputs3 = F.interpolate((inputs3), (img_width, img_height), mode="bilinear", align_corners=False)

        # transform depth to uint8
        #inputs2 = F.interpolate((inputs2), (img_width, img_height), mode="bilinear", align_corners=False)
        
        mask = inputs3 > 0.15
        #mask = inputs2 < 10.0
        print("MIN: " + str(inputs2[mask].min()))
        print("MAX: "  +str(inputs2[mask].max()))
        print("MEAN: "  +str(inputs2[mask].mean()))
        norm_factor = None
        ref_min = None
        if ref_depth_norm_fac == None and ref_depth_min == None:
            #norm_factor = (inputs2[inputs2<10.0].max() - inputs2[inputs2<10.0].min()) # * 255
            norm_factor = (inputs2.max() - inputs2.min()) # * 255
            ref_min = inputs2.min()
            #inputs2 = (inputs2 - inputs2.min()) / (inputs2.max() - inputs2.min()) * 255
            inputs2 = (inputs2 - ref_min) / norm_factor * 255
        else:
            inputs2 = (inputs2 - ref_depth_min) / ref_depth_norm_fac * 255
        # TODO use normalization factor of input for every object
        #inputs2[mask] = (inputs2[mask] - inputs2[mask].min()) / (inputs2[mask].max() - inputs2[mask].min()) * 255
        
        #inputs2 = inputs2 / inputs2.max() * 255
        #print("MIN: " + str(inputs2.min()))
        #print("MAX: " + str(inputs2.max()))
        inputs2 = inputs2.to(torch.uint8)
        #
        
        # add images
        transform = T.ToPILImage()
        image = transform(inputs[0])
        image2 = transform(inputs2[0])
        image3 = transform(inputs3[0])

        figure.paste(image, (0 * img_width, 0 * img_height))
        figure2.paste(image2, (0 * img_width, 0 * img_height))
        figure3.paste(image3, (0 * img_width, 0 * img_height))

        try:
            #figure.save(r"debug/diffModelDebug" + str(string) + r".jpg")
            #dp = str(self.path)
            #name = dp.split('/')[-1] # get last element as name
            #figure.save(self.path + '/' + self.object_name + '_trellis' + '.png')
            angle = str(int(iter_angle))
            figure.save(image_path + "_" + angle + "_color.png")
            figure2.save(image_path + "_" + angle + "_depth.png")
            figure3.save(image_path + "_" + angle + "_alpha.png")
        except OSError:
            print("Cannot save image")
        
        ####################################
        ######### write images end #########
        ####################################

        return ref_min, norm_factor # used for input only depth renderings
                                                                                                                                                        
