import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cv2 import erode
from scipy.ndimage import binary_erosion

from mvdream.camera_utils import get_camera, convert_opengl_to_blender, normalize_camera
from mvdream.model_zoo import build_model
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
#from .modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor
from mvdream.ldm.interface import extract_into_tensor
import torchvision.transforms.functional

#from guidance.diffusion_blending import DiffusionBlend

import torchvision
import torchvision.transforms as T 
from PIL import Image

import matplotlib.pyplot as plt

from diffusers import DDIMScheduler
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from torch import autocast

import cv2

class MVDream(nn.Module):
    def __init__(
        self,
        device,
        model_name='sd-v2.1-base-4view',
        ckpt_path=None,
        #t_range=[0.001, 0.98],
        t_range=[0.02, 0.98],
    ):
        super().__init__()

        self.device = device
        self.model_name = model_name
        self.ckpt_path = ckpt_path

        self.model = build_model(self.model_name, ckpt_path=self.ckpt_path).eval().to(self.device)
        self.model.device = device
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.dtype = torch.float32
        self.debug_step = 0
        self.train_steps = 0
        self.timelapse_imgs = []

        # CUSTOM
        #self.num_train_timesteps = 1000
        #TODO
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.all_steps = []

        self.noise = None

        self.latents_noisy = None

        self.embeddings = {}

        self.scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", subfolder="scheduler", torch_dtype=self.dtype
        )
        #CUSTOM
        #self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #   "stabilityai/stable-diffusion-2-inpainting", torch_dtype=self.dtype
        #)
        #self.scheduler = self.pipe.scheduler

        # CUSTOM
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts).repeat(4,1,1)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts).repeat(4,1,1)
        self.embeddings['pos'] = pos_embeds
        self.embeddings['neg'] = neg_embeds
    
    @torch.no_grad()
    def get_inverse_text_embeds(self, uncond, inverse_text):
        uncond_embeds = self.encode_text(uncond)
        inverse_embeds = self.encode_text(inverse_text)
        self.embeddings['uncond'] = uncond_embeds
        self.embeddings['inverse_text'] = inverse_embeds
    
    def encode_text(self, prompt):
        # prompt: [str]
        embeddings = self.model.get_learned_conditioning(prompt).to(self.device)
        return embeddings
    
    @torch.no_grad()
    def refine(self, pred_rgb, camera,
               guidance_scale=25, steps=50, strength=0.8, 
        customLoss=None,
        static_images=None,
        static_depth_images=None,
        current_cam_hors=[0, 0, 0, 0],
        captured_angles_hor=[0, 0]
        ):

        def batch_write_images_to_drive(input1, input2, input3, index=-1, string=""):

                import PIL.Image

                img_width = input1.shape[2]
                img_height = input1.shape[3]
                # create figure
                figure = PIL.Image.new('RGB', (img_width * 4, img_height * 3), color=(255, 255, 255))
                inputs = [input1, input2, input3]

                # add images
                for i in range(0,3):
                    for j, img in enumerate(inputs[i]):
                        transform = T.ToPILImage()
                        image = transform(img)
                        figure.paste(image, (j * img_width, i * img_height))
                

                figure.save(r"debug/refine_model_debug" + str(string) + ".jpg")

        batch_size = pred_rgb.shape[0]
        real_batch_size = batch_size // 4
        pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)

        '''
        ################# BLENDED DIFFUSION ? ##################################
        ########################################################################
        # CUSTOM
        img_width = static_images[0].shape[1]
        img_height = static_images[0].shape[2]
        if(img_width > 256):
            img_width = 256
        if(img_height > 256):
            img_height = 256
        # do blending with static part
        static_region = self.retrieve_gs_depth_static_only(static_images, static_depth_images, img_width, img_height)
        static_images = torch.stack((static_images))
        static_images = F.interpolate((static_images), (img_width, img_height), mode="bilinear", align_corners=False)
        static_images = static_images[:,:3]
        current_cam_hors = torch.FloatTensor(current_cam_hors)
        valid_cams = torch.logical_and(current_cam_hors >= captured_angles_hor[0], current_cam_hors <= captured_angles_hor[1]) #and current_cam_hors <= captured_angles_hor[1]
        valid_cams = valid_cams.nonzero() # get indices
        print("VALID CAMS: " + str(valid_cams))
        #latents_dec = self.decode_latents(latents)
        loss = 0
        if (valid_cams.shape[0] != 0):
            for valid_cam in valid_cams:
                #if(self.train_steps % 10 == 0):
                #    target = target
                #else:
                #target[valid_cam][static_region[valid_cam].int().bool()] = (static_images[valid_cam] * alphas_stat[valid_cam,:3])[static_region[valid_cam,:3].int().bool()]#static_depth[0,:3]
                #write_images_to_drive(static_region, string="_static_region")
                bool_mask = static_region[valid_cam].int().bool().squeeze(0)
                #alphas_stat = alphas_stat[:,:3]
                #loss += F.mse_loss(latents_dec[valid_cam, bool_mask], static_images[valid_cam, bool_mask], reduction='sum')
                #imgs[valid_cam, bool_mask] = static_images[valid_cam, bool_mask]
                pred_rgb_256[valid_cam, bool_mask] = static_images[valid_cam, bool_mask]
        ################# BLENDED DIFFUSION ? ##################################
        ########################################################################
        '''

        latents = self.encode_imgs(pred_rgb_256.to(self.dtype))
        # latents = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.dtype)

        # CUSTOM
        #self.scheduler.set_timesteps(steps)
        #steps *= 5
        self.scheduler.set_timesteps(steps)
        init_step = int(steps * strength)
        latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])

        camera = camera[:, [0, 2, 1, 3]] # to blender convention (flip y & z axis)
        camera[:, 1] *= -1
        camera = normalize_camera(camera).view(batch_size, 16)
        camera = camera.repeat(2, 1)

        embeddings = torch.cat([self.embeddings['neg'].repeat(real_batch_size, 1, 1), self.embeddings['pos'].repeat(real_batch_size, 1, 1)], dim=0)
        context = {"context": embeddings, "camera": camera, "num_frames": 4}

        # TODO: same technique as in train_step, but here maybe just overlay the static part, or we have to somehow fix the uvs in the static region
        
        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
    
            latent_model_input = torch.cat([latents] * 2)
            
            tt = torch.cat([t.unsqueeze(0).repeat(batch_size)] * 2).to(self.device)

            noise_pred = self.model.apply_model(latent_model_input, tt, context)

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # CUSTOM
        img_width = static_images[0].shape[1]
        img_height = static_images[0].shape[2]
        if(img_width > 256):
            img_width = 256
        if(img_height > 256):
            img_height = 256
        # do blending with static part
        static_region = self.retrieve_gs_depth_static_only(static_images, static_depth_images, img_width, img_height)
        static_images = torch.stack((static_images))
        static_images = F.interpolate((static_images), (img_width, img_height), mode="bilinear", align_corners=False)
        static_images = static_images[:,:3]
        current_cam_hors = torch.FloatTensor(current_cam_hors)
        valid_cams = torch.logical_and(current_cam_hors >= captured_angles_hor[0], current_cam_hors <= captured_angles_hor[1]) #and current_cam_hors <= captured_angles_hor[1]
        valid_cams = valid_cams.nonzero() # get indices
        print("VALID CAMS: " + str(valid_cams))
        latents_dec = self.decode_latents(latents)
        loss = 0
        if (valid_cams.shape[0] != 0):
            for valid_cam in valid_cams:
                #if(self.train_steps % 10 == 0):
                #    target = target
                #else:
                #target[valid_cam][static_region[valid_cam].int().bool()] = (static_images[valid_cam] * alphas_stat[valid_cam,:3])[static_region[valid_cam,:3].int().bool()]#static_depth[0,:3]
                #write_images_to_drive(static_region, string="_static_region")
                bool_mask = static_region[valid_cam].int().bool().squeeze(0)
                #alphas_stat = alphas_stat[:,:3]
                #loss += F.mse_loss(latents_dec[valid_cam, bool_mask], static_images[valid_cam, bool_mask], reduction='sum')
                imgs[valid_cam, bool_mask] = static_images[valid_cam, bool_mask]
                #pred_rgb_256[valid_cam, bool_mask] = static_images[valid_cam, bool_mask]

        
        #CUSTOM
        batch_write_images_to_drive(pred_rgb_256, imgs, static_images)
        return imgs, loss
    
    def retrieve_gs_depth_static_only(self, static_images, static_depth_images, img_width, img_height):
         with torch.no_grad():
            ######### ADD both binary masked images #############
            #imgs_blended = customLoss.blend_images(dynamic_images, static_images, dynamic_depth_images, static_depth_images)
            #dynamic_depth, static_depth = customLoss.blend_images(dynamic_images, static_images, dynamic_depth_images, static_depth_images)
            '''
            # TODO blend grad and static regions of rendering, so grad results in zero in those regions
            static_depth = torch.stack((static_depth)).detach()
            i0 = torch.vstack((static_depth[0], static_depth[0], static_depth[0], static_depth[0])).unsqueeze(0)
            i1 = torch.vstack((static_depth[1], static_depth[1], static_depth[1], static_depth[1])).unsqueeze(0)
            i2 = torch.vstack((static_depth[2], static_depth[2], static_depth[2], static_depth[2])).unsqueeze(0)
            i3 = torch.vstack((static_depth[3], static_depth[3], static_depth[3], static_depth[3])).unsqueeze(0)
            static_depth = torch.vstack((i0, i1, i2, i3))
            dynamic_depth = torch.stack((dynamic_depth)).detach()
            i0 = torch.vstack((dynamic_depth[0], dynamic_depth[0], dynamic_depth[0], dynamic_depth[0])).unsqueeze(0)
            i1 = torch.vstack((dynamic_depth[1], dynamic_depth[1], dynamic_depth[1], dynamic_depth[1])).unsqueeze(0)
            i2 = torch.vstack((dynamic_depth[2], dynamic_depth[2], dynamic_depth[2], dynamic_depth[2])).unsqueeze(0)
            i3 = torch.vstack((dynamic_depth[3], dynamic_depth[3], dynamic_depth[3], dynamic_depth[3])).unsqueeze(0)
            dynamic_depth = torch.vstack((i0, i1, i2, i3))
            inverted_static_depth = static_depth.clone().detach()
            inverted_static_depth[static_depth == 0] = 1.0
            inverted_static_depth[static_depth == 1] = 0.0
            ##################################################################
            ##################################################################

            ######################## BLENDED DIFFUSION #######################
            static_images = torch.stack((static_images)).detach()
            # TODO maybe instead of multiplying by alpha, gaussian blur the binary mask directly ?
            #static_images_interp = F.interpolate((static_images * static_depth), (256, 256), mode="bilinear", align_corners=False)[:, :3]

            # TODO add noisy static regions
            # TODO blend target with static image regions
            alphas_dyn = torch.stack((dynamic_images))[:, 3:].detach()
            alphas_dyn = torch.repeat_interleave(alphas_dyn, 4, dim=1)
            alphas_stat = static_images[:, 3:].detach()
            alphas_stat = F.interpolate((alphas_stat), (img_width, img_height), mode="bilinear", align_corners=False)
            alphas_stat = torch.repeat_interleave(alphas_stat, 4, dim=1)
            inverted_static_depth = F.interpolate((inverted_static_depth * 1.0), (img_width, img_height), mode="bilinear", align_corners=False)[:, :3]
            #inverted_static_depth = F.interpolate((inverted_static_depth * 1.0), (32, 32), mode="bilinear", align_corners=False)
            dynamic_depth = F.interpolate((dynamic_depth * 1.0), (img_width, img_height), mode="bilinear", align_corners=False)
            static_depth = F.interpolate((static_depth * 1.0), (img_width, img_height), mode="bilinear", align_corners=False)
            static_images = F.interpolate((static_images), (img_width, img_height), mode="bilinear", align_corners=False)[:, :3]
            #static_images = F.interpolate((static_images), (256, 256), mode="bilinear", align_corners=False)
            images_static = static_images[:, :3] * static_depth[:, :3]#F.interpolate((static_images * static_depth), (256, 256), mode="bilinear", align_corners=False)[:, :3]
            #images_static = static_images * static_depth #F.interpolate((static_images * static_depth), (256, 256), mode="bilinear", align_corners=False)[:, :3]
            ###################################################################
        
            #with torch.no_grad():
            '''
            static_images = torch.stack((static_images))
            alphas_stat = static_images[:, 3:]
            alphas_stat = F.interpolate((alphas_stat), (img_width, img_height), mode="bilinear", align_corners=False)
            alphas_stat = torch.repeat_interleave(alphas_stat, 4, dim=1)

            static_depth_images = torch.stack(static_depth_images)
            static_depth_images /= torch.max(static_depth_images) # to range [0, 1]
            # depth map is in weird range, so normalize it
            #mean = torch.mean(static_depth_images, dim=(1, 2, 3))
            #std = torch.std(static_depth_images, dim=(1, 2, 3))
            #transform = T.Compose([T.Normalize(mean, std)])
            #static_depth_images = transform(static_depth_images)
            static_depth_images = F.interpolate((static_depth_images * 1.0), (img_width, img_height), mode="bilinear", align_corners=False)
            alphas_stat = alphas_stat[:,:1]
            #static_depth_images *= alphas_stat
            static_depth_images = (static_depth_images < 1.0).bool().int() * 1.0 # invert binary mask
            static_region = (static_depth_images > 0.9).bool().int() * 1.0 # * alphas_stat
            static_region = torch.repeat_interleave((static_region), (3), dim=1)

            return static_region
    
    def retrieve_gs_depth_images(self, customLoss, static_images, dynamic_images, static_depth_images, dynamic_depth_images, img_width, img_height):
        with torch.no_grad():
            ######### ADD both binary masked images #############
            #imgs_blended = customLoss.blend_images(dynamic_images, static_images, dynamic_depth_images, static_depth_images)
            dynamic_depth, static_depth = customLoss.blend_images(dynamic_images, static_images, dynamic_depth_images, static_depth_images)

            # TODO blend grad and static regions of rendering, so grad results in zero in those regions
            static_depth = torch.stack((static_depth)).detach()
            i0 = torch.vstack((static_depth[0], static_depth[0], static_depth[0], static_depth[0])).unsqueeze(0)
            i1 = torch.vstack((static_depth[1], static_depth[1], static_depth[1], static_depth[1])).unsqueeze(0)
            i2 = torch.vstack((static_depth[2], static_depth[2], static_depth[2], static_depth[2])).unsqueeze(0)
            i3 = torch.vstack((static_depth[3], static_depth[3], static_depth[3], static_depth[3])).unsqueeze(0)
            static_depth = torch.vstack((i0, i1, i2, i3))
            dynamic_depth = torch.stack((dynamic_depth)).detach()
            i0 = torch.vstack((dynamic_depth[0], dynamic_depth[0], dynamic_depth[0], dynamic_depth[0])).unsqueeze(0)
            i1 = torch.vstack((dynamic_depth[1], dynamic_depth[1], dynamic_depth[1], dynamic_depth[1])).unsqueeze(0)
            i2 = torch.vstack((dynamic_depth[2], dynamic_depth[2], dynamic_depth[2], dynamic_depth[2])).unsqueeze(0)
            i3 = torch.vstack((dynamic_depth[3], dynamic_depth[3], dynamic_depth[3], dynamic_depth[3])).unsqueeze(0)
            dynamic_depth = torch.vstack((i0, i1, i2, i3))
            inverted_static_depth = static_depth.clone().detach()
            inverted_static_depth[static_depth == 0] = 1.0
            inverted_static_depth[static_depth == 1] = 0.0
            ##################################################################
            ##################################################################

            ######################## BLENDED DIFFUSION #######################
            static_images = torch.stack((static_images)).detach()
            # TODO maybe instead of multiplying by alpha, gaussian blur the binary mask directly ?
            #static_images_interp = F.interpolate((static_images * static_depth), (256, 256), mode="bilinear", align_corners=False)[:, :3]

            # TODO add noisy static regions
            # TODO blend target with static image regions
            alphas_dyn = torch.stack((dynamic_images))[:, 3:].detach()
            alphas_dyn = torch.repeat_interleave(alphas_dyn, 4, dim=1)
            alphas_stat = static_images[:, 3:].detach()
            alphas_stat = F.interpolate((alphas_stat), (img_width, img_height), mode="bilinear", align_corners=False)
            alphas_stat = torch.repeat_interleave(alphas_stat, 4, dim=1)
            inverted_static_depth = F.interpolate((inverted_static_depth * 1.0), (img_width, img_height), mode="bilinear", align_corners=False)[:, :3]
            #inverted_static_depth = F.interpolate((inverted_static_depth * 1.0), (32, 32), mode="bilinear", align_corners=False)
            dynamic_depth = F.interpolate((dynamic_depth * 1.0), (img_width, img_height), mode="bilinear", align_corners=False)
            static_depth = F.interpolate((static_depth * 1.0), (img_width, img_height), mode="bilinear", align_corners=False)
            static_images = F.interpolate((static_images), (img_width, img_height), mode="bilinear", align_corners=False)[:, :3]
            #static_images = F.interpolate((static_images), (256, 256), mode="bilinear", align_corners=False)
            images_static = static_images[:, :3] * static_depth[:, :3]#F.interpolate((static_images * static_depth), (256, 256), mode="bilinear", align_corners=False)[:, :3]
            #images_static = static_images * static_depth #F.interpolate((static_images * static_depth), (256, 256), mode="bilinear", align_corners=False)[:, :3]
            ###################################################################
        
            #with torch.no_grad():

            static_depth_images = torch.stack(static_depth_images)
            static_depth_images /= torch.max(static_depth_images) # to range [0, 1]
            # depth map is in weird range, so normalize it
            #mean = torch.mean(static_depth_images, dim=(1, 2, 3))
            #std = torch.std(static_depth_images, dim=(1, 2, 3))
            #transform = T.Compose([T.Normalize(mean, std)])
            #static_depth_images = transform(static_depth_images)
            static_depth_images = F.interpolate((static_depth_images * 1.0), (img_width, img_height), mode="bilinear", align_corners=False)
            alphas_stat = alphas_stat[:,:1]
            #static_depth_images *= alphas_stat
            static_depth_images = (static_depth_images < 1.0).bool().int() * 1.0 # invert binary mask
            static_region = (static_depth_images > 0.9).bool().int() * 1.0 # * alphas_stat
            static_region = torch.repeat_interleave((static_region), (3), dim=1)

            return static_region

    def train_step(
        self,
        pred_rgb, # [B, C, H, W], B is multiples of 4
        timelapse_img,
        camera, # [B, 4, 4]
        customLoss,
        step_ratio=None,
        guidance_scale=100,
        as_latent=False,
        dynamic_images=None,
        static_images=None,
        dynamic_depth_images=None,
        static_depth_images=None,
        current_cam_hors=[0, 0, 0, 0],
        captured_angles_hor=[0, 0],
        object_params=None,
        only_dynamic_splats=False,
    ):
        
        # CUSTOM
        def write_images_to_drive(input_tensor, index=-1, string=""):

                import PIL.Image

                img_width = input_tensor.shape[1]
                img_height = input_tensor.shape[2]
                # create figure
                figure = PIL.Image.new('RGB', (img_width, img_height), color=(255, 255, 255))

                # add images
                #for i, img in enumerate(input_tensor):
                transform = T.ToPILImage()
                image = transform(input_tensor)
                figure.paste(image, (0, 0))

                try:
                    #figure.save(r"debug/diffModelDebug" + str(string) + r".jpg")
                    figure.save(str(object_params.data_path) + "/" + str(string) + ".jpg")
                except OSError:
                    print("Cannot save image")

        #CUSTOM
        #self.base_ratio = 1.0 / 500.0 # 1 / overall iterations
        self.train_steps += 1
        #self.num_train_timesteps = 200
        #TODO BEWARE !!!! self.num_train_time_steps also has to be adjusted 
        # above in the init section!! since max and min steps depend on it!!
        #self.num_train_timesteps = 400# * (1.0 - step_ratio*4.0)
        #self.num_train_timesteps = 400
        #if self.train_steps % 2 == 0:
        #    guidance_scale = 75.0
        #if self.train_steps < 400:
        #    guidance_scale = 100.0
        #    step_ratio *= 0.5
        #if self.train_steps >= 400 and self.train_steps < 1000:
        #    guidance_scale = 100.0
        #    guidance_scale = 5.0
            #step_ratio *= 1.2
        #if self.train_steps >= 400 and self.train_steps <= 500:
        #    guidance_scale = 25.0
        #if self.train_steps >= 450:
        #    guidance_scale = 2
        #self.all_steps = np.append(self.all_steps, step_ratio) # collect all steps for graph plot

        # CUSTOM
        with torch.no_grad():
            img_width = static_images[0].shape[1]
            img_height = static_images[0].shape[2]
            if(img_width > 256):
                img_width = 256
            if(img_height > 256):
                img_height = 256
            #
        
        batch_size = pred_rgb.shape[0]
        real_batch_size = batch_size // 4
        pred_rgb = pred_rgb.to(self.dtype)

        if as_latent:
            latents = F.interpolate(pred_rgb, (32, 32), mode="bilinear", align_corners=False) * 2 - 1
        else:
            # interp to 256x256 to be fed into vae.
            pred_rgb_256 = F.interpolate(pred_rgb, (img_width, img_height), mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_256)

        t = 0
        '''
        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (real_batch_size,), dtype=torch.long, device=self.device).repeat(4)

        '''
        if step_ratio is not None:
            max_linear_anneal_iters = 700
            if self.train_steps <= max_linear_anneal_iters:
                # dreamtime-like
                #t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
                step_ratio = min(1, self.train_steps / max_linear_anneal_iters) # do not use too low "t" at the beginning
                t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
                #t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)
                t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            #elif self.train_steps > 700 and self.train_steps <= 900: # after gradADreamer
            #    step_ratio = min(1, self.train_steps / 900)
            #    t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            #    t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
                #guidance_scale *= 1.5
            elif self.train_steps > max_linear_anneal_iters and self.train_steps <= 1000: # after gradADreamer
                # t ~ U(0.02, 0.98)
                t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)
                #guidance_scale *= 2.0
            else:
                # t ~ U(0.02, 0.50) # after gradADreamer
                t = torch.randint(self.min_step, int(self.max_step * 0.5) + 1, (batch_size,), dtype=torch.long, device=self.device)
                #guidance_scale *= 2.0

        else:
            t = torch.randint(self.min_step, self.max_step + 1, (real_batch_size,), dtype=torch.long, device=self.device).repeat(4)


        # camera = convert_opengl_to_blender(camera)
        # flip_yz = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).unsqueeze(0)
        # camera = torch.matmul(flip_yz.to(camera), camera)
        camera = camera[:, [0, 2, 1, 3]] # to blender convention (flip y & z axis)
        camera[:, 1] *= -1
        camera = normalize_camera(camera).view(batch_size, 16)

        ###############
        # sampler = DDIMSampler(self.model)
        # shape = [4, 32, 32]
        # c_ = {"context": self.embeddings['pos']}
        # uc_ = {"context": self.embeddings['neg']}

        # # print(camera)

        # # camera = get_camera(4, elevation=0, azimuth_start=0)
        # # camera = camera.repeat(batch_size // 4, 1).to(self.device)

        # # print(camera)

        # c_["camera"] = uc_["camera"] = camera
        # c_["num_frames"] = uc_["num_frames"] = 4

        # latents_, _ = sampler.sample(S=30, conditioning=c_,
        #                                 batch_size=batch_size, shape=shape,
        #                                 verbose=False, 
        #                                 unconditional_guidance_scale=guidance_scale,
        #                                 unconditional_conditioning=uc_,
        #                                 eta=0, x_T=None)

        # # Img latents -> imgs
        # imgs = self.decode_latents(latents_)  # [4, 3, 256, 256]
        # import kiui
        # kiui.vis.plot_image(imgs)
        ###############

        with torch.no_grad():

            #static_region = self.retrieve_gs_depth_images(customLoss=customLoss, static_images=static_images,
            #                                             dynamic_images=dynamic_images, static_depth_images=static_depth_images,
            #                                             dynamic_depth_images=dynamic_depth_images, img_width=img_width, img_height=img_height)
            
            # CHECK if first camera angle is in a "good" angle, such that we can overlay the static part completely
            current_cam_hors = torch.FloatTensor(current_cam_hors)#
            print(current_cam_hors.shape)
            print(captured_angles_hor[0])
            #valid_cams = torch.ge(current_cam_hors, captured_angles_hor[0])
            valid_cams = np.zeros(4)
            if (captured_angles_hor[0] > captured_angles_hor[1]):
                for i in range(0, 4):
                    if (current_cam_hors[i] > captured_angles_hor[1]):
                        isTrue = current_cam_hors[i] >= captured_angles_hor[0]
                        if (isTrue):
                            valid_cams[i] = 1#np.append(valid_cams, i)
                    else:
                        valid_cams[i] = 1#np.append(valid_cams, i)
                valid_cams = torch.from_numpy(valid_cams)
            else:
                valid_cams = torch.logical_and(current_cam_hors >= captured_angles_hor[0], current_cam_hors <= captured_angles_hor[1]) #and current_cam_hors <= captured_angles_hor[1]
            valid_cams = valid_cams.nonzero() # get indices+
            #valid_cams = torch.FloatTensor(valid_cams)
            print("valid cams shape: " + str(valid_cams))
            print("VALID CAMS: " + str(valid_cams))
            print("Current cam HORS: " + str(current_cam_hors))
        
        # #################################################################

        camera = camera.repeat(2, 1)
        #mask_image = static_region
        embeddings = torch.cat([self.embeddings['neg'].repeat(real_batch_size, 1, 1), self.embeddings['pos'].repeat(real_batch_size, 1, 1)], dim=0)
        context = {"context": embeddings, "camera": camera, "num_frames": 4}

        #'''
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():

            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.model.q_sample(latents, t, noise)
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            # import kiui
            # kiui.lo(latent_model_input, t, context['context'], context['camera'])
            
            #noise_pred = self.model.apply_model(latent_model_input, tt, context)

            # PREDICT
            noise_pred = self.model.apply_model(latent_model_input, tt, context)

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            ###################################################################
            #'''
            # CFG rescale copied from pipeline_stable_diffusion.py
            if object_params.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = self.rescale_noise_cfg(noise_pred, noise_pred_cond, guidance_rescale=object_params.guidance_rescale)
            #####

        # CUSTOM
        with_recon_loss = False
        # paper reconstruction loss is equivalent to standard SDS formulation w * (noise_pred - noise)
        if with_recon_loss:
            #noise_pred_before = noise_pred.clone()
            # CFG RESCALING https://github.com/DSaurus/threestudio-mvdream/blob/main/guidance/mvdream_guidance.py
            #write_images_to_drive(self.decode_latents(self.model.predict_start_from_noise(latents_noisy, t, noise_pred)), string="_noise_pred_before_rescale")
            '''latents_recon = self.cfg_rescale(latents_noisy, noise_pred_pos, noise_pred, t)'''
            #write_images_to_drive(self.decode_latents(self.model.predict_start_from_noise(latents_noisy, t, noise_pred)), string="_noise_pred_after_rescale")
            #write_images_to_drive(self.decode_latents(self.model.predict_start_from_noise(latents_recon, t, noise_pred)), string="_noise_pred_after_rescale")

            # calculate loss
            '''
            loss = 0.5 * F.mse_loss(latents.float(), latents_recon.detach(), reduction="sum") / latents.shape[0]
            grad = torch.autograd.grad(loss, latents, retain_graph=True)[0]
            '''

            # just for debug
            target = (latents - grad).detach()
        else:
            # CUSTOM
            w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)
            
            # Original SDS
            #grad = w * 40.0 * (noise_pred - noise)
            #grad = w * (noise_pred - noise)
            grad = (noise_pred - noise)
            #grad = w * (noise_pred - noise) * (1.0 + step_ratio * 4.0)
            grad = torch.nan_to_num(grad)

            # seems important to avoid NaN...
            # grad = grad.clamp(-1, 1)

            # TODO i think this target calculation is the correct one!
            #target = (latents - grad).detach()

            # INVERSE of q_sample with predicted noise instead of random noise 
            #noise = default(noise, lambda: torch.randn_like(x_start))
            # x_t = a * x_0 + b * e, -> inverse: x_0 = (x_t - b * e) / a, where a = (extract_into_tensor(self.model.sqrt_alphas_cumprod, t, latents_noisy.shape), b = extract_into_tensor(self.model.sqrt_one_minus_alphas_cumprod, t, latents_noisy.shape)
            # target = x_0, latents_noisy = x_t, noise_pred = e
            target = (latents - grad).detach()
            #target = ((latents_noisy - extract_into_tensor(self.model.sqrt_one_minus_alphas_cumprod, t, latents_noisy.shape) * noise_pred) / (extract_into_tensor(self.model.sqrt_alphas_cumprod, t, latents_noisy.shape))).detach()

            #target = self.encode_imgs(target).detach()
            #latents = self.decode_latents(latents)
            #TODO use mask as target (everything except static part)
            #if (self.train_steps % 2 != 0):
            loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]
            #else:
            #    loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]
            #    loss -= 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]
            #loss = F.mse_loss(latents.float(), target, reduction='sum') * 1000.0

            # 2ND LOSS
            static_alpha = None
            bool_mask_images = np.zeros((4, 3, img_height, img_width))
            bool_mask_images = torch.tensor((bool_mask_images), dtype=torch.float32).detach()
            if (self.train_steps % 2 == 0 and only_dynamic_splats == False):
                #latents_dec = self.decode_latents(latents).detach()
                latents_dec = self.decode_latents(latents) # don't use detach() here !
                #target_dec = self.decode_latents(target).detach()
                with torch.no_grad():
                    static_images = torch.stack((static_images)).detach()
                    static_images = F.interpolate((static_images), (img_width, img_height), mode="bilinear", align_corners=False)
                    static_alpha = torch.repeat_interleave(static_images[:,3:], 3, 1)
                    static_images = static_images[:,:3]
                if (valid_cams.shape[0] != 0):
                    for valid_cam in valid_cams:
                        #if(self.train_steps % 10 == 0):
                        #    target = target
                        #else:
                        #target[valid_cam][static_region[valid_cam].int().bool()] = (static_images[valid_cam] * alphas_stat[valid_cam,:3])[static_region[valid_cam,:3].int().bool()]#static_depth[0,:3]
                        #write_images_to_drive(static_region, string="_static_region")
                        with torch.no_grad():
                            static_alpha = static_alpha > 0.5
                            #static_alpha = static_region
                            bool_mask = static_alpha[valid_cam].int()#static_region[valid_cam].int()
                            #write_images_to_drive(bool_mask.squeeze(0) * 1.0, string="mask")
                            #'''
                            kernel = np.ones((3, 3), dtype=np.float32)
                            kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0))
                            bool_mask = bool_mask[:, 0].unsqueeze(0).float().cpu()
                            #bool_mask = 1 - torch.clamp(torch.nn.functional.conv2d(1 - bool_mask, kernel_tensor, padding=(1,1)), 0, 1)
                            #bool_mask = 1 - torch.clamp(torch.nn.functional.conv2d(1 - bool_mask, kernel_tensor, padding=(1,1)), 0, 1)
                            #bool_mask = 1 - torch.clamp(torch.nn.functional.conv2d(1 - bool_mask, kernel_tensor, padding=(1,1)), 0, 1)
                            #write_images_to_drive(bool_mask.squeeze(0), string="mask_eroded")
                            bool_mask = bool_mask.squeeze(0).bool()
                            bool_mask = torch.repeat_interleave(bool_mask, 3, 0)
                            #bool_mask_images.append(bool_mask)
                            bool_mask_images[valid_cam] = bool_mask.float()
                            #'''
                            #bool_mask = bool_mask.squeeze(0).bool()
                        #alphas_stat = alphas_stat[:,:3]
                        # TODO reenable
                        #loss += 0.5 * F.mse_loss(latents_dec[valid_cam, bool_mask], static_images[valid_cam, bool_mask], reduction='sum') / latents_dec.shape[0]
                        loss += F.mse_loss(latents_dec[valid_cam, bool_mask], static_images[valid_cam, bool_mask], reduction='sum')
                        #loss += 0.5 * F.mse_loss(pred_rgb[valid_cam, bool_mask], static_images[valid_cam, bool_mask], reduction='sum') / latents_dec.shape[0]
                        #loss += 0.5 * F.mse_loss(target_dec[valid_cam, bool_mask], static_images[valid_cam, bool_mask], reduction='sum') / target_dec.shape[0]
                        #loss += F.mse_loss(latents_dec[valid_cam], static_images[valid_cam], reduction='sum')

            ############### Custom loss on features of dynamic splats to resemble existing static part #############
            #dynamic_images = latents
            #static_images = self.encode_imgs(static_images)
            #if (self.train_steps < 50):
            #latents_dec = self.decode_latents(latents)
            #loss += F.mse_loss(latents_dec.float(), static_images, reduction='sum') * 0.1
            
            

        #CUSTOM
        if object_params.DEBUG and self.train_steps % object_params.DEBUG_VIS_INTERVAL == 0:
            with torch.no_grad():
                self.debug_step += 1
                if self.debug_step % 1 == 0:
                        noisy_input = self.decode_latents(latents_noisy).detach()
                        #gs_renders = self.decode_latents(latents)
                        gs_renders = self.decode_latents(latents).detach()
                        ######TODO remove comment latent_output = self.decode_latents(self.model.predict_start_from_noise(latents_noisy_before_blended_diff, t, noise_pred_before_blended_diff)).detach().cpu()
                        latent_output = self.decode_latents(noise_pred - noise).detach()
                        #blended_output = self.decode_latents(self.model.predict_start_from_noise(latents_noisy, t, noise_pred))
                        blended_output = self.decode_latents(self.model.predict_start_from_noise(latents_noisy, t, noise_pred)).detach()
                        target_debug = self.decode_latents(target).detach()
                        
                        #batch_write_images_to_drive(noisy_input, gs_renders, target_debug, latent_output, blended_output, string=r"_batch_debug")
                        self.batch_write_images_to_drive(noisy_input, gs_renders, target_debug, latent_output, bool_mask_images, timelapse_img, object_params, string=r"_batch_debug")
                        #write_images_to_drive(static_region, string="_static_depth_images")
                        
                        print("good Horizontal angles: " + str(current_cam_hors))
                        print("guidance scale (mvdream): ", guidance_scale)
                        print("t: ", t[0])
                        print("num timesteps ", self.num_train_timesteps)
                        #print(torch.cuda.memory_summary())
                        self.debug_step = 0
        

        return loss
    
    # copied from pipeline_stable_diffusion.py
    def rescale_noise_cfg(self, noise_cfg, noise_pred_text, guidance_rescale=0.0):
        """
        Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        return noise_cfg
    
    # CUSTOM
    def batch_write_images_to_drive(self, input1, input2, input3, input4, input5, timelapse_img, object_params, index=-1, string=""):

            import PIL.Image

            img_width = input1.shape[2]
            img_height = input1.shape[3]
            # create figure
            figure = PIL.Image.new('RGB', (img_width * 4, img_height * 5), color=(255, 255, 255))
            #figure = PIL.Image.new('RGB', (512 * 4, 512 * 5), color=(255, 255, 255))
            inputs = [input1, input2, input3, input4, input5]

            #inputs = F.interpolate((inputs), (512, 512), mode="bilinear", align_corners=False)

            # add images
            for i in range(0,5):
                for j, img in enumerate(inputs[i]):
                    transform = T.ToPILImage()
                    image = transform(img)
                    figure.paste(image, (j * img_width, i * img_height))

            #figure2 = PIL.Image.new('RGB', (512 * 4, 512), color=(255, 255, 255))
            figure2 = PIL.Image.new('RGB', (1024, 1024), color=(255, 255, 255))
            if self.train_steps % 2 == 0:
                #for j, img in enumerate(inputs[1]):
                #for j, img in enumerate(inputs[1]):
                    #img = F.interpolate(img.unsqueeze(0), (512, 512), mode="bilinear", align_corners=False)
                img = F.interpolate(timelapse_img, (1024, 1024), mode="bilinear", align_corners=False)
                transform = T.ToPILImage()
                image = transform(img[0])
                figure2.paste(image, (0, 0))
                self.timelapse_imgs.append(figure2)

            try:
                #figure.save(r"debug/diffModelDebug" + str(string) + r".jpg")
                figure.save(str(object_params.data_path) + '/diffModelDebug.jpg')
                if (self.train_steps == object_params.max_steps):
                    #for x in range(0, 30):
                    #    self.timelapse_imgs.append(self.timelapse_imgs[len(self.timelapse_imgs) - 1])
                    out = cv2.VideoWriter(str(object_params.data_path) + '/timelapse_MVDREAM_coarse.mp4', cv2.VideoWriter.fourcc(*'mp4v'), 30, (1024, 1024))
                    #ext_frames = np.repeat(self.timelapse_imgs, 60)
                    for frame in self.timelapse_imgs:
                        out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
                    out.release()
                    self.timelapse_imgs[0].save(str(object_params.data_path) + '/timelapseDebug_MVDREAM_coarse.gif', save_all=True, append_images=self.timelapse_imgs[1:], optimize=False, duration=250, loop=0)
            except OSError:
                print("Cannot save image")
         

    def decode_latents(self, latents):
        imgs = self.model.decode_first_stage(latents)
        imgs = ((imgs + 1) / 2).clamp(0, 1)
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, 256, 256]
        imgs = 2 * imgs - 1
        latents = self.model.get_first_stage_encoding(self.model.encode_first_stage(imgs))
        return latents # [B, 4, 32, 32]

    @torch.no_grad()
    def prompt_to_img(
        self,
        prompts,
        negative_prompts="",
        height=256,
        width=256,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
        elevation=0,
        azimuth_start=0,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
        
        batch_size = len(prompts) * 4

        # Text embeds -> img latents
        sampler = DDIMSampler(self.model)
        shape = [4, height // 8, width // 8]
        c_ = {"context": self.encode_text(prompts).repeat(4,1,1)}
        uc_ = {"context": self.encode_text(negative_prompts).repeat(4,1,1)}

        camera = get_camera(4, elevation=elevation, azimuth_start=azimuth_start)
        camera = camera.repeat(batch_size // 4, 1).to(self.device)

        c_["camera"] = uc_["camera"] = camera
        c_["num_frames"] = uc_["num_frames"] = 4

        latents, _ = sampler.sample(S=num_inference_steps, conditioning=c_,
                                        batch_size=batch_size, shape=shape,
                                        verbose=False, 
                                        unconditional_guidance_scale=guidance_scale,
                                        unconditional_conditioning=uc_,
                                        eta=0, x_T=None)

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [4, 3, 256, 256]
        
        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--negative", default="", type=str)
    parser.add_argument("--steps", type=int, default=30)
    opt = parser.parse_args()

    device = torch.device("cuda")

    sd = MVDream(device)

    while True:
        imgs = sd.prompt_to_img(opt.prompt, opt.negative, num_inference_steps=opt.steps)

        grid = np.concatenate([
            np.concatenate([imgs[0], imgs[1]], axis=1),
            np.concatenate([imgs[2], imgs[3]], axis=1),
        ], axis=0)

        # visualize image
        plt.imshow(grid)
        plt.show()
