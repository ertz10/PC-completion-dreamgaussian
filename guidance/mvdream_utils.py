import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mvdream.camera_utils import get_camera, convert_opengl_to_blender, normalize_camera
from mvdream.model_zoo import build_model
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
import torchvision.transforms.functional

#from guidance.diffusion_blending import DiffusionBlend

import torchvision
import torchvision.transforms as T 
from PIL import Image

import matplotlib.pyplot as plt

from diffusers import DDIMScheduler
from diffusers import StableDiffusionPipeline
from torch import autocast

class MVDream(nn.Module):
    def __init__(
        self,
        device,
        model_name='sd-v2.1-base-4view',
        ckpt_path=None,
        t_range=[0.001, 0.98],
        #t_range=[0.02, 0.98],
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

        # CUSTOM
        #self.num_train_timesteps = 1000
        #TODO
        self.num_train_timesteps = 800
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.all_steps = []

        self.embeddings = {}

        self.scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", subfolder="scheduler", torch_dtype=self.dtype
        )

        # CUSTOM
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts).repeat(4,1,1)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts).repeat(4,1,1)
        self.embeddings['pos'] = pos_embeds
        self.embeddings['neg'] = neg_embeds
    
    def encode_text(self, prompt):
        # prompt: [str]
        embeddings = self.model.get_learned_conditioning(prompt).to(self.device)
        return embeddings
    
    @torch.no_grad()
    def refine(self, pred_rgb, camera,
               guidance_scale=20, steps=100, strength=0.8,
        ):

        def batch_write_images_to_drive(input1, input2, index=-1, string=""):

                import PIL.Image

                img_width = input1.shape[2]
                img_height = input1.shape[3]
                # create figure
                figure = PIL.Image.new('RGB', (img_width * 4, img_height * 2), color=(255, 255, 255))
                inputs = [input1, input2]

                # add images
                for i in range(0,2):
                    for j, img in enumerate(inputs[i]):
                        transform = T.ToPILImage()
                        image = transform(img)
                        figure.paste(image, (j * img_width, i * img_height))
                

                figure.save("debug/refine_model_debug" + str(string) + ".jpg")

        batch_size = pred_rgb.shape[0]
        real_batch_size = batch_size // 4
        pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
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

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
    
            latent_model_input = torch.cat([latents] * 2)
            
            tt = torch.cat([t.unsqueeze(0).repeat(batch_size)] * 2).to(self.device)

            noise_pred = self.model.apply_model(latent_model_input, tt, context)

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents) # [1, 3, 512, 512]
        #CUSTOM
        batch_write_images_to_drive(pred_rgb_256, imgs)
        return imgs

    def train_step(
        self,
        pred_rgb, # [B, C, H, W], B is multiples of 4
        camera, # [B, 4, 4]
        customLoss,
        step_ratio=None,
        guidance_scale=100, #8 had good results so far, without normalization
        as_latent=False,
        dynamic_images=None,
        static_images=None,
        dynamic_depth_images=None,
        static_depth_images=None
    ):
        
        # CUSTOM
        def write_images_to_drive(input_tensor, index=-1, string=""):

                import PIL.Image

                img_width = input_tensor.shape[2]
                img_height = input_tensor.shape[3]
                # create figure
                figure = PIL.Image.new('RGB', (img_width * 4, img_height), color=(255, 255, 255))

                # add images
                for i, img in enumerate(input_tensor):
                    transform = T.ToPILImage()
                    image = transform(img)
                    figure.paste(image, (i * img_width, 0))

                figure.save("debug/diff_model_debug" + str(string) + ".jpg")
                
        # CUSTOM
        def batch_write_images_to_drive(input1, input2, input3, input4, input5, index=-1, string=""):

                import PIL.Image

                img_width = input1.shape[2]
                img_height = input1.shape[3]
                # create figure
                figure = PIL.Image.new('RGB', (img_width * 4, img_height * 5), color=(255, 255, 255))
                inputs = [input1, input2, input3, input4, input5]

                # add images
                for i in range(0,5):
                    for j, img in enumerate(inputs[i]):
                        transform = T.ToPILImage()
                        image = transform(img)
                        figure.paste(image, (j * img_width, i * img_height))
                

                figure.save("debug/diff_model_debug" + str(string) + ".jpg")

        #CUSTOM
        #self.base_ratio = 1.0 / 500.0 # 1 / overall iterations
        self.train_steps += 1
        self.num_train_timesteps = 200
        #TODO BEWARE !!!! self.num_train_time_steps also has to be adjusted 
        # above in the init section!! since max and min steps depend on it!!
        self.num_train_timesteps = 800# * (1.0 - step_ratio*4.0)
        #self.num_train_timesteps = 400
        #if self.train_steps < 150:
        #    step_ratio *= 0.5
        #if self.train_steps >= 150 and self.train_steps < 300:
        #    guidance_scale = 5.0
            #step_ratio *= 1.2
        #if self.train_steps >= 400 and self.train_steps <= 500:
        #    guidance_scale = 3.0
        #if self.train_steps >= 450:
        #    guidance_scale = 2
        #self.all_steps = np.append(self.all_steps, step_ratio) # collect all steps for graph plot

        
        batch_size = pred_rgb.shape[0]
        real_batch_size = batch_size // 4
        pred_rgb = pred_rgb.to(self.dtype)

        if as_latent:
            latents = F.interpolate(pred_rgb, (32, 32), mode="bilinear", align_corners=False) * 2 - 1
        else:
            # interp to 256x256 to be fed into vae.
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            ################ CUSTOM gaussian blurring ? ################
            #gaussian_kernel_size = 1 if (142 - (self.train_steps * 2 - 1)) < 1 else (142 - (self.train_steps * 2 - 1))
            #xp = np.arange(600)
            #fp = np.arange(200)
            #for i in range(0, 200):
            #    if(fp[i] % 2 == 1):
            #        fp[i] = i # is odd
            #    else:
            #        fp[i] = i - 1 # when even
            #fp[0] = 1
            # truncate
            #step = np.floor(200/64).astype(int)
            #fp = fp[0:64]
            # repeat each element by step
            #fp = np.repeat(fp, step)
            #gaussian_kernel_size = 1#64 - fp[self.train_steps] if self.train_steps < 190 else 1
            #pred_rgb_256 = torchvision.transforms.functional.gaussian_blur(pred_rgb_256, int(gaussian_kernel_size))
            ############################################################
            latents = self.encode_imgs(pred_rgb_256)

        t = 0
        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            #TODO adjust t
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
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

        camera = camera.repeat(2, 1)
        embeddings = torch.cat([self.embeddings['neg'].repeat(real_batch_size, 1, 1), self.embeddings['pos'].repeat(real_batch_size, 1, 1)], dim=0)
        context = {"context": embeddings, "camera": camera, "num_frames": 4}

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            #CUSTOM
            #t = (t/10).to(torch.int64)
            # GOOD resource https://colab.research.google.com/drive/1dlgggNa5Mz8sEAGU0wFCHhGLFooW_pf1#scrollTo=pj33ZTHKUYIx
            # https://www.reddit.com/r/StableDiffusion/comments/xalo78/fixing_excessive_contrastsaturation_resulting/ #
            #################### TODO try some normalization to tackle high SD contrast ? #####################
            #latents = latents / (latents.max() / 2.0)
            ###################################################################################################
            latents_noisy = self.model.q_sample(latents, t, noise)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            ######### ADD both binary masked images #############
            #imgs_blended = customLoss.blend_images(dynamic_images, static_images, dynamic_depth_images, static_depth_images)
            dynamic_depth, static_depth = customLoss.blend_images(dynamic_images, static_images, dynamic_depth_images, static_depth_images)

            #write_images_to_drive(torch.stack((dynamic_images))[:, 3:], string="_depth_compare")

            # TODO blend grad and static regions of rendering, so grad results in zero in those regions
            ######TODO remove comment static_depth = torch.stack((static_depth))
            ######TODO remove comment i0 = torch.vstack((static_depth[0], static_depth[0], static_depth[0], static_depth[0])).unsqueeze(0)
            ######TODO remove comment i1 = torch.vstack((static_depth[1], static_depth[1], static_depth[1], static_depth[1])).unsqueeze(0)
            ######TODO remove comment i2 = torch.vstack((static_depth[2], static_depth[2], static_depth[2], static_depth[2])).unsqueeze(0)
            ######TODO remove comment i3 = torch.vstack((static_depth[3], static_depth[3], static_depth[3], static_depth[3])).unsqueeze(0)
            ######TODO remove comment static_depth = torch.vstack((i0, i1, i2, i3))
            ######TODO remove comment dynamic_depth = torch.stack((dynamic_depth))
            ######TODO remove comment i0 = torch.vstack((dynamic_depth[0], dynamic_depth[0], dynamic_depth[0], dynamic_depth[0])).unsqueeze(0)
            ######TODO remove comment i1 = torch.vstack((dynamic_depth[1], dynamic_depth[1], dynamic_depth[1], dynamic_depth[1])).unsqueeze(0)
            ######TODO remove comment i2 = torch.vstack((dynamic_depth[2], dynamic_depth[2], dynamic_depth[2], dynamic_depth[2])).unsqueeze(0)
            ######TODO remove comment i3 = torch.vstack((dynamic_depth[3], dynamic_depth[3], dynamic_depth[3], dynamic_depth[3])).unsqueeze(0)
            ######TODO remove comment dynamic_depth = torch.vstack((i0, i1, i2, i3))
            ######TODO remove comment inverted_static_depth = static_depth.clone().detach()
            ######TODO remove comment inverted_static_depth[static_depth == 0] = 1.0
            ######TODO remove comment inverted_static_depth[static_depth == 1] = 0.0
            #####################################################
            ####################

            # import kiui
            # kiui.lo(latent_model_input, t, context['context'], context['camera'])
            
            #noise_pred = self.model.apply_model(latent_model_input, tt, context)

            ######################## BLENDED DIFFUSION #######################
            ######TODO remove comment static_images = torch.stack((static_images))
            # TODO maybe instead of multiplying by alpha, gaussian blur the binary mask directly ?
            #inverted_static_depth_interp = F.interpolate((inverted_static_depth), (256, 256), mode="bilinear", align_corners=False)[:, :3]
            #static_images_interp = F.interpolate((static_images * static_depth), (256, 256), mode="bilinear", align_corners=False)[:, :3]
            ######TODO remove comment static_images = F.interpolate((static_images), (256, 256), mode="bilinear", align_corners=False)[:, :3]
            ######TODO remove comment static_images = self.encode_imgs(static_images)
            ######TODO remove comment latents_noisy_static = self.model.q_sample(static_images, t, noise)
            #latent_model_input_static = torch.cat([latents_noisy_static] * 2)

            # TODO add noisy static regions
            # TODO blend target with static image regions
            ######TODO remove comment alphas = torch.stack((dynamic_images))[:, 3:].detach()
            ######TODO remove comment alphas = torch.repeat_interleave(alphas, 4, dim=1)
            # TODO maybe instead of multiplying by alpha, gaussian blur the binary mask directly ?
            ######TODO remove comment inverted_static_depth = F.interpolate((inverted_static_depth * 1.0), (256, 256), mode="bilinear", align_corners=False)[:, :3]
            ######TODO remove comment dynamic_depth = F.interpolate((dynamic_depth * 1.0), (256, 256), mode="bilinear", align_corners=False)
            ######TODO remove comment static_depth = F.interpolate((static_depth * 1.0), (256, 256), mode="bilinear", align_corners=False)
            ######TODO remove comment static_images = self.decode_latents(static_images)
            ######TODO remove comment static_images_interp = static_images[:, :3] * static_depth[:, :3]#F.interpolate((static_images * static_depth), (256, 256), mode="bilinear", align_corners=False)[:, :3]
            ######TODO remove comment static_images = self.encode_imgs(static_images)
            
            #static_depth = self.encode_imgs(static_depth[:,:3] * 1.0)
            #dynamic_depth = self.encode_imgs(dynamic_depth[:,:3] * 1.0)

            # decode first
            ######TODO remove comment latents_noisy = self.decode_latents(latents_noisy)
            ######TODO remove comment latents_noisy_before_blended_diff = torch.clone(latents_noisy)
            ######TODO remove comment latents_noisy_static = self.decode_latents(latents_noisy_static)
            # background + dynamic region + static region
            ######TODO remove comment bg_mask = torch.logical_and(dynamic_depth < 1.0, static_depth < 1.0).bool().int() * 1.0
            ######TODO remove comment latents_noisy = latents_noisy * bg_mask[:,:3] + latents_noisy * dynamic_depth[:,:3] + latents_noisy_static.detach() * static_depth[:,:3]
            # encode again
            ######TODO remove comment latents_noisy = self.encode_imgs(latents_noisy)
            ######TODO remove comment latents_noisy_before_blended_diff = self.encode_imgs(latents_noisy_before_blended_diff)
            #

            # PREDICT
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.model.apply_model(latent_model_input, tt, context)
            #

            ######TODO remove comment static_depth = torch.cat([static_depth] * 2)
            #inverted_static_depth = torch.cat([inverted_static_depth] * 2)
            ######TODO remove comment dynamic_depth = torch.cat([dynamic_depth] * 2)
            ######TODO remove comment latents_noisy_static = torch.cat([latents_noisy_static] * 2)
            ######TODO remove comment bg_mask = torch.cat([bg_mask] * 2)

            # decode first
            ######TODO remove comment noise_pred = self.decode_latents(noise_pred)
            ######TODO remove comment noise_pred_before_blended_diff = torch.clone(noise_pred)
            # background + dynamic region + static region
            ######TODO remove comment noise_pred = noise_pred * bg_mask[:,:3] + noise_pred * dynamic_depth[:,:3] + latents_noisy_static[:,:3].detach() * static_depth[:,:3]
            # encode again
            ######TODO remove comment noise_pred = self.encode_imgs(noise_pred)
            ######TODO remove comment noise_pred_before_blended_diff = self.encode_imgs(noise_pred_before_blended_diff)
            ######TODO remove comment latents_noisy_static = self.encode_imgs(latents_noisy_static)
            #
            ###################################################################

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

            # CUSTOM
            ######TODO remove comment noise_pred_uncond, noise_pred_pos = noise_pred_before_blended_diff.chunk(2)
            ######TODO remove comment noise_pred_before_blended_diff = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
            #

        # CUSTOM
        with_recon_loss = False
        # paper reconstruction loss is equivalent to standard SDS formulation w * (noise_pred - noise)
        if with_recon_loss:
            #noise_pred_before = noise_pred.clone()
            # CFG RESCALING https://github.com/DSaurus/threestudio-mvdream/blob/main/guidance/mvdream_guidance.py
            write_images_to_drive(self.decode_latents(self.model.predict_start_from_noise(latents_noisy, t, noise_pred)), string="_noise_pred_before_rescale")
            latents_recon = self.cfg_rescale(latents_noisy, noise_pred_pos, noise_pred, t)
            #write_images_to_drive(self.decode_latents(self.model.predict_start_from_noise(latents_noisy, t, noise_pred)), string="_noise_pred_after_rescale")
            write_images_to_drive(self.decode_latents(self.model.predict_start_from_noise(latents_recon, t, noise_pred)), string="_noise_pred_after_rescale")

            # calculate loss
            loss = 0.5 * F.mse_loss(latents.float(), latents_recon.detach(), reduction="sum") / latents.shape[0]
            grad = torch.autograd.grad(loss, latents, retain_graph=True)[0]

            # just for debug
            target = (latents - grad).detach()
        else:
            # CUSTOM
            w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)
            # Original SDS
            #grad = w * 40.0 * (noise_pred - noise)
            grad = (noise_pred - noise)
            #grad = w * (noise_pred - noise) * (1.0 + step_ratio * 4.0)
            grad = torch.nan_to_num(grad)

            # seems important to avoid NaN...
            # grad = grad.clamp(-1, 1)

            target = (latents - grad).detach()

            # CUSTOM target blending
            ######TODO remove comment target = (self.decode_latents(target) * inverted_static_depth) + (static_images_interp.detach() * 1.0) # use detach for static_images_interp, otherwise it will throw gradient error
            ######TODO remove comment target = self.encode_imgs(target)
            #

            loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]

        # from https://github.com/DSaurus/threestudio-mvdream/blob/main/guidance/mvdream_guidance.py
        #grad = torch.autograd.grad(loss, latents, retain_graph=True)[0]

        #CUSTOM
        self.debug_step += 1
        if self.debug_step % 1 == 0:
                #write_images_to_drive(imgs, string="_latents_denoised")
                #write_images_to_drive(imgs_target, string="_target")
                #write_images_to_drive(imgs_latents, string="_latents")
                #write_images_to_drive(imgs_noise_pred, string="_noise_pred_blended_diffusion")
                noisy_input = self.decode_latents(latents_noisy).detach().cpu()
                gs_renders = self.decode_latents(latents).detach().cpu()
                ######TODO remove comment latent_output = self.decode_latents(self.model.predict_start_from_noise(latents_noisy_before_blended_diff, t, noise_pred_before_blended_diff)).detach().cpu()
                #latent_output = self.decode_latents(noise_pred - noise).detach().cpu()
                blended_output = self.decode_latents(self.model.predict_start_from_noise(latents_noisy, t, noise_pred)).detach().cpu()
                target_debug = self.decode_latents(target).detach().cpu()
                batch_write_images_to_drive(noisy_input, gs_renders, target_debug, blended_output, blended_output, string="_batch_debug")
                #write_images_to_drive(imgs_static_depth, string="_static_depth")
                #write_images_to_drive(imgs_dynamic_depth, string="_dynamic_depth")
                print("guidance scale (mvdream): ", guidance_scale)
                print("t: ", t[0])
                print("num timesteps ", self.num_train_timesteps)
                self.debug_step = 0
        

        return loss
    
    # # CFG RESCALING https://github.com/DSaurus/threestudio-mvdream/blob/main/guidance/mvdream_guidance.py
    def cfg_rescale(self, latents_noisy,
                    noise_pred_pos, #x_neg
                    noise_pred, #x_pos
                    t,
                    rescale_strength = 0.7):
        
        # reconstruct x0
        latents_recon = self.model.predict_start_from_noise(latents_noisy, t, noise_pred)
        latents_recon_nocfg = self.model.predict_start_from_noise(latents_noisy, t, noise_pred_pos)

        latents_recon_nocfg_reshape = latents_recon_nocfg.view(-1, 4, *latents_recon_nocfg.shape[1:]) # * unpacks values
        latents_recon_reshape = latents_recon.view(-1, 4, *latents_recon.shape[1:])
        sigma_nocfg = latents_recon_nocfg_reshape.std([1,2,3,4], keepdim=True) + 1e-8
        sigma_cfg = latents_recon_reshape.std([1,2,3,4], keepdim=True) + 1e-8
        factor = sigma_nocfg / sigma_cfg

        latents_recon_adjust = latents_recon.clone() * factor.squeeze(1).repeat_interleave(4, dim=0)
        # actual rescaling
        return (rescale_strength * latents_recon_adjust) + (1.0 - rescale_strength) * latents_recon
         

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
