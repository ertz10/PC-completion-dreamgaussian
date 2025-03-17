from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
)
from diffusers.utils.import_utils import is_xformers_available

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as T 
from torch.cuda.amp import custom_bwd, custom_fwd
from PIL import Image
import cv2

from .sd_step import *


# Lucid Dreamer
class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class StableDiffusion(nn.Module):
    def __init__(
        self,
        device,
        fp16=True,
        vram_O=False,
        sd_version="2.1",
        hf_key=None,
        max_t_range=0.98,
        t_range=[0.02, 0.98],
    ):
        super().__init__()

        self.device = device
        self.precision_t = torch.float16 if fp16 else torch.float32
        self.sd_version = sd_version

        if hf_key is not None:
            print(f"[INFO] using hugging face custom model key: {hf_key}")
            model_key = hf_key
        elif self.sd_version == "2.1":
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == "2.0":
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == "1.5":
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(
                f"Stable-diffusion version {self.sd_version} not supported."
            )

        self.dtype = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype
        )

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype
        )

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.scheduler.set_timesteps(self.num_train_timesteps, device=device)
        self.timesteps = torch.flip(self.scheduler.timesteps, dims=(0, ))
        self.sche_func = ddim_step

        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience
        self.betas = 1 - self.alphas # = 1 - alphas

        self.warmup_step = int(self.num_train_timesteps*(max_t_range-t_range[1]))

        self.noise_gen = torch.Generator(self.device)
        self.noise_gen.manual_seed(0)

        self.embeddings = {}

        self.debug_step = 0
        self.train_steps = 0
        self.timelapse_imgs = []

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts)
        self.embeddings['pos'] = pos_embeds
        self.embeddings['neg'] = neg_embeds

        # directional embeddings
        for d in ['front', 'side', 'back']:
            embeds = self.encode_text([f'{p}, {d} view' for p in prompts])
            self.embeddings[d] = embeds

    @torch.no_grad()
    def get_inverse_text_embeds(self, uncond, inverse_text):
        uncond_embeds = self.encode_text(uncond)
        inverse_embeds = self.encode_text(inverse_text)
        self.embeddings['uncond'] = uncond_embeds
        self.embeddings['inverse_text'] = inverse_embeds
    
    def encode_text(self, prompt):
        # prompt: [str]
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    @torch.no_grad()
    def refine(self, pred_rgb,
               guidance_scale=100, steps=50, strength=0.8, object_params=None
        ):

        # CUSTOM
        def batch_write_images_to_drive(input1, input2, index=-1, string=""):

                import PIL.Image

                img_width = input1.shape[2]
                img_height = input1.shape[3]
                # create figure
                inputs = [input1, input2]
                figure = PIL.Image.new('RGB', (img_width, img_height * len(inputs)), color=(255, 255, 255))
                #figure = PIL.Image.new('RGB', (512 * 4, 512 * 5), color=(255, 255, 255))
                #inputs = [input1, input2, input3, input4, input5]
                

                #inputs = F.interpolate((inputs), (512, 512), mode="bilinear", align_corners=False)

                # add images
                for i in range(0,len(inputs)):
                    for j, img in enumerate(inputs[i]):
                        transform = T.ToPILImage()
                        image = transform(img)
                        figure.paste(image, (j * img_width, i * img_height))

                figure2 = PIL.Image.new('RGB', (512, 512), color=(255, 255, 255))
                if self.train_steps % 10 == 0:
                    for j, img in enumerate(inputs[0]):
                        img = F.interpolate(img.unsqueeze(0), (512, 512), mode="bilinear", align_corners=False)
                        transform = T.ToPILImage()
                        image = transform(img[0])
                        figure2.paste(image, (j * 512, 0))
                    self.timelapse_imgs.append(figure2)

                try:
                    #figure.save(r"debug/diffModelDebug" + str(string) + r".jpg")
                    figure.save(str(object_params.data_path) + '/diffModelDebug_tex_refine.jpg')
                    if (self.train_steps == object_params.max_steps_tex_refine):
                        for x in range(0, 30):
                            self.timelapse_imgs.append(self.timelapse_imgs[len(self.timelapse_imgs) - 1])
                        # save timelapse
                        #self.timelapse_imgs[0].save('debug/timelapseDebug.gif', save_all=True, append_images=self.timelapse_imgs[1:], optimize=False, duration=250, loop=0)
                        self.timelapse_imgs[0].save(str(object_params.data_path) + '/timelapseDebug_tex_refine.gif', save_all=True, append_images=self.timelapse_imgs[1:], optimize=False, duration=250, loop=0)
                except OSError:
                    print("Cannot save image")

        batch_size = pred_rgb.shape[0]
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb_512.to(self.dtype))
        # latents = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.dtype)

        self.scheduler.set_timesteps(steps)
        init_step = int(steps * strength)
        latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])
        embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
    
            latent_model_input = torch.cat([latents] * 2)

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=embeddings,
            ).sample

            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Debug output
        batch_write_images_to_drive(pred_rgb_512, imgs)

        return imgs
    
    def retrieve_gs_depth_images(self, customLoss, static_images, dynamic_images, static_depth_images, dynamic_depth_images, img_width, img_height):
        with torch.no_grad():
            ######### ADD both binary masked images #############
            #imgs_blended = customLoss.blend_images(dynamic_images, static_images, dynamic_depth_images, static_depth_images)
            dynamic_depth, static_depth = customLoss.blend_images(dynamic_images, static_images, dynamic_depth_images, static_depth_images)

            # TODO blend grad and static regions of rendering, so grad results in zero in those regions
            static_depth = torch.stack((static_depth)).detach()
            dynamic_depth = torch.stack((dynamic_depth)).detach()
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
            alphas_stat = static_images[:, 3:].detach()
            alphas_stat = F.interpolate((alphas_stat), (img_width, img_height), mode="bilinear", align_corners=False)
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
        



        
    def add_noise_with_cfg(self, latents, noise, 
                           ind_t, ind_prev_t, 
                           text_embeddings=None, cfg=1.0, 
                           delta_t=1, inv_steps=1,
                           is_noisy_latent=False,
                           eta=0.0):
        
        #text_embeddings = text_embeddings.to(self.precision_t)
        if cfg <= 1.0:
            uncond_text_embedding = text_embeddings.reshape(2, -1, text_embeddings.shape[-2], text_embeddings.shape[-1])[1]

        unet = self.unet

        if is_noisy_latent:
            prev_noisy_lat = latents
        else:
            prev_noisy_lat = self.scheduler.add_noise(latents, noise, self.timesteps[ind_prev_t])

        cur_ind_t = ind_prev_t
        cur_noisy_lat = prev_noisy_lat

        pred_scores = []

        for i in range(inv_steps):
            # pred noise
            cur_noisy_lat_ = self.scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(self.precision_t)
            
            if cfg > 1.0:
                latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_])
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
                unet_output = unet(latent_model_input, timestep_model_input, 
                                encoder_hidden_states=text_embeddings).sample
                
                uncond, cond = torch.chunk(unet_output, chunks=2)
                
                unet_output = cond + cfg * (uncond - cond) # reverse cfg to enhance the distillation
            else:
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(cur_noisy_lat_.shape[0], 1).reshape(-1)
                unet_output = unet(cur_noisy_lat_, timestep_model_input, 
                                    encoder_hidden_states=uncond_text_embedding).sample

            pred_scores.append((cur_ind_t, unet_output))

            next_ind_t = min(cur_ind_t + delta_t, ind_t)
            cur_t, next_t = self.timesteps[cur_ind_t], self.timesteps[next_ind_t]
            delta_t_ = next_t-cur_t if isinstance(self.scheduler, DDIMScheduler) else next_ind_t-cur_ind_t

            cur_noisy_lat = self.sche_func(self.scheduler, unet_output, cur_t, cur_noisy_lat, -delta_t_, eta).prev_sample
            cur_ind_t = next_ind_t

            del unet_output
            torch.cuda.empty_cache()

            if cur_ind_t == ind_t:
                break

        return prev_noisy_lat, cur_noisy_lat, pred_scores[::-1]
    







    def train_step(
        self,
        pred_rgb,
        timelapse_img,
        step_ratio=None,
        guidance_scale=7.5,
        as_latent=False,
        customLoss=None,
        vers=None, hors=None,
        dynamic_images=None,
        static_images=None,
        dynamic_depth_images=None,
        static_depth_images=None,
        current_cam_hors=[0, 0, 0, 0],
        captured_angles_hor=[0, 0],
        object_params=None,
        only_dynamic_splats=False,
        tex_refine=False
    ):
        
        # CUSTOM
        def batch_write_images_to_drive(input1, input2, input3, input4, input5, index=-1, string="", batch_size=1):

                import PIL.Image

                img_width = input1.shape[2]
                img_height = input1.shape[3]
                # create figure
                inputs = [input1, input2, input3, input4, input5]
                figure = PIL.Image.new('RGB', (img_width * batch_size, img_height * len(inputs)), color=(255, 255, 255))
                #figure = PIL.Image.new('RGB', (512 * 4, 512 * 5), color=(255, 255, 255))
                #inputs = [input1, input2, input3, input4, input5]
                

                #inputs = F.interpolate((inputs), (512, 512), mode="bilinear", align_corners=False)

                # add images
                for i in range(0,len(inputs)):
                    for j, img in enumerate(inputs[i]):
                        transform = T.ToPILImage()
                        image = transform(img)
                        figure.paste(image, (j * img_width, i * img_height))

                figure2 = PIL.Image.new('RGB', (1024, 1024), color=(255, 255, 255))
                if self.train_steps % 2 == 0:
                    for j, img in enumerate(inputs[1]):
                        #img = F.interpolate(img.unsqueeze(0), (512, 512), mode="bilinear", align_corners=False)
                        img = F.interpolate(timelapse_img, (1024, 1024), mode="bilinear", align_corners=False)
                        transform = T.ToPILImage()
                        image = transform(img[0])
                        figure2.paste(image, (j * 1024, 0))
                    self.timelapse_imgs.append(figure2)

                try:
                    #figure.save(r"debug/diffModelDebug" + str(string) + r".jpg")
                    if (tex_refine):
                        figure.save(str(object_params.data_path) + '/diffModelDebug_TEX_refine.jpg')
                    else:
                        figure.save(str(object_params.data_path) + '/diffModelDebug_SD_refine.jpg')
                    max_steps = object_params.max_steps_tex_refine if tex_refine else object_params.max_steps_refine
                    if (self.train_steps == max_steps):
                        #for x in range(0, max_steps):
                        #    self.timelapse_imgs.append(self.timelapse_imgs[len(self.timelapse_imgs) - 1])
                        # save timelapse
                        #self.timelapse_imgs[0].save('debug/timelapseDebug.gif', save_all=True, append_images=self.timelapse_imgs[1:], optimize=False, duration=250, loop=0)
                        if (tex_refine):
                            out = cv2.VideoWriter(str(object_params.data_path) + '/timelapse_TEX_refine.mp4', cv2.VideoWriter.fourcc(*'mp4v'), 30, (1024, 1024))
                            #ext_frames = np.repeat(self.timelapse_imgs, 60)
                            for frame in self.timelapse_imgs:
                                out.write(frame)
                            out.release()
                            self.timelapse_imgs[0].save(str(object_params.data_path) + '/timelapseDebug_TEX_refine.gif', save_all=True, append_images=self.timelapse_imgs[1:], optimize=False, duration=250, loop=0)
                        else:
                            out = cv2.VideoWriter(str(object_params.data_path) + '/timelapse_SD_refine.mp4', cv2.VideoWriter.fourcc(*'mp4v'), 30, (1024, 1024))
                            #ext_frames = np.repeat(self.timelapse_imgs, 60)
                            for frame in self.timelapse_imgs:
                                out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
                            out.release()
                            self.timelapse_imgs[0].save(str(object_params.data_path) + '/timelapseDebug_SD_refine.gif', save_all=True, append_images=self.timelapse_imgs[1:], optimize=False, duration=250, loop=0)
                except OSError:
                    print("Cannot save image")

        self.train_steps += 1
        
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        # TODO change num_train_timesteps
        #self.num_train_timesteps = 50
        with torch.no_grad():
            if (tex_refine):
                if step_ratio is not None:
                    #if self.train_steps <= 2000:
                    # dreamtime-like
                    # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
                    t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
                    t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
                else:
                    t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)
            else:
                if step_ratio is not None:
                    max_linear_anneal_iters = 700
                    if self.train_steps <= max_linear_anneal_iters:
                        # dreamtime-like
                        # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
                        step_ratio = min(1, self.train_steps / max_linear_anneal_iters) # after GradeADreamer (max_linear_anneal_iter)
                        t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
                        t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
                        #t = torch.randint(self.min_step, int(self.max_step * 0.5) + 1, (batch_size,), dtype=torch.long, device=self.device)
                    #elif self.train_steps > 700 and self.train_steps <= 900: # after gradADreamer
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
                    t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)
                
                
                
            # TODO remove after testing LUCID DREAMER stuff
            warm_up_rate = 1. - min(self.train_steps / object_params.warmup_iter, 1.)
            #t = torch.randint(self.min_step, self.max_step + int(self.warmup_step * warm_up_rate), (batch_size,), dtype=torch.long, generator=self.noise_gen, device=self.device)
            t = torch.randint(self.min_step, self.max_step + int(self.warmup_step * warm_up_rate), (1,), dtype=torch.long, generator=self.noise_gen, device=self.device)

            # predict the noise residual with unet, NO grad!
            # After LUCID Dreamer
            if object_params.ism:
                inverse_text_embeddings = torch.cat([self.embeddings['uncond'].expand(batch_size, -1, -1), self.embeddings['inverse_text'].expand(batch_size, -1, -1)])
                if object_params.annealing_intervals:
                    current_delta_t = int(object_params.delta_t + (warm_up_rate)*(object_params.delta_t_start - object_params.delta_t))
                else:
                    current_delta_t = object_params.delta_t


            ############ Masking ###########    
            static_region = self.retrieve_gs_depth_images(customLoss=customLoss, static_images=static_images,
                                                          dynamic_images=dynamic_images, static_depth_images=static_depth_images,
                                                          dynamic_depth_images=dynamic_depth_images, img_width=512, img_height=512)
            
            # filter cameras which "can see" the visible static part
            valid_cams = self.filter_cams(current_cam_hors, captured_angles_hor)

            # add noise
            noise = torch.randn_like(latents)
            #latents_noisy = self.scheduler.add_noise(latents, noise, t)

            if object_params.annealing_intervals:
                current_delta_t =  int(object_params.delta_t + (warm_up_rate)*(object_params.delta_t_start - object_params.delta_t))
            else:
                current_delta_t =  object_params.delta_t

            #ind_t = t.clone()
            #ind_prev_t = torch.randint(self.min)

            prev_t = max(t - current_delta_t, torch.ones_like(t) * 0)
            if not object_params.ism: # normal Score distillation

                #prev_latents_noisy = self.scheduler.add_noise(latents, noise, prev_t)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                target = noise
            else:
                # step 1: sample x_s with larger steps
                xs_delta_t = object_params.xs_delta_t if object_params.xs_delta_t is not None else object_params
                xs_inv_steps = object_params.xs_inv_steps if object_params.xs_inv_steps is not None else int(np.ceil(prev_t / xs_delta_t))
                starting_idx = max(prev_t - xs_delta_t * xs_inv_steps, torch.ones_like(t) * 0)

                _, prev_latents_noisy, pred_scores_xs = self.add_noise_with_cfg(latents, noise, prev_t, starting_idx, inverse_text_embeddings, 
                                                                                object_params.denoise_guidance_scale, xs_delta_t, xs_inv_steps, eta=object_params.xs_eta)

                _, latents_noisy, pred_scores_xt = self.add_noise_with_cfg(prev_latents_noisy, noise, t, prev_t, inverse_text_embeddings, 
                                                                                object_params.denoise_guidance_scale, current_delta_t, 1, is_noisy_latent=True)

                pred_scores = pred_scores_xt + pred_scores_xs
                target = pred_scores[0][1]
                #target = pred_scores[1][1]
            
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            #tt = torch.cat([t] * 2)
            tt = torch.cat([t] * 2 * batch_size)
            # TODO scale model input ? lucid dreamer line 412
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, tt[0])


            if hors is None:
                embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])
            else:
                def _get_dir_ind(h):
                    if abs(h) < 60: return 'front'
                    elif abs(h) < 120: return 'side'
                    else: return 'back'

                embeddings = torch.cat([self.embeddings[_get_dir_ind(h)] for h in hors] + [self.embeddings['neg'].expand(batch_size, -1, -1)])

            noise_pred = self.unet(
                latent_model_input.to(torch.float16), tt, encoder_hidden_states=embeddings
            ).sample

            # perform guidance (high scale from paper!)
            # TODO which is first, noise_pred_cond or noise_pred_uncond ???? mvdream says noise_pred_uncond first
            #noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # CFG rescale copied from pipeline_stable_diffusion.py
            #if object_params.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            #    noise_pred = self.rescale_noise_cfg(noise_pred, noise_pred_cond, guidance_rescale=object_params.guidance_rescale)
            #####

        #early_phase = True if self.train_steps <= object_params.max_steps_refine / 2 else False
        #early_phase_tex = True if self.train_steps <= object_params.max_steps_tex_refine / 2 else False
        early_phase = True
        early_phase_tex = True
        '''
        if (tex_refine):
            # After gradAdreamer/fantasia3d https://github.com/Gorilla-Lab-SCUT/Fantasia3D/blob/main/geometry/dlmesh.py
            #https://github.com/trapoom555/GradeADreamer/blob/main/gradeadreamer/geometry/dmtet.py
            # w = 1 / (1 - self.alphas[t]).view(batch_size, 1, 1, 1)
            # hint (dreamfusion): sigma_t^2 = 1 - alpha_t^2
            #w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(batch_size, 1, 1, 1) #early phase
            if (early_phase_tex):
                w = ((1 - self.alphas[t]) * (self.alphas[t]) ** 0.5).view(batch_size, 1, 1, 1)
            else: #late phase
                w = (1 / (1 - self.alphas[t])).view(batch_size, 1, 1, 1) #weight strategy equal to one (after GradeADreamer)
        else:
            #TODO implement early phase w and later phase w after Fanatasia3d
            # w(t), sigma_t^2
            if (early_phase):
                w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)            
            else: # late phase
                w = ((1 - self.alphas[t]) * (self.alphas[t]) ** 0.5).view(batch_size, 1, 1, 1)
        '''

        # LUCID DREAMER
        w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)
        #w = (((1 - self.alphas[t]) / self.alphas[t]) ** 0.5).view(batch_size, 1, 1, 1)
        
        #grad = w * (noise_pred - noise)
        grad = w(self.alphas[t]) * (noise_pred - target) # target = noise in case of !ISM
        #grad = w * (lambda_t * delta_inv + guidance_scale * (noise_pred - noise))
        grad = torch.nan_to_num(grad)

        if (torch.isinf(grad).any()):
            print("Stop")
        if (torch.isnan(grad).any()):
            print("Stop")

        # seems important to avoid NaN...
        # grad = grad.clamp(-1, 1)

        #target = (latents - grad).detach()
        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]
        #loss = SpecifyGradient.apply(latents, grad)

        if (only_dynamic_splats == False):
            latents_dec = self.decode_latents(latents)
            with torch.no_grad():
                static_images = torch.stack((static_images)).detach()
                static_images = F.interpolate((static_images), (512, 512), mode="bilinear", align_corners=False)
                static_images = static_images[:,:3]
            if (valid_cams.shape[0] != 0):
                for valid_cam in valid_cams:
                    with torch.no_grad():
                        bool_mask = static_region[valid_cam].int()
                        #write_images_to_drive(bool_mask.squeeze(0) * 1.0, string="mask")
                        '''
                        bool_mask = static_region[valid_cam].int().bool().squeeze(0)
                        bool_mask = static_region[valid_cam].int().squeeze(0)
                        bool_mask = bool_mask.cpu().numpy()
                            #kernel = np.ones((5, 5), np.uint8) 
                            #bool_mask = erode(bool_mask, kernel=kernel) #cv2
                            # dilate bool mask
                        bool_mask = torch.tensor(binary_erosion(bool_mask, iterations=1))
                        write_images_to_drive(bool_mask * 1.0, string="mask_eroded")
                        bool_mask = bool_mask.bool()
                        '''
                        #'''
                        kernel = np.ones((3, 3), dtype=np.float32)
                        kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0))
                        bool_mask = bool_mask[:, 0].unsqueeze(0).float().cpu()
                        bool_mask = 1 - torch.clamp(torch.nn.functional.conv2d(1 - bool_mask, kernel_tensor, padding=(1,1)), 0, 1)
                        bool_mask = 1 - torch.clamp(torch.nn.functional.conv2d(1 - bool_mask, kernel_tensor, padding=(1,1)), 0, 1)
                        bool_mask = 1 - torch.clamp(torch.nn.functional.conv2d(1 - bool_mask, kernel_tensor, padding=(1,1)), 0, 1)
                        #write_images_to_drive(bool_mask.squeeze(0), string="mask_eroded")
                        bool_mask = bool_mask.squeeze(0).bool()
                        bool_mask = torch.repeat_interleave(bool_mask, 3, 0)
                    #alphas_stat = alphas_stat[:,:3]
                    # TODO reenable
                    #loss += F.mse_loss(latents_dec[valid_cam, bool_mask].float(), static_images[valid_cam, bool_mask], reduction='sum')

        #CUSTOM
        if object_params.DEBUG:
            with torch.no_grad():
                self.debug_step += 1
                if self.debug_step % 1 == 0:
                        noisy_input = self.decode_latents(latents_noisy.to(torch.float16))
                        #gs_renders = self.decode_latents(latents)
                        gs_renders = self.decode_latents(latents)
                        ######TODO remove comment latent_output = self.decode_latents(self.model.predict_start_from_noise(latents_noisy_before_blended_diff, t, noise_pred_before_blended_diff)).detach().cpu()
                        latent_output = self.decode_latents(noise_pred - noise)
                        #blended_output = self.decode_latents(self.model.predict_start_from_noise(latents_noisy, t, noise_pred))
                        #blended_output = self.decode_latents(self.model.predict_start_from_noise(latents_noisy, t, noise_pred))
                        target_debug = self.decode_latents(target.to(torch.float16))
                        
                        #batch_write_images_to_drive(noisy_input, gs_renders, target_debug, latent_output, blended_output, string=r"_batch_debug")
                        batch_write_images_to_drive(noisy_input, gs_renders, target_debug, latent_output, static_region, string=r"_batch_debug", batch_size=batch_size)
                        #write_images_to_drive(static_region, string="_static_depth_images")
                        
                        print("good Horizontal angles: " + str(current_cam_hors))
                        print("guidance scale: ", guidance_scale)
                        print("t: ", t[0])
                        print("num timesteps ", self.num_train_timesteps)
                        #print(torch.cuda.memory_summary())
                        self.debug_step = 0
        return loss
    
    def filter_cams(self, current_cam_hors, captured_angles_hor):
        # CHECK if first camera angle is in a "good" angle, such that we can overlay the static part completely
            current_cam_hors = torch.FloatTensor(current_cam_hors)#
            print(current_cam_hors.shape)
            print(captured_angles_hor[0])
            #valid_cams = torch.ge(current_cam_hors, captured_angles_hor[0])
            valid_cams = np.zeros(4)
            if (captured_angles_hor[0] > captured_angles_hor[1]):
                for i in range(0, 1):
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
            #if (valid_cams.shape[0] != 0):
            #    for valid_cam in valid_cams:
                    # overlay complete static region (only for first image)
                    # TODO remove this
                    #valid_cams = valid_cams
                    #latents[valid_cams][static_region[valid_cams].int().bool()] = (static_images[valid_cams] * alphas_stat[valid_cams,:3])[static_region[valid_cams,:3].int().bool()]
                    #latents[valid_cams][static_region[valid_cams].int().bool()] = static_images[valid_cams][static_region[valid_cams].int().bool()]
            #latents = self.encode_imgs(latents)
            ######### Masking end ##############
            return valid_cams
    

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

    @torch.no_grad()
    def produce_latents(
        self,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if latents is None:
            latents = torch.randn(
                (
                    1,
                    self.unet.in_channels,
                    height // 8,
                    width // 8,
                ),
                device=self.device,
            )

        batch_size = latents.shape[0]
        self.scheduler.set_timesteps(num_inference_steps)
        embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=embeddings
            ).sample

            # perform guidance
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(
        self,
        prompts,
        negative_prompts="",
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        self.get_text_embeds(prompts, negative_prompts)
        
        # Text embeds -> img latents
        latents = self.produce_latents(
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

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
    parser.add_argument(
        "--sd_version",
        type=str,
        default="2.1",
        choices=["1.5", "2.0", "2.1"],
        help="stable diffusion version",
    )
    parser.add_argument(
        "--hf_key",
        type=str,
        default=None,
        help="hugging face Stable diffusion model key",
    )
    parser.add_argument("--fp16", action="store_true", help="use float16 for training")
    parser.add_argument(
        "--vram_O", action="store_true", help="optimization for low VRAM usage"
    )
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device("cuda")

    sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
