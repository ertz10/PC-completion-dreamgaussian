import torch
import numpy as np
import math

import torchvision.transforms as T

# finer depth rendering from https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth.git
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from sh_utils import eval_sh, SH2RGB, RGB2SH

class GaussianCustomRenderer:
    '''
    Adapted from gs_renderer.py
    '''

    def __init__(self, sh_degree=3, white_background=True, radius=1):
        
        self.sh_degree = sh_degree
        self.white_background = white_background
        self.radius = radius

        self.bg_color = torch.tensor(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )

    def render_static(
        self,
        gaussians,
        viewpoint_camera,
        render_type,
        scaling_modifier=1.0,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        #CUSTOM
        # static_points_mask needs adjustment after densification, vstack ...
        #test = np.ones(len(gaussians.get_xyz))
        #static_points_mask = np.hstack((test, gaussians.static_points_mask[gaussians.static_points_mask == 0]))
        #static_points_mask = np.hstack((np.ones(len(gaussians.get_xyz)), gaussians.static_points_mask[gaussians.static_points_mask == 0]))
        #dynamic_gaussians = gaussians.get_xyz
        static_gaussians = gaussians.get_xyz#[static_points_mask == 0]
        active_gaussians = None
        '''
        # TODO select max and min gaussians on each axis, and append to each rendering, for static and dynamic!
        maxima = torch.argmax(torch.vstack((dynamic_gaussians, static_gaussians)), axis=0)
        minima = torch.argmin(torch.vstack((dynamic_gaussians, static_gaussians)), axis=0)
        max_gaussians = torch.vstack((dynamic_gaussians, static_gaussians))[maxima]
        min_gaussians = torch.vstack((dynamic_gaussians, static_gaussians))[minima]
        #TODO maybe try with static synthetically created gaussians, take corners as coordinates
        top_1 = torch.tensor((1.0, 1.0, 1.0), device='cuda')
        top_2 = torch.tensor((1.0, 1.0, -1.0), device='cuda')
        top_3 = torch.tensor((-1.0, 1.0, -1.0), device='cuda')
        top_4 = torch.tensor((-1.0, 1.0, 1.0), device='cuda')
        bot_1 = torch.tensor((-1.0, 1.0, -1.0), device='cuda')
        bot_2 = torch.tensor((1.0, -1.0, 1.0), device='cuda')
        bot_3 = torch.tensor((1.0, -1.0, -1.0), device='cuda')
        bot_4 = torch.tensor((-1.0, -1.0, -1.0), device='cuda')
        max_gaussians = torch.vstack((top_1, top_2, top_3, top_4))
        min_gaussians = torch.vstack((bot_1, bot_2, bot_3, bot_4))
        '''
        active_gaussians = static_gaussians#torch.vstack((static_gaussians, max_gaussians, min_gaussians))

        screenspace_points = (
            torch.zeros_like(
                active_gaussians,#CUSTOM,
                #self.gaussians.get_xyz,
                dtype=active_gaussians.dtype,
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
            sh_degree=gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        # TODO add some finer depth rendering https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        #means3D = self.gaussians.get_xyz
        #means2D = screenspace_points
        #opacity = self.gaussians.get_opacity
        # CUSTOM 
        #means3D = torch.vstack((gaussians.get_xyz, gaussians.original_xyz[static_points_mask == 0]))
        #means2D = screenspace_points
        #opa = gaussians.opacity_activation(gaussians.original_opacity[static_points_mask == 0])
        #opacity = torch.vstack((gaussians.get_opacity, opa))
        means3D = active_gaussians
        means2D = screenspace_points
        active_opac = None
        #min_gaussians_opac = torch.vstack((gaussians._opacity, gaussians.original_opacity))[minima]
        #max_gaussians_opac = torch.vstack((gaussians._opacity, gaussians.original_opacity))[maxima]
        full_opac = torch.ones((len(gaussians._opacity), 1), device='cuda')
        #active_opac = gaussians.opacity_activation(torch.vstack((gaussians.original_opacity, max_gaussians_opac, min_gaussians_opac)))
        active_opac = gaussians.get_opacity#gaussians.opacity_activation(torch.vstack((gaussians._opacity, full_opac[:4], full_opac[:4])))
        opacity = active_opac

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = gaussians.get_covariance(scaling_modifier)
        else:
            scales = gaussians.get_scaling
            rotations = gaussians.get_rotation

        #CUSTOM
        #sca = gaussians.scaling_activation(gaussians.original_scaling[static_points_mask == 0])
        #rot = gaussians.rotation_activation(gaussians.original_rotation[static_points_mask == 0])
        #scales = torch.vstack((scales, sca))
        #rotations = torch.vstack((rotations, rot))
        #min_gaussians_scaling = torch.vstack((gaussians._scaling, gaussians.original_scaling))[minima]
        #max_gaussians_scaling = torch.vstack((gaussians._scaling, gaussians.original_scaling))[maxima]
        #min_gaussians_rotation = torch.vstack((gaussians._rotation, gaussians.original_rotation))[minima]
        #max_gaussians_rotation = torch.vstack((gaussians._rotation, gaussians.original_rotation))[maxima]
        sc = torch.tensor((-6.0, -6.0, -6.0), device='cuda')
        ro = torch.tensor((0.0, 0.0, 0.0, 0.0), device='cuda')
        uniform_scaling = torch.vstack((sc, sc, sc, sc))
        uniform_rotation = torch.vstack((ro, ro, ro, ro))
        #active_scales = gaussians.scaling_activation(torch.vstack((gaussians.original_scaling, max_gaussians_scaling, min_gaussians_scaling)))
        #active_rotations = gaussians.rotation_activation(torch.vstack((gaussians.original_rotation, max_gaussians_rotation, min_gaussians_rotation)))
        active_scales = gaussians.get_scaling
        active_rotations = gaussians.get_rotation
        scales = active_scales
        rotations = active_rotations

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = gaussians.get_features.transpose(1, 2).view(
                    -1, 3, (gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = gaussians.get_xyz - viewpoint_camera.camera_center.repeat(
                    gaussians.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                # TODO create default features for all corners of the cube
                #min_gaussians_features = gaussians.get_features[minima] #gets all features, masked by min gaussians
                #max_gaussians_features = gaussians.get_features[maxima]
                #shs = torch.vstack((gaussians.get_features))
                shs = torch.cat((gaussians._features_dc, gaussians._features_rest), dim=1)
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
        def write_image_to_drive(input_tensor, index=None, type="static"):
                # transform torch tensor to rgb image and write to drive for DEBUG purposes
                transform = T.ToPILImage()
                img = transform(input_tensor[0])
                #img = img.save("debug/train_step_debug" + str(index) +".jpg")
                img = img.save("debug/gaussian_raster_depth_debug_" + str(type) + ".jpg")

        #write_image_to_drive(rendered_depth, type=render_type)

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

    def render_custom(
        self,
        gaussians,
        viewpoint_camera,
        render_type,
        scaling_modifier=1.0,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
        only_dynamic_splats=False
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        #CUSTOM
        # static_points_mask needs adjustment after densification, vstack ...
        #test = np.ones(len(gaussians.get_xyz))
        #static_points_mask = np.hstack((test, gaussians.static_points_mask[gaussians.static_points_mask == 0]))
        if(only_dynamic_splats == False):
            static_points_mask = np.hstack((np.ones(len(gaussians.get_xyz)), gaussians.static_points_mask[gaussians.static_points_mask == 0]))
        else:
            static_points_mask = np.ones(len(gaussians.get_xyz))
        dynamic_gaussians = gaussians.get_xyz
        static_gaussians = gaussians.original_xyz#[static_points_mask == 0]
        active_gaussians = None
        # TODO select max and min gaussians on each axis, and append to each rendering, for static and dynamic!
        if(only_dynamic_splats == False):
            maxima = torch.argmax(torch.vstack((dynamic_gaussians, static_gaussians)), axis=0)
            minima = torch.argmin(torch.vstack((dynamic_gaussians, static_gaussians)), axis=0)
            max_gaussians = torch.vstack((dynamic_gaussians, static_gaussians))[maxima]
            min_gaussians = torch.vstack((dynamic_gaussians, static_gaussians))[minima]
        else:
            maxima = torch.argmax(dynamic_gaussians, axis=0)
            minima = torch.argmin(dynamic_gaussians, axis=0)
            max_gaussians = dynamic_gaussians[maxima]
            min_gaussians = dynamic_gaussians[minima]
        #TODO maybe try with static synthetically created gaussians, take corners as coordinates
        top_1 = torch.tensor((1.0, 1.0, 1.0), device='cuda')
        top_2 = torch.tensor((1.0, 1.0, -1.0), device='cuda')
        top_3 = torch.tensor((-1.0, 1.0, -1.0), device='cuda')
        top_4 = torch.tensor((-1.0, 1.0, 1.0), device='cuda')
        bot_1 = torch.tensor((-1.0, 1.0, -1.0), device='cuda')
        bot_2 = torch.tensor((1.0, -1.0, 1.0), device='cuda')
        bot_3 = torch.tensor((1.0, -1.0, -1.0), device='cuda')
        bot_4 = torch.tensor((-1.0, -1.0, -1.0), device='cuda')
        max_gaussians = torch.vstack((top_1, top_2, top_3, top_4))
        min_gaussians = torch.vstack((bot_1, bot_2, bot_3, bot_4))
        if(render_type == "dynamic"):
            active_gaussians = torch.vstack((dynamic_gaussians, max_gaussians, min_gaussians))
        if(render_type == "static"):
            active_gaussians = torch.vstack((static_gaussians, max_gaussians, min_gaussians))

        screenspace_points = (
            torch.zeros_like(
                active_gaussians,#CUSTOM,
                #self.gaussians.get_xyz,
                dtype=active_gaussians.dtype,
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
            sh_degree=gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        # TODO add some finer depth rendering https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        #means3D = self.gaussians.get_xyz
        #means2D = screenspace_points
        #opacity = self.gaussians.get_opacity
        # CUSTOM 
        #means3D = torch.vstack((gaussians.get_xyz, gaussians.original_xyz[static_points_mask == 0]))
        #means2D = screenspace_points
        #opa = gaussians.opacity_activation(gaussians.original_opacity[static_points_mask == 0])
        #opacity = torch.vstack((gaussians.get_opacity, opa))
        means3D = active_gaussians
        means2D = screenspace_points
        active_opac = None
        #min_gaussians_opac = torch.vstack((gaussians._opacity, gaussians.original_opacity))[minima]
        #max_gaussians_opac = torch.vstack((gaussians._opacity, gaussians.original_opacity))[maxima]
        full_opac = torch.ones((len(gaussians._opacity), 1), device='cuda')
        if(render_type == "dynamic"):
            #active_opac = gaussians.opacity_activation(torch.vstack((gaussians._opacity, max_gaussians_opac, min_gaussians_opac)))
            active_opac = gaussians.opacity_activation(torch.vstack((gaussians._opacity, full_opac[:4], full_opac[:4])))
        if(render_type == "static"):
            #active_opac = gaussians.opacity_activation(torch.vstack((gaussians.original_opacity, max_gaussians_opac, min_gaussians_opac)))
            active_opac = gaussians.opacity_activation(torch.vstack((gaussians.original_opacity, full_opac[:4], full_opac[:4])))
        opacity = active_opac

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = gaussians.get_covariance(scaling_modifier)
        else:
            scales = gaussians.get_scaling
            rotations = gaussians.get_rotation

        #CUSTOM
        #sca = gaussians.scaling_activation(gaussians.original_scaling[static_points_mask == 0])
        #rot = gaussians.rotation_activation(gaussians.original_rotation[static_points_mask == 0])
        #scales = torch.vstack((scales, sca))
        #rotations = torch.vstack((rotations, rot))
        #min_gaussians_scaling = torch.vstack((gaussians._scaling, gaussians.original_scaling))[minima]
        #max_gaussians_scaling = torch.vstack((gaussians._scaling, gaussians.original_scaling))[maxima]
        #min_gaussians_rotation = torch.vstack((gaussians._rotation, gaussians.original_rotation))[minima]
        #max_gaussians_rotation = torch.vstack((gaussians._rotation, gaussians.original_rotation))[maxima]
        sc = torch.tensor((-6.0, -6.0, -6.0), device='cuda')
        ro = torch.tensor((0.0, 0.0, 0.0, 0.0), device='cuda')
        uniform_scaling = torch.vstack((sc, sc, sc, sc))
        uniform_rotation = torch.vstack((ro, ro, ro, ro))
        if(render_type == "dynamic"):
            #active_scales = scales#gaussians.scaling_activation(scales)
            #active_rotations = rotations#gaussians.rotation_activation(rotations)
            #active_scales = torch.vstack((scales, gaussians.scaling_activation(max_gaussians_scaling),
            #                              gaussians.scaling_activation(min_gaussians_scaling)))
            #active_rotations = torch.vstack((rotations, gaussians.rotation_activation(max_gaussians_rotation), 
            #                                 gaussians.rotation_activation(min_gaussians_rotation)))
            active_scales = torch.vstack((scales, gaussians.scaling_activation(uniform_scaling), gaussians.scaling_activation(uniform_scaling)))
            active_rotations = torch.vstack((rotations, gaussians.rotation_activation(uniform_rotation), gaussians.rotation_activation(uniform_rotation)))
        if(render_type == "static"):
            #active_scales = gaussians.scaling_activation(torch.vstack((gaussians.original_scaling, max_gaussians_scaling, min_gaussians_scaling)))
            #active_rotations = gaussians.rotation_activation(torch.vstack((gaussians.original_rotation, max_gaussians_rotation, min_gaussians_rotation)))
            active_scales = gaussians.scaling_activation(torch.vstack((gaussians.original_scaling, uniform_scaling, uniform_scaling)))
            active_rotations = gaussians.rotation_activation(torch.vstack((gaussians.original_rotation, uniform_rotation, uniform_rotation)))
        scales = active_scales
        rotations = active_rotations

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = gaussians.get_features.transpose(1, 2).view(
                    -1, 3, (gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = gaussians.get_xyz - viewpoint_camera.camera_center.repeat(
                    gaussians.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                # TODO create default features for all corners of the cube
                min_gaussians_features = gaussians.get_features[minima] #gets all features, masked by min gaussians
                max_gaussians_features = gaussians.get_features[maxima]
                if(render_type == "dynamic"):
                    shs = torch.vstack((gaussians.get_features[static_points_mask == 1], max_gaussians_features, min_gaussians_features, max_gaussians_features[:1])) # very hacky TODO make it right
                if(render_type == "static"):
                    shs = torch.vstack((gaussians.get_features[static_points_mask == 0], max_gaussians_features, min_gaussians_features, max_gaussians_features[:1]))
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
        def write_image_to_drive(input_tensor, index=None, type="static"):
                # transform torch tensor to rgb image and write to drive for DEBUG purposes
                transform = T.ToPILImage()
                img = transform(input_tensor[0])
                #img = img.save("debug/train_step_debug" + str(index) +".jpg")
                img = img.save("debug/gaussian_raster_depth_debug_" + str(type) + ".jpg")

        #write_image_to_drive(rendered_depth, type=render_type)

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