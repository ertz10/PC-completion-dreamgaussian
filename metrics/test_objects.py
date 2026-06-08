
import os, sys
sys.path.append('../')


from gsply_loader import GSPLY_Handler
import glob
import argparse
from omegaconf import OmegaConf

import numpy as np

class test_capsule:

    def __init__(self, test_name, opt_object_path_prefix, opt_path, path, image_folder_path, image_suffix, 
                 object_suffix, opt_object_alignment_prefix, mesh_path_prefix, *renderArgs):
        
        self.test_name = test_name
        
        self.opt_object_path_prefix = opt_object_path_prefix
    
        self.opt_object_alignment_prefix = opt_object_alignment_prefix
        self.opt_path = opt_path
        self.opt_object = None #OmegaConf.merge(OmegaConf.load(opt_object_path))
        self.opt = OmegaConf.merge(OmegaConf.load(opt_path))
        self.opt_alignment = None
        self.path = path
        self.object_path = "" # will be filled later
        self.object_suffix = object_suffix
        self.object_name = ""
        self.image_path = "" # --,,--
        self.image_folder_path = image_folder_path
        self.image_suffix = image_suffix
        self.data_handler = GSPLY_Handler(path, "", None, opt=self.opt)
        #self.data_handler.initialize(renderArgs)
        self.renderArgs = list(renderArgs)

        self.mesh_path_prefix = mesh_path_prefix

    def loadObjectConf(self, object):
        self.opt_object = OmegaConf.merge(OmegaConf.load(self.opt_object_path_prefix + str(object)))
        self.data_handler.opt_object = self.opt_object

    def loadOptAlignmentConf(self, object):
        print("LOADING OPT_ALIGNMENT: " + str(self.opt_object_alignment_prefix + str(object) + "_config.yaml"))
        self.opt_alignment = OmegaConf.merge(OmegaConf.load(self.opt_object_alignment_prefix + str(object) + "_config.yaml"))
        self.data_handler.opt_alignment = self.opt_alignment

    def loadMesh(self, object):
        print("Loading MESH: " + self.mesh_path_prefix + str(object) + "/" + str(object) + str(self.object_suffix))
        self.data_handler.loadMesh(self.mesh_path_prefix + str(object) + "/" + str(object) + str(self.object_suffix))

    def loadRenderer(self):
        print(str(self.path))
        print(str(self.object_name))
        self.renderArgs[1] = self.path + self.object_name + "/" + self.object_name + self.object_suffix
        print("RENDER ARGS " + str(self.renderArgs))
        print("OPT_ALIGNMENT PATH: " + str(self.opt_alignment))
        self.data_handler.loadRenderer(self.opt_object, self.opt_alignment, *self.renderArgs)


if __name__ == "__main__":
    
    
    
    #test_objects = ["wooden_bench"]
    #test_objects = ["fish"]
    #
    #test_objects = ["bear", "bicycle", "bonsai", "garden_desk", "train", "truck"]
    #test_objects = [""]
    #test_partial_meshes = ["diner_seats", "flip_flop", "orc_warrior", "pixel_cat", "trumpet"]

    # [isRawPCD=False, input=None, num_pts=5000, radius=0.5, AABB=np.array((0, 1, 0, 1, 0, 1)), spatial_lr_scale=1, no_transform=False, 
    # no_rotation=False, blob_init_size=None, num_pts_init=None, flip_z=False, normalize=True, transform_splats_only=False]
    args = [False, "", 5000, 0.5, np.array((0, 1, 0, 1, 0, 1)), 1, False, False, 0.0001, 1, False, True, False]
    test =                      "input"
    opt_object_path_prefix =    "../data/BACKUPS/full_pipe/"
    opt_path =                  "../configs/text_mv.yaml"
    path =                      "../data/BACKUPS/full_pipe/"
    image_folder_path =         "../data/test_images/" + test + "/"
    image_suffix =              "_" + test
    obj_suffix =                "_cropped.ply"
    #obj_suffix =                "_static.ply"

    opt_alignment_prefix =      None
    mesh_path =                 None
    test_capsule_input = test_capsule(test, opt_object_path_prefix, opt_path, path,
                                    image_folder_path, image_suffix, obj_suffix, opt_alignment_prefix, mesh_path, *args)
    

    # our test objects and capture from different angles
    # load gsply's and capture reference image
    #False, object_path, no_transform=True, no_rotation=True, blob_init_size=0.0001, num_pts_init=1, flip_z=False, normalize=False, transform_splats_only=False
    args = [False, "", 5000, 0.5, np.array((0, 1, 0, 1, 0, 1)), 1, True, True, 0.0001, 1, False, False, False]
    test =                      "full"
    opt_object_path_prefix =    "../data/BACKUPS/full_pipe/"
    opt_path =                  "../configs/text_mv.yaml"
    path =                      "../data/BACKUPS/full_pipe/"
    image_folder_path =         "../data/test_images/" + test + "/"
    image_suffix =              "_" + test
    obj_suffix =                "_final.ply"

    opt_alignment_prefix =      None
    mesh_path =                 None
    test_capsule_full = test_capsule(test, opt_object_path_prefix, opt_path, path,
                                    image_folder_path, image_suffix, obj_suffix, opt_alignment_prefix, mesh_path, *args)
    
    # ABLATION: no preserve loss
    args = [False, "", 5000, 0.5, np.array((0, 1, 0, 1, 0, 1)), 1, True, True, 0.0001, 1, False, False, False]
    test =                      "no_preserve_loss"
    opt_object_path_prefix =    "../data/BACKUPS/" + test + "/"
    opt_path =                  "../configs/text_mv.yaml"
    path =                      "../data/BACKUPS/" + test + "/"
    image_folder_path =         "../data/test_images/" + test + "/"
    image_suffix =              "_" + test

    opt_alignment_prefix =      None
    mesh_path =                 None
    test_capsule_npl = test_capsule(test, opt_object_path_prefix, opt_path, path,
                                    image_folder_path, image_suffix, obj_suffix, opt_alignment_prefix, mesh_path, *args)
    
    # ABLATION: no preserve loss no init no schedule
    args = [False, "", 5000, 0.5, np.array((0, 1, 0, 1, 0, 1)), 1, True, True, 0.0001, 1, False, False, False]
    test =                      "no_preserve_no_init_no_schedule"
    opt_object_path_prefix =    "../data/BACKUPS/" + test + "/"
    opt_path =                  "../configs/text_mv.yaml"
    path =                      "../data/BACKUPS/" + test + "/"
    image_folder_path =         "../data/test_images/" + test + "/"
    image_suffix =              "_" + test

    opt_alignment_prefix =      None
    mesh_path =                 None
    test_capsule_nplnoinos = test_capsule(test, opt_object_path_prefix, opt_path, path,
                                    image_folder_path, image_suffix, obj_suffix, opt_alignment_prefix, mesh_path, *args)
    
    # ABLATION: no schedule
    args = [False, "", 5000, 0.5, np.array((0, 1, 0, 1, 0, 1)), 1, True, True, 0.0001, 1, False, False, False]
    test =                      "no_schedule"
    opt_object_path_prefix =    "../data/BACKUPS/" + test + "/"
    opt_path =                  "../configs/text_mv.yaml"
    path =                      "../data/BACKUPS/" + test + "/"
    image_folder_path =         "../data/test_images/" + test + "/"
    image_suffix =              "_" + test

    opt_alignment_prefix =      None
    mesh_path =                 None
    test_capsule_nos = test_capsule(test, opt_object_path_prefix, opt_path, path,
                                    image_folder_path, image_suffix, obj_suffix, opt_alignment_prefix, mesh_path, *args)

    # TRELLIS # ATTENTION, Trellis and Trellis MV needs manual transformation (rotation) of local gaussian splat coordinates by config, since gaussian splats are not perfectly circular
    # load gsply's and capture reference image
    # [isRawPCD=False, input=None, num_pts=5000, radius=0.5, AABB=np.array((0, 1, 0, 1, 0, 1)), spatial_lr_scale=1, no_transform=False, 
    # no_rotation=False, blob_init_size=None, num_pts_init=None, flip_z=False, normalize=True, transform_splats_only=False]
    args = [False, "", 5000, 0.5, np.array((0, 1, 0, 1, 0, 1)), 1, True, False, 0.0001, 1, False, False, True]
    test =                      "trellis"
    opt_object_path_prefix =    "../data/BACKUPS/full_pipe/"
    opt_path =                  "../configs/text_mv.yaml"
    path =                      "../data/metrics/baselines/" + test + "/"
    image_folder_path =         "../data/test_images/baselines/" + test + "/"
    image_suffix =              "_" + test
    obj_suffix =                "_aligned.ply"

    opt_alignment_prefix =      "../data/metrics/baselines/" + test + "/configs/"
    mesh_path =                 None
    test_capsule_trellis = test_capsule(test, opt_object_path_prefix, opt_path, path,
                                    image_folder_path, image_suffix, obj_suffix, opt_alignment_prefix, mesh_path, *args)
    
    # TRELLIS MV # ATTENTION, Trellis and Trellis MV needs manual transformation (rotation) by config, since gaussian splats are not perfectly circular
    # load gsply's and capture reference image
    args = [False, "", 5000, 0.5, np.array((0, 1, 0, 1, 0, 1)), 1, True, False, 0.0001, 1, False, False, True]
    test =                      "trellis_mv"
    opt_object_path_prefix =    "../data/BACKUPS/full_pipe/"
    opt_path =                  "../configs/text_mv.yaml"
    path =                      "../data/metrics/baselines/" + test + "/"
    image_folder_path =         "../data/test_images/baselines/" + test + "/"
    image_suffix =              "_" + test
    obj_suffix =                "_mv_aligned.ply"

    opt_alignment_prefix =      "../data/metrics/baselines/" + test + "/configs/"
    mesh_path =                 None
    test_capsule_trellis_mv = test_capsule(test, opt_object_path_prefix, opt_path, path,
                                    image_folder_path, image_suffix, obj_suffix, opt_alignment_prefix, mesh_path, *args)
    

    # InstantMesh, use mesh loader here ???
    # load gsply's and capture reference image
    args = [False, "", 5000, 0.5, np.array((0, 1, 0, 1, 0, 1)), 1, True, False, 0.0001, 1, False, False, True]
    test =                      "instantMesh"
    opt_object_path_prefix =    "../data/BACKUPS/full_pipe/"
    opt_path =                  "../configs/text_mv.yaml"
    path =                      "../data/metrics/baselines/" + test + "/"
    image_folder_path =         "../data/test_images/baselines/" + test + "/"
    image_suffix =              "_" + test
    obj_suffix =                "_aligned.ply"

    opt_alignment_prefix =      "../data/metrics/baselines/" + test + "/configs/"
    mesh_path =                 None
    test_capsule_instantmesh = test_capsule(test, opt_object_path_prefix, opt_path, path,
                                    image_folder_path, image_suffix, obj_suffix, opt_alignment_prefix, mesh_path, *args)
    
    # TripoSG, use mesh loader here
    # load gsply's and capture reference image
    args = [False, "", 5000, 0.5, np.array((0, 1, 0, 1, 0, 1)), 1, True, False, 0.0001, 1, False, False, True]
    test =                      "tripoSG"
    opt_object_path_prefix =    "../data/BACKUPS/full_pipe/"
    opt_path =                  "../configs/text_mv.yaml"
    path =                      "../data/metrics/baselines/" + test + "/"
    image_folder_path =         "../data/test_images/baselines/" + test + "/"
    image_suffix =              "_" + test
    obj_suffix =                "_aligned.ply"

    opt_alignment_prefix =      "../data/metrics/baselines/" + test + "/configs/"
    mesh_path =                 None #"../data/metrics/baselines/" + test + "/"
    test_capsule_triposg = test_capsule(test, opt_object_path_prefix, opt_path, path,
                                    image_folder_path, image_suffix, obj_suffix, opt_alignment_prefix, mesh_path, *args)
    

    #test_capsules = [test_capsule_input, test_capsule_full, test_capsule_nplnoinos, test_capsule_nos, test_capsule_npl,
    #                 test_capsule_trellis, test_capsule_trellis_mv, test_capsule_instantmesh, test_capsule_triposg]
    test_capsules = [test_capsule_input, test_capsule_full, test_capsule_nplnoinos, test_capsule_nos, test_capsule_npl,
                     test_capsule_trellis, test_capsule_trellis_mv, test_capsule_instantmesh, test_capsule_triposg]
    #test_capsules = [
    #                 test_capsule_trellis, test_capsule_trellis_mv, test_capsule_instantmesh, test_capsule_triposg]
    #test_capsules = [
    #                 test_capsule_trellis_mv, test_capsule_instantmesh, test_capsule_triposg]

    #test_capsules = [
    #                 test_capsule_instantmesh, test_capsule_triposg]
    #test_capsules = [test_capsule_triposg]
    #test_capsules = [test_capsule_trellis]
    #test_capsules = [test_capsule_trellis_mv]

    #test_capsules = [test_capsule_instantmesh]

    #test_capsules = [test_capsule_instantmesh]
    #test_capsules = [test_capsule_nplnoinos]

    # use with input _cropped
    test_objects = ["shoe", "couch_blender", "vase", "elephant", "hocker", "banana_tuna", 
                    "chicken", "plant", "pumpkins", "knife_block", "rubiks_cube", "headset", 
                    "leather_book", "hat", "sponge", "coffee_mug", "bread", "fish",
                    "bear", "bicycle", "bonsai", "garden_desk", "train", "truck",
                    "diner_seats", "flip_flop", "orc_warrior", "pixel_cat", "trumpet"]
                
    # use with input _static
    #test_objects = ["coffee_machine", "globe", "sofa", "wardrobe", "wooden_bench",
    #                "flower", "lego_bulldozer", "onions", "pot_plant", "wood_bowl"]
    
    #test_objects = ["bear", "bicycle", "bonsai", "garden_desk", "train", "truck", "banana_tuna"] # run again with _cropped instead of _static
    
    #test_objects = ["coffee_machine", "flip_flop"]

    from omegaconf import OmegaConf

    ref_depth_min = np.zeros((len(test_objects), 8)) # reference input images' norm factors
    ref_depth_norm_fac = np.zeros((len(test_objects), 8)) # reference input images' norm factors
    for tc_item in test_capsules:
        idx = 0
        for object in test_objects:
            tc_item.object_name = object
            tc_item.loadObjectConf(str(object) + "/conf.yaml")

            if not tc_item.opt_object_alignment_prefix == None:
                tc_item.loadOptAlignmentConf(str(object))

            if not tc_item.mesh_path_prefix == None:
                tc_item.loadMesh(str(object))
                tc_item.object_path = tc_item.mesh_path_prefix + str(object) + "/" + str(object) + tc_item.object_suffix

            if not tc_item.path == None:
                tc_item.loadRenderer()
                tc_item.object_path = tc_item.path + str(object) + "/" + str(object) + tc_item.object_suffix

            opt_path = tc_item.opt_object
            opt_object = tc_item.opt_object
            opt = tc_item.opt

            # load gsply
            #tc_item.path = tc_item.path + str(object) + "/" + str(object) # + "/" + str(object) + ".ply"


            
            tc_item.data_handler.object_name = str(object)
            tc_item.image_path = tc_item.image_folder_path + str(object) + "/" + str(object) + tc_item.image_suffix
            for i in range(0,8):
                azimuth = tc_item.opt_object.reference_angle_hor + 45.0 * i
                iter_angle = 45.0 * i
                if tc_item.test_name == "input":
                    # additionally save norm factors of input depth
                    ref_depth_min[idx, i], ref_depth_norm_fac[idx, i] = tc_item.data_handler.train_step(tc_item.image_path, azimuth, iter_angle)
                else:
                    #print("ref_depth_min: ", ref_depth_min)
                    #print("ref_depth_norm_fac.shape: ",  ref_depth_norm_fac)
                    tc_item.data_handler.train_step(tc_item.image_path, azimuth, iter_angle, ref_depth_min=ref_depth_min[idx, i], ref_depth_norm_fac=ref_depth_norm_fac[idx, i]) 
            idx += 1







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
