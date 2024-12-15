import os
from main import GUI

import subprocess
from subprocess import Popen

#class TestSet:

if __name__ == "__main__":
    #import argparse
    #from omegaconf import OmegaConf
    # https://pavolkutaj.medium.com/how-to-attach-debugger-to-python-script-called-from-terminal-in-visual-studio-code-ddd377d99456
    #input("Press enter to start ... (this prompt enables attaching the Python DEBUGGER!)")

    #parser = argparse.ArgumentParser()
    #parser.add_argument("--config", required=True, help="path to the yaml config file")
    #args, extras = parser.parse_known_args()

    # override default config from cli
    #opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    #parser.add_argument("--object_conf", required=True, help="path to the object's config file")
    #args, extras = parser.parse_known_args()
    #opt_object = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    #gui = GUI(opt, opt_object)

    #if opt.gui:
    #    gui.render()
    #else:
    #    gui.train(opt.iters)

    # test objects
    ########## SHOE ###############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a beige winter shoe with beige shoelaces, dark brown shoe sole, natural colors\" "
    save_path = "save_path=shoe "
    load = "load=data/shoe/shoe_cropped.ply "
    object_conf = "--object_conf=data/shoe/conf.yaml"

    # send command
    command = cmd + prompt + save_path + load + object_conf
    #s = subprocess.run(command, capture_output=True)
    #p = Popen(command)
    #p.communicate()
    #print(s.stdout)

    ########## COUCH ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a red leather couch with three seats\" "
    save_path = "save_path=couch "
    load = "load=data/couch_blender/couch_cropped.ply "
    object_conf = "--object_conf=data/couch_blender/conf.yaml"

    command = cmd + prompt + save_path + load + object_conf
    #p = Popen(command)
    #p.communicate()

    ########## VASE ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a photo of a white matte vase with a plant in it\" "
    save_path = "save_path=vase "
    load = "load=data/vase/vase_cropped.ply "
    object_conf = "--object_conf=data/vase/conf.yaml"

    command = cmd + prompt + save_path + load + object_conf
    #p = Popen(command)
    #p.communicate()

    ########## ELEPHANT ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a white knitted plush elephant\" "
    save_path = "save_path=elephant "
    load = "load=data/elephant/elephant_cropped_v2.ply "
    object_conf = "--object_conf=data/elephant/conf.yaml"

    #command = cmd + prompt + save_path + load + object_conf
    #p = Popen(command)
    #p.communicate()

    ########## HOCKER ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a photo of a square dark green footrest from ikea\" "
    save_path = "save_path=hocker "
    load = "load=data/hocker/hocker_cropped.ply "
    object_conf = "--object_conf=data/hocker/conf.yaml"

    command = cmd + prompt + save_path + load + object_conf
    #p = Popen(command)
    #p.communicate()

    ########## BANANA TUNA ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"two connected bananas and a tuna can next to them\" "
    save_path = "save_path=banana_tuna "
    load = "load=data/banana_tuna/banana_tuna_cropped.ply "
    object_conf = "--object_conf=data/banana_tuna/conf.yaml"

    command = cmd + prompt + save_path + load + object_conf
    #p = Popen(command)
    #p.communicate()
#
    ########## CHICKEN ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a photo of a brownish and white toy chicken\" "
    save_path = "save_path=chicken "
    load = "load=data/chicken/chicken_cropped.ply "
    object_conf = "--object_conf=data/chicken/conf.yaml"

    command = cmd + prompt + save_path + load + object_conf
    #p = Popen(command)
    #p.communicate()

    ########## PLANT ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a snake plant in a beige coarsely braided pot\" "
    save_path = "save_path=plant "
    load = "load=data/plant/plant_cropped.ply "
    object_conf = "--object_conf=data/plant/conf.yaml"

    command = cmd + prompt + save_path + load + object_conf
    #p = Popen(command)
    #p.communicate()

    ########## PUMPKINS ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a DLSR picture of two pumpkins\" "
    save_path = "save_path=pumpkins "
    load = "load=data/pumpkins/pumpkins_cropped.ply "
    object_conf = "--object_conf=data/pumpkins/conf.yaml"

    command = cmd + prompt + save_path + load + object_conf
    p = Popen(command)
    p.communicate()

    ########## KNIFE BLOCK ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a DLSR picture a dark wooden knife block with knifes in it\" "
    save_path = "save_path=knife_block "
    load = "load=data/knife_block/knife_block_cropped.ply "
    object_conf = "--object_conf=data/knife_block/conf.yaml"

    command = cmd + prompt + save_path + load + object_conf
    #p = Popen(command)
    #p.communicate()