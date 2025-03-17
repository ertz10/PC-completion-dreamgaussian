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


    shoe = False
    couch = False
    vase = False
    elephant = False
    hocker = False
    banana_tuna = False
    chicken = False
    plant = False
    pumpkins = False
    knife_block = False
    rubiks_cube = False

    shoe = True
    couch = True
    #vase = True
    #elephant = True
    #hocker = True
    #banana_tuna = True
    #chicken = True
    #plant = True
    #pumpkins = True
    #knife_block = True
    #rubiks_cube = True





    # test objects
    ########## SHOE ###############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a beige winter shoe with beige shoelaces, dark brown shoe sole, natural colors\" "
    save_path = "save_path=shoe "
    object_conf = "--object_conf=data/shoe/conf.yaml"

    # send command
    command = cmd + prompt + save_path + object_conf
    if shoe:
        p = Popen(command)
        p.communicate()

    ########## COUCH ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a red leather couch with three seats\" "
    save_path = "save_path=couch "
    object_conf = "--object_conf=data/couch_blender/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if couch:
        p = Popen(command)
        p.communicate()

    ########## VASE ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a photo of a white matte ceramic vase with a smooth surface with a plant in it\" "
    save_path = "save_path=vase "
    object_conf = "--object_conf=data/vase/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if vase:
        p = Popen(command)
        p.communicate()

    ########## ELEPHANT ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a pure white knitted toy elephant\" "
    save_path = "save_path=elephant "
    object_conf = "--object_conf=data/elephant/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if elephant:
        p = Popen(command)
        p.communicate()

    ########## HOCKER ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a photo of a square dark green footrest from ikea\" "
    save_path = "save_path=hocker "
    object_conf = "--object_conf=data/hocker/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if hocker:
        p = Popen(command)
        p.communicate()

    ########## BANANA TUNA ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"two connected bananas and a tuna can next to them\" "
    save_path = "save_path=banana_tuna "
    object_conf = "--object_conf=data/banana_tuna/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if banana_tuna:
        p = Popen(command)
        p.communicate()
#
    ########## CHICKEN ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a photo of a brownish toy chicken\" "
    save_path = "save_path=chicken "
    object_conf = "--object_conf=data/chicken/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if chicken:
        p = Popen(command)
        p.communicate()

    ########## PLANT ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a snake plant in a beige coarsely braided pot\" "
    save_path = "save_path=plant "
    object_conf = "--object_conf=data/plant/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if plant:
        p = Popen(command)
        p.communicate()

    ########## PUMPKINS ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"two hokkaido pumpkins placed next to each other\" "
    save_path = "save_path=pumpkins "
    object_conf = "--object_conf=data/pumpkins/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if pumpkins:
        p = Popen(command)
        p.communicate()

    ########## KNIFE BLOCK ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a DLSR picture a dark wooden knife block with knifes in it\" "
    save_path = "save_path=knife_block "
    object_conf = "--object_conf=data/knife_block/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if knife_block:
        p = Popen(command)
        p.communicate()

    ########## RUBIKS CUBE ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a rubiks cube\" "
    save_path = "save_path=rubiks_cube "
    object_conf = "--object_conf=data/rubiks_cube/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if rubiks_cube:
        p = Popen(command)
        p.communicate()