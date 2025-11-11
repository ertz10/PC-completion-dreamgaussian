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
    headset = False
    tennis_ball = False
    flashlight = False
    leather_book = False
    hat = False
    sponge = False
    coffee_mug = False
    bread = False
    fish = False

    shoe = True #new
    couch = True #new
    vase = True #new
    elephant = True #new
    hocker = True #new
    banana_tuna = True
    chicken = True # try again later, seems to build too many splats over time, reduce!
    plant = True
    pumpkins = True #new
    knife_block = True
    rubiks_cube = True #new
    headset = True # TODO create crop less from the original gs file
    #tennis_ball = True # CHECKPOINT
    #flashlight = True
    leather_book = True########
    hat = True
    sponge = True
    coffee_mug = True
    bread = True
    fish = True





    # test objects
    ########## SHOE ###############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a beige winter shoe with beige shoelaces, dark brown shoe sole, natural colors\" "
    #prompt = "prompt=\"a DSLR foto of a bright beige winter shoe, dark brown shoe sole, natural colors, desaturated\" "
    #prompt = "prompt=\"desaturated\" "
    save_path = "save_path=shoe "
    object_conf = "--object_conf=data/shoe/conf.yaml"

    # send command
    command = cmd + prompt + save_path + object_conf
    if shoe:
        p = Popen(command)
        p.communicate()

    ########## COUCH ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a red leather couch with two seats\" "
    save_path = "save_path=couch "
    object_conf = "--object_conf=data/couch_blender/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if couch:
        p = Popen(command)
        p.communicate()

    ########## VASE ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a white matte ceramic flower pot with a smooth surface without pattern and with a plant in it\" "
    save_path = "save_path=vase "
    object_conf = "--object_conf=data/vase/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if vase:
        p = Popen(command)
        p.communicate()

    ########## ELEPHANT ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a white knitted plush elephant with two ears\" "
    save_path = "save_path=elephant "
    object_conf = "--object_conf=data/elephant/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if elephant:
        p = Popen(command)
        p.communicate()

    ########## HOCKER ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a photo of a rectangular dark green footrest made of microfiber cloth\" "
    save_path = "save_path=hocker "
    object_conf = "--object_conf=data/hocker/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if hocker:
        p = Popen(command)
        p.communicate()

    ########## BANANA TUNA ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"two old bananas that are connected and a tuna can\" "
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
    prompt = "prompt=\"a DLSR picture a dark wooden kitchen knife block with knifes in it\" "
    save_path = "save_path=knife_block "
    object_conf = "--object_conf=data/knife_block/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if knife_block:
        p = Popen(command)
        p.communicate()

    ########## RUBIKS CUBE ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a rubiks cube with random colors on each side\" "
    save_path = "save_path=rubiks_cube "
    object_conf = "--object_conf=data/rubiks_cube/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if rubiks_cube:
        p = Popen(command)
        p.communicate()

    ########## HEADSET ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a black computer headset with microphone\" "
    save_path = "save_path=headset "
    object_conf = "--object_conf=data/headset/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if headset:
        p = Popen(command)
        p.communicate()

    ########## TENNIS BALL ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a DLSR photo of a classic green tennis ball, desaturated\" "
    save_path = "save_path=tennis_ball "
    object_conf = "--object_conf=data/tennis_ball/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if tennis_ball:
        p = Popen(command)
        p.communicate()

    ########## FLASHLIGHT ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a flashlight with a black and silver casing\" "
    save_path = "save_path=flashlight "
    object_conf = "--object_conf=data/flashlight/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if flashlight:
        p = Popen(command)
        p.communicate()

    ########## LEATHER BOOK ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a book with a dark red leather cover with subtle ornaments\" "
    save_path = "save_path=leather_book "
    object_conf = "--object_conf=data/leather_book/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if leather_book:
        p = Popen(command)
        p.communicate()

    ########## Hat ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a foto of a round safari bush hat, beige color, desaturated, natural light\" "
    save_path = "save_path=hat "
    object_conf = "--object_conf=data/hat/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if hat:
        p = Popen(command)
        p.communicate()


    ########## Sponge ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a foto of a bright yellow rectangular kitchen sponge with small pores and a small green top part\" "
    save_path = "save_path=sponge "
    object_conf = "--object_conf=data/sponge/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if sponge:
        p = Popen(command)
        p.communicate()

    
    ########## Coffee mug ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a coffee mug with noisy dark brown color, desaturated\" "
    save_path = "save_path=coffee_mug "
    object_conf = "--object_conf=data/coffee_mug/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if coffee_mug:
        p = Popen(command)
        p.communicate()


    ########## bread ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a foto of a bread with one side cut\" "
    save_path = "save_path=bread "
    object_conf = "--object_conf=data/bread/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if bread:
        p = Popen(command)
        p.communicate()


    ########## fish ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a fish made of ceramic with a long tail and scales, bright blue color, desaturated\" "
    save_path = "save_path=fish "
    object_conf = "--object_conf=data/fish/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if fish:
        p = Popen(command)
        p.communicate()