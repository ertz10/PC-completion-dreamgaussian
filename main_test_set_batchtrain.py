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

    bear = False
    truck = False
    garden_desk = False
    train = False
    bicycle = False
    bonsai = False

    # partial meshes
    orc_warrior = False
    trumpet = False
    diner_seats = False
    pixel_cat = False
    hot_chocolate = False
    flip_flop = False

    # lidar scans
    wooden_bench = False
    christmas_tree = False
    coffee_machine = False
    sofa = False
    wardrobe = False
    lamp = False
    fruit_bowl = False
    globe = False

    # mono depth
    lego_bulldozer = False
    flower = False
    antique_bowl = False
    onions = False
    pot_plant = False
    wood_bowl = False   

    #shoe = True #new
    #couch = True #new
    #vase = True #new
    #elephant = True #new
    #hocker = True #new
    #banana_tuna = True
    #chicken = True # try again later, seems to build too many splats over time, reduce!
    #plant = True
    #pumpkins = True #new
    #knife_block = True
    #rubiks_cube = True #new
    #headset = True # TODO create crop less from the original gs file
    #tennis_ball = True # CHECKPOINT
    #flashlight = True
    #leather_book = True########
    #hat = True
    #sponge = True
    #coffee_mug = True
    #bread = True
    #fish = True

    bear = True
    truck = True
    garden_desk = True
    train = True
    bicycle = True
    bonsai = True

    '''
    # ablation looks good with partial meshes, no reprocess needed!
    orc_warrior = True
    trumpet = True
    diner_seats = True
    pixel_cat = True
    flip_flop = True
    '''
    # reprocess ablation on all lidar scans
    # lidar scans
    wooden_bench = True
    coffee_machine = True
    sofa = True
    wardrobe = True
    globe = True
    
    # mono depth
    lego_bulldozer = True
    flower = True # no reprocess needed
    onions = True # no reprocess needed
    pot_plant = True
    wood_bowl = True
    
    




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




    ########## bear ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a statue of a bear standing on its feet, made of stone\" "
    save_path = "save_path=bear "
    object_conf = "--object_conf=data/bear/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if bear:
        p = Popen(command)
        p.communicate()



    ########## truck ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"an old bright blue pickup truck\" "
    save_path = "save_path=truck "
    object_conf = "--object_conf=data/truck/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if truck:
        p = Popen(command)
        p.communicate()


    ########## garden desk ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a round garden table made of dark wood, with a vase on top\" "
    save_path = "save_path=garden_desk "
    object_conf = "--object_conf=data/garden_desk/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if garden_desk:
        p = Popen(command)
        p.communicate()

    
    ########## train ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"an old american freight train\" "
    save_path = "save_path=train "
    object_conf = "--object_conf=data/train/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if train:
        p = Popen(command)
        p.communicate()


    ########## bicycle ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a white thin bicycle in a park\" "
    save_path = "save_path=bicycle "
    object_conf = "--object_conf=data/bicycle/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if bicycle:
        p = Popen(command)
        p.communicate()


    ########## bonsai ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a bonsai made of lego\" "
    save_path = "save_path=bonsai "
    object_conf = "--object_conf=data/bonsai/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if bonsai:
        p = Popen(command)
        p.communicate()


    #########################################################################
    ######################### partial meshes ################################
    #########################################################################
    ########## orc warrior ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a stylized orc warrior\" "
    save_path = "save_path=PARTIAL_MESHES/orc_warrior "
    object_conf = "--object_conf=data/PARTIAL_MESHES/orc_warrior/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if orc_warrior:
        p = Popen(command)
        p.communicate()

    ########## trumpet ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a metal trumpet\" "
    save_path = "save_path=PARTIAL_MESHES/trumpet "
    object_conf = "--object_conf=data/PARTIAL_MESHES/trumpet/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if trumpet:
        p = Popen(command)
        p.communicate()

    ########## diner seats ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"two vintage diner seats with a table in the middle\" "
    save_path = "save_path=PARTIAL_MESHES/diner_seats "
    object_conf = "--object_conf=data/PARTIAL_MESHES/diner_seats/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if diner_seats:
        p = Popen(command)
        p.communicate()

    ########## pixel cat ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a 3 dimensional voxelated cat like toy\" "
    save_path = "save_path=PARTIAL_MESHES/pixel_cat "
    object_conf = "--object_conf=data/PARTIAL_MESHES/pixel_cat/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if pixel_cat:
        p = Popen(command)
        p.communicate()

    ########## hot chocolate ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a hot chocolate with marshmallows\" "
    save_path = "save_path=PARTIAL_MESHES/hot_chocolate "
    object_conf = "--object_conf=data/PARTIAL_MESHES/hot_chocolate/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if hot_chocolate:
        p = Popen(command)
        p.communicate()

    ########## flip flop ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a flip flop with a toe straps\" "
    save_path = "save_path=PARTIAL_MESHES/flip_flop "
    object_conf = "--object_conf=data/PARTIAL_MESHES/flip_flop/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if flip_flop:
        p = Popen(command)
        p.communicate()

    #########################################################################
    ######################### lidar scans ###################################
    #########################################################################

    ########## wooden bench ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a wooden bench\" "
    save_path = "save_path=LIDAR/wooden_bench "
    object_conf = "--object_conf=data/LIDAR/wooden_bench/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if wooden_bench:
        p = Popen(command)
        p.communicate()

    ########## christmas tree ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a decorated christmas tree\" "
    save_path = "save_path=LIDAR/christmas_tree "
    object_conf = "--object_conf=data/LIDAR/christmas_tree/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if christmas_tree:
        p = Popen(command)
        p.communicate()

    ########## coffee machine ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a full automatic red and silver coffee machine\" "
    save_path = "save_path=LIDAR/coffee_machine "
    object_conf = "--object_conf=data/LIDAR/coffee_machine/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if coffee_machine:
        p = Popen(command)
        p.communicate()

    ########## sofa ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a brown sofa with pillows\" "
    save_path = "save_path=LIDAR/sofa "
    object_conf = "--object_conf=data/LIDAR/sofa/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if sofa:
        p = Popen(command)
        p.communicate()

    ########## wardrobe ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a vintage wooden wardrobe\" "
    save_path = "save_path=LIDAR/wardrobe "
    object_conf = "--object_conf=data/LIDAR/wardrobe/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if wardrobe:
        p = Popen(command)
        p.communicate()


    ########## lamp ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a lamp with a vintage lampshade\" "
    save_path = "save_path=LIDAR/lamp "
    object_conf = "--object_conf=data/LIDAR/lamp/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if lamp:
        p = Popen(command)
        p.communicate()

    ########## fruit_bowl ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a glass bowl fruits inside\" "
    save_path = "save_path=LIDAR/fruit_bowl "
    object_conf = "--object_conf=data/LIDAR/fruit_bowl/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if fruit_bowl:
        p = Popen(command)
        p.communicate()

    ########## fruit_bowl ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a world globe with brownish, desaturated colors\" "
    save_path = "save_path=LIDAR/globe "
    object_conf = "--object_conf=data/LIDAR/globe/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if globe:
        p = Popen(command)
        p.communicate()


    ########## lego bulldozer ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a yellow lego bulldozer\" "
    save_path = "save_path=MONO_DEPTH/lego_bulldozer "
    object_conf = "--object_conf=data/MONO_DEPTH/lego_bulldozer/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if lego_bulldozer:
        p = Popen(command)
        p.communicate()

    ########## flower ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"an orange flower\" "
    save_path = "save_path=MONO_DEPTH/flower "
    object_conf = "--object_conf=data/MONO_DEPTH/flower/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if flower:
        p = Popen(command)
        p.communicate()

    ########## antique bowl ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a standing antique ceramic bowl with intricate details\" "
    save_path = "save_path=MONO_DEPTH/antique_bowl "
    object_conf = "--object_conf=data/MONO_DEPTH/antique_bowl/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if antique_bowl:
        p = Popen(command)
        p.communicate()

    ########## onions ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"three onions packed in a net\" "
    save_path = "save_path=MONO_DEPTH/onions "
    object_conf = "--object_conf=data/MONO_DEPTH/onions/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if onions:
        p = Popen(command)
        p.communicate()

    ########## pot plant  ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a plant in a brown clay pot\" "
    save_path = "save_path=MONO_DEPTH/pot_plant "
    object_conf = "--object_conf=data/MONO_DEPTH/pot_plant/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if pot_plant:
        p = Popen(command)
        p.communicate()

    ########## wood bowl ##############
    cmd = "python main.py --config configs/text_mv.yaml "
    prompt = "prompt=\"a wooden bowl with fruits inside\" "
    save_path = "save_path=MONO_DEPTH/wood_bowl "
    object_conf = "--object_conf=data/MONO_DEPTH/wood_bowl/conf.yaml"

    command = cmd + prompt + save_path + object_conf
    if wood_bowl:
        p = Popen(command)
        p.communicate()