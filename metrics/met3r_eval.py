from torchmetrics.multimodal.clip_score import CLIPScore
import torch
import PIL
import numpy as np

import torch.nn.functional as F

import open_clip
'''def compute_lpips(im0, im1):
    
    loss_fn = lpips.LPIPS(net='alex')

    d = loss_fn.forward(im0, im1) # normalize images to [-1, 1] before computing 
    return d
'''

#def compute_clip_score(im0, im1):
#    metric = CLIPScore(model_name_or_path='openai/clip-vit-base-patch16')
#    score = metric(im0, im1) / 100.0
#    return score.detach().round() 

'''
def compute_mse(im0, im1, mask): # im1 gt
    #im0 = im0.unsqueeze(0)
    #im1 = im1.unsqueeze(0)
    #print(str(im0.shape) + ", " + str(im1.shape))
    
    return F.mse_loss(im1 * mask, im0 * mask, reduction='mean') / im1.shape[0] * 100.0


def compute_open_clip_score(im0, im1):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()
    #tokenizer = open_clip.get_tokenizer('ViT-B-32')

    image0 = preprocess(im0).unsqueeze(0)
    image1 = preprocess(im1).unsqueeze(0)

    with torch.no_grad(), torch.autocast("cuda"):
        image0_features = model.encode_image(image0)
        image1_features = model.encode_image(image1)
        image0_features /= image0_features.norm(dim=-1, keepdim=True)
        image1_features /= image1_features.norm(dim=-1, keepdim=True)

        #score = (100.0 * image0_features @ image1_features.T)#.softmax(dim=-1)

        #print("SHAPE: " + str(image0_features.shape) + ", " + str(image1_features.shape))
        score = 100.0 * torch.dot(image0_features.squeeze(0), image1_features.squeeze(0))

    return score
'''

def compute_met3r_score(im0, im1):
    


def get_images_3(im0_path, im1_path, alpha_path=None):
    im0 = Image.open(im0_path)
    im1 = Image.open(im1_path)
    if alpha_path is not None:
        alpha = Image.open(alpha_path)
    im0 = fn.to_tensor(im0)# * 2.0 - 1.0 # to [-1, 1] range
    im1 = fn.to_tensor(im1)# * 2.0 - 1.0
    if alpha_path is not None:
        alpha = fn.to_tensor(alpha)

    ##print("IM0 HIGH: " + str(torch.max(im0)), "LOW: "  + str(torch.min(im0)))
    #print("IM1 HIGH: " + str(torch.max(im1)), "LOW: "  + str(torch.min(im1)))

    return im0, im1, alpha

def get_images(im0_path, im1_path):
    im0 = Image.open(im0_path)
    im1 = Image.open(im1_path)
    im0 = fn.to_tensor(im0)# * 2.0 - 1.0 # to [-1, 1] range
    im1 = fn.to_tensor(im1)# * 2.0 - 1.0

    ##print("IM0 HIGH: " + str(torch.max(im0)), "LOW: "  + str(torch.min(im0)))
    #print("IM1 HIGH: " + str(torch.max(im1)), "LOW: "  + str(torch.min(im1)))

    return im0, im1

def get_image(img_path):
    im = Image.open(img_path)
    im = fn.to_tensor(im)
    return im


def compute_metrics(model='', objects=None, folder_path=''):
    # ABLATION: no preservation loss
    if os.path.exists("data/metrics/met3r_" + model + ".txt"):
        os.remove("data/metrics/met3r_" + model + ".txt")
    #if os.path.exists("data/metrics/mse_depth_" + model + ".txt"):
    #    os.remove("data/metrics/mse_depth_" + model + ".txt")
    #if os.path.exists("data/metrics/mse_rgb_" + model + ".txt"):
    #    os.remove("data/metrics/mse_rgb_" + model + ".txt")

    #clip_scores = []
    #mse_depth_scores = []
    #mse_rgb_scores = []

    met3r_scores = []
    # TRELLIS
    for object in objects:
        #print("Max val: ", im0.max())
        #print("Min val: ", im0.min())
        im0_path = "data/test_images/input/" + str(object) + "/" + str(object) + "_input_0_color.png"
        im1_path = "data/test_images/" + folder_path + "/" + model + "/" + str(object) + "/" + str(object) + "_" + model + "_0_color.png"

        im2_path = "data/test_images/input/" + str(object) + "/" + str(object) + "_input_0_color.png"
        im3_path = "data/test_images/" + folder_path + "/" + model + "/" + str(object) + "/" + str(object) + "_" + model + "_90_color.png"

        im4_path = "data/test_images/input/" + str(object) + "/" + str(object) + "_input_0_color.png"
        im5_path = "data/test_images/" + folder_path + "/" + model + "/" + str(object) + "/" + str(object) + "_" + model + "_180_color.png"

        im6_path = "data/test_images/input/" + str(object) + "/" + str(object) + "_input_0_color.png"
        im7_path = "data/test_images/" + folder_path + "/" + model + "/" + str(object) + "/" + str(object) + "_" + model + "_270_color.png"


        depth0_path = "data/test_images/input/" + str(object) + "/" + str(object) + "_input_0_depth.png"
        depth1_path = "data/test_images/" + folder_path + "/" + model + "/" + str(object) + "/" + str(object) + "_" + model + "_0_depth.png"
        mask_path = "data/test_images/input/" + str(object) + "/" + str(object) + "_input_0_alpha.png"
        object_mask_path = "data/test_images/" + folder_path + "/" + model + "/" + str(object) + "/" + str(object) + "_" + model + "_0_alpha.png"
        alpha_path = "data/" + str(object) + "/" + str(object) + "_static_mask.png"
        im0, im1, mask = get_images_3(im0_path, im1_path, mask_path)
        object_mask = get_image(object_mask_path)
        im2, im3 = get_images(im2_path, im3_path)
        im4, im5 = get_images(im4_path, im5_path)
        im6, im7 = get_images(im6_path, im7_path)
        depth0, depth1 = get_images(depth0_path, depth1_path)
        #im0 = im0[:, :, :256] # static image, take only front facing image portion (first image) 
        #im1 = im1[:, :, 256:] # another perspective different form the front view (sensor view)
        #alpha = alpha[:, :, :256]
        #im1 = ( ((im1 + 1.0) / 2.0) * alpha + ((im0 + 1.0) / 2.0) * (torch.ones((3, 256, 256), dtype=torch.float) - alpha) ) * 2.0 - 1.0
        
        # binarize alpha to create mask
        mask = mask > 0.55 # make threshold stronger ? 
        # TODO combine with test objects masks to account only for regions inside both masks, instead of calculating regions that are in the background
        object_mask = object_mask > 0.55
        mask = torch.logical_and(mask, object_mask) # intersect both masks
        mask = mask.int() 
        #object_mask = object_mask.int()

        #debug_im = fn.to_pil_image(im1) #fn.to_pil_image((im1 + 1.0) / 2.0)
        #debug_im2 = fn.to_pil_image(im0)
        #debug_im3 = fn.to_pil_image(depth0)
        #debug_im4 = fn.to_pil_image(depth1)
        #debug_im5 = fn.to_pil_image(mask.float())
        #debug_im.save("data/metrics/clip_debug_" + str(model) + "/" + str(object) + "/" + str(object) + "_im1.png")
        #debug_im2.save("data/metrics/clip_debug_" + str(model) + "/" + str(object) + "/" + str(object) + "_im0.png")
        #debug_im3.save("data/metrics/mse_debug_" + str(model) + "/" + str(object) + "/" + str(object) + "_depth0.png")
        #debug_im4.save("data/metrics/mse_debug_" + str(model) + "/" + str(object) + "/" + str(object) + "_depth1.png")
        #debug_im5.save("data/metrics/mse_debug_" + str(model) + "/" + str(object) + "/" + str(object) + "_mask.png")

        
        imgs = [im0, im1, im2, im3, im4, im5, im6, im7]
        '''
        intermed_score = 0.0
        start_index = 0
        for i in range(0, 4):
            im0_clip = fn.to_pil_image(imgs[start_index])
            im1_clip = fn.to_pil_image(imgs[start_index+1])
            #clip_score = compute_open_clip_score(im0_clip, im1_clip)
            intermed_score += compute_open_clip_score(im0_clip, im1_clip)
            #print("start index: " + str(start_index))
            start_index += 2
        clip_score = intermed_score / (len(imgs)/2.0)
        clip_scores = np.append(clip_scores, clip_score.item())
        with open("data/metrics/clip_" + model + ".txt", "a") as f:
            for i in range(0, 20 - len(str(object))):
                f.write(" ")
            f.write(str(object) + ": " + str(clip_score.item()) + "\n")

        mse_depth = compute_mse(depth0, depth1, mask)
        mse_depth_scores = np.append(mse_depth_scores, mse_depth.item())
        with open("data/metrics/mse_depth_" + model + ".txt", "a") as f:
            for i in range(0, 20 - len(str(object))):
                f.write(" ")
            f.write(str(object) + ": " + str(mse_depth.item()) + "\n")

        mse_rgb = compute_mse(im0, im1, mask)
        mse_rgb_scores = np.append(mse_rgb_scores, mse_rgb.item())
        with open("data/metrics/mse_rgb_" + model + ".txt", "a") as f:
            for i in range(0, 20 - len(str(object))):
                f.write(" ")
            f.write(str(object) + ": " + str(mse_rgb.item()) + "\n")

    with open("data/metrics/clip_" + model + ".txt", "a") as f:
        f.write("\nMean: " + str(np.mean(clip_scores)))

    with open("data/metrics/mse_depth_" + model + ".txt", "a") as f:
        f.write("\nMean: " + str(np.mean(mse_depth_scores)))

    with open("data/metrics/mse_rgb_" + model + ".txt", "a") as f:
        f.write("\nMean: " + str(np.mean(mse_rgb_scores)))
        '''

    met3r_score = compute_met3r_score(im0, im1, mask)
    met3r_scores = np.append(met3r_scores, met3r_score.item())


if __name__ == "__main__":

    from PIL import Image
    import torchvision.transforms.functional as fn

    import os

    test_objects = ["shoe", "couch_blender", "vase", "elephant", "hocker", "banana_tuna", 
                    "chicken", "plant", "pumpkins", "knife_block", "rubiks_cube", "headset", 
                    "leather_book", "hat", "sponge", "coffee_mug", "bread", "fish",
                    "bear", "bicycle", "bonsai", "garden_desk", "train", "truck",
                    "diner_seats", "flip_flop", "orc_warrior", "pixel_cat", "trumpet",
                    "coffee_machine", "globe", "sofa", "wardrobe", "wooden_bench",
                    "flower", "lego_bulldozer", "onions", "pot_plant", "wood_bowl"]
    
    with torch.no_grad():
        #for object in test_objects:
            #print("Max val: ", im0.max())
            #print("Min val: ", im0.min())
        compute_metrics("input", objects=test_objects)
        compute_metrics("full", objects=test_objects)
        compute_metrics("trellis", objects=test_objects, folder_path="baselines/")
        compute_metrics("instantmesh", objects=test_objects, folder_path="baselines/")
        compute_metrics("trellis_mv", objects=test_objects, folder_path="baselines/")
        compute_metrics("tripoSG", objects=test_objects, folder_path="baselines/")
        #compute_metrics("no_preserve_loss", objects=test_objects)
        compute_metrics("no_preserve_no_init_no_schedule", objects=test_objects)
        #compute_metrics("no_schedule", objects=test_objects)

            



