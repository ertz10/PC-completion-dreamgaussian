import lpips
import torch
import PIL
def compute_lpips(im0, im1):
    
    loss_fn = lpips.LPIPS(net='alex')

    d = loss_fn.forward(im0, im1) # normalize images to [-1, 1] before computing 
    return d

def get_images(im0_path, im1_path, alpha_path):
    im0 = Image.open(im0_path)
    im1 = Image.open(im1_path)
    alpha = Image.open(alpha_path)
    im0 = fn.to_tensor(im0) * 2.0 - 1.0 # to [-1, 1] range
    im1 = fn.to_tensor(im1) * 2.0 - 1.0
    alpha = fn.to_tensor(alpha)

    return im0, im1, alpha

if __name__ == "__main__":

    from PIL import Image
    import torchvision.transforms.functional as fn

    import os
    if os.path.exists("data/metrics/lpips.txt"):
        os.remove("data/metrics/lpips.txt")

    test_objects = ["shoe", "couch_blender", "vase", "elephant", "hocker", "banana_tuna", "chicken", "plant", "pumpkins", "knife_block", "rubiks_cube", "headset", "leather_book", "hat", "sponge", "coffee_mug", "bread", "fish"]

    for object in test_objects:
        #print("Max val: ", im0.max())
        #print("Min val: ", im0.min())
        im0_path = "data/" + str(object) + "/" + str(object) + "_static.png"
        im1_path = "data/" + str(object) + "/" + str(object) + "_optimized.png"
        alpha_path = "data/" + str(object) + "/" + str(object) + "_static_mask.png"
        im0, im1, alpha = get_images(im0_path, im1_path, alpha_path)
        im0 = im0[:, :, :256] # static image, take only front facing image portion (first image) 
        im1 = im1[:, :, :256] # full image
        alpha = alpha[:, :, :256]
        im1 = ( ((im1 + 1.0) / 2.0) * alpha + ((im0 + 1.0) / 2.0) * (torch.ones((3, 256, 256), dtype=torch.float) - alpha) ) * 2.0 - 1.0
        debug_im = fn.to_pil_image((im1 + 1.0) / 2.0)
        debug_im.save("data/" + str(object) + "/" + str(object) + "_im1.png")
        
        lpips_val = compute_lpips(im0, im1)
        with open("data/metrics/lpips.txt", "a") as f:
            for i in range(0, 20 - len(str(object))):
                f.write(" ")
            f.write(str(object) + ": " + str(lpips_val.item()) + "\n")

    