import torch

def compute_loss_rgb_ssim(pred, batch_data):
    from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
    from torchvision.transforms.functional import resize
    rgb_target = batch_data['rgb_gt'].clone()
    unknown_mask = torch.isnan(rgb_target)
    rgb_target[unknown_mask] = pred[unknown_mask]
    #pred_resized = resize(pred, size=[256, 256], antialias=True)
    #rgb_target_resized = resize(rgb_target, size=[256, 256], antialias=True)
    rgb_loss = 1.0 - ms_ssim(pred_resized, rgb_target_resized, data_range=1.0, size_average=False, win_size=5)

    # broadcast to input B,H,W size
    b = rgb_loss.shape[0]
    h = rgb_target.shape[2]
    w = rgb_target.shape[3]
    rgb_loss = rgb_loss[:, None, None].broadcast_to((b, h, w))
    return rgb_loss


if __name__ == "__main__":

    from PIL import Image
    import torchvision.transforms.functional as fn

    import os
    if os.path.exists("data/metrics/msssim.txt"):
        os.remove("data/metrics/msssim.txt")

    test_objects = ["shoe", "couch_blender", "vase", "elephant", "hocker", "banana_tuna", "chicken", "plant", "pumpkins", "knife_block", "rubiks_cube", "headset", "tennis_ball", "flashlight", "leather_book", "hat", "sponge", "coffee_mug", "bread"]

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