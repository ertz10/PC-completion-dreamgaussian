if __name__ == "__main__":
    import os
    import glob
    import argparse
    from gaussianSplatsRender import GaussianCustomRenderer
    from gs_renderer import GaussianModel, MiniCam, Renderer
    import cv2

    import torch

    from cam_utils import orbit_camera, OrbitCamera
    import numpy as np

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='logs', type=str, help='Directory where gaussian splat ply files are stored')
    parser.add_argument('--out', default='videos', type=str, help='Directory where videos will be saved')
    parser.add_argument('--object_params', type=str)
    args = parser.parse_args()

    opt_object = OmegaConf.merge(OmegaConf.load(args.object_params))

    out = args.out
    os.makedirs(out, exist_ok=True)

    #files = glob.glob(f'{args.dir}/*.obj')
    file = f'{args.file}'
    #for f in files:
    name = os.path.basename(file)
    # first stage model, ignore
    #if name.endswith('_model.ply'): 
    #    continue
    print(f'[INFO] process {name}')
    #TODO implement own rendering with increasing horizontal values, call gaussian renderer
    #os.system(f"python -m kiui.render {file} --save_video {os.path.join(out, name.replace('.ply', '.mp4'))} ")

    # create gaussian model
    #gaussians = GaussianModel(0, opt_object)
    #gaussians.load_static_ply(file, 1)

    #gaussian_renderer_custom = GaussianCustomRenderer()
    gaussian_renderer = Renderer(opt_object=opt_object) # also loads gaussian model
    gaussian_renderer.gaussians.load_static_ply(file, 1)


    cam = OrbitCamera(1024, 1024, r=2.5, fovy=49.1)

    images = []
    for i in range(0, 1200):
        hor_i = i * 360 / 1200
        pose = orbit_camera(45, hor_i, cam.radius)
        cur_cam = MiniCam(pose, 1024, 1024, cam.fovy, cam.fovx, cam.near, cam.far)
        
        images.append(gaussian_renderer.render(viewpoint_camera=cur_cam, bg_color=torch.Tensor([1, 1, 1])))

    try:
        #for x in range(0, 30):
        #    self.timelapse_imgs.append(self.timelapse_imgs[len(self.timelapse_imgs) - 1])
        out = cv2.VideoWriter(str(opt_object.data_path) + '/360_view.mp4', cv2.VideoWriter.fourcc(*'mp4v'), 30, (1024, 1024))
        #ext_frames = np.repeat(self.timelapse_imgs, 60)
        for frame in images:
            out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        out.release()
    except OSError:
        print("Cannot save image")
        
