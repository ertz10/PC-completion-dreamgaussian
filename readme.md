# GSComplete

This repository is based on [DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation](https://arxiv.org/abs/2309.16653).

### [Project Page](https://dreamgaussian.github.io) | [Arxiv](https://arxiv.org/abs/2309.16653)

### News

## Install

```bash
pip install -r requirements.txt

# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# simple-knn
pip install ./simple-knn

# nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast/

# To use MVdream, also install:
pip install git+https://github.com/bytedance/MVDream


Tested on:

- Windows 10 with torch 2.1 & CUDA 11.8 on a 3090.

## Usage

Text-to-3D:

```bash
### test objects (comment/comment for certain objects)
python main_test_set_batchtrain.py


Helper scripts:


## Tips
* The world & camera coordinate system is the same as OpenGL:
```
    World            Camera        
  
     +y              up  target                                              
     |               |  /                                            
     |               | /                                                
     |______+x       |/______right                                      
    /                /         
   /                /          
  /                /           
 +z               forward           

elevation: in (-90, 90), from +y to -y is (-90, 90)
azimuth: in (-180, 180), from +z to +x is (0, 90)
```

## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [threestudio](https://github.com/threestudio-project/threestudio)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
- [dearpygui](https://github.com/hoffstadt/DearPyGui)

## Citation
