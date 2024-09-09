import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', default='logs', type=str, help='Directory where gaussian splat ply files are stored')
parser.add_argument('--out', default='videos', type=str, help='Directory where videos will be saved')
args = parser.parse_args()

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
os.system(f"python -m kiui.render {file} --save_video {os.path.join(out, name.replace('.ply', '.mp4'))} ")