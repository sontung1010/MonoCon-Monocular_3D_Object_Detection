import os
import sys
import torch
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from engine.monocon_engine import MonoconEngine
from utils.engine_utils import tprint, load_cfg, generate_random_seed, set_random_seed
from utils.kitti_convert_utils import *

# Arguments
parser = argparse.ArgumentParser('MonoCon Tester for KITTI 3D Object Detection Dataset')
parser.add_argument('--config_file',
                    type=str,
                    help="Path of the config file (.yaml)")
parser.add_argument('--checkpoint_file', 
                    type=str,
                    help="Path of the checkpoint file (.pth)")
parser.add_argument('--gpu_id', type=int, default=0, help="Index of GPU to use for testing")
parser.add_argument('--test_normal', action='store_true')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--save_dir', 
                    type=str,
                    help="Path of the directory to save the visualized results")

args = parser.parse_args()


# Some Torch Settings
torch_version = int(torch.__version__.split('.')[1])
if torch_version >= 7:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# Load Config
cfg = load_cfg(args.config_file)
# breakpoint()
cfg.DATA.TEST_SPLIT = 'test'
cfg.GPU_ID = args.gpu_id


# Set Benchmark
# If this is set to True, it may consume more memory. (Default: True)
if cfg.get('USE_BENCHMARK', True):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    tprint(f"CuDNN Benchmark is enabled.")


# Set Random Seed
seed = cfg.get('SEED', -1)
seed = generate_random_seed(seed)
set_random_seed(seed)

tprint(f"Using Random Seed {seed}")



engine = MonoconEngine(cfg, auto_resume=False, is_test=True, use_org=args.test_normal)
engine.load_checkpoint(args.checkpoint_file, verbose=True)


if(args.test_normal):
    filename  = str(args.save_dir) + '/normal/merged_test_normal.txt'
    vis_dir  = str(args.save_dir) + '/normal'
    # Initialize Engine
else:
    filename  = str(args.save_dir) + '/bonus/merged_test_bonus.txt'
    vis_dir  = str(args.save_dir) + '/bonus'
    # Initialize Engine
    # engine = MonoconEngine(cfg, auto_resume=False, is_test=True)
    # engine.load_checkpoint(args.checkpoint_file, verbose=True)

# # Evaluate
if args.evaluate:
    tprint("Mode: Evaluation")
    # op = engine.evaluate()
    eval_container, img_meta_list = engine.evaluate_test()
    
    kitti_3d_to_file(eval_container, img_meta_list, filename, single_file=True)


# Visualize
if args.visualize:
    tprint("Mode: Visualization")
    engine.visualize_test(vis_dir, args.test_normal,  draw_items=['2d', '3d', 'bev'])
    # engine.visualize_test(vis_dir, args.test_normal,  draw_items=['2d'])