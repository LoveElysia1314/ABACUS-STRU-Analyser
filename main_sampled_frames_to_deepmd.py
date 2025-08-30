import sys
import os
import importlib.util

repo_root = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(repo_root, 'src')
io_dir = os.path.join(src_dir, 'io')
target_file = os.path.join(io_dir, 'sampled_frames_to_deepmd.py')

spec = importlib.util.spec_from_file_location("sampled_frames_to_deepmd", target_file)
sampled_frames_to_deepmd = importlib.util.module_from_spec(spec)
sys.modules["sampled_frames_to_deepmd"] = sampled_frames_to_deepmd
spec.loader.exec_module(sampled_frames_to_deepmd)

if __name__ == "__main__":
    sampled_frames_to_deepmd.main()