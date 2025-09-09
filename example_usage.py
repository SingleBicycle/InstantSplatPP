#!/usr/bin/env python3
"""
Example usage script demonstrating how to use different prior models
with the updated init_geo.py and init_test_pose.py scripts.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run a command and print the output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(f"Success: {result.stdout}")
    return result.returncode == 0

def main():
    """Demonstrate usage of different prior models."""
    
    # Example parameters
    source_path = "./assets/sora/Art"  # Path to your images
    model_path = "./output/example"    # Output directory
    device = "cuda"
    
    print("=== Example Usage of Different Prior Models ===\n")
    
    # Example 1: Using MASt3R (default)
    print("1. Using MASt3R model (default):")
    cmd = [
        "python", "init_geo.py",
        "--source_path", source_path,
        "--model_path", f"{model_path}_mast3r",
        "--model_type", "mast3r",
        "--device", device
    ]
    # run_command(cmd)  # Uncomment to actually run
    
    print("\n2. Using VGGT model:")
    cmd = [
        "python", "init_geo.py",
        "--source_path", source_path,
        "--model_path", f"{model_path}_vggt",
        "--model_type", "vggt",
        "--device", device
    ]
    # run_command(cmd)  # Uncomment to actually run
    
    print("\n3. Using LSM model:")
    cmd = [
        "python", "init_geo.py",
        "--source_path", source_path,
        "--model_path", f"{model_path}_lsm",
        "--model_type", "lsm",
        "--device", device
    ]
    # run_command(cmd)  # Uncomment to actually run
    
    print("\n4. Using SpatialTracker v2 model:")
    cmd = [
        "python", "init_geo.py",
        "--source_path", source_path,
        "--model_path", f"{model_path}_spatial_tracker_v2",
        "--model_type", "spatial_tracker_v2",
        "--device", device
    ]
    # run_command(cmd)  # Uncomment to actually run
    
    print("\n=== Available Model Types ===")
    print("- mast3r: MASt3R model (default)")
    print("- vggt: VGGT model")
    print("- lsm: LSM model")
    print("- spatial_tracker_v2: SpatialTracker v2 model")
    
    print("\n=== Default Checkpoint Paths ===")
    print("- mast3r: ./mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    print("- vggt: ./vggt/checkpoints/VGGT_model.pth")
    print("- lsm: ./lsm/checkpoints/LSM_model.pth")
    print("- spatial_tracker_v2: ./spatial_tracker_v2/checkpoints/SpatialTrackerV2_model.pth")
    
    print("\n=== Custom Checkpoint Path ===")
    print("You can also specify a custom checkpoint path:")
    print("python init_geo.py --source_path <path> --model_path <path> --model_type vggt --ckpt_path /path/to/custom/checkpoint.pth")
    
    print("\n=== Same functionality available in init_test_pose.py ===")
    print("The same model selection is also available in init_test_pose.py for test pose initialization.")

if __name__ == "__main__":
    main()
