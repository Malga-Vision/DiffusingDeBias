import argparse
from DiffusionFreeGuidance.TrainCondition import train, eval

# Default model configuration
DEFAULT_MODEL_CONFIG = {
    "state": "train",  # or eval
    "iterations": 100000,
    "batch_size": 32,
    "T": 1000,
    "channel": 128,
    "channel_mult": [1, 2, 2, 2],
    "num_res_blocks": 2,
    "dropout": 0.15,
    "lr": 1e-4,
    "multiplier": 2.5,
    "beta_1": 1e-4,
    "beta_T": 0.028,
    "img_size": 64,
    "grad_clip": 1.0,
    "device": "cuda:0",
    "w": 1,
    "save_dir": "./ModelCheckpoints",
    "load_weights": None,
    "sampled_dir": "./SyntheticImages/",
    "images_to_sample": 1000,
    "nrow": 10,
    "freq_save": 10000,
    "dataset": "waterbirds",
    "data_dir": "./data",
}

def parse_args():
    """Parse command-line arguments to override default configurations."""
    parser = argparse.ArgumentParser(description="Model Configuration")
    
    # Add arguments corresponding to the model config keys
    parser.add_argument("--state", type=str, choices=["train", "eval"], help="Mode: train or eval")
    parser.add_argument("--iterations", type=int, help="Number of training iterations")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--T", type=int, help="Number of time steps")
    parser.add_argument("--channel", type=int, help="Base number of channels")
    parser.add_argument("--dropout", type=float, help="Dropout rate")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--img_size", type=int, help="Image size")
    parser.add_argument("--device", type=str, help="Device to use, e.g., 'cuda:0' or 'cpu'")
    parser.add_argument("--w", type=int, help="Classifier-free guided diffusion strength")
    parser.add_argument("--save_dir", type=str, help="Directory to save model checkpoints")
    parser.add_argument("--freq_save", type=int, help="Iteration frequency to save model checkpoints")
    parser.add_argument("--load_weights", type=str, help="Path to model checkpoint to be used for evaluation")
    parser.add_argument("--sampled_dir", type=str, help="Directory to save sampled images")
    parser.add_argument("--images_to_sample", type=int, help="Number of images to sample")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--data_dir", type=str, help="Directory containing dataset")

    return parser.parse_args()

def update_config(default_config, args):
    """Update the default configuration with command-line arguments."""
    config = default_config.copy()
    for key, value in vars(args).items():
        if value is not None:  # Only override if an argument is provided
            config[key] = value
    return config


def main():
    # Parse arguments
    args = parse_args()
    
    # Update configuration
    modelConfig = update_config(DEFAULT_MODEL_CONFIG, args)
    print(f"Experiment Configuration: {modelConfig}")

    # Run train or eval based on the state
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)

if __name__ == '__main__':
    main()


# python runDDB.py

# python runDDB.py --state train --iterations 100000 --batch_size 32 --dataset waterbirds --img_size 64 --device cuda:0
# python runDDB.py --state eval --load_weights ./ModelCheckpoints/waterbirds/ckpt_100000_iterations.pt --batch_size 100 --dataset waterbirds --img_size 64 --device cuda:0