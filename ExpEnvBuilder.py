import argparse
import json
import os
import sys
from pathlib import Path
import shutil


def parse_config(path: str) -> tuple[dict, dict, dict, Path, Path, Path, Path]:
    config = None
    resources = None
    training_config = None
    
    with open(path, "r") as config_file:
        config = json.load(config_file)
        resources = config["resources"]
        training_config = config["training_config"]
    
    experiment_dir = Path(config["model_name"])
    checkpoints_dir_path = experiment_dir / "checkpoints"
    logs_dir_path = experiment_dir / "logs"
    config_path = experiment_dir / "config.json"
    
    return config, resources, training_config, experiment_dir, checkpoints_dir_path, logs_dir_path, config_path


def write_slurm_script(path: str) -> None:
    config, resources, training_config, experiment_dir, checkpoints_dir_path, logs_dir_path, config_path = parse_config(path)
    
    script_template = \
    f"""#!/bin/bash
#SBATCH --account={os.getlogin()} # Account
#SBATCH --job-name={config["model_name"]} # Job name
#SBATCH --partition=only-one-gpu # Partition name (e.g.only-one-gpu,ulow)

# RESOURCES
#SBATCH --ntasks={resources["ntasks"]} # How many tasks
#SBATCH --cpus-per-task={resources["cpus_per_task"]} # How many CPU cores per task
#SBATCH --mem={resources["mem"]} # Job memory request
#SBATCH --gres={resources["gres"]} # How many GPUs (0..1)
#SBATCH --time={resources["time"]} # Time limit hrs:min:sec

# OUTPUT FILES
#SBATCH --output={logs_dir_path.absolute()}/out_%x_%j.log # Standard output and error log, with job name and id

### Definitions
export SHRDIR="/scratch_share/islab/Dubini"

### Software dependencies
module purge
module load amd/gcc-8.5.0/miniforge3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis_venv

### Header
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "Log directory: {logs_dir_path.absolute()}"
echo "Working directory: {config['project_dir']}"
echo "Checkpoints directory: {checkpoints_dir_path.absolute()}"
echo "========================================"
echo "Description:"
echo "{config['description']}"
echo "========================================"
echo "Job parameters:"
echo "Model type: {config['model_type']}"
echo "Batch size: {training_config['batch_size']}"
echo "Number of epochs: {training_config['num_epochs']}"
echo "Learning rate: {training_config['learning_rate']}"
echo "Learning rate drop factor: {training_config['lr_drop_factor']}"
echo "Learning rate drop patience: {training_config['lr_drop_patience']}"
echo "========================================"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "Python: $(python --version 2>&1)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA build: $(python -c 'import torch; print(torch.version.cuda)')"
echo "========================================"

### Executable script
cd "{config['project_dir']}" || exit 1

python ./main.py \\
    --model_type {config['model_type']} \\
    --checkpoints_dir '{checkpoints_dir_path.absolute()}' \\
    --dataset_dir "{config['dataset_dir']}" \\
    --split_dir "{config['split_dir']}" \\
    --model_name "{config['model_name']}"  \\
    --batch_size {training_config['batch_size']}  \\
    --num_epochs {training_config['num_epochs']} \\
    --learning_rate {training_config['learning_rate']} \\
    --lr_drop_factor {training_config['lr_drop_factor']} \\
    --lr_drop_patience {training_config['lr_drop_patience']} \\
    --num_workers "{resources['cpus_per_task']}"

### Footer
echo "Finished: $(date)"
"""

    with open(experiment_dir / "script.slurm", "w") as slurm_file:
        slurm_file.write(script_template)


def build_experiment_env(path: str) -> None:
    _, _, _,experiment_dir, checkpoints_dir_path, logs_dir_path, config_path = parse_config(path)
    
    if experiment_dir.exists():
        raise FileExistsError(f"Experiment '{experiment_dir.name}' already exists.")
    
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir_path.mkdir(parents=True, exist_ok=True)
    logs_dir_path.mkdir(parents=True, exist_ok=True)
    
    write_slurm_script(path)
    
    shutil.copy(path, config_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SLURM script from JSON configuration.")
    parser.add_argument("config_path", type=str, help="Path to the JSON configuration file.")
    parser.add_argument("--update_script", action="store_true", help="Update the SLURM script if it already exists.")
    args = parser.parse_args()
    
    if not os.path.isfile(args.config_path):
        print(f"Error: Configuration file '{args.config_path}' does not exist.")
        sys.exit(1)
     
    if args.update_script:
        write_slurm_script(args.config_path)
    else:
        build_experiment_env(args.config_path)
