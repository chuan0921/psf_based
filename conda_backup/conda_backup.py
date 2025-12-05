import subprocess
import os
import datetime

# Get current date and time, format as string for folder naming
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# BACKUP_DIR will be a new folder named with the current date and time for each backup
BACKUP_DIR = f'/home/r13k47025/hcy_research/psf_based/conda_backup/{now}'  # Automatically name folder with current date and time

# Ensure the backup directory exists; create if it does not exist
os.makedirs(BACKUP_DIR, exist_ok=True)

def get_conda_envs():
    """
    Retrieve the names of all conda environments, excluding the 'base' environment.

    Returns:
        list: A list of conda environment names (str).
    """
    # Use the conda command to list all environments
    result = subprocess.run(['conda', 'env', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        # Print error message if the command fails
        print(f"Error listing conda environments: {result.stderr}")
        return []

    # Extract environment names, skipping any lines containing only 'base' environment
    envs = [line.split()[0] for line in result.stdout.splitlines() if line and 'base' not in line]
    return envs

def backup_env(env_name):
    """
    Backup the configuration of a specific conda environment to a .yml file.

    Args:
        env_name (str): The name of the conda environment to back up.
    """
    print(f"Backing up environment: {env_name}")
    backup_file = os.path.join(BACKUP_DIR, f"{env_name}_environment.yml")
    # Export the environment configuration using conda
    result = subprocess.run(['conda', 'env', 'export', '--name', env_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        # Save the environment configuration to a file
        with open(backup_file, 'w') as f:
            f.write(result.stdout)
        print(f"Backup of environment '{env_name}' saved to {backup_file}")
    else:
        # Print error message if the backup fails
        print(f"Failed to backup environment '{env_name}': {result.stderr}")

def backup_all_envs():
    """
    List all conda environments and back up each environment's configuration.
    """
    envs = get_conda_envs()
    if not envs:
        print("No conda environments found.")
        return

    for env in envs:
        backup_env(env)

if __name__ == "__main__":
    # Entry point: backup all conda environments when the script is run directly
    backup_all_envs()

# Example for restoring an environment:
# To restore an environment from a .yml file, use the following command:
# conda env create -f /path/to/your_backup_file.yml

