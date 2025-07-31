import argparse
import datetime
import os
import subprocess

COMMAND_FILE = "/mnt/weka/home/kyle.chickering/code/nanoGPT/paramaterized_train.py"

parser = argparse.ArgumentParser()

parser.add_argument('--config_generator_file', type=str, required=True)
parser.add_argument('--max_concurrent', type=int, default=30)

args = parser.parse_args()

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_logging_dir = "/mnt/weka/home/kyle.chickering/code/nanoGPT/mup_paper_experiments/slurm_logs"
logging_dir = f"{base_logging_dir}/{now}"
orchastrator_dir = f"{logging_dir}/orchastrator"

os.makedirs(base_logging_dir, exist_ok=True)
os.makedirs(logging_dir, exist_ok=True)
os.makedirs(orchastrator_dir, exist_ok=True)

# run the config to get configs as a string
def run_config_generator(config_generator_file):
    """Run the config generator script and return its output."""
    try:
        result = subprocess.run(
            ['python', config_generator_file],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running config generator: {e}")
        return None

configs = run_config_generator(args.config_generator_file)
if configs is None:
    print("Failed to generate configurations. Exiting.")
    exit(1)

num_experiments = len(configs.split('\n'))

sbatch_headers = f"""#!/bin/bash

#SBATCH --array=0-{num_experiments-1}%{min(args.max_concurrent, num_experiments)}
#SBATCH --job-name=kyle_orchestrator
#SBATCH --time=50:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output={orchastrator_dir}/%A_%a.out
#SBATCH --error={orchastrator_dir}/%A_%a.err
#SBATCH --mem=8G
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH --distribution=block:block

"""

parsed_config_str = ""
for config in configs.split('\n'):
    parsed_config_str += f"    '{config.strip()}'\n"

if parsed_config_str.endswith('\n'):
    parsed_config_str = parsed_config_str[:-1]

config_str = f"""CONFIGS=(
{parsed_config_str}
)

"""

parsing_str = f"""CONFIG_JSON=\"${{CONFIGS[$SLURM_ARRAY_TASK_ID]}}\"
# Parse the individual configs
ARGS=\"\"
while IFS='=' read -r key value; 
do
    key=$(echo \"$key\" | xargs)
    value=$(echo \"$value\" | xargs)

    if [ -z \"$key\" ]; 
    then
        continue
    fi
    
    key=\"${{key%\\"}}\"
    key=\"${{key#\\"}}\"

    if [ "$value" == "true" ]; 
    then
        ARGS+=" --$key"
    elif [ "$value" == "false" ]; 
    then
        true # no-op
    elif [ "$value" == "null" ]; 
    then
        true # no-op
    else
        ESCAPED_VALUE=$(printf %q "$value")
        ARGS+=" --$key ${{ESCAPED_VALUE}}"
    fi
done < <(echo "$CONFIG_JSON" | jq -r 'to_entries[] | .key + "=" + (.value | @json)')

ARGS+=" --sbatch_logging_dir {logging_dir}"

"""

command_str = """echo $ARGS\n"""
command_str += f"""python {COMMAND_FILE} $ARGS"""

shell_script = sbatch_headers + config_str + parsing_str + command_str

# print(sbatch_headers + config_str + parsing_str + command_str)

# print(shell_script)

try:
    process = subprocess.run(
        ['sbatch'], input=shell_script, text=True, capture_output=True, check=True
    )
    print(f"Job submitted successfully: {process.stdout.strip()}", flush=True)
except subprocess.CalledProcessError as e:
    print(f"Error submitting job: {e.stderr.strip()}", flush=True)
    exit(1)