#!/bin/bash
#SBATCH --job-name=specr_aime_32b_greedy9               # Job name
#SBATCH --output="/home/rp2773/slurm_logs/%A.out"       # Standard output log
#SBATCH --error="/home/rp2773/slurm_logs/%A.err"         # Standard error log
#SBATCH --ntasks=1                            # Number of tasks (1 process)
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task
#SBATCH --gres=gpu:4                        # Number of GPUs to allocate
##SBATCH --constraint="gpu80"
#SBATCH --time=7:00:00                        # Time limit (24 hours max)
#SBATCH --mem=100G                            # Memory allocation (adjust as needed)
#SBATCH --mail-user=ruipan@princeton.edu  # Your email
#SBATCH --mail-type=ALL  # Options: BEGIN, END, FAIL, REQUEUE, TIME_LIMIT, etc.
#SBATCH --partition=pli
#SBATCH --account=specreason

# Starts a vLLM server and captures logs; fill placeholders before running.
LOG_DIR="vllm_logs"  # or any directory you prefer
export HF_HOME=/raid0-data/jiayi_tian
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi

python - <<'PY'
import vllm
print(vllm.__file__)
PY

# Function to check if a server is ready
wait_for_server() {
    local port=$1
    while true; do
        # Try to connect to the server
        curl -s http://localhost:$port/v1/models > /dev/null
        if [ $? -eq 0 ]; then
            echo "Server on port $port is ready!"
            break
        else
            echo "Waiting for server on port $port to start..."
            sleep 10  # Wait 10 seconds before retrying
        fi
    done
}

#################################################
export CUDA_VISIBLE_DEVICES=4,5,6,7

LOG_FILE="${LOG_DIR}/CUDA${CUDA_DEVICE}_server_$(date +%Y%m%d_%H%M%S).out"
TEMPLATE_FILE=template/deepseekr1.jinja  # NOTE: change to the template file of your model, e.g. "./template/your_model.jinja"

TP_SIZE=4  # applies for both models

# launch 32b model and 1.5b model one by one
export BASE_MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"  # "Qwen/QwQ-32B" or "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" or "NovaSky-AI/Sky-T1-32B-Preview"
export SMALL_MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" or "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
export BASE_MODEL_PORT=40000
export SMALL_MODEL_PORT=40001
vllm serve "$BASE_MODEL_NAME" --dtype auto -tp "$TP_SIZE" --max_model_len 12000 --gpu-memory-utilization 0.6 --enable-prefix-caching --host 0.0.0.0 --port $BASE_MODEL_PORT --max-num-seqs 64 --chat-template $TEMPLATE_FILE &
VLLM_BASE_PID=$!
wait_for_server $BASE_MODEL_PORT
nvidia-smi
STATIC_STEER_ENABLE=0 \
vllm serve "$SMALL_MODEL_NAME" --dtype auto -tp "$TP_SIZE" --max_model_len 12000 --gpu-memory-utilization 0.2 --enable-prefix-caching --host 0.0.0.0 --port $SMALL_MODEL_PORT --max-num-seqs 64 --chat-template $TEMPLATE_FILE &
VLLM_SMALL_PID=$!
wait_for_server $SMALL_MODEL_PORT


# PORT=40000  # NOTE: change to the port of your model, e.g. 11125
# pid=$(lsof -ti tcp:$PORT)

# if [ -n "$pid" ]; then
#   # Free the port if another process is already listening.
#   echo "Port $PORT is occupied, process PID: $pid"
#   echo "Kill the process..."
#   kill -9 $pid
# fi

# MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"  # NOTE: change to your model path/name, e.g. org/model-name
# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE vllm serve $MODEL \
#   --dtype auto \
#   -tp 4 \
#   --max-model-len 8192 \
#   --chat-template $TEMPLATE_FILE \
#   --gpu_memory_utilization 0.6 \
#   --port $PORT \
#   --host 0.0.0.0 \
#   --enable-prefix-caching &


# PORT=40001  # NOTE: change to the port of your model, e.g. 11125
# pid=$(lsof -ti tcp:$PORT)

# if [ -n "$pid" ]; then
#   # Free the port if another process is already listening.
#   echo "Port $PORT is occupied, process PID: $pid"
#   echo "Kill the process..."
#   kill -9 $pid
# fi
# MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # NOTE: change to your model path/name, e.g. org/model-name
# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE vllm serve $MODEL \
#   --dtype auto \
#   -tp 4 \
#   --max-model-len 8192 \
#   --chat-template $TEMPLATE_FILE \
#   --gpu_memory_utilization 0.1 \
#   --port $PORT \
#   --host 0.0.0.0 \
#   --enable-prefix-caching &
