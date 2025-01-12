# Please note!!!
# The code here is only an example. 
# Due to anonymous review requirements, we have anonymized 
# the distributed training platform of our organization. 
# If you need to run code, please refer to the 
# multi-GPU task submission method that suits you.

workdir=$(cd $(dirname $0); pwd)
echo "workdir: "${workdir}
echo ""


source path_to_your_dir/general/config.sh

# Distributed cluster queue name
QUEUE="name_of_your_queue"
WORLD_SIZE=8 # 8 GPUs
echo "         QUEUE: "${QUEUE}

deepspeed_config="path_to_your_dir/general/deepspeed.json"
cluster_config="path_to_your_dir/general/cluster.json"
echo "      WORLD_SIZE: "${WORLD_SIZE}
echo "deepspeed_config: "${deepspeed_config}
echo "  cluster_config: "${cluster_config}
echo ""


train_stage=1 # [1, 2]
safe_samples=3000
output_path="safe_llava_res/model_checkpoints/test_output"
model_path="llava-hf/llava-1.5-7b-hf"


data_path="./dataloader/example.jsonl"
output_dir="path_to_your_output_dir"
image_folder="path_to_your_dir_of_images"

args="--root_path ${output_dir} \
--data_path ${data_path} \
--model_path ${model_path} \
--image_folder ${image_folder} \
--seed 256 \
--deepspeed_config ${deepspeed_config} \
--train_stage ${train_stage} \
--model_version v1.5 \
--safe_samples ${safe_samples}"
echo "args: "${args}
echo ""

# use distributed training platform
command_to_call_Distributed_training_platform --queue=${QUEUE} \
                  --project=safe_vlm \
                  --entry=train.py \
                  --worker_count=${WORLD_SIZE}  \
                  --user_params="$args" \
                  --file.cluster_file=${cluster_config} \
                  --algo_name=pytorch220 \
                  --job_name=safe_vlm_test \