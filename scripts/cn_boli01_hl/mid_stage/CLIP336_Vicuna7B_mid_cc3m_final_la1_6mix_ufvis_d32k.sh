#!/bin/bash
NAS_REGION="vl-research-cn-boli01-hl"
# set up wandb
export WANDB_API_KEY=a651c244635bc6f913ab654af3f0eebaecdc9381
export WANDB_ENTITY=llava-vl
export WANDB_PROJECT=llava-next
export WANDB_MODE=online
export PYTHONWARNINGS="ignore"

export ACCELERATE_DEBUG_MODE="1"
export HF_HOME=/mnt/bn/${NAS_REGION}/workspace/.cache/huggingface
export HF_TOKEN="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"
export HF_HUB_ENABLE_HF_TRANSFER="1"

############### Prepare Envs #################
cd /mnt/bn/${NAS_REGION}/workspace/boli01/projects/LLaVA_Next
python3 -m pip install --upgrade pip;

python3 -m pip install -e ".[train]"

python3 -m pip install ninja
python3 -m pip install flash-attn --no-build-isolation

alias python=python3
############### Show Envs ####################

nvidia-smi
# 取 worker0 第一个 port
ports=($(echo $METIS_WORKER_0_PORT | tr ',' ' '))
port=${ports[0]}
port_in_cmd="$(echo "${METIS_WORKER_0_PORT:-2222}" | awk -F',' '{print $1}')"
random_port_in_cmd="$(shuf -i 26000-27000 -n 1)"

echo "total workers: ${ARNOLD_WORKER_NUM}"
echo "cur worker id: ${ARNOLD_ID}"
echo "gpus per worker: ${ARNOLD_WORKER_GPU}"
echo "master ip: ${METIS_WORKER_0_HOST}"
echo "master port: ${port}"
echo "master port in cmd: ${port_in_cmd}"

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
# export NCCL_IB_HCA=${ARNOLD_RDMA_DEVICE}
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN

PORT=26000
GPUS="0,1,2,3,4,5,6,7"

wandb login a651c244635bc6f913ab654af3f0eebaecdc9381
wandb online

################ Arnold Jobs ################

LLM_VERSION="lmsys/vicuna-7b-v1.5"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
PROMPT_VERSION=plain
PRETRAIN_DATA_VERSION="blip558k"

############### Pretrain ################

BASE_RUN_NAME="ds_llava-vicuna-7b-v1-5-clip_large_336px-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

# Stage 1.5
MID_STAGE_DATA="cc3m"
PROMPT_VERSION="vicuna_v1"
MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mid_${MID_STAGE_DATA}_${PROMPT_VERSION}_ufvis_d4k"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"
torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${random_port_in_cmd}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $LLM_VERSION \
    --version $PROMPT_VERSION \
    --data_path /mnt/bn/vl-research-cn-boli01-hl/data/llava_instruct/cc3m_llava34b_recap_2421640.json \
    --image_folder /mnt/bn/vl-research-cn-boli01-hl/data/llava_data/cc3m/images \
    --pretrain_mm_mlp_adapter="/mnt/bn/${NAS_REGION}/checkpoints/projectors/${BASE_RUN_NAME}/mm_projector.bin" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir /mnt/bn/${NAS_REGION}/checkpoints/$MID_RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 20000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb

# Stage 2
PROMPT_VERSION="vicuna_v1"
FINAL_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mid_${MID_STAGE_DATA}_${PROMPT_VERSION}_final_la1_6mix_ufvis_d4k"
echo "FINAL_RUN_NAME: ${FINAL_RUN_NAME}"
torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${random_port_in_cmd}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path /mnt/bn/${NAS_REGION}/checkpoints/${MID_RUN_NAME} \
    --version $PROMPT_VERSION \
    --data_path /mnt/bn/${NAS_REGION}/data/llava_instruct/llava_158k_detailv3_reinstall_gpt4v24k_wild15k_mixdocvqa_dca45k_synden40k_cococaps20k_sg40kt2k_ori_rep_ai2d_filtered_794409.json \
    --image_folder /mnt/bn/${NAS_REGION}/data/llava_data \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower="${VISION_MODEL_VERSION}" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $FINAL_RUN_NAME \
    --output_dir /mnt/bn/${NAS_REGION}/checkpoints/$FINAL_RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 20000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb

############### Upload ################
function azcopy_upload() {
    # Assuming the first argument is SRC and the second is TGT
    local SRC="$1"
    local TGT="$2"
    local SAS_TOKEN="?sv=2023-01-03&st=2023-12-23T13%3A48%3A31Z&se=2024-06-30T13%3A48%3A00Z&sr=c&sp=racwdxltf&sig=K77ocq6Ram1uYMenQJZJl%2BBayH%2Bg4e10Raci6wzQY3M%3D"
    # Executing the azcopy command with the provided SRC and TGT
    /mnt/bn/${NAS_REGION}/software/azcopy copy "$SRC" "https://chunyldev.blob.core.windows.net/output/$TGT$SAS_TOKEN" --recursive --overwrite=ifSourceNewer
}

azcopy_upload "/mnt/bn/vl-research/workspace/boli01/projects/LLaVA_Next/project_checkpoints/${FINAL_RUN_NAME}" "projects/llava_data/checkpoints/"

################ Evaluation ################
# sh /mnt/bn/${NAS_REGION}/workspace/boli01/projects/lmms-eval/scripts/configure_envs.sh
# cd /mnt/bn/${NAS_REGION}/workspace/boli01/projects/lmms-eval
# which python3

# accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
#     --model llava \
#     --model_args pretrained="/mnt/bn/${NAS_REGION}/workspace/boli01/projects/LLaVA_Next/project_checkpoints/${FINAL_RUN_NAME},conv_template=vicuna_v1" \
#     --tasks ai2d,chartqa,docvqa_val,mathvista_testmini,mme,mmmu_val,textcaps_val,scienceqa_img \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix ${FINAL_RUN_NAME} \
#     --output_path ./logs/ \
#     --wandb_args 'project=lmms-eval,job_type=eval,entity=llava-vl';