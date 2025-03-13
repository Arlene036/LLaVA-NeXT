export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

export NUM_GPUS=1 # <<<<<<< 每个节点上的GPU数量
export NNODES=1 # <<<<<<< 分布式训练中节点的数量
export RANK=0 # <<<<<<< 当前节点在分布式训练中的排名
export ADDR=127.0.0.1 # <<<<<<< 分布式训练中master node的IP地址
export PORT=29500 # <<<<<<< 节点间通信的网络端口

LLM_VERSION="Qwen/Qwen2-7B-Instruct" 
# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# ############### Pretrain ################

# BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
# echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################

# Stage 2
PROMPT_VERSION="qwen_1_5"
RUN_NAME="llava-onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-ov_stage_am9" 
MODEL_CHECKPOINT="lmms-lab/llava-onevision-qwen2-7b-ov"
DATASET_PATH="/home/renyy/projects/VideoX_wyq/scanqa_map_data.yaml" # <<<<<<< dataset yaml
IMAGE_FOLDER="/mnt/gaia/datasets/project_data/ScanQA_map/data_processed/map_output/bev" # <<<<<<< image folder
VIDEO_FOLDER="/mnt/gaia/datasets/project_data/ScanNet/scans/"  # <<<<<<< video folder
MM_TUNABLE_PARTS="mm_vision_tower,mm_mlp_adapter,mm_language_model" # >>>>> TODO: to training
LR=1e-5 # >>>>> TODO: to tuning, 1e-5 for 7b
VIDEO_FPS=30 # <<<<<< video fps for ScanNet

echo "MODEL_CHECKPOINT: ${MODEL_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $MODEL_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $DATASET_PATH\
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --mm_tunable_parts=$MM_TUNABLE_PARTS \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir /mnt/bn/vl-research/checkpoints/onevision/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --video_fps $VIDEO_FPS
exit 0;

# You can delete the sdpa attn_implementation if you want to use flash attn
