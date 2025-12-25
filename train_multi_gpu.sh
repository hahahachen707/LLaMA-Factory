set -x
export PYTHONUNBUFFERED=1
ulimit -s unlimited  # 防止栈溢出
export USE_FLASH_ATTN=1  # 启用 Flash Attention
export NCCL_SHM_DISABLE=1 # 禁用 NCCL 共享内存，防止容器内 /dev/shm 不足

# 显存优化配置：尝试减少碎片
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# CUDNN 优化配置：禁用 benchmark 防止资源耗尽
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_BENCHMARK=0
export TORCH_CUDNN_DETERMINISTIC=1
# 如果上面不行，可以尝试把下面这行注释去掉，强制禁用 CUDNN（极慢但稳）
# export TORCH_BACKEND_CUDNN_ENABLED=0

# 修改NCCL环境变量
export NCCL_DEBUG=WARN  # 只显示警告和错误
export NCCL_IB_DISABLE=1  # 禁用IB
export NCCL_SOCKET_IFNAME=lo  # 使用本地回环
export OMP_NUM_THREADS=16
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=OFF  # 关闭分布式调试
export NCCL_P2P_DISABLE=1  # 禁用P2P
export NCCL_BLOCKING_WAIT=1  # 使用阻塞等待
export TORCH_DISTRIBUTED_ELASTIC_RENDEZVOUS_BACKEND=c10d  # 使用c10d作为rendezvous后端


# llamafactory-cli train examples/train_lora/qwen3vl_lora_sft.yaml
torchrun --nnodes 1 \
--node_rank 0 --nproc_per_node 4 \
--master_addr 127.0.0.1 --master_port 6047 \
--rdzv_backend=c10d \
--rdzv_endpoint 127.0.0.1:6047 \
--rdzv_id=job \
--rdzv_conf="timeout=300" \
--max_restarts=0 \
/apdcephfs_cq12/share_1150325/hahahachen/work/LLaMA-Factory/src/llamafactory/launcher.py \
examples/train_lora/qwen3vl_lora_sft.yaml
