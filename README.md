# openvla-LoRA
Fine-tuning OpenVLA using LoRA for parameter-efficient robotic policy learning.

## Installation

```
docker run --gpus all -it --name openvla -w /workspace -v "C:\Users\idong\src":/workspace nvcr.io/nvidia/pytorch:23.10-py3

# Create and activate conda environment
conda create -n openvla python=3.10 -y
conda activate openvla

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

cd repos/openvla
pip install -e .

pip install "flash-attn==2.5.5" --no-build-isolation
```

## Download Dataset ([BridgeData V2](https://rail-berkeley.github.io/bridgedata/))

```
# 
wget -P ./data https://rail.eecs.berkeley.edu/datasets/bridge_release/data/scripted_6_18.zip

# Sample
wget -P ./data https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/bridge_dataset-train.tfrecord-00000-of-01024

# Download meta data
wget -r -np -nd -A "*.json" -P ./data https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/
```

## Try #1

nvcr.io/nvidia/pytorch:24.01-py3 컨테이너로 환경 생성

docker run --gpus all -it --name openvla -w /workspace -v ~/src/psu:/workspace nvcr.io/nvidia/dia/pytorch:24.01-py3

pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

pip install transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10

pip install jupyter ipykernel

Set Environment

export CUDA_HOME=/usr/local/cuda-12.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

pip install "flash-attn==2.5.5" --no-build-isolation

\# clone openvla repo
cd ./repos/openvla
pip install -e .

Finetune OpenVLA via LoRA
Download https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

지금 `scripted` 데이터를 TFRecord로 이미 만드셨잖아요? **우선은 그 TFRecord를 그대로 `finetune.py`에 넣어보세요.** * **만약 에러가 안 나고 학습이 시작된다면:** 아까 사용한 변환 스크립트가 이미 RLDS와 유사한 구조로 만들어준 것이니 그대로 진행하시면 됩니다.

convert script dataset to TFRecord
python bridge_data_v2/data_processing/bridgedata_raw_to_numpy.py --input_path ./data/scripted_raw/ --output_path ./data/scripted_numpy_output --depth 2

CUDA_VISIBLE_DEVICES="" python bridge_data_v2/data_processing/bridgedata_numpy_to_tfrecord.py --input_path ./data/scripted_numpy_output/ --output_path ./data/scripted_tf_output --depth 2

torchrun --standalone --nnodes 1 --nproc-per-node 1 openvla/vla-scripts/finetune.py  --vla_path "openvla/openvla-7b"  --data_root_dir "./data/scripted_tf_output" --dataset_name "bridge_scripted_tf" --run_root_dir "./experiments/bridge_finetune" --adapter_tmp_dir "./tmp/adapters" --lora_rank 32 --batch_size 2 --grad_accumulation_steps 1 --learning_rate 5e-4 --image_aug True --save_steps 100

실패하면 아래를 해야된다
convert script dataset to RLDS
git clone https://github.com/kvablack/dlimp.git

아니면 그냥 주어진 brige_org 데이터 셋을 사용해야할거같다.. > it works..
wget -r -nH --cut-dirs=4 --reject="index.html*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0

torchrun --standalone --nnodes 1 --nproc-per-node 1 openvla/vla-scripts/finetune.py  --vla_path "openvla/openvla-7b"  --data_root_dir "./data" --dataset_name bridge_orig --run_root_dir "./experiments/bridge_finetune" --adapter_tmp_dir "./tmp/adapters" --lora_rank 32 --batch_size 2 --grad_accumulation_steps 1 --learning_rate 5e-4 --image_aug True --save_steps 100



[Docker 생성]
docker run -it --rm --name sdi_gpu-0 -v /home/sdi/openvla/:/workspace --gpus "device=0" nvcr.io/nvidia/pytorch:23.10-py3