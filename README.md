# openvla-LoRA
Fine-tuning OpenVLA using LoRA for parameter-efficient robotic policy learning. For computational efficieny and simplicity, the model is trained on 100 sample trajectories with 300 epochs, taking approximately over 48 hours.

## Installation

```
# Run docker container
docker run --gpus all -it --name openvla-LoRA -w /workspace -v ./:/workspace nvcr.io/nvidia/dia/pytorch:24.01-py3

# Install prerequisite
pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

pip install transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10

[optional] pip install jupyter ipykernel

cd ./repos/openvla
pip install -e .

# Set environment
export CUDA_HOME=/usr/local/cuda-12.3 
export PATH=$CUDA_HOME/bin:$PATH 
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install flash-attention
pip install "flash-attn==2.5.5" --no-build-isolation
```

## Download Dataset ([BridgeData V2](https://rail-berkeley.github.io/bridgedata/))

```
wget -P ./data https://rail.eecs.berkeley.edu/datasets/bridge_release/data/scripted_6_18.zip
```

## Fine-tuning OpenVLA with General Instruction Sets

```
python Pretrain-openvla-7b.py
```

## Inference

```
python Inference-sample-100.py --checkpoint ./checkpoints/openvla-lora-epoch-01-000000-sample-100-rows/ --output ./outputs/openvla-lora-epoch-01-000000-sample-100-rows
```

## Evaluate

```
python Evaluate.py --json ./outputs/inference_results-openvla-lora-epoch-0301-000000-sample-100-rows.json
```