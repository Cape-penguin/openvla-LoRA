# openvla-LoRA
Fine-tuning OpenVLA using LoRA for parameter-efficient robotic policy learning. For computational efficieny and simplicity, the model is trained on 100 sample trajectories with 300 epochs, taking approximately over 48 hours.

## Installation
To ensure environment consistency, the following Docker setup is recommended.
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
In this experiment, BridgeData V2 is used for training. You can fetch the specific scripted trajectory subset using the following:
```
wget -P ./data https://rail.eecs.berkeley.edu/datasets/bridge_release/data/scripted_6_18.zip
```

## Fine-tuning OpenVLA with General Instruction Sets
Fine-tune the 7B model using synthetic instruction augmentation and general instruction sets.
```
python Pretrain-openvla-7b.py
```

## Inference & Evaluation
Generate quantitative results through the inference script followed by the evaluation protocol.
```
# Generate model predictions
python Inference-sample-100.py --checkpoint ./checkpoints/openvla-lora-epoch-01-000000-sample-100-rows/ --output ./outputs/openvla-lora-epoch-01-000000-sample-100-rows

# Derive performance metrics
python Evaluate.py --json ./outputs/inference_results-openvla-lora-epoch-0301-000000-sample-100-rows.json
```

## Reference
For full implementation details and results, please refer to [paper](https://arxiv.org/abs/2603.16044)

```
@misc{shin2026enhancinglinguisticgeneralizationvla,
      title={Enhancing Linguistic Generalization of VLA: Fine-Tuning OpenVLA via Synthetic Instruction Augmentation}, 
      author={Dongik Shin},
      year={2026},
      eprint={2603.16044},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2603.16044}, 
}
```