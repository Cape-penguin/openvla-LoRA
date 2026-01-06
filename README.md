# openvla-LoRA
Fine-tuning OpenVLA using LoRA for parameter-efficient robotic policy learning.

## Installation

```
# Create and activate conda environment
conda create -n openvla python=3.10 -y
conda activate openvla

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

cd repos/openvla
pip install -e .
```