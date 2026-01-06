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

## Download Dataset ([BridgeData V2](https://rail-berkeley.github.io/bridgedata/))

```
wget -P ./data https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/bridge_dataset-train.tfrecord-00000-of-01024

# Download meta data
wget -r -np -nd -A "*.json" -P ./data https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/
```