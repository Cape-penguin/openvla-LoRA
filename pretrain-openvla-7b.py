import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model

processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

# 1. 모델 로드
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# 2. LoRA 설정
# OpenVLA의 언어 모델 파트(Llama)의 특정 레이어를 타겟팅합니다.
config = LoraConfig(
    r=32,                         # Rank
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Attention 레이어 타겟
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 3. LoRA 모델로 변환
vla = get_peft_model(vla, config)
print(f"trainable parameters: {vla.print_trainable_parameters()}")

class BridgeV2Dataset(Dataset):
    def __init__(self, traj_dir, instruction, processor, vla_config):
        self.processor = processor
        self.instruction = instruction
        self.vla_config = vla_config # vla.config 전달
        
        # 데이터 로드
        with open(os.path.join(traj_dir, "policy_out.pkl"), "rb") as f:
            raw_data = pickle.load(f)
        self.actions = [d['actions'] for d in raw_data] if isinstance(raw_data[0], dict) else raw_data
        self.img_dir = os.path.join(traj_dir, "images0")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        # 1. 이미지 로드 및 전처리
        image = Image.open(os.path.join(self.img_dir, f"im_{idx}.jpg")).convert("RGB")
        inputs = self.processor(self.instruction, image, return_tensors="pt")
        
        # 2. 액션 양자화 (Action -> Token IDs)
        # OpenVLA 7B는 보통 31000번 이후의 256개 토큰을 사용합니다.
        raw_action = np.array(self.actions[idx], dtype=np.float32)
        
        # 공식 가이드에 따른 정규화 및 토큰화 로직
        # OpenVLA는 [-1, 1] 범위를 256개 bin으로 나눕니다.
        bin_indices = np.clip((raw_action + 1.0) / 2.0 * 255, 0, 255).astype(np.int32)
        
        # OpenVLA 7B의 액션 토큰 시작 인덱스는 보통 31000입니다.
        # 정확한 값은 vla.config.vocab_size - 256 근처입니다.
        action_token_ids = torch.tensor(bin_indices + 31000, dtype=torch.long)

        # 3. 레이블 구성 (중요!)
        # 전체 길이는 input_ids와 동일하게 만듭니다.
        input_ids = inputs["input_ids"].squeeze(0)
        labels = torch.full_like(input_ids, -100) # 모든 구간을 무시(-100)로 초기화
        # OpenVLA는 보통 시퀀스의 맨 마지막 7개 토큰이 액션입니다.
        # 마지막 7자리에 실제 액션 토큰 정답을 넣습니다.
        labels[-7:] = action_token_ids

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": labels
        }
# 1. 초기 설정
TRAJ_PATH = "./data/scripted_raw/sweep_12-03/2022-12-04_14-56-20/raw/traj_group0/traj0"
INST = "In order to pick up the can, the robot should"

# 2. 데이터셋 및 데이터로더 생성
train_dataset = BridgeV2Dataset(
    traj_dir=TRAJ_PATH,
    instruction=INST,
    processor=processor,
    vla_config=vla.config
)

def collate_fn(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch])
    }

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=2, 
    shuffle=True, 
    collate_fn=collate_fn
)

# 3. 동작 확인
batch = next(iter(train_dataloader))
print(f"Input IDs shape: {batch['input_ids'].shape}")     # [BS, Seq_Len]
print(f"Pixel Values shape: {batch['pixel_values'].shape}") # [BS, 3, 224, 224]
print(f"Labels shape: {batch['labels'].shape}")           # [BS, 7] (7 action tokens)

from torch.optim import AdamW

optimizer = AdamW(vla.parameters(), lr=2e-5)
vla.to('cuda')
vla.train()
for batch in train_dataloader:
    # 데이터를 GPU로 이동
    input_ids = batch["input_ids"].to("cuda")
    pixel_values = batch["pixel_values"].to("cuda", dtype=torch.bfloat16)
    labels = batch["labels"].to("cuda")

    # Forward Pass
    # OpenVLA는 input_ids와 pixel_values를 받아 마지막에 액션 토큰을 예측하도록 설계됨
    outputs = vla(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
    
    loss = outputs.loss
    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"Loss: {loss.item()}")

    break

print("test done")