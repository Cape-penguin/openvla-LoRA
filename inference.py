import os
import pickle
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model

# 1. 모델 로드
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

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

df = pd.read_csv('./instruction.csv', )

with open(os.path.join(df["path"].to_list()[0], "policy_out.pkl"), "rb") as f:
    raw_data = pickle.load(f)

actions = [d['actions'] for d in raw_data]
action = actions[0]

im_pth = os.path.join(df["path"].to_list()[0], f"images0/im_{0}.jpg")
image = Image.open(im_pth).convert("RGB")

prompt = "In order to pick up the object, the robot should"

vla.eval()
device = "cuda:0"
print(f"device={device}")

processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

inputs = processor(prompt, image, return_tensors="pt", ).to(device, dtype=torch.bfloat16)
input_ids = inputs["input_ids"].squeeze(0)
labels = torch.full_like(input_ids, -100)

raw_action = np.array(action, dtype=np.float32)
bin_indices = np.clip((raw_action + 1.0) / 2.0 * 255, 0, 255).astype(np.int32)
action_token_ids = torch.tensor(bin_indices + 31000, dtype=torch.long)
labels[-7:] = action_token_ids

print(f"labels:{labels}")

device = torch.device("cuda")
vla.to(device)
vla.eval()

inputs = {k: v.to(device, dtype=torch.bfloat16 if v.dtype == torch.float32 else v.dtype) for k, v in inputs.items()}

with torch.no_grad():
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig")

print(action)