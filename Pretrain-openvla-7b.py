import os
import pickle
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model

device = "cuda:0"
print(f"device={device}")

# 1. Load model
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map=device
)

config = LoraConfig(
    r=32,                                                    # Rank
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Attention Layer Target
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 3. Transform to LoRA
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
vla.eval()

inputs = {k: v.to(device, dtype=torch.bfloat16 if v.dtype == torch.float32 else v.dtype) for k, v in inputs.items()}

# Design for 1 sample (not batch level)
with torch.no_grad():
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig")

print(action)


# Load Dataset
class Traj:
    def __init__(self, traj_dir, instruction):
        self.path = traj_dir
        self.instruction = instruction
        self.img_dir = os.path.join(traj_dir, "images0")
        self.img_len = len(os.listdir(os.path.join(traj_dir, "images0")))
        with open(os.path.join(traj_dir, "policy_out.pkl"), "rb") as f:
            raw_data = pickle.load(f)
        self.actions = [d['actions'] for d in raw_data]

        assert len(self.actions) == (self.img_len - 1)
    
    def __len__(self):
        return len(self.actions)
    
    def getitem(self, idx):
        return os.path.join(self.img_dir, f"im_{idx}.jpg"), self.instruction
    
    def getitems(self):
        ims = []
        inst = []
        for idx in range(self.img_len - 1):
            i, s = self.getitem(idx)
            ims.append(i)
            inst.append(s)
        return ims, self.actions, inst

class BridgeDatasetV2(Dataset):
    def __init__(self, traj_dirs, instructions, processor, ):
        self.processor = processor
        self.instructions = instructions
        # self.vla_config = vla_config
        self.traj_dirs = traj_dirs
        self.trajs = self.load_trajs()

        self.ims = []
        self.actions = []
        self.INST = []

        for traj in self.trajs:
            I, A, inst = traj.getitems()
            self.ims.extend(I)
            self.actions.extend(A)
            self.INST.extend(inst)
        print(f"initialize BridgeDatasetV2, number of trajectories: {len(self.trajs)}, total sample size: {len(self.actions)}.")

    def load_trajs(self, ):
        trajs = []
        for traj_dir, inst in zip(self.traj_dirs, self.instructions):
            obj = Traj(traj_dir, inst)
            trajs.append(obj)
        return trajs

    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        image = Image.open(self.ims[idx]).convert("RGB")
        inputs = self.processor(self.INST[idx], image, return_tensors="pt")

        raw_action = np.array(self.actions[idx], dtype=np.float32)
        bin_indices = np.clip((raw_action + 1.0) / 2.0 * 255, 0, 255).astype(np.int32)
        action_token_ids = torch.tensor(bin_indices + 31000, dtype=torch.long)
        
        prompt_ids = inputs["input_ids"].squeeze(0)
        input_ids = torch.cat([prompt_ids, action_token_ids], dim=0)
        
        labels = torch.full_like(input_ids, -100)
        
        labels[-7:] = action_token_ids

        return {
            "input_ids": input_ids,
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": labels
        }
    
def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    
    input_ids = [item["input_ids"] for item in batch]
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    attention_mask = padded_input_ids.ne(processor.tokenizer.pad_token_id).long()

    return {
        "input_ids": padded_input_ids,
        "pixel_values": pixel_values,
        "labels": padded_labels,
        "attention_mask": attention_mask
    }

df = pd.read_csv('./instruction.csv', )[:100]
print(f"Dataframe shape: {df.shape}")

train_dataset = BridgeDatasetV2(
    traj_dirs=df['path'].to_list(),
    instructions=df['instruction'].to_list(),
    processor=processor,
)

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=2, 
    shuffle=True, 
    collate_fn=collate_fn
)


print(f"DataLoader Length: {len(train_dataset)}")
batch = next(iter(train_dataloader))
print(f"Input IDs shape: {batch['input_ids'].shape}")       # [BS, Seq_Len]
print(f"Pixel Values shape: {batch['pixel_values'].shape}") # [BS, 3, 224, 224]
print(f"Labels shape: {batch['labels'].shape}")             # [BS, 7] (7 action tokens)

print(f'infer batch level')

batch = {k: v.to(device, dtype=torch.bfloat16 if v.dtype == torch.float32 else v.dtype) for k, v in batch.items()}

input_ids = batch['input_ids']
pixel_values = batch['pixel_values']
labels = batch['labels']

outputs = vla(
    input_ids=batch["input_ids"],
    pixel_values=batch["pixel_values"].to(torch.bfloat16),
    labels=batch["labels"]
)

print('outputs:', outputs.logits.shape)

loss = outputs.loss
loss.backward()
print(f"Current Loss: {loss.item()}")

predicted_ids = torch.argmax(outputs.logits, dim=-1)
print("Predicted Action Tokens:", predicted_ids[0, -7:])
print("Actual Target Tokens:   ", batch["labels"][0, -7:])


from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm

num_epochs = 1000
# batch_size = 16  # Adjust batch size, batch 2 is fine wiht 40G VRAM
gradient_accumulation_steps = 4  # 16 * 4 = 64 batch size effect
learning_rate = 5e-5  # Recommend LoRA lr

optimizer = AdamW(vla.parameters(), lr=learning_rate)

num_training_steps = num_epochs * len(train_dataloader) // gradient_accumulation_steps
lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=int(0.03 * num_training_steps),
    num_training_steps=num_training_steps,
)

print("Starting Training...")
vla.train()

for epoch in range(num_epochs):
    epoch_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
    
    

    for step, batch in enumerate(progress_bar):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        outputs = vla(
            input_ids=batch["input_ids"],
            pixel_values=batch["pixel_values"].to(torch.bfloat16),
            labels=batch["labels"],
            attention_mask=batch.get("attention_mask")
        )
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        # Gradient Accumulation
        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
        epoch_loss += loss.item() * gradient_accumulation_steps
        progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})

        if epoch % 20 == 0 and step == 0:
            print(f"[{epoch}/{num_epochs}] save checkpoints")
            vla.save_pretrained(f"./checkpoints/openvla-lora-epoch-{epoch+1:04}-{step:06}-sample-100-rows")

    avg_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

print("Training Complete!")
