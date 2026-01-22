import argparse
import os
import pickle
import json
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=False, default='checkpoints/openvla-lora-epoch-01-000000/')
parser.add_argument("--output", type=str, default='')
args = parser.parse_args()

device = "cuda:0"
print(f"device={device}")


vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map=device
)


print(f"resume checkpoints from '{args.checkpoint}'")
vla = PeftModel.from_pretrained(vla, args.checkpoint)
vla.eval()

processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)


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
        return os.path.join(self.img_dir, f"im_{idx}.jpg"), self.instruction, self.path
    
    def getitems(self):
        ims = []
        inst = []
        path = []
        for idx in range(self.img_len - 1):
            i, s, p = self.getitem(idx)
            ims.append(i)
            inst.append(s)
            path.append(p)
        return ims, self.actions, inst, path

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
        self.paths = []

        for traj in self.trajs:
            I, A, inst, p = traj.getitems()
            self.ims.extend(I)
            self.actions.extend(A)
            self.INST.extend(inst)
            self.paths.extend(p)
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
            "labels": labels,
            "path": self.paths[idx],
            "im_path": self.ims[idx]
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
        "attention_mask": attention_mask,
        "path": [item["path"] for item in batch],
        "im_path": [item["im_path"] for item in batch]
    }

df = pd.read_csv('./instruction.csv', )[:100]
print(f"Dataframe shape: {df.shape}")

train_dataset = BridgeDatasetV2(
    traj_dirs=df['path'].to_list(),
    instructions=df['instruction'].to_list(),
    processor=processor,
)

eval_dataloader = DataLoader(
    train_dataset, 
    batch_size=1, 
    shuffle=False, 
    collate_fn=collate_fn
)


print(f"DataLoader Length: {len(train_dataset)}")
batch = next(iter(eval_dataloader))
print(f"Input IDs shape: {batch['input_ids'].shape}")       # [BS, Seq_Len]
print(f"Pixel Values shape: {batch['pixel_values'].shape}") # [BS, 3, 224, 224]
print(f"Labels shape: {batch['labels'].shape}")             # [BS, 7] (7 action tokens)

from tqdm import tqdm

results = []

print("Starting Inference...")
with torch.no_grad():
    for i, batch in enumerate(tqdm(eval_dataloader)):
        
        input_ids = batch["input_ids"].to(device)
        pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
        
        
        # [BS, Seq_Len] 
        gt_token_ids = batch["labels"][batch["labels"] != -100].tolist()
        
        
        
        pred_action = vla.predict_action(
            input_ids=input_ids[0:1], 
            pixel_values=pixel_values[0:1], 
            unnorm_key="bridge_orig"
        )
        
        
        # OpenVLA: (token_id - 31000) / 255 * 2.0 - 1.0
        gt_action = [(tid - 31000) / 255.0 * 2.0 - 1.0 for tid in gt_token_ids]
        
        results.append({
            "sample_idx": i,
            "im_path": train_dataset.ims[i],
            "instruction": train_dataset.INST[i],
            "gt_action": np.array(gt_action).tolist(),
            "pred_action": pred_action.tolist() # to list
        })

        with open(f"inference_results-{args.output}.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(results[-1], ensure_ascii=False) + "\n")



#
#output_df = pd.DataFrame(results)
#output_df.to_json(f"inference_results-{args.output}.json", orient="records", indent=4)
#print(f"Inference results saved to 'inference_results-{args.output}.json'.")

# (MSE)
mse = np.mean([np.mean((np.array(r['gt_action']) - np.array(r['pred_action']))**2) for r in results])
print(f"Average Mean Squared Error (MSE): {mse:.6f}")
