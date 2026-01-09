# # Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# # > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
# from transformers import AutoModelForVision2Seq, AutoProcessor
# from PIL import Image

# import torch

# # Load Processor & VLA
# processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
# vla = AutoModelForVision2Seq.from_pretrained(
#     "openvla/openvla-7b", 
#     attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
#     torch_dtype=torch.bfloat16, 
#     low_cpu_mem_usage=True, 
#     trust_remote_code=True
# ).to("cuda:0")

# # Grab image input & format prompt
# image: Image.Image = get_from_camera(...)
# prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"

# # Predict Action (7-DoF; un-normalize for BridgeData V2)
# inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
# action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

# # Execute...
# robot.act(action, ...)

# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda:0")

# 2. 입력 데이터 준비 (이미지 + 명령어)
image = Image.open("data/scripted_raw/sweep_12-03/2022-12-04_14-56-20/raw/traj_group0/traj0/images0/im_0.jpg") 
# 로봇 카메라 이미지
prompt = "In order to pick up the can, the robot should" # Bridge 데이터셋 스타일 프롬프트

# 3. 추론 실행
inputs = processor(prompt, image, return_tensors="pt").to("cuda", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

print(f"Predicted Action: {action}") # [x, y, z, roll, pitch, yaw, gripper]

# Predicted Action: [-6.56135805e-04  1.18787507e-02  2.87665951e-03  9.44003262e-03
#  -2.75751861e-02  6.60170097e-02  9.96078431e-01]

