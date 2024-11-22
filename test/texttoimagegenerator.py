# -*- coding: utf-8 -*-
"""TextToImageGenerator.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1T7WemEtSlsFxaJ4pDGKesz9_LtoUhsl1
"""

!pip install --upgrade diffusers transformers -q

from pathlib import Path
import tqdm
import torch
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cv2

class CFG:
    device = "cuda"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400,400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12

image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16,
    revision="fp16", use_auth_token='your_hugging_face_auth_token', guidance_scale=9
)
image_gen_model = image_gen_model.to(CFG.device)

def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    return image

# !pip install googletrans==4.0.0-rc1 -q

from googletrans import Translator

def translation(txt):
  translator = Translator()
  ban_dich = translator.translate(txt, src='vi', dest='en')
  return ban_dich.text

txt="naruto"
generate_image(translation(txt), image_gen_model)

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import pipeline

# Khởi tạo pipeline Stable Diffusion với scheduler được cập nhật
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe= pipe.to("cuda")

# Khởi tạo pipeline tạo prompt
prompt_generator = pipeline("text-generation", model="gpt2")

# Tạo prompt
prompt = prompt_generator("a 3D model of a", num_return_sequences=1, max_length=30)

# Tạo mô hình 3D
with torch.autocast("cuda"):
    image = pipe(prompt[0]["generated_text"], num_inference_steps=50, guidance_scale=7.5).images[0]

# Hiển thị mô hình 3D
image.show()

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2", torch_dtype=torch.float16, device="cuda"
)

prompt = prompt_generator(
    "a 3D model of a",
    num_return_sequences=1,
    max_length=30,
    truncation=True
)

!pip install torch torchvision torchaudio
!pip install pytorch3d
!pip install dreamfusion

!git config --global user.name "Thanh An"
!git config --global user.email "caqonguyenthanhan.aaa@gmail.com"

!pip install git+https://github.com/facebookresearch/dreamfusion.git

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import pipeline

# Khởi tạo pipeline Stable Diffusion với scheduler được cập nhật
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe= pipe.to("cuda")

# Khởi tạo pipeline tạo prompt
prompt_generator = pipeline("text-generation", model="gpt2")

# Tạo prompt
prompt = prompt_generator("a 3D model of a", num_return_sequences=1, max_length=30)

# Sử dụng DreamFusion để tạo mô hình 3D (giả định bạn đã cài đặt DreamFusion)
model_3d = dreamfusion.generate_model(prompt[0]["generated_text"])

# Hiển thị hoặc lưu mô hình 3D
model_3d.show()