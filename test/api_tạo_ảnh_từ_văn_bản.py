# -*- coding: utf-8 -*-
import os
import json
from PIL import Image
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
from diffusers import StableDiffusionPipeline, DDPMScheduler
from torch.optim import AdamW
from accelerate import Accelerator
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from ultralytics import YOLO

# Tải mô hình YOLOv5
model_yolo = YOLO("yolov5s.pt")  # Thay bằng mô hình YOLO khác nếu cần

# Đường dẫn đến tập dữ liệu
dataset_path = "/content/drive/MyDrive/txttoimage11/dataset"

def analyze_images(dataset_path, model):
    """
    Phân tích hình ảnh trong thư mục, dự đoán nhãn bằng mô hình,
    và lưu nhãn vào file JSON tương ứng.
    """
    for filename in os.listdir(dataset_path):
        # Kiểm tra phần mở rộng của file
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(dataset_path, filename)
            label_path = os.path.splitext(image_path)[0] + ".json"

            # Kiểm tra xem file nhãn đã tồn tại
            if not os.path.exists(label_path):
                try:
                    # Tải và phân tích ảnh
                    image = Image.open(image_path)
                    results = model(image)

                    # Trích xuất nhãn từ kết quả
                    labels = []
                    for *xyxy, conf, cls in results.xyxy[0]:
                        label = {
                            "class_id": int(cls),
                            "name": model.names[int(cls)],
                            "bbox": [int(x) for x in xyxy],
                            "confidence": float(conf)
                        }
                        labels.append(label)

                    # Lưu nhãn vào file JSON
                    with open(label_path, "w") as f:
                        json.dump(labels, f)

                    print(f"Lưu nhãn: {label_path}")
                except Exception as e:
                    print(f"Lỗi xử lý {filename}: {e}")
            else:
                print(f"File nhãn đã tồn tại: {label_path}")

# Gọi hàm phân tích ảnh
analyze_images(dataset_path, model_yolo)

def preprocess(image):
    """
    Tiền xử lý ảnh: chuyển đổi ảnh PIL sang tensor và chuẩn hóa.
    """
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),  # Thay đổi kích thước ảnh
        transforms.ToTensor(),         # Chuyển ảnh PIL sang tensor
        transforms.Normalize([0.5], [0.5])  # Chuẩn hóa dữ liệu (-1 đến 1)
    ])
    return preprocess(image)

def preprocess_function(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    examples["pixel_values"] = images

    # Trích xuất nhãn từ các tệp JSON
    labels = []
    for i, image in enumerate(examples["image"]):
        filename = examples["image_file_path"][i]
        label_path = os.path.splitext(filename)[0] + ".json"
        with open(label_path, "r") as f:
            label = json.load(f)
        labels.append(label)
    examples["labels"] = labels

    return examples

# Tải dataset
dataset = load_dataset("imagefolder", data_dir=dataset_path)

# Tiền xử lý dữ liệu
dataset = dataset.map(preprocess_function, batched=True, num_proc=4)

# Tạo DataLoader
train_dataloader = DataLoader(dataset["train"], batch_size=4, shuffle=True)

# Tải mô hình Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")

# Sử dụng DPMSolverMultistepScheduler
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

# Di chuyển model sang GPU nếu có
accelerator = Accelerator()
device = accelerator.device
pipe.to(device)

# Optimizer
optimizer = AdamW(pipe.unet.parameters(), lr=5e-6)

# Số epochs
num_epochs = 10

# Vòng lặp huấn luyện
for epoch in range(num_epochs):
    pipe.train()
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for step, batch in progress_bar:
        # Di chuyển dữ liệu sang thiết bị (GPU hoặc CPU)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"]

        # Tạo noise ngẫu nhiên
        noise = torch.randn(pixel_values.shape).to(device)

        # Tạo timestep ngẫu nhiên
        timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (pixel_values.shape[0],)).long().to(device)

        # Thêm noise vào ảnh gốc
        noisy_images = pipe.scheduler.add_noise(pixel_values, noise, timesteps)

        # Dự đoán noise
        noise_pred = pipe.unet(noisy_images, timesteps, encoder_hidden_states=pipe.text_encoder.encode(labels)).sample

        # Tính loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # Backpropagation
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        # Hiển thị loss trên progress bar
        progress_bar.set_postfix({"loss": loss.item()})

# Lưu mô hình sau khi huấn luyện
pipe.save_pretrained("/content/drive/MyDrive/txttoimage11")

def image_to_base64(image):
    """
    Chuyển đổi ảnh PIL sang định dạng base64.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

app = Flask(__name__)

# Tải mô hình Stable Diffusion đã huấn luyện
pipe = StableDiffusionPipeline.from_pretrained("/content/drive/MyDrive/txttoimage11")
pipe = pipe.to("cuda")

@app.route('/generate', methods=['POST'])
def generate():
    description = request.form['description']
    # Xử lý mô tả và tạo prompt (ví dụ: thêm "no background")
    prompt = f"{description}, no background"
    image = pipe(prompt).images[0]
    image_base64 = image_to_base64(image)
    return jsonify({'image': image_base64})  # Trả về ảnh

if __name__ == '__main__':
    app.run()