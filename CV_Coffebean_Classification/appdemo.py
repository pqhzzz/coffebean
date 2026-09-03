# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import time

model_path = r'D:\coffe bean\coffee_roasting_model.pth'
class_names = ['Dark', 'Green', 'Light', 'Medium'] 

model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(num_ftrs, 4))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

def predict_coffee(image):
    if image is None:
        return None, "Vui lòng tải ảnh lên!"
        
    start_time = time.time()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_t = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_t)
        prob = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    end_time = time.time()
    process_time = round(end_time - start_time, 3)
    time_str = f"⏱️ Thời gian xử lý: **{process_time} giây**"
    
    result_dict = {class_names[i]: float(prob[i]) for i in range(len(class_names))}
    return result_dict, time_str

with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("<h1 style='text-align: center;'>☕ HỆ THỐNG PHÂN LOẠI CẤP ĐỘ RANG CÀ PHÊ</h1>")
    gr.Markdown("<p style='text-align: center;'>Mô hình Học sâu: <b>ResNet50</b> | Đồ án môn Thị giác Máy tính</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Tải ảnh hoặc sử dụng Webcam")
            predict_btn = gr.Button("🔍 Phân loại ngay", variant="primary")
            
        with gr.Column(scale=1):
            label_output = gr.Label(num_top_classes=4, label="Kết quả Nhận diện")
            time_output = gr.Markdown("⏱️ Thời gian xử lý: Đang chờ...")

    predict_btn.click(
        fn=predict_coffee, 
        inputs=img_input, 
        outputs=[label_output, time_output]
    )

if __name__ == "__main__":
    interface.launch()