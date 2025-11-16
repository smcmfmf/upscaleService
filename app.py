import os
import io
import base64
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import cv2
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

app = Flask(__name__)

class DnCNN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=20):
        super(DnCNN, self).__init__()
        self.in_conv = nn.Conv2d(in_nc, nc, 3, 1, 1, bias=True)
        self.conv_list = nn.ModuleList([nn.Conv2d(nc, nc, 3, 1, 1, bias=True) for _ in range(nb - 2)])
        self.out_conv = nn.Conv2d(nc, out_nc, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_in = x
        x = self.relu(self.in_conv(x))
        for conv in self.conv_list:
            x = self.relu(conv(x))
        x = self.out_conv(x)
        return x_in - x

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 노이즈 필터 (DnCNN)
dncnn = DnCNN().to(device)
dncnn.load_state_dict(torch.load('weights/dncnn_sigma2_color.pth', map_location=device), strict=False)
dncnn.eval()

# 업스케일러 (Real-ESRGAN)
rrdb_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path='weights/RealESRGAN_x4plus.pth',
    model=rrdb_model,
    tile=0,
    pre_pad=0,
    half=False
)

def pil_to_base64(img_pil):
    buff = io.BytesIO()
    img_pil.save(buff, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buff.getvalue()).decode("utf-8")

def cv2_to_base64(img_cv2):
    _, buffer = cv2.imencode(".png", img_cv2)
    return "data:image/png;base64," + base64.b64encode(buffer).decode("utf-8")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upscale', methods=['POST'])
def upscale():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        img = Image.open(file.stream).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np.transpose(2,0,1)).unsqueeze(0).to(device)

        # 1. DnCNN 노이즈 제거
        with torch.no_grad():
            denoised = dncnn(img_tensor).clamp(0,1)
        denoised_np = denoised.squeeze().cpu().numpy().transpose(1,2,0)
        denoised_bgr = cv2.cvtColor((denoised_np*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # 2. Real-ESRGAN 업스케일링
        upscaled, _ = upsampler.enhance(denoised_bgr, outscale=4)

        # 결과를 base64로 변환
        original_b64 = pil_to_base64(img)
        denoised_b64 = cv2_to_base64(denoised_bgr)
        upscaled_b64 = cv2_to_base64(upscaled)

        return jsonify({
            'original': original_b64,
            'denoised': denoised_b64,
            'upscaled': upscaled_b64
        })
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
