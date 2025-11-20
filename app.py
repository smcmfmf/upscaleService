import os
import io
import base64
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import cv2
import shutil
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
    temp_dir = 'temp_upscale_parts'
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        img = Image.open(file.stream).convert("RGB")

        # 1. 이미지 2배 축소 (Downscaling)
        w, h = img.size
        img = img.resize((w // 2, h // 2), Image.LANCZOS)

        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np.transpose(2,0,1)).unsqueeze(0).to(device)

        # 2. DnCNN 노이즈 제거
        with torch.no_grad():
            denoised = dncnn(img_tensor).clamp(0,1)
        denoised_np = denoised.squeeze().cpu().numpy().transpose(1,2,0)
        denoised_bgr = cv2.cvtColor((denoised_np*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # 3. Real-ESRGAN 업스케일링
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        h_curr, w_curr, c = denoised_bgr.shape
        rows, cols = 3, 4
        h_step = h_curr // rows
        w_step = w_curr // cols
        
        pad_size = 32 

        upscaled_parts_paths = []

        for r in range(rows):
            row_paths = []
            for c in range(cols):
                y_start = r * h_step
                y_end = (r + 1) * h_step if r < rows - 1 else h_curr
                x_start = c * w_step
                x_end = (c + 1) * w_step if c < cols - 1 else w_curr

                y_pad_start = max(0, y_start - pad_size)
                y_pad_end = min(h_curr, y_end + pad_size)
                x_pad_start = max(0, x_start - pad_size)
                x_pad_end = min(w_curr, x_end + pad_size)

                crop_padded = denoised_bgr[y_pad_start:y_pad_end, x_pad_start:x_pad_end]

                crop_upscaled_padded, _ = upsampler.enhance(crop_padded, outscale=4)

                crop_y_offset = (y_start - y_pad_start) * 4
                crop_x_offset = (x_start - x_pad_start) * 4
                
                core_h_scaled = (y_end - y_start) * 4
                core_w_scaled = (x_end - x_start) * 4

                crop_upscaled_core = crop_upscaled_padded[
                    crop_y_offset : crop_y_offset + core_h_scaled,
                    crop_x_offset : crop_x_offset + core_w_scaled
                ]

                part_filename = os.path.join(temp_dir, f"part_{r}_{c}.png")
                cv2.imwrite(part_filename, crop_upscaled_core)
                row_paths.append(part_filename)
            
            upscaled_parts_paths.append(row_paths)

        full_h = h_curr * 4
        full_w = w_curr * 4
        final_upscaled = np.zeros((full_h, full_w, 3), dtype=np.uint8)

        current_y = 0
        for r in range(rows):
            current_x = 0
            row_height = 0
            for c in range(cols):
                part_path = upscaled_parts_paths[r][c]
                part_img = cv2.imread(part_path)
                
                if part_img is None:
                    raise Exception(f"Failed to load part image: {part_path}")

                ph, pw, _ = part_img.shape
                
                final_upscaled[current_y : current_y + ph, current_x : current_x + pw] = part_img
                
                current_x += pw
                row_height = ph
            
            current_y += row_height

        upscaled = final_upscaled

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

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
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)