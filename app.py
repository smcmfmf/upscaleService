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
print(f"Using device: {device}")

denoise_models = {}

try:
    print("Loading DnCNN Sigma 2...")
    dncnn_sigma2 = DnCNN(nb=17).to(device)
    dncnn_sigma2.load_state_dict(torch.load('weights/dncnn_sigma2_color.pth', map_location=device), strict=False)
    dncnn_sigma2.eval()
    denoise_models['dncnn_sigma2'] = {
        'model': dncnn_sigma2,
        'name': 'DnCNN Light',
        'description': '약한 노이즈 제거 (Sigma 2)'
    }
    print("DnCNN Sigma 2 loaded")
except Exception as e:
    print(f"DnCNN Sigma 2 failed: {e}")

try:
    print("Loading DnCNN 25...")
    dncnn_25 = DnCNN(nb=20).to(device)
    dncnn_25.load_state_dict(torch.load('weights/dncnn_25.pth', map_location=device), strict=False)
    dncnn_25.eval()
    denoise_models['dncnn_25'] = {
        'model': dncnn_25,
        'name': 'DnCNN Medium',
        'description': '중간 노이즈 제거 (Noise Level 25)'
    }
    print("DnCNN 25 loaded")
except Exception as e:
    print(f"DnCNN 25 failed: {e}")

try:
    print("Loading DnCNN 50...")
    dncnn_50 = DnCNN(nb=20).to(device)
    dncnn_50.load_state_dict(torch.load('weights/dncnn_50.pth', map_location=device), strict=False)
    dncnn_50.eval()
    denoise_models['dncnn_50'] = {
        'model': dncnn_50,
        'name': 'DnCNN Strong',
        'description': '강한 노이즈 제거 (Noise Level 50)'
    }
    print("DnCNN 50 loaded")
except Exception as e:
    print(f"DnCNN 50 failed: {e}")

denoise_models['none'] = {
    'model': None,
    'name': 'No Denoising',
    'description': '노이즈 제거 생략'
}

upscale_models = {}

try:
    print("Loading RealESRGAN x4...")
    rrdb_x4 = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upscale_models['realesrgan_x4'] = {
        'model': RealESRGANer(
            scale=4,
            model_path='weights/RealESRGAN_x4plus.pth',
            model=rrdb_x4,
            tile=0,
            pre_pad=0,
            half=False
        ),
        'scale': 4,
        'name': 'RealESRGAN x4',
        'description': '4배 업스케일 (균형잡힌 품질)'
    }
    print("RealESRGAN x4 loaded")
except Exception as e:
    print(f"RealESRGAN x4 failed: {e}")

try:
    print("Loading RealESRGAN Anime x4...")
    rrdb_anime = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    upscale_models['realesrgan_anime'] = {
        'model': RealESRGANer(
            scale=4,
            model_path='weights/RealESRGAN_x4plus_anime_6B.pth',
            model=rrdb_anime,
            tile=0,
            pre_pad=0,
            half=False
        ),
        'scale': 4,
        'name': 'RealESRGAN Anime x4',
        'description': '4배 업스케일 (애니메이션/일러스트 특화)'
    }
    print("RealESRGAN Anime x4 loaded")
except Exception as e:
    print(f"RealESRGAN Anime x4 failed: {e}")

print(f"\n=== Loaded Models Summary ===")
print(f"Denoise models: {len([k for k in denoise_models.keys() if k != 'none'])}")
print(f"Upscale models: {len(upscale_models)}")
print("=" * 30 + "\n")

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

@app.route('/models', methods=['GET'])
def get_models():
    """사용 가능한 모델 목록 반환"""
    denoise_list = [
        {'id': k, 'name': v['name'], 'description': v['description']}
        for k, v in denoise_models.items()
    ]
    upscale_list = [
        {'id': k, 'name': v['name'], 'description': v['description'], 'scale': v['scale']}
        for k, v in upscale_models.items()
    ]
    return jsonify({
        'denoise_models': denoise_list,
        'upscale_models': upscale_list
    })

@app.route('/upscale', methods=['POST'])
def upscale():
    temp_dir = 'temp_upscale_parts'
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        denoise_model_id = request.form.get('denoise_model', 'dncnn_sigma2')
        upscale_model_id = request.form.get('upscale_model', 'realesrgan_x4')
        downscale = request.form.get('downscale', 'true').lower() == 'true'

        print(f"\n=== Processing Request ===")
        print(f"Denoise model: {denoise_model_id}")
        print(f"Upscale model: {upscale_model_id}")
        print(f"Downscale: {downscale}")

        if denoise_model_id not in denoise_models:
            return jsonify({'error': f'Invalid denoise model: {denoise_model_id}'}), 400
        if upscale_model_id not in upscale_models:
            return jsonify({'error': f'Invalid upscale model: {upscale_model_id}'}), 400

        denoise_info = denoise_models[denoise_model_id]
        upscale_info = upscale_models[upscale_model_id]

        img = Image.open(file.stream).convert("RGB")
        original_size = img.size
        print(f"Original size: {original_size}")

        if downscale:
            w, h = img.size
            img = img.resize((w // 2, h // 2), Image.LANCZOS)
            print(f"Downscaled to: {img.size}")

        img_np = np.array(img).astype(np.float32) / 255.0
        
        if denoise_info['model'] is not None:
            print("Applying denoising...")
            img_tensor = torch.from_numpy(img_np.transpose(2,0,1)).unsqueeze(0).to(device)
            with torch.no_grad():
                denoised = denoise_info['model'](img_tensor).clamp(0,1)
            denoised_np = denoised.squeeze().cpu().numpy().transpose(1,2,0)
            print("Denoising complete")
        else:
            print("Skipping denoising")
            denoised_np = img_np

        denoised_bgr = cv2.cvtColor((denoised_np*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        print("Starting upscaling...")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        h_curr, w_curr, c = denoised_bgr.shape
        rows, cols = 3, 4
        h_step = h_curr // rows
        w_step = w_curr // cols
        pad_size = 32
        scale_factor = upscale_info['scale']

        upscaled_parts_paths = []

        for r in range(rows):
            row_paths = []
            for col in range(cols):
                y_start = r * h_step
                y_end = (r + 1) * h_step if r < rows - 1 else h_curr
                x_start = col * w_step
                x_end = (col + 1) * w_step if col < cols - 1 else w_curr

                y_pad_start = max(0, y_start - pad_size)
                y_pad_end = min(h_curr, y_end + pad_size)
                x_pad_start = max(0, x_start - pad_size)
                x_pad_end = min(w_curr, x_end + pad_size)

                crop_padded = denoised_bgr[y_pad_start:y_pad_end, x_pad_start:x_pad_end]

                crop_upscaled_padded, _ = upscale_info['model'].enhance(crop_padded, outscale=scale_factor)

                crop_y_offset = (y_start - y_pad_start) * scale_factor
                crop_x_offset = (x_start - x_pad_start) * scale_factor
                
                core_h_scaled = (y_end - y_start) * scale_factor
                core_w_scaled = (x_end - x_start) * scale_factor

                crop_upscaled_core = crop_upscaled_padded[
                    crop_y_offset : crop_y_offset + core_h_scaled,
                    crop_x_offset : crop_x_offset + core_w_scaled
                ]

                part_filename = os.path.join(temp_dir, f"part_{r}_{col}.png")
                cv2.imwrite(part_filename, crop_upscaled_core)
                row_paths.append(part_filename)
            
            upscaled_parts_paths.append(row_paths)

        full_h = h_curr * scale_factor
        full_w = w_curr * scale_factor
        final_upscaled = np.zeros((full_h, full_w, 3), dtype=np.uint8)

        current_y = 0
        for r in range(rows):
            current_x = 0
            row_height = 0
            for col in range(cols):
                part_path = upscaled_parts_paths[r][col]
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

        print(f"Upscaling complete. Final size: {upscaled.shape[:2][::-1]}")
        print("=" * 30 + "\n")

        original_b64 = pil_to_base64(img)
        denoised_b64 = cv2_to_base64(denoised_bgr)
        upscaled_b64 = cv2_to_base64(upscaled)

        return jsonify({
            'original': original_b64,
            'denoised': denoised_b64,
            'upscaled': upscaled_b64,
            'settings': {
                'denoise_model': denoise_info['name'],
                'upscale_model': upscale_info['name'],
                'scale': scale_factor,
                'downscaled': downscale
            }
        })

    except Exception as e:
        print("Error:", e)
        import traceback
        traceback.print_exc()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)