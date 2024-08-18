import argparse
import os
import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from model import GeneratorRRDB
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

# Argument parser
parser = argparse.ArgumentParser(description='Test Images in Directory')
parser.add_argument('--upscale_factor', default=5, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['CPU', 'GPU'], help='using CPU or GPU')
parser.add_argument('--test_dir', type=str, default='/home/wj/works/SR-project/WSdata/test/', help='directory containing LR and HR images')
parser.add_argument('--output_dir', type=str, default='./output_b32_40_2000_0.01_0_128/', help='directory to save the output images')
parser.add_argument('--model_name', default='generator_39.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

# Set options
UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
TEST_DIR = opt.test_dir
OUTPUT_DIR = opt.output_dir
MODEL_NAME = opt.model_name

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
model = GeneratorRRDB(channels=3, num_res_blocks=16).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('saved_models/b32_40_2000_0.01_0_128/' + MODEL_NAME, map_location=lambda storage, loc: storage))

# Get image paths
lr_images = sorted([os.path.join(TEST_DIR, 'LR', x) for x in os.listdir(os.path.join(TEST_DIR, 'LR')) if x.endswith(('png', 'jpg', 'jpeg'))])
hr_images = sorted([os.path.join(TEST_DIR, 'HR', x) for x in os.listdir(os.path.join(TEST_DIR, 'HR')) if x.endswith(('png', 'jpg', 'jpeg'))])

mse_total = 0
ssim_total = 0
psnr_total = 0
num_images = len(lr_images)

for lr_image_path, hr_image_path in zip(lr_images, hr_images):
    lr_image = Image.open(lr_image_path).convert('RGB')
    hr_image = Image.open(hr_image_path).convert('RGB')

    lr_image = ToTensor()(lr_image).unsqueeze(0)
    hr_image = ToTensor()(hr_image).unsqueeze(0)

    if TEST_MODE:
        lr_image = lr_image.cuda()
        hr_image = hr_image.cuda()

    with torch.no_grad():
        sr_image = model(lr_image)
        sr_image = torch.clamp(sr_image, min=0, max=1)

    # Calculate MSE and RMSE using torch
    mse = torch.nn.functional.mse_loss(sr_image, hr_image).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    mse_total += mse

    # Convert tensors back to numpy arrays for SSIM and PSNR
    sr_image_np = sr_image[0].cpu().numpy().transpose(1, 2, 0)
    hr_image_np = hr_image[0].cpu().numpy().transpose(1, 2, 0)

    # Calculate SSIM and PSNR with appropriate win_size and data_range for small images
    win_size = min(7, hr_image_np.shape[0], hr_image_np.shape[1])  # 이미지 크기보다 크지 않도록 설정
    ssim_value = ssim(hr_image_np, sr_image_np, channel_axis=2, win_size=win_size, data_range=1.0)
    psnr_value = psnr(hr_image_np, sr_image_np, data_range=1.0)
    ssim_total += ssim_value
    psnr_total += psnr_value

    print(f'Processed {os.path.basename(lr_image_path)}: MSE={mse:.4f}, RMSE={rmse:.4f}, SSIM={ssim_value:.4f}, PSNR={psnr_value:.4f}')

    # Convert tensors back to PIL images
    lr_image_pil = ToPILImage()(lr_image[0].cpu())
    sr_image_pil = ToPILImage()(sr_image[0].cpu())
    hr_image_pil = ToPILImage()(hr_image[0].cpu())

    # Resize LR image to match HR and SR dimensions
    lr_image_resized = lr_image_pil.resize((100, 100), Image.BICUBIC)

    # Create a new image with enough width to hold all three images
    total_width = lr_image_resized.width + sr_image_pil.width + hr_image_pil.width
    max_height = max(lr_image_resized.height, sr_image_pil.height, hr_image_pil.height)

    new_image = Image.new('RGB', (total_width, max_height))

    # Paste the images side by side
    new_image.paste(lr_image_resized, (0, 0))
    new_image.paste(sr_image_pil, (lr_image_resized.width, 0))
    new_image.paste(hr_image_pil, (lr_image_resized.width + sr_image_pil.width, 0))

    # Save the output image
    output_image_path = os.path.join(OUTPUT_DIR, 'out_srf_' + str(UPSCALE_FACTOR) + '_' + os.path.basename(lr_image_path))
    new_image.save(output_image_path)

    print(f'Saved to {output_image_path}')

# Average MSE, RMSE, SSIM, and PSNR calculation
mse_avg = mse_total / num_images
rmse_avg = np.sqrt(mse_avg)
ssim_avg = ssim_total / num_images
psnr_avg = psnr_total / num_images

print(f'Average MSE: {mse_avg:.4f}')
print(f'Average RMSE: {rmse_avg:.4f}')
print(f'Average SSIM: {ssim_avg:.4f}')
print(f'Average PSNR: {psnr_avg:.4f}')
