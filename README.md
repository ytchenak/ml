# SwinIR vs Swin2SR

## Resources
test images: https://drive.google.com/drive/folders/1EGdMvvT-JjOnQm7SHOuSRXlKf5R-WC2A?usp=sharing
- Extract to test_images

SwinIR models: https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0
- put inside `SwinIR/model_zoo`

Swin2SR models: https://github.com/mv-lab/swin2sr/releases/tag/v0.0.1
- put inside `swin2sr/model_zoo`

Optimized models: https://drive.google.com/drive/u/0/folders/1pMsGgihLrFXDFI8Q3DX52YjLgiEj2TwR

## Test

Run the following:

- `python SwinIR/main_test_swinir.py --task classical_sr --scale 2 --model_path SwinIR/model_zoo/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth --folder_lq test_images/urban100x2a --folder_gt test_images/urban100`

- `python SwinIR/main_test_swinir.py --task real_sr --scale 4 --model_path SwinIR/model_zoo/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth --folder_lq test_images/urban100x4a --folder_gt test_images/urban100`
 
- `python Swin2SR/main_test_swin2sr.py --task classical_sr --scale 2 --training_patch_size 64 --model_path SwinIR/model_zoo/Swin2SR_ClassicalSR_X2_64.pth --folder_lq test_images/urban100x2a --folder_gt test_images/urban100`

- `python Swin2SR/main_test_swin2sr.py --task real_sr --scale 4 --model_path Swin2SR/model_zoo/Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth --folder_lq test_images/urban100x4a --folder_gt test_images/urban100`
