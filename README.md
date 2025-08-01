## 1. Create python venv
- Using UV
```bash
uv venv
```
- Active the venv
```bash
source .venv/bin/active
```

- Install requirements.txt
```bash
uv pip install -r model/requirements.txt
```

## 2. Create Dataset
Please change `--input-image`, `--fft-image` to the correct path and `--out-root` to your desired directory.
```bash
python dataset/create_inpainting_dataset.py \
  --input-image ./ulam_spiral_liouville.png \
  --fft-image ./ulam_spiral_fft.png \
  --out-root ./ulam_training_data \
  --num-samples 256 \
  --crop-size 512 \
  --augment \
  --mask-ratio 0.5
```

FADM
```bash
python dataset/create_inpainting_dataset.py \
  --input-image /home/user/liouville/dataset/ulam_spiral_liouville.png \
  --out-root /home/user/liouville/dataset/ulam_training_data_FADM \
  --num-samples 256 \
  --crop-size 512 \
  --augment \
  --mask-ratio 0.5
```

## 3. Run Finetuned
Please change `--instance_data_dir`, `--output_dir` to your desired full paths.
```bash 
python model/train_dreambooth_inpaint.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-inpainting" \
  --instance_data_dir="../dataset/ulam_training_data/images" \
  --instance_prompt="ulamsprial pattern" \
  --output_dir="../finetuned_model" \
  --resolution=512 \
  --train_batch_size=1 \
  --num_train_epochs=10 \
  --learning_rate=5e-6 \
  --gradient_accumulation_steps=1
```

Focal Loss
```bash
python model/train_inpainting_unet_combined_loss.py --data_root dataset/ulam_training_data_seperate/ --img_size 256 --learning_rate 5e-6 --batch_size 1 --epochs 50 --alpha 0.25 --gamma 2.0 --lambda_f 1.0
```

## 4. Test 
Please change the `model_path`, `image_path` and `mask_path` to the correct path and model in `model/test_inpainting.py`
```bash
python model/test_inpainting.py
```

Focal Loss
```bash
python model/test_inpainting_unet_combined_loss.py --model_path inpainting_unet_3ch.pt --image_path dataset/ulam_training_data_seperate/test/images/sample_0006.png --mask_path dataset/ulam_training_data_seperate/test/masks/sample_0006.png --fft_path dataset/ulam_training_data_seperate/test/ffts/sample_0006.png
```