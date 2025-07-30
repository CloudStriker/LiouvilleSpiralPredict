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
uv pip install -r /model/requirements.txt
```

## 2. Create Dataset
```bash
python dataset/create_inpainting_dataset.py --input-image ./ulam_spiral_liouville.png --fft-image ./ulam_spiral_fft.png --out-root ./ulam_training_data --num-samples 256 --crop-size 512 --augment --mask-ratio 0.5
```

## 3. Run Finetuned
```bash
python model/train_dreambooth_inpaint.py   --pretrained_model_name_or_path="runwayml/stable-diffusion-inpainting" --instance_data_dir="../dataset/ulam_training_data/images" --instance_prompt="ulamsprial pattern" --output_dir="../finetuned_model" --resolution=512 --train_batch_size=1 --num_train_epochs=10 --learning_rate=5e-6 --gradient_accumulation_steps=1
```

## 4. Test 
```bash
python model/test_inpainting.py
```