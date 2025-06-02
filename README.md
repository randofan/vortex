# VORTEX Painting-Year Prediction

End-to-end pipeline that mirrors the VORTEX paper (ViT + LoRA + CORAL).

```bash
# 1 env
conda create -n vortex python=3.10 -y
conda activate vortex
pip install -r requirements.txt

# 2 data → downloads & dedupes images
python scripts/download_images.py --csv data/paintings.csv --out data/images

# 3 train
python vortex/train.py --csv data/clean.csv --epochs 30

# 4 hyper-param tuning
python vortex/tune.py --csv data/clean.csv --random-trials 50 --timeout 3600
```

# TODOs

- [ ] Check LoRa target module names in model.py:75
```bash
python - << 'EOF'
from transformers import AutoModel
model = AutoModel.from_pretrained("facebook/dinov2-giant")
# look for anything with query/key/value/proj/dense
names = [n for n,_ in model.named_modules() 
         if any(tok in n for tok in ("query","key","value","proj","dense"))]
print("\n".join(sorted(set(names))))
EOF
```