# [ICCV 2025] Know "No" Better: A Data-Driven Approach for Enhancing Negation Awareness in CLIP
[arXiv](https://arxiv.org/abs/2501.10913) • [Hugging Face Model](https://huggingface.co/jerryray/negationclip)

> **TL;DR.** To address the negation issue in CLIP, we propose data generation pipelines and validate their effectiveness using public and novel benchmarks. Our approach enhances negation awareness and extends to diverse multimodal tasks.

---

## Status
- ✅ Preprint available on [arXiv](https://arxiv.org/abs/2501.10913)  
- ✅ NegationCLIP checkpoints available on [Hugging Face](https://huggingface.co/jerryray/negationclip)  
- ✅ Data generation & fine-tuning scripts included  
- 🚧 Benchmarks and downstream applications (coming soon)  
- 🗓️ Presentation at **ICCV 2025**, Honolulu, Hawaii  

---

## ⚙️ Installation

```bash
git clone https://github.com/jerryray/NegationCLIP.git
cd NegationCLIP
conda env create -f environment.yml
conda activate negationclip
```

---

## 🧠 Data Generation (Negation-inclusive Captions)

The script generates captions with explicit negation from COCO using **LLaMA-3** and **LLaVA-v1.6-Mistral-7B**.

```bash
python src/data_generation.py \
  --caption_path /path/to/COCO/captions_train2014.json \
  --image_dir /path/to/COCO/train2014 \
  --output_dir ./output
```

**Options**  
- `--use_random_object`: randomly select absent objects (instead of contextual ones)  

---

## 🏋️ Fine-Tuning (NegationCLIP)

Fine-tune the **text encoder** of CLIP on negation-inclusive captions:

```bash
python src/clip_finetune.py \
  --json_path ./annotations/negationclip_captions_train2014.json \
  --image_dir /path/to/train2014 \
  --output_dir ./checkpoints \
  --clip_model "ViT-B/32" \
```

**Outputs**  
- Best model automatically saved when validation loss improves.

---

## 🧱 Directory Structure

```
negationclip/
├── src/
│   ├── clip_finetune.py
│   └── data_generation.py
├── annotations/
│   └── negationclip_captions_train2014.json
├── requirements.txt
├── environment.yml
├── README.md
└── LICENSE
```

---

## 📦 Model Access

- **Hugging Face:** [jerryray/negationclip](https://huggingface.co/jerryray/negationclip)  
- **Model Type:** Fine-tuned CLIP (ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px)  

---