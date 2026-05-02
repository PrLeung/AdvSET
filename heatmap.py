import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from ruamel.yaml import YAML
from torchvision import transforms
from transformers import BertForMaskedLM

from dataset import pre_caption, compute_model_specific_attention, _normalize_attention, _resize_attention
from eval_AET import load_model


yaml = YAML(typ='safe')


def overlay_heatmap(rgb_np, heatmap_np, alpha=0.45):
    cmap = plt.get_cmap('jet')
    color = cmap(heatmap_np)[..., :3]
    blended = (1 - alpha) * rgb_np + alpha * color
    return np.clip(blended, 0, 1)


def build_model_input_transform(model_name):
    if model_name in {'ALBEF', 'TCL'}:
        return transforms.Compose([
            transforms.Resize((384, 384), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Retrieval_flickr.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--source_text_encoder', default='bert-base-uncased')
    parser.add_argument('--albef_ckpt', default='./checkpoints/albef_flickr.pth')
    parser.add_argument('--tcl_ckpt', default='./checkpoints/tcl_flickr.pth')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--output_dir', default='./attention_vis_10')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
    config = yaml.load(open(args.config, 'r'))

    ann = json.load(open(config['test_file'], 'r'))
    samples = ann[:min(args.num_samples, len(ann))]

    # load models
    import eval_AET as eval_module
    eval_module.config = config
    model_names = ['ALBEF', 'TCL', 'CLIP_ViT']
    models = {}
    for m in model_names:
        model, _, tokenizer = load_model(args, m, args.source_text_encoder, device)
        model = model.to(device).eval()
        models[m] = (model, tokenizer)

    os.makedirs(args.output_dir, exist_ok=True)

    for i, item in enumerate(samples):
        image_path = os.path.join(config['image_root'], item['image'])
        caption = pre_caption(item['caption'][0], 30)

        img_pil = Image.open(image_path).convert('RGB')
        base_show = img_pil.resize((384, 384), Image.BICUBIC)
        base_np = np.asarray(base_show).astype(np.float32) / 255.0

        overlays = {}
        for m in model_names:
            model, tokenizer = models[m]
            tfm = build_model_input_transform(m)
            image_tensor = tfm(img_pil).to(device)

            with torch.enable_grad():
                attn = compute_model_specific_attention(
                    image_tensor=image_tensor,
                    caption=caption,
                    model_name=m,
                    source_model=model,
                    device=device,
                    tokenizer=tokenizer,
                )
            attn = _normalize_attention(attn)
            attn = _resize_attention(attn, 384).detach().cpu().numpy()
            overlays[m] = overlay_heatmap(base_np, attn)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(base_np)
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(overlays['ALBEF'])
        axes[1].set_title('ALBEF Attention')
        axes[1].axis('off')

        axes[2].imshow(overlays['TCL'])
        axes[2].set_title('TCL Attention')
        axes[2].axis('off')

        axes[3].imshow(overlays['CLIP_ViT'])
        axes[3].set_title('CLIP_ViT Attention')
        axes[3].axis('off')

        fig.suptitle(f"sample={i} | image={item['image']}\ncaption={caption}", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f'compare_{i:02d}.png'), dpi=200)
        plt.close(fig)

    print(f'Saved {len(samples)} comparison figures to: {args.output_dir}')


if __name__ == '__main__':
    main()
