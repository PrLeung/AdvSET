import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from ruamel.yaml import YAML

from dataset import compute_model_specific_attention, pre_caption, _normalize_attention, _resize_attention
from eval_AET import load_model


yaml = YAML(typ='safe')


def overlay_heatmap(rgb_np, heatmap_np, alpha=0.45):
    cmap = plt.get_cmap('jet')
    color = cmap(heatmap_np)[..., :3]
    blended = (1 - alpha) * rgb_np + alpha * color
    return np.clip(blended, 0, 1)


def get_image_tensor(img_pil, model_name):
    if model_name in {'ALBEF', 'TCL'}:
        img = img_pil.resize((384, 384), Image.BICUBIC)
    else:
        img = img_pil.resize((224, 224), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


def main():
    parser = argparse.ArgumentParser(description='Test dataset attention maps for ALBEF/TCL/CLIP_ViT.')
    parser.add_argument('--config', default='./configs/Retrieval_flickr.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--source_text_encoder', default='bert-base-uncased')
    parser.add_argument('--albef_ckpt', default='./checkpoints/albef_flickr.pth')
    parser.add_argument('--tcl_ckpt', default='./checkpoints/tcl_flickr.pth')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--output_dir', default='./attention_vis_10')
    parser.add_argument('--verbose', action='store_true', help='print intermediate debug information')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
    config = yaml.load(open(args.config, 'r'))

    print('[INFO] ===== Attention compare test start =====')
    print(f'[INFO] config={args.config}')
    print(f'[INFO] device={device}')
    print(f'[INFO] output_dir={args.output_dir}')

    # load models via existing eval pipeline helper
    import eval_AET as eval_module
    eval_module.config = config

    model_names = ['ALBEF', 'TCL', 'CLIP_ViT']
    models = {}
    tokenizers = {}
    for name in model_names:
        print(f'[INFO] loading model: {name}')
        model, _, tokenizer = load_model(args, name, args.source_text_encoder, device)
        models[name] = model.to(device).eval()
        tokenizers[name] = tokenizer
        print(f'[INFO] model ready: {name}')

    ann = yaml.load(open(config['test_file'], 'r')) if config['test_file'].endswith(('.yaml', '.yml')) else None
    if ann is None:
        import json
        ann = json.load(open(config['test_file'], 'r'))

    os.makedirs(args.output_dir, exist_ok=True)

    samples = ann[: min(args.num_samples, len(ann))]
    print(f'[INFO] total annotations={len(ann)}, selected_samples={len(samples)}')

    for idx, item in enumerate(samples):
        image_rel_path = item['image']
        caption = pre_caption(item['caption'][0], 30)
        image_path = os.path.join(config['image_root'], image_rel_path)

        print(f"[INFO] sample {idx+1}/{len(samples)} | image={image_rel_path}")
        if args.verbose:
            print(f"[DEBUG] caption={caption}")
            print(f"[DEBUG] image_path={image_path}")

        img_pil = Image.open(image_path).convert('RGB')
        base = img_pil.resize((384, 384), Image.BICUBIC)
        base_np = np.asarray(base).astype(np.float32) / 255.0

        overlays = {}
        for model_name in model_names:
            if args.verbose:
                print(f"[DEBUG] computing attention for model={model_name}")
            tensor = get_image_tensor(img_pil, model_name).to(device)
            with torch.enable_grad():
                attn = compute_model_specific_attention(
                    image_tensor=tensor,
                    caption=caption,
                    model_name=model_name,
                    source_model=models[model_name],
                    device=device,
                    tokenizer=tokenizers[model_name],
                )
            if args.verbose:
                print(f"[DEBUG] raw attn shape ({model_name})={tuple(attn.shape)}")
                print(f"[DEBUG] raw attn min/max ({model_name})={attn.min().item():.6f}/{attn.max().item():.6f}")
            attn = _normalize_attention(attn)
            attn = _resize_attention(attn, 384).detach().cpu().numpy()
            if args.verbose:
                print(f"[DEBUG] resized attn shape ({model_name})={attn.shape}")
                print(f"[DEBUG] resized attn min/max ({model_name})={attn.min():.6f}/{attn.max():.6f}")
            overlays[model_name] = overlay_heatmap(base_np, attn)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(base_np)
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(overlays['ALBEF'])
        axes[1].set_title('ALBEF')
        axes[1].axis('off')

        axes[2].imshow(overlays['TCL'])
        axes[2].set_title('TCL')
        axes[2].axis('off')

        axes[3].imshow(overlays['CLIP_ViT'])
        axes[3].set_title('CLIP_ViT')
        axes[3].axis('off')

        fig.suptitle(f"sample={idx} | image={image_rel_path}\ncaption={caption}", fontsize=10)
        fig.tight_layout()
        out_path = os.path.join(args.output_dir, f'compare_{idx:02d}.png')
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f'[INFO] saved: {out_path}')

    print(f'[INFO] ===== Done. Saved {len(samples)} figures to {args.output_dir} =====')


if __name__ == '__main__':
    main()
