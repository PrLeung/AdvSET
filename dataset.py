import json
import os
import re
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from models import clip


def pre_caption(caption, max_words):
    caption = re.sub(r"([,.'!?\"()*#:;~])", '', caption.lower())
    caption = caption.replace('-', ' ').replace('/', ' ').replace('<person>', 'person')
    caption = re.sub(r"\s{2,}", ' ', caption)
    caption = caption.rstrip('\n').strip(' ')
    words = caption.split(' ')
    if len(words) > max_words:
        caption = ' '.join(words[:max_words])
    return caption


def _load_image(path):
    return Image.open(path).convert('RGB')


def _normalize_attention(attn: torch.Tensor) -> torch.Tensor:
    attn = attn.float()
    attn_min = attn.min()
    attn_max = attn.max()
    return (attn - attn_min) / (attn_max - attn_min + 1e-8)


def _resize_attention(attn: torch.Tensor, size: int) -> torch.Tensor:
    attn = attn.unsqueeze(0).unsqueeze(0)
    attn = F.interpolate(attn, size=(size, size), mode='bilinear', align_corners=False)
    return attn[0, 0]


def _clip_interpret(image_tensor, caption, model, device, start_layer=-1):
    text = clip.tokenize([caption]).to(device)
    images = image_tensor.unsqueeze(0)
    logits_per_image, _ = model(images, text)
    one_hot = torch.ones_like(logits_per_image)
    score = torch.sum(one_hot * logits_per_image)
    model.zero_grad(set_to_none=True)

    blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    if start_layer == -1:
        start_layer = len(blocks) - 1

    num_tokens = blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, device=device, dtype=blocks[0].attn_probs.dtype).unsqueeze(0)

    for i, blk in enumerate(blocks):
        if i < start_layer:
            continue
        grad = torch.autograd.grad(score, [blk.attn_probs], retain_graph=True, allow_unused=True)[0]
        if grad is None:
            continue
        attn = blk.attn_probs.detach()
        cam = (grad.detach() * attn).clamp(min=0)
        cam = cam.reshape(1, -1, cam.shape[-2], cam.shape[-1]).mean(dim=1)
        R = R + torch.bmm(cam, R)

    relevance = R[:, 0, 1:]
    side = int(relevance.shape[-1] ** 0.5)
    return relevance.reshape(side, side)


def _albef_tcl_interpret(image_tensor, caption, model, tokenizer, device):
    # Register hook on last ViT block for attention/gradient
    last_block = model.visual_encoder.blocks[-1]

    text_input = tokenizer([caption], padding='max_length', truncation=True, max_length=30, return_tensors='pt').to(device)
    image = image_tensor.unsqueeze(0)

    out_img = model.visual_encoder(image, register_blk=len(model.visual_encoder.blocks) - 1)
    image_feat = F.normalize(model.vision_proj(out_img[:, 0, :]), dim=-1)

    text_out = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, return_dict=True, mode='text')
    text_feat = F.normalize(model.text_proj(text_out.last_hidden_state[:, 0, :]), dim=-1)

    score = (image_feat * text_feat).sum()
    model.zero_grad(set_to_none=True)
    score.backward(retain_graph=True)

    attn = last_block.attn.get_attention_map()      # [B, heads, tokens, tokens]
    grad = last_block.attn.get_attn_gradients()     # [B, heads, tokens, tokens]
    if attn is None or grad is None:
        raise RuntimeError('ALBEF/TCL attention hooks are empty; check VisionTransformer register_blk path')

    cam = (attn * grad).clamp(min=0).mean(dim=1)    # [B, tokens, tokens]
    relevance = cam[:, 0, 1:]                       # CLS -> patches
    side = int(relevance.shape[-1] ** 0.5)
    return relevance.reshape(side, side).detach()


def compute_model_specific_attention(image_tensor, caption, model_name, source_model, device, tokenizer=None):
    if model_name == 'CLIP_ViT':
        return _clip_interpret(image_tensor, caption, source_model, device)
    if model_name in {'ALBEF', 'TCL'}:
        if tokenizer is None:
            tokenizer = source_model.tokenizer
        return _albef_tcl_interpret(image_tensor, caption, source_model, tokenizer, device)
    raise ValueError(f'Unsupported model_name for attention: {model_name}')


class paired_dataset(Dataset):
    def __init__(
        self,
        ann_file,
        transform,
        image_root,
        processed_image_root,
        processed_attn_root,
        device,
        max_words=30,
        model_name=None,
        source_attn_model=None,
    ):
        if source_attn_model is None:
            raise ValueError('source_attn_model is required; fallback is disabled')

        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.processed_image_root = processed_image_root
        self.processed_attn_root = processed_attn_root
        self.device = device
        self.max_words = max_words
        self.model_name = model_name
        self.source_attn_model = source_attn_model.to(device).eval()

        self.text: List[str] = []
        self.image: List[str] = []
        self.image_ids: List[str] = []
        self.txt2img: Dict[int, int] = {}
        self.img2txt: Dict[int, List[int]] = {}

        self.attn_target_size = 384 if model_name in {'ALBEF', 'TCL'} else 224

        os.makedirs(self.processed_image_root, exist_ok=True)
        os.makedirs(self.processed_attn_root, exist_ok=True)

        # Model-specific tensor transforms for attention computation
        self.clip_attn_transform = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        self.albef_tcl_attn_transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

        txt_id = 0
        for i, ann in tqdm(enumerate(self.ann), total=len(self.ann), desc='Processing Images'):
            self.img2txt[i] = []
            image_name = ann['image']
            self.image.append(image_name)
            self.image_ids.append(image_name)

            image_path = os.path.join(self.image_root, image_name)
            transformed_image_path = os.path.join(self.processed_image_root, image_name)
            os.makedirs(os.path.dirname(transformed_image_path), exist_ok=True)

            source_image = _load_image(image_path)
            if not os.path.exists(transformed_image_path):
                transformed = self.transform(source_image)
                transforms.ToPILImage()(transformed).save(transformed_image_path)

            if self.model_name == 'CLIP_ViT':
                attn_image = self.clip_attn_transform(source_image).to(self.device)
            else:
                attn_image = self.albef_tcl_attn_transform(source_image).to(self.device)

            for caption in ann['caption']:
                processed_caption = pre_caption(caption, self.max_words)
                self.text.append(processed_caption)
                self.txt2img[txt_id] = i
                self.img2txt[i].append(txt_id)

                attn_matrix_path = os.path.join(self.processed_attn_root, f'{image_name}_text{txt_id}.npy')
                os.makedirs(os.path.dirname(attn_matrix_path), exist_ok=True)
                if not os.path.exists(attn_matrix_path):
                    with torch.enable_grad():
                        relevance_matrix = compute_model_specific_attention(
                            attn_image,
                            processed_caption,
                            self.model_name,
                            self.source_attn_model,
                            self.device,
                        )
                    relevance_matrix = _normalize_attention(relevance_matrix)
                    relevance_matrix = _resize_attention(relevance_matrix, self.attn_target_size)
                    np.save(attn_matrix_path, relevance_matrix.detach().cpu().numpy())
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.processed_image_root, self.image[index])
        image = transforms.ToTensor()(_load_image(image_path))

        text_ids = self.img2txt[index]
        texts = [self.text[i] for i in text_ids]

        attn_matrices = []
        for text_id in text_ids:
            attn_matrix_path = os.path.join(self.processed_attn_root, f'{self.image[index]}_text{text_id}.npy')
            image_relevance = torch.from_numpy(np.load(attn_matrix_path)).float()
            attn_matrices.append(_normalize_attention(image_relevance))

        attn_matrices = torch.stack(attn_matrices)
        averaged_attn_matrices = attn_matrices.mean(dim=0)
        return image, texts, self.image_ids[index], index, text_ids, averaged_attn_matrices

    def collate_fn(self, batch):
        imgs, txt_groups, img_name, img_ids, text_ids_groups, attn_matrices_groups = list(zip(*batch))
        imgs = torch.stack(imgs, 0)
        return imgs, txt_groups, img_name, list(img_ids), text_ids_groups, attn_matrices_groups


class paired_dataset2(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for i, ann in enumerate(self.ann):
            self.img2txt[i] = []
            self.image.append(ann['image'])
            for caption in ann['caption']:
                self.text.append(pre_caption(caption, self.max_words))
                self.txt2img[txt_id] = i
                self.img2txt[i].append(txt_id)
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image[index])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        text_ids = self.img2txt[index]
        texts = [self.text[i] for i in self.img2txt[index]]
        return image, texts, index, text_ids, self.image[index]

    def collate_fn(self, batch):
        imgs, txt_groups, img_ids, text_ids_groups, image_paths = list(zip(*batch))
        imgs = torch.stack(imgs, 0)
        return imgs, txt_groups, list(img_ids), text_ids_groups, image_paths


class pair_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []

        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for i, ann in enumerate(self.ann):
            self.img2txt[i] = []
            for caption in ann['caption']:
                self.image.append(ann['image'])
                self.text.append(pre_caption(caption, self.max_words))
                self.txt2img[txt_id] = i
                self.img2txt[i].append(txt_id)
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image[index])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        text = self.text[index]

        return image, text, index


class pair_dataset2(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []

        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for i, ann in enumerate(self.ann):
            self.img2txt[i] = []
            for caption in ann['caption']:
                self.image.append(ann['image'])
                self.text.append(pre_caption(caption, self.max_words))
                self.txt2img[txt_id] = i
                self.img2txt[i].append(txt_id)
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image[index])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        text = self.text[index]

        return image, text, index, self.image[index]
