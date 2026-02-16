import json 
import os
import re

import torch 
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset
from models import clip

def pre_caption(caption,max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

# class paired_dataset(Dataset):
#     def __init__(self, ann_file, transform, image_root, max_words=30):
#         self.ann = json.load(open(ann_file, 'r'))
#         self.transform = transform
#         self.image_root = image_root
#         self.max_words = max_words

#         self.text = []
#         self.image = []

#         self.txt2img = {}
#         self.img2txt = {}

#         txt_id = 0
#         for i, ann in enumerate(self.ann):
#             self.img2txt[i] = []
#             self.image.append(ann['image'])
#             for j, caption in enumerate(ann['caption']):
#                 self.text.append(pre_caption(caption, self.max_words))
#                 self.txt2img[txt_id] = i
#                 self.img2txt[i].append(txt_id)
#                 txt_id += 1

#     def __len__(self):
#         return len(self.image)

#     def __getitem__(self, index):
#         image_path = os.path.join(self.image_root, self.image[index])
#         image = Image.open(image_path).convert('RGB')
#         image = self.transform(image)
#         text_ids =  self.img2txt[index]
#         texts = [self.text[i] for i in self.img2txt[index]]
#         return image, texts, index, text_ids

#     def collate_fn(self, batch):
#         imgs, txt_groups, img_ids, text_ids_groups = list(zip(*batch))        
#         imgs = torch.stack(imgs, 0)
#         return imgs, txt_groups, list(img_ids), text_ids_groups

import os
import json
import torch
from PIL import Image
from torchvision import transforms


import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm  # 导入tqdm



def interpret(image, text, model, device, start_layer=-1):
    batch_size = text.shape[0]

    images = image.repeat(batch_size, 1, 1, 1)
    logits_per_image, logits_per_text = model(images, text)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.to(device) * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    if start_layer == -1: 
        # calculate index of last layer 
        start_layer = len(image_attn_blocks) - 1
    
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
            continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True, allow_unused=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]

    return image_relevance

class paired_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, processed_image_root, processed_attn_root, device, max_words=30, model_name=None):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.processed_image_root = processed_image_root
        self.processed_attn_root = processed_attn_root
        # self.model=model
        self.device = device
        self.text = []
        self.image = []
        self.max_words=max_words
        self.model_name=model_name
        self.image_ids=[]

        self.txt2img = {}
        self.img2txt = {}
        model, _ = clip.load("ViT-B/32", device=device, jit=False)
        txt_id = 0
        os.makedirs(self.processed_image_root, exist_ok=True)  # 保存处理过的图像的目录
        os.makedirs(self.processed_attn_root, exist_ok=True)  # 保存相关性矩阵的目录

        # 假设 self.ann 是一个列表或可迭代对象，你可以用 tqdm 包装它
        for i, ann in tqdm(enumerate(self.ann), total=len(self.ann), desc="Processing Images"):
            self.img2txt[i] = []
            self.image.append(ann['image'])
            self.image_ids.append(ann['image'])
            image_path = os.path.join(self.image_root, ann['image'])
            
            # 目标文件夹中的图像路径
            transformed_image_path = os.path.join(self.processed_image_root, ann['image'])

            # 如果目标文件夹中没有处理过的图像，就进行图像预处理
            if not os.path.exists(transformed_image_path):
                
                image = Image.open(image_path).convert('RGB')
                
                # 对图像进行预处理
                image_transformed = self.transform(image)

                # 保存处理过的图像到新的文件夹
                transformed_image = transforms.ToPILImage()(image_transformed)  # 将tensor转回PIL图像
                transformed_image.save(transformed_image_path)
                
                if self.model_name in ['ALBEF', 'TCL']:
                    n_px = model.visual.input_resolution
                    clip_transform=transforms.Compose([
                        transforms.Resize(n_px, interpolation=Image.BICUBIC),
                        transforms.CenterCrop(n_px),
                        transforms.ToTensor(),       
                    ])
                    image_transformed=clip_transform(image)
                image_tensor = image_transformed.to(self.device)
                for _, caption in enumerate(ann['caption']):
                    processed_caption = pre_caption(caption, self.max_words)
                    self.text.append(processed_caption)
                    self.txt2img[txt_id] = i
                    self.img2txt[i].append(txt_id)
                    text = clip.tokenize([processed_caption]).to(device)
                    relevance_matrix = interpret(image_tensor, text, model, self.device)
                    if self.model_name in ['ALBEF', 'TCL']:
                        dim = int(relevance_matrix.numel() ** 0.5)
                        relevance_matrix = relevance_matrix.reshape(1, 1, dim, dim)
                        relevance_matrix = torch.nn.functional.interpolate(relevance_matrix, size=384, mode='bilinear')
                        relevance_matrix = relevance_matrix.reshape(384, 384).cuda().data
                    else:
                        dim = int(relevance_matrix.numel() ** 0.5)
                        relevance_matrix = relevance_matrix.reshape(1, 1, dim, dim)
                        relevance_matrix = torch.nn.functional.interpolate(relevance_matrix, size=224, mode='bilinear')
                        relevance_matrix = relevance_matrix.reshape(224, 224).cuda().data
                    # 保存每个文本的相关性矩阵到指定目录，命名为 'image_name_text_id.npy'
                    attn_matrix_path = os.path.join(self.processed_attn_root, f"{ann['image']}_text{txt_id}.npy")
                    np.save(attn_matrix_path, relevance_matrix.cpu().numpy())
                    txt_id+=1
            else:
                for _, caption in enumerate(ann['caption']):
                    processed_caption = pre_caption(caption, self.max_words)
                    self.text.append(processed_caption)
                    self.txt2img[txt_id] = i
                    self.img2txt[i].append(txt_id)
                    txt_id+=1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        # 直接从保存的图像文件夹加载处理后的图像
        image_path = os.path.join(self.processed_image_root, self.image[index])
        image = Image.open(image_path).convert('RGB')
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        
        # 加载与图像对应的相关性矩阵
        text_ids = self.img2txt[index]
        texts = [self.text[i] for i in self.img2txt[index]]

        attn_matrices = []
        for text_id in text_ids:
            # 加载与文本对应的相关性矩阵
            attn_matrix_path = os.path.join(self.processed_attn_root, f"{self.image[index]}_text{text_id}.npy")
            image_relevance = np.load(attn_matrix_path)
            image_relevance=torch.from_numpy(image_relevance).float()
            image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
            attn_matrices.append(image_relevance)
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
            for j, caption in enumerate(ann['caption']):
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
        text_ids =  self.img2txt[index]
        texts = [self.text[i] for i in self.img2txt[index]]
        return image, texts, index, text_ids, self.image[index]

    def collate_fn(self, batch):
        imgs, txt_groups, img_ids, text_ids_groups,image_paths = list(zip(*batch))        
        imgs = torch.stack(imgs, 0)
        return imgs, txt_groups, list(img_ids), text_ids_groups,image_paths

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
            for j, caption in enumerate(ann['caption']):
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
            for j, caption in enumerate(ann['caption']):
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

        return image, text, index,self.image[index]