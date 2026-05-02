import argparse
import os
import csv
import pickle
from ruamel.yaml import YAML

yaml=YAML(typ='safe')
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from transformers import BertForMaskedLM
from torchvision import transforms
from PIL import Image

from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from models import clip

import utils
import copy
import time

from SA_AET import Attacker, ImageAttacker, TextAttacker, precompute_cluster_centers
from dataset import paired_dataset
from sklearn.cluster import KMeans


def get_cluster_info_from_embeds(image_embeds, cluster_centers, top_k=4, eps=0.0000001):
    """
    从图像嵌入和聚类中心计算聚类信息（类别和最近的k个类别索引）
    
    Args:
        image_embeds: 图像嵌入 [batch_size, embed_dim]
        cluster_centers: 聚类中心 [num_clusters, embed_dim]
        top_k: 选择最近的 k 个聚类中心
        eps: 数值稳定性参数
    
    Returns:
        cluster_info: 包含类别和最近k个类别索引的字典
    """
    device = image_embeds.device
    cluster_centers = cluster_centers.to(device)
    
    # Ensure cluster_centers has the same dtype as image_embeds
    cluster_centers = cluster_centers.to(image_embeds.dtype)
    
    # 归一化
    image_embeds_norm = torch.sqrt(torch.sum(image_embeds ** 2, dim=1, keepdim=True))
    image_embeds = image_embeds / (image_embeds_norm + eps)
    image_embeds[image_embeds != image_embeds] = 0
    
    cluster_centers_norm = torch.sqrt(torch.sum(cluster_centers ** 2, dim=1, keepdim=True))
    cluster_centers = cluster_centers / (cluster_centers_norm + eps)
    
    # 计算每个样本到所有聚类中心的余弦相似度
    sample_cluster_sim = torch.mm(image_embeds, cluster_centers.t())  # [batch_size, num_clusters]
    
    # 选择最近的 top_k 个聚类中心
    top_k_values, top_k_indices = torch.topk(sample_cluster_sim, top_k, dim=1)  # [batch_size, top_k]
    
    cluster_info = {
        'cluster_id': top_k_indices[:, 0].cpu().tolist(),  # 最近的聚类中心索引（类别）
        'top_k_indices': top_k_indices.cpu().tolist(),  # 最近的k个聚类中心索引
        'top_k_similarities': top_k_values.cpu().tolist()  # 最近的k个聚类中心的相似度
    }
    
    return cluster_info

def get_projection_matrix(all_txt_supervisions):
    U, S, V = torch.svd(all_txt_supervisions.T.to(torch.float32))
    U,S,V=U.half(),S.half(),V.half()
    projection_matrix = U[:, 1:len(U)] @ U[:, 1:len(U)].t()
    return projection_matrix.half()

def get_projection_matrix_2(all_txt_supervisions):
    U, S, V = torch.svd(all_txt_supervisions.T.to(torch.float32))
    U,S,V=U.half(),S.half(),V.half()
    projection_matrix = U[:, 1:len(U)] @ U[:, 1:len(U)].t() # 投影到语义空间
    return projection_matrix.half()


def get_cluster_cache_path(cache_dir, dataset_name, model_name, num_clusters, seed):
    cache_name = f"clusters_{dataset_name}_{model_name}_k{num_clusters}_seed{seed}.pt"
    return os.path.join(cache_dir, cache_name)


def load_or_compute_cluster_centers(all_image_embeds, num_clusters, device, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        cache_data = torch.load(cache_path, map_location='cpu')
        cached_centers = cache_data.get('cluster_centers')
        cached_labels = cache_data.get('cluster_labels')
        meta = cache_data.get('meta', {})
        if (
            cached_centers is not None
            and meta.get('num_clusters') == num_clusters
            and meta.get('num_samples') == int(all_image_embeds.shape[0])
            and meta.get('feature_dim') == int(all_image_embeds.shape[1])
        ):
            print(f"Loaded cluster centers from cache: {cache_path}")
            return cached_centers.to(device), cached_labels

    cluster_centers, cluster_labels = precompute_cluster_centers(all_image_embeds, num_clusters=num_clusters, device=device)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({
            'cluster_centers': cluster_centers.cpu(),
            'cluster_labels': cluster_labels,
            'meta': {
                'num_clusters': num_clusters,
                'num_samples': int(all_image_embeds.shape[0]),
                'feature_dim': int(all_image_embeds.shape[1]),
            }
        }, cache_path)
        print(f"Saved cluster centers to cache: {cache_path}")
    return cluster_centers, cluster_labels

def save_adversarial_images(adv_images, image_ids, save_dir="adv_images"):
    """
    保存对抗性图像到指定目录，最多保存 max_images 张
    """
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

    num_saved = len(os.listdir(save_dir))  # 获取已有图片数量，避免覆盖
    for i, img in enumerate(adv_images):        
        img_pil = toImage(img)  # 转换为 PIL.Image
        # 只使用图像的文件名，不包含路径
        img_filename = os.path.basename(image_ids[i])
        save_path = os.path.join(save_dir, img_filename)  # 格式化命名
        img_pil.save(save_path)  # 保存图像
        num_saved += 1

    print(f"Saved {num_saved} adversarial images to {save_dir}")

def toImage(norm_img):
    pil_array = (norm_img * 255).to(torch.uint8).cpu().numpy()
    pil_img=Image.fromarray(np.transpose(pil_array, (1, 2, 0)))
    return pil_img

def retrieval_eval(model, ref_model, t_models, t_ref_models, t_test_transforms, data_loader, tokenizer, t_tokenizers, device, args,config, result_base_dir, save_cluster_info=False, use_topological_loss=True, use_attention_weighted=True, num_clusters=30):
    model.to(device)
    ref_model.to(device)
    model.float()
    model.eval()
    ref_model.eval()

    for t_model, t_ref_model in zip(t_models, t_ref_models):
        t_model.to(device)
        t_ref_model.to(device)
        t_model.float()
        t_model.eval()
        t_ref_model.eval()    

    print('Computing features for evaluation adv...')

    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    img_attacker = ImageAttacker(
        images_normalize,
        eps=8/255,
        steps=10,
        step_size=2/255,
        use_attention_weighted=use_attention_weighted,
        use_topological_loss=use_topological_loss
    )

    max_length = 30 if args.source_model in ['ALBEF', 'TCL'] else 77 
    txt_attacker = TextAttacker(ref_model, tokenizer, cls=False, max_length=max_length, number_perturbation=1,
                                topk=10, threshold_pred_score=0.3)
    attacker = Attacker(model, img_attacker, txt_attacker)

    print('Prepare memory')
    num_text = len(data_loader.dataset.text)
    num_image = len(data_loader.dataset.ann)

    s_feat_dict = {}
    if args.source_model in ['ALBEF', 'TCL']:
        s_feat_dict['s_image_feats'] = torch.zeros(num_image, config['embed_dim'])
        s_feat_dict['s_image_embeds'] = torch.zeros(num_image, 577, 768)
        s_feat_dict['s_text_feats'] = torch.zeros(num_text, config['embed_dim'])
        s_feat_dict['s_text_embeds'] = torch.zeros(num_text, 30, 768)
        s_feat_dict['s_text_atts'] = torch.zeros(num_text, 30).long()
    else:
        s_feat_dict['s_image_feats'] = torch.zeros(num_image, model.visual.output_dim)
        s_feat_dict['s_text_feats'] = torch.zeros(num_text, model.visual.output_dim)

    t_feat_dicts = []
    t_model_names = copy.deepcopy(args.model_list)
    t_model_names.remove(args.source_model)
    for t_model_name,t_model in zip(t_model_names,t_models):
        t_feat_dict = {}
        if t_model_name in ['ALBEF', 'TCL']:
            t_feat_dict['t_image_feats'] = torch.zeros(num_image, config['embed_dim'])
            t_feat_dict['t_image_embeds'] = torch.zeros(num_image, 577, 768)
            t_feat_dict['t_text_feats'] = torch.zeros(num_text, config['embed_dim'])
            t_feat_dict['t_text_embeds'] = torch.zeros(num_text, 30, 768)
            t_feat_dict['t_text_atts'] = torch.zeros(num_text, 30).long()
        else:
            t_feat_dict['t_image_feats'] = torch.zeros(num_image, t_model.visual.output_dim)
            t_feat_dict['t_text_feats'] = torch.zeros(num_text, t_model.visual.output_dim)
        t_feat_dicts.append(t_feat_dict)

    if args.scales is not None:
        scales = [float(itm) for itm in args.scales.split(',')]
        print(scales)
    else:
        scales = None

    print('Forward')

    need_cluster_data = use_topological_loss or save_cluster_info
    if need_cluster_data:
        print("Computing image embeddings for clustering...")
    all_image_embeds = []
    all_texts_all = []
    for batch_idx, (images, texts_group, images_name, images_ids, text_ids_groups, attn_matrices) in enumerate(data_loader):
        for index_text in range(len(texts_group)):
            all_texts_all += texts_group[index_text]
        if need_cluster_data:
            images = images.to(device)
            with torch.no_grad():
                if args.source_model in ['ALBEF', 'TCL']:
                    images_output = model.inference_image(images_normalize(images))
                else:
                    images_output = model.inference_image(images_normalize(images))
                image_embeds = images_output['image_feat']
                all_image_embeds.append(image_embeds.cpu())
            print(f"Processed batch {batch_idx+1}/{len(data_loader)} for clustering")

    if need_cluster_data:
        all_image_embeds = torch.cat(all_image_embeds, dim=0)
        print(f"Total image embeddings for clustering: {all_image_embeds.shape}")
    
    adv_texts_dict = {}
    
    # 用于存储聚类信息
    cluster_info_dict = {}

    num_samples = int(0.4 * len(all_texts_all))

       #使用这些索引来选取张量中的数据
    all_texts = random.sample(all_texts_all, num_samples)
    # all_texts=all_texts_all

    batch_size = 3000  # 每个批次的大小
    n = len(all_texts)  # 总文本数量
    # 初始化输出变量
    all_texts_output = {
        'text_embed': None,
        'text_feat': None
    }
    with torch.no_grad():
        # 按批次处理并合并
        for i in range(0, n, batch_size):
            batch_texts = all_texts[i:i+batch_size]  # 获取当前批次的文本
            batch_texts_input = attacker.txt_attacker.tokenizer(batch_texts, padding='max_length', truncation=True,
                                                            max_length=max_length, return_tensors="pt").to(device)
            batch_texts_output = attacker.model.inference_text(batch_texts_input)
            
            # 如果all_texts_output为空，则初始化它
            if all_texts_output['text_embed'] is None:
                all_texts_output['text_embed'] = batch_texts_output['text_embed']
                all_texts_output['text_feat'] = batch_texts_output['text_feat']
            else:
                # 否则直接合并当前批次的结果
                all_texts_output['text_embed'] = torch.cat([all_texts_output['text_embed'], batch_texts_output['text_embed']], dim=0)
                all_texts_output['text_feat'] = torch.cat([all_texts_output['text_feat'], batch_texts_output['text_feat']], dim=0)
        all_txt_supervisions = all_texts_output['text_feat']

        assert all_txt_supervisions.shape[0] == len(all_texts)

    projection_matrix=get_projection_matrix(all_txt_supervisions)

    # 使用图像嵌入计算聚类中心（可缓存）
    if need_cluster_data:
        print("Computing cluster centers from image embeddings...")
        cluster_cache_path = None
        if not args.disable_cluster_cache:
            cluster_cache_path = get_cluster_cache_path(
                args.cache_dir, args.dataset_name, args.source_model, num_clusters, args.seed
            )
        cluster_centers, cluster_labels = load_or_compute_cluster_centers(
            all_image_embeds, num_clusters=num_clusters, device=device, cache_path=cluster_cache_path
        )
        print(f"Source model cluster centers computed, shape: {cluster_centers.shape}")
        # 更新ImageAttacker的聚类中心
        img_attacker.cluster_centers = cluster_centers
    
    # 为每个 target model 计算聚类中心
    t_cluster_centers_dict = {}
    if save_cluster_info:
        print("Computing cluster centers for target models...")
        for t_model_name, t_model, t_test_transform in zip(t_model_names, t_models, t_test_transforms):
            print(f"Computing cluster centers for {t_model_name}...")
            all_t_image_embeds = []
            
            # 重置数据加载器
            data_loader_iter = iter(data_loader)
            for batch_idx, (images, texts_group, images_name, images_ids, text_ids_groups, attn_matrices) in enumerate(data_loader_iter):
                images = images.to(device)
                # 转换图像格式以适应 target model
                t_img_list = []
                for img in images:
                    t_img_list.append(t_test_transform(img))
                t_images = torch.stack(t_img_list, 0).to(device)
                
                with torch.no_grad():
                    t_images_norm = images_normalize(t_images)
                    if t_model_name in ['ALBEF', 'TCL']:
                        t_images_output = t_model.inference_image(t_images_norm)
                    else:
                        t_images_output = t_model.inference_image(t_images_norm)
                    t_image_embeds = t_images_output['image_feat']
                    all_t_image_embeds.append(t_image_embeds.cpu())
                print(f"Processed batch {batch_idx+1}/{len(data_loader)} for {t_model_name} clustering")
            
            all_t_image_embeds = torch.cat(all_t_image_embeds, dim=0)
            print(f"Total image embeddings for {t_model_name} clustering: {all_t_image_embeds.shape}")
            
            # 计算该 target model 的聚类中心（可缓存）
            t_cluster_cache_path = None
            if not args.disable_cluster_cache:
                t_cluster_cache_path = get_cluster_cache_path(
                    args.cache_dir, args.dataset_name, t_model_name, num_clusters, args.seed
                )
            t_cluster_centers, t_cluster_labels = load_or_compute_cluster_centers(
                all_t_image_embeds, num_clusters=num_clusters, device=device, cache_path=t_cluster_cache_path
            )
            t_cluster_centers_dict[t_model_name] = t_cluster_centers
            print(f"{t_model_name} cluster centers computed, shape: {t_cluster_centers.shape}")
    
    # 初始化CSV文件并写入表头
    csv_path = os.path.join(result_base_dir, "adv_texts.csv")
    max_texts_per_image = 0
    # 先遍历一次数据加载器来确定最大文本数量
    for _, texts_group, _, _, _, _ in data_loader:
        for texts in texts_group:
            max_texts_per_image = max(max_texts_per_image, len(texts))
    
    # 写入表头
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["image_name"] + [f"adv_text_{i+1}" for i in range(max_texts_per_image)]
        writer.writerow(header)
    
    # 重置数据加载器
    data_loader = iter(data_loader)
    
    for batch_idx, (images, texts_group, images_name, images_ids, text_ids_groups, attn_matrices) in enumerate(data_loader):
        print(f'--------------------> batch:{batch_idx}/{len(data_loader)}')
        texts_ids = []
        txt2img = []
        texts = []
        for i in range(len(texts_group)):
            texts += texts_group[i]
            texts_ids += text_ids_groups[i]
            txt2img += [i]*len(text_ids_groups[i])
        images = images.to(device)

        attack_result = attacker.attack(images, texts, txt2img,projection_matrix, device=device,
                                                max_length=max_length, scales=scales, attn_matrices=attn_matrices,
                                                return_cluster_info=save_cluster_info)
        
        if save_cluster_info:
            adv_images, adv_texts, execuate_time, cluster_info = attack_result
            # 收集聚类信息
            for i, img_name in enumerate(images_name):
                img_filename = os.path.basename(img_name)
                cluster_info_dict[img_filename] = {
                    'clean': {
                        'cluster_id': cluster_info['clean']['cluster_id'][i],
                        'top_k_indices': cluster_info['clean']['top_k_indices'][i],
                        'top_k_similarities': cluster_info['clean']['top_k_similarities'][i]
                    },
                    'adversarial': {
                        'cluster_id': cluster_info['adversarial']['cluster_id'][i],
                        'top_k_indices': cluster_info['adversarial']['top_k_indices'][i],
                        'top_k_similarities': cluster_info['adversarial']['top_k_similarities'][i]
                    },
                    'target_models': {}
                }
        else:
            adv_images, adv_texts, execuate_time = attack_result
        
        adv_images_dir = os.path.join(result_base_dir, "adv_images")
        
        save_adversarial_images(adv_images, images_name, save_dir=adv_images_dir)
        
        texts_per_image = len(texts) // len(images_name)
        for i, img_name in enumerate(images_name):
            start_idx = i * texts_per_image
            end_idx = start_idx + texts_per_image
            img_adv_texts = adv_texts[start_idx:end_idx]
            if img_name not in adv_texts_dict:
                adv_texts_dict[img_name] = []
            adv_texts_dict[img_name].extend(img_adv_texts)
            
            # 立即写入CSV文件
            img_filename = os.path.basename(img_name)
            row = [img_filename] + adv_texts_dict[img_name]
            while len(row) < max_texts_per_image + 1:
                row.append("")
            with open(csv_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        with torch.no_grad():
            s_adv_images_norm = images_normalize(adv_images)
            if args.source_model in ['ALBEF', 'TCL']:
                adv_texts_input = tokenizer(adv_texts, padding='max_length', truncation=True, max_length=30, 
                                            return_tensors="pt").to(device)            
                s_output_img = model.inference_image(s_adv_images_norm)
                s_output_txt = model.inference_text(adv_texts_input)

                s_feat_dict['s_image_feats'][images_ids] = s_output_img['image_feat'].cpu().detach()
                s_feat_dict['s_image_embeds'][images_ids] = s_output_img['image_embed'].cpu().detach()
                s_feat_dict['s_text_feats'][texts_ids] = s_output_txt['text_feat'].cpu().detach()
                s_feat_dict['s_text_embeds'][texts_ids] = s_output_txt['text_embed'].cpu().detach()
                s_feat_dict['s_text_atts'][texts_ids] = adv_texts_input.attention_mask.cpu().detach()
            else:
                output = model.inference(s_adv_images_norm, adv_texts)
                s_feat_dict['s_image_feats'][images_ids] = output['image_feat'].cpu().float().detach()
                s_feat_dict['s_text_feats'][texts_ids] = output['text_feat'].cpu().float().detach()

            for t_model_name,t_model,t_feat_dict,t_test_transform in zip(t_model_names,t_models,t_feat_dicts,t_test_transforms):
                t_adv_img_list = []
                for itm in adv_images:
                    t_adv_img_list.append(t_test_transform(itm))
                t_adv_imgs = torch.stack(t_adv_img_list, 0).to(device)            
                t_adv_images_norm = images_normalize(t_adv_imgs)
                if t_model_name in ['ALBEF', 'TCL']:
                    adv_texts_input = tokenizer(adv_texts, padding='max_length', truncation=True, max_length=30, 
                                        return_tensors="pt").to(device)            
                    t_output_img = t_model.inference_image(t_adv_images_norm)
                    t_output_txt = t_model.inference_text(adv_texts_input)
                    t_feat_dict['t_image_feats'][images_ids] = t_output_img['image_feat'].cpu().detach()
                    t_feat_dict['t_image_embeds'][images_ids] = t_output_img['image_embed'].cpu().detach()
                    t_feat_dict['t_text_feats'][texts_ids] = t_output_txt['text_feat'].cpu().detach()
                    t_feat_dict['t_text_embeds'][texts_ids] = t_output_txt['text_embed'].cpu().detach()
                    t_feat_dict['t_text_atts'][texts_ids] = adv_texts_input.attention_mask.cpu().detach()
                else:
                    output = t_model.inference(t_adv_images_norm, adv_texts)
                    t_feat_dict['t_image_feats'][images_ids] = output['image_feat'].cpu().float().detach()
                    t_feat_dict['t_text_feats'][texts_ids] = output['text_feat'].cpu().float().detach()
                
                # 获取攻击前后在 target model 中的聚类信息
                if save_cluster_info and t_model_name in t_cluster_centers_dict:
                    t_cluster_centers = t_cluster_centers_dict[t_model_name]
                    
                    # 获取原始图像在 target model 中的聚类信息
                    t_clean_img_list = []
                    for itm in images:
                        t_clean_img_list.append(t_test_transform(itm))
                    t_clean_imgs = torch.stack(t_clean_img_list, 0).to(device)
                    t_clean_images_norm = images_normalize(t_clean_imgs)
                    
                    with torch.no_grad():
                        if t_model_name in ['ALBEF', 'TCL']:
                            t_clean_output = t_model.inference_image(t_clean_images_norm)
                        else:
                            t_clean_output = t_model.inference_image(t_clean_images_norm)
                        t_clean_embeds = t_clean_output['image_feat']
                        
                        if t_model_name in ['ALBEF', 'TCL']:
                            t_adv_output = t_model.inference_image(t_adv_images_norm)
                        else:
                            t_adv_output = t_model.inference_image(t_adv_images_norm)
                        t_adv_embeds = t_adv_output['image_feat']
                    
                    # 计算原始图像的聚类信息
                    t_clean_cluster_info = get_cluster_info_from_embeds(t_clean_embeds, t_cluster_centers)
                    # 计算对抗图像的聚类信息
                    t_adv_cluster_info = get_cluster_info_from_embeds(t_adv_embeds, t_cluster_centers)
                    
                    # 保存到 cluster_info_dict
                    for i, img_name in enumerate(images_name):
                        img_filename = os.path.basename(img_name)
                        if img_filename in cluster_info_dict:
                            cluster_info_dict[img_filename]['target_models'][t_model_name] = {
                                'clean': {
                                    'cluster_id': t_clean_cluster_info['cluster_id'][i],
                                    'top_k_indices': t_clean_cluster_info['top_k_indices'][i],
                                    'top_k_similarities': t_clean_cluster_info['top_k_similarities'][i]
                                },
                                'adversarial': {
                                    'cluster_id': t_adv_cluster_info['cluster_id'][i],
                                    'top_k_indices': t_adv_cluster_info['top_k_indices'][i],
                                    'top_k_similarities': t_adv_cluster_info['top_k_similarities'][i]
                                }
                            }

    s_score_matrix_i2t = None
    s_score_matrix_t2i = None
    if args.source_model in ['ALBEF', 'TCL']:
        s_score_matrix_i2t, s_score_matrix_t2i = retrieval_score(model, s_feat_dict['s_image_feats'], s_feat_dict['s_image_embeds'], s_feat_dict['s_text_feats'],
                                                        s_feat_dict['s_text_embeds'], s_feat_dict['s_text_atts'], num_image, num_text, device=device)
        s_score_matrix_i2t = s_score_matrix_i2t.cpu().numpy()
        s_score_matrix_t2i = s_score_matrix_t2i.cpu().numpy()
    else:
        s_sims_matrix = s_feat_dict['s_image_feats'] @ s_feat_dict['s_text_feats'].t()
        s_score_matrix_i2t = s_sims_matrix.cpu().numpy()
        s_score_matrix_t2i = s_sims_matrix.t().cpu().numpy()
    
    t_score_matrix_i2ts= [] 
    t_score_matrix_t2is= []
    for t_model_name,t_feat_dict,t_model in zip(t_model_names,t_feat_dicts,t_models):
        if t_model_name in ['ALBEF', 'TCL']:
            t_score_matrix_i2t, t_score_matrix_t2i = retrieval_score(t_model, t_feat_dict['t_image_feats'], t_feat_dict['t_image_embeds'], t_feat_dict['t_text_feats'],
                                                        t_feat_dict['t_text_embeds'], t_feat_dict['t_text_atts'], num_image, num_text, device=device)
            t_score_matrix_i2ts.append(t_score_matrix_i2t.cpu().numpy())
            t_score_matrix_t2is.append(t_score_matrix_t2i.cpu().numpy())
        else:
            t_sims_matrix = t_feat_dict['t_image_feats'] @ t_feat_dict['t_text_feats'].t()
            t_score_matrix_i2t = t_sims_matrix.cpu().numpy()
            t_score_matrix_t2i = t_sims_matrix.t().cpu().numpy()
            t_score_matrix_i2ts.append(t_score_matrix_i2t)
            t_score_matrix_t2is.append(t_score_matrix_t2i)

    # 保存聚类信息
    if save_cluster_info and cluster_info_dict:
        cluster_info_path = os.path.join(result_base_dir, "cluster_info.csv")
        with open(cluster_info_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            # 写入表头
            header = [
                "image_name",
                "clean_cluster_id",
                "clean_top_k_indices",
                "clean_top_k_similarities",
                "adv_cluster_id",
                "adv_top_k_indices",
                "adv_top_k_similarities"
            ]
            
            # 为每个 target model 添加列
            for t_model_name in t_model_names:
                header.extend([
                    f"{t_model_name}_clean_cluster_id",
                    f"{t_model_name}_clean_top_k_indices",
                    f"{t_model_name}_clean_top_k_similarities",
                    f"{t_model_name}_adv_cluster_id",
                    f"{t_model_name}_adv_top_k_indices",
                    f"{t_model_name}_adv_top_k_similarities"
                ])
            
            writer.writerow(header)
            
            # 写入数据
            for img_name in sorted(cluster_info_dict.keys()):
                info = cluster_info_dict[img_name]
                row = [
                    img_name,
                    info['clean']['cluster_id'],
                    str(info['clean']['top_k_indices']),
                    str(info['clean']['top_k_similarities']),
                    info['adversarial']['cluster_id'],
                    str(info['adversarial']['top_k_indices']),
                    str(info['adversarial']['top_k_similarities'])
                ]
                
                # 添加每个 target model 的聚类信息
                for t_model_name in t_model_names:
                    if t_model_name in info.get('target_models', {}):
                        t_info = info['target_models'][t_model_name]
                        row.extend([
                            t_info['clean']['cluster_id'],
                            str(t_info['clean']['top_k_indices']),
                            str(t_info['clean']['top_k_similarities']),
                            t_info['adversarial']['cluster_id'],
                            str(t_info['adversarial']['top_k_indices']),
                            str(t_info['adversarial']['top_k_similarities'])
                        ])
                    else:
                        row.extend(["", "", "", "", "", ""])
                
                writer.writerow(row)
        
        print(f"Cluster info saved to {cluster_info_path}")

    return s_score_matrix_i2t, s_score_matrix_t2i, \
        t_score_matrix_i2ts, t_score_matrix_t2is

@torch.no_grad()
def retrieval_score(model, image_feats, image_embeds, text_feats, text_embeds, text_atts, num_image, num_text, device=None):
    if device is None:
        device = image_embeds.device

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation Direction Similarity With Bert Attack:'

    sims_matrix = image_feats @ text_feats.t()
    score_matrix_i2t = torch.full((num_image, num_text), -100.0).to(device)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_embeds[i].repeat(config['k_test'], 1, 1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds=text_embeds[topk_idx].to(device),
                                    attention_mask=text_atts[topk_idx].to(device),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    mode='fusion'
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[i, topk_idx] = score

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((num_text, num_image), -100.0).to(device)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_embeds[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds=text_embeds[i].repeat(config['k_test'], 1, 1).to(device),
                                    attention_mask=text_atts[i].repeat(config['k_test'], 1).to(device),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    mode='fusion'
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[i, topk_idx] = score

    return score_matrix_i2t, score_matrix_t2i

@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, img2txt, txt2img, model_name, args):
    # Images->Text
    with open(f'./temp/{model_name}_scores_i2t.pkl', 'wb') as file:
        pickle.dump(scores_i2t, file)
    with open(f'./temp/{model_name}_scores_t2i.pkl', 'wb') as file:
        pickle.dump(scores_t2i, file)
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)


    after_attack_tr1 = np.where(ranks < 1)[0]
    after_attack_tr5 = np.where(ranks < 5)[0]
    after_attack_tr10 = np.where(ranks < 10)[0]
    
    original_rank_index_path = args.original_rank_index_path
    origin_tr1 = np.load(os.path.join(original_rank_index_path, f'{model_name}_tr1_rank_index.npy'))
    origin_tr5 = np.load(os.path.join(original_rank_index_path, f'{model_name}_tr5_rank_index.npy'))
    origin_tr10 = np.load(os.path.join(original_rank_index_path, f'{model_name}_tr10_rank_index.npy'))

    asr_tr1 = round(100.0 * len(np.setdiff1d(origin_tr1, after_attack_tr1)) / len(origin_tr1), 2) 
    asr_tr5 = round(100.0 * len(np.setdiff1d(origin_tr5, after_attack_tr5)) / len(origin_tr5), 2)
    asr_tr10 = round(100.0 * len(np.setdiff1d(origin_tr10, after_attack_tr10)) / len(origin_tr10), 2)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]


    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    after_attack_ir1 = np.where(ranks < 1)[0]
    after_attack_ir5 = np.where(ranks < 5)[0]
    after_attack_ir10 = np.where(ranks < 10)[0]

    origin_ir1 = np.load(os.path.join(original_rank_index_path, f'{model_name}_ir1_rank_index.npy'))
    origin_ir5 = np.load(os.path.join(original_rank_index_path, f'{model_name}_ir5_rank_index.npy'))
    origin_ir10 = np.load(os.path.join(original_rank_index_path, f'{model_name}_ir10_rank_index.npy'))

    asr_ir1 = round(100.0 * len(np.setdiff1d(origin_ir1, after_attack_ir1)) / len(origin_ir1), 2) 
    asr_ir5 = round(100.0 * len(np.setdiff1d(origin_ir5, after_attack_ir5)) / len(origin_ir5), 2)
    asr_ir10 = round(100.0 * len(np.setdiff1d(origin_ir10, after_attack_ir10)) / len(origin_ir10), 2)


    eval_result = {'txt_r1_ASR (txt_r1)': f'{asr_tr1}({tr1})',
                   'txt_r5_ASR (txt_r5)': f'{asr_tr5}({tr5})',
                   'txt_r10_ASR (txt_r10)': f'{asr_tr10}({tr10})',
                   'img_r1_ASR (img_r1)': f'{asr_ir1}({ir1})',
                   'img_r5_ASR (img_r5)': f'{asr_ir5}({ir5})',
                   'img_r10_ASR (img_r10)': f'{asr_ir10}({ir10})'}
    return eval_result

def load_model(args,model_name,text_encoder, device):
    tokenizer = BertTokenizer.from_pretrained("/home/myang/SA-AET/BLIP/bert")
    ref_model = BertForMaskedLM.from_pretrained(text_encoder)    
    if model_name in ['ALBEF', 'TCL']:
        model = ALBEF(config=config, text_encoder=text_encoder, tokenizer=tokenizer)
        model_ckpt = args.albef_ckpt if model_name == 'ALBEF' else args.tcl_ckpt
        checkpoint = torch.load(model_ckpt, map_location='cpu')
    ### load checkpoint
    else:
        model_name = 'ViT-B/16' if model_name == 'CLIP_ViT' else 'RN101'
        model, preprocess = clip.load(model_name, device=device, jit=False)
        model.set_tokenizer(tokenizer)
        return model, ref_model, tokenizer
    
    try:
        state_dict = checkpoint['model']
    except:
        state_dict = checkpoint

    if model_name == 'TCL':
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 

    for key in list(state_dict.keys()):
        if 'bert' in key:
            encoder_key = key.replace('bert.', '')
            state_dict[encoder_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    
    return model, ref_model, tokenizer

def _extract_metric_value(metric_str):
    # metric_str format example: "99.90(12.34)" -> return 99.90
    try:
        if '(' in metric_str and ')' in metric_str:
            return float(metric_str.split('(')[0])
        return float(metric_str)
    except Exception:
        return None


def _format_experiment_row(args, metrics_by_model):
    model_order = ['ALBEF', 'TCL', 'CLIP_ViT']
    run_name = os.path.basename(args.result_base_dir)
    row = [run_name, args.source_model, args.experiment]
    for model_name in model_order:
        m = metrics_by_model.get(model_name, {})
        row.extend([
            str(m.get('txt@1', '')),
            str(m.get('img@1', '')),
        ])
    return "| " + " | ".join(row) + " |\n"


def _extract_source_rows_from_baseline(baseline_path, source_model):
    if not os.path.exists(baseline_path):
        return []
    rows = []
    with open(baseline_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped.startswith('|'):
                continue
            if stripped.startswith('|---'):
                continue
            if stripped.startswith('| Source |'):
                continue
            cells = [c.strip() for c in stripped.strip('|').split('|')]
            if len(cells) >= 2 and cells[0] == source_model:
                rows.append(line)
    return rows


def _write_result_md(result_md_path, args, metrics_by_model):
    header = "| Source | Method | ALBEF TR R@1 | ALBEF IR R@1 | TCL TR R@1 | TCL IR R@1 | CLIP_ViT TR R@1 | CLIP_ViT IR R@1 |\n"
    sep = "|---|---|---:|---:|---:|---:|---:|---:|\n"

    baseline_rows = _extract_source_rows_from_baseline(args.baseline_md_path, args.source_model)
    new_row = _format_experiment_row(args, metrics_by_model)

    with open(result_md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Result for Source Model: {args.source_model}\n\n")
        f.write(header)
        f.write(sep)
        for r in baseline_rows:
            f.write(r)
        f.write(new_row)


def eval_asr(model, ref_model, tokenizer, t_models, t_ref_models, t_tokenizers, t_test_transforms, data_loader, device, args, config, result_base_dir, save_cluster_info=False, use_topological_loss=True, use_attention_weighted=True, num_clusters=30):
    print("Start eval")
    start_time = time.time()
    
    score_i2t, score_t2i, t_score_i2ts, t_score_t2is = retrieval_eval(model, ref_model, t_models, t_ref_models, t_test_transforms,
                                                                   data_loader, tokenizer, t_tokenizers, device, args,config, result_base_dir, save_cluster_info, use_topological_loss, use_attention_weighted, num_clusters)

    result_md_path = os.path.join(result_base_dir, "result.md")
    args.result_base_dir = result_base_dir

    metrics_by_model = {}
    source_result = itm_eval(score_i2t, score_t2i, data_loader.dataset.img2txt, data_loader.dataset.txt2img, args.source_model, args)
    metrics_by_model[args.source_model] = {
        "txt@1": _extract_metric_value(source_result.get('txt_r1_ASR (txt_r1)', '')),
        "img@1": _extract_metric_value(source_result.get('img_r1_ASR (img_r1)', '')),
    }

    t_model_names = copy.deepcopy(args.model_list)
    t_model_names.remove(args.source_model)
    for t_model_name, t_score_i2t, t_score_t2i in zip(t_model_names, t_score_i2ts, t_score_t2is):
        t_result = itm_eval(t_score_i2t, t_score_t2i, data_loader.dataset.img2txt, data_loader.dataset.txt2img, t_model_name, args)
        metrics_by_model[t_model_name] = {
            "txt@1": _extract_metric_value(t_result.get('txt_r1_ASR (txt_r1)', '')),
            "img@1": _extract_metric_value(t_result.get('img_r1_ASR (img_r1)', '')),
        }

    _write_result_md(result_md_path, args, metrics_by_model)

    torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluate time {}'.format(total_time_str))

def main(args, config, use_topological_loss=True, use_attention_weighted=True):
    torch.cuda.set_device(args.cuda_id)
    device = torch.device('cuda')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 从配置文件路径中提取数据集的名字
    config_filename = os.path.basename(args.config)
    dataset_name = config_filename.split('_')[1].split('.')[0]  # 例如从'Retrieval_flickr.yaml'中提取'flickr'
    args.dataset_name = dataset_name
    result_base_dir = os.path.join("results", f"{timestamp}_{args.source_model}_{dataset_name}_{args.experiment}")
    os.makedirs(result_base_dir, exist_ok=True)
    
    adv_images_dir = os.path.join(result_base_dir, "adv_images")
    os.makedirs(adv_images_dir, exist_ok=True)
    
    print(f"Results will be saved to: {result_base_dir}")
    
    print("Creating Source Model")
    model, ref_model, tokenizer = load_model(args,args.source_model,args.source_text_encoder, device)

    print("Creating Target Model")
    t_models = []
    t_ref_models = []
    t_tokenizers = []
    t_model_names = copy.deepcopy(args.model_list)
    print(t_model_names)
    t_model_names.remove(args.source_model)
    for t_model_name in t_model_names:
        t_model, t_ref_model, t_tokenizer = load_model(args,t_model_name, args.target_text_encoder, device)
        t_models.append(t_model)
        t_ref_models.append(t_ref_model)
        t_tokenizers.append(t_tokenizer)
   
    #### Dataset ####
    print("Creating dataset")
    
    s_test_transform = None
    if args.source_model in ['ALBEF', 'TCL']:
        s_test_transform = transforms.Compose([
            transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
            transforms.ToTensor(),        
        ])
    else:
        n_px = model.visual.input_resolution
        s_test_transform = transforms.Compose([
            transforms.Resize(n_px, interpolation=Image.BICUBIC),
            transforms.CenterCrop(n_px),
            transforms.ToTensor(),       
        ])

    t_test_transforms = []
    for index,t_model_name in enumerate(t_model_names):
        if t_model_name in ['ALBEF', 'TCL']:
            t_test_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
                transforms.ToTensor(),  
            ])
            t_test_transforms.append(t_test_transform)
        else:
            t_model = t_models[index]
            t_n_px = t_model.visual.input_resolution
            t_test_transform = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.Resize(t_n_px, interpolation=Image.BICUBIC),
                transforms.CenterCrop(t_n_px),
                transforms.ToTensor(),
            ])
            t_test_transforms.append(t_test_transform)
    
    

    processed_image_root = os.path.join(config.get('processed_image_root', 'processed_dataset/'), args.source_model)
    dataset_device = device

    # 统一注意力目录：attention_root/{source_model}/
    # 为兼容历史配置，若没有 attention_root，则回退到旧键。
    attention_root = config.get('attention_root')
    if attention_root is None:
        legacy_attn_root = config['fused_attn_root'] if args.source_model in ['ALBEF', 'TCL'] else config['aligned_attn_root']
        attention_root = legacy_attn_root
    processed_attn_root = os.path.join(attention_root, args.source_model)

    test_dataset = paired_dataset(
        config['test_file'],
        s_test_transform,
        config['image_root'],
        processed_image_root,
        processed_attn_root,
        device=dataset_device,
        model_name=args.source_model,
        attn_heatmap_mode=args.attn_heatmap_mode,
        source_attn_model=model
    )
    # return
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=4, collate_fn=test_dataset.collate_fn)

    eval_asr(
        model,
        ref_model,
        tokenizer,
        t_models,
        t_ref_models,
        t_tokenizers,
        t_test_transforms,
        test_loader,
        device,
        args,
        config,
        result_base_dir,
        save_cluster_info=args.save_cluster_info,
        use_topological_loss=use_topological_loss,
        use_attention_weighted=use_attention_weighted,
        num_clusters=args.num_clusters
    )


def resolve_experiment_switches(args):
    """
    统一实验开关：
    - main:          全模块开启
    - wo_topo:       关闭拓扑损失
    - wo_attn:       关闭注意力调制
    - wo_topo_attn:  同时关闭拓扑损失与注意力调制
    """
    mode = args.experiment
    if mode == 'main':
        return True, True
    if mode == 'wo_topo':
        return False, True
    if mode == 'wo_attn':
        return True, False
    if mode == 'wo_topo_attn':
        return False, False
    raise ValueError(f"Unknown experiment mode: {mode}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Retrieval_flickr.yaml')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--cuda_id', default=0, type=int)

    parser.add_argument('--model_list', nargs='+', type=str, default=['ALBEF', 'TCL', 'CLIP_ViT'])
    parser.add_argument('--source_model', default='ALBEF', type=str)
    parser.add_argument('--source_text_encoder', default='bert-base-uncased', type=str)   
    parser.add_argument('--target_text_encoder', default='bert-base-uncased', type=str)

    parser.add_argument('--albef_ckpt', default='./checkpoints/albef_flickr.pth', type=str) 
    parser.add_argument('--tcl_ckpt', default='./checkpoints/tcl_flickr.pth', type=str)    
 
    parser.add_argument('--original_rank_index_path', default='./std_eval_idx/flickr30k/')  
    parser.add_argument('--scales', type=str, default='0.5,0.75,1.25,1.5')
    parser.add_argument('--save_cluster_info', action='store_true', help='保存聚类信息到CSV文件')
    parser.add_argument('--no_attention_weighted', action='store_true', help='禁用注意力加权（用于消融实验）')
    parser.add_argument('--no_topological_loss', action='store_true', help='禁用拓扑损失（用于消融实验）')
    parser.add_argument('--num_clusters', default=30, type=int, help='聚类数量（用于计算topological loss）')
    parser.add_argument('--cache_dir', default='./cache', type=str, help='聚类中心缓存目录')
    parser.add_argument('--disable_cluster_cache', action='store_true', help='禁用聚类中心缓存')
    parser.add_argument(
        '--attn_heatmap_mode',
        default='unified',
        choices=['unified', 'per_model'],
        help='注意力热图模式：unified=统一热图（原方法），per_model=按模型分别热图'
    )
    parser.add_argument(
        '--experiment',
        default='main',
        choices=['main', 'wo_topo', 'wo_attn', 'wo_topo_attn'],
        help='一键实验模式：主实验或两项消融'
    )
    parser.add_argument('--baseline_md_path', default='./results/baseline_table.md', type=str,
                        help='论文/基线结果表(md)，用于提取对应source的历史行')
    args = parser.parse_args()

    use_topological_loss, use_attention_weighted = resolve_experiment_switches(args)

    # 若用户显式传入旧开关，则在 experiment 的基础上进一步关闭对应模块
    if args.no_topological_loss:
        use_topological_loss = False
    if args.no_attention_weighted:
        use_attention_weighted = False

    config = yaml.load(open(args.config, 'r'))

    main(args, config, use_topological_loss=use_topological_loss, use_attention_weighted=use_attention_weighted)
