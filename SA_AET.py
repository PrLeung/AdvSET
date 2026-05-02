import numpy as np
import torch
import torch.nn as nn

import copy
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

import torch

import torch

import torch

def dynamic_scaling(A, gamma=2.0, mode='exponential'):
    """注意力矩阵动态范围调整"""
    if mode == 'exponential':
        return torch.exp(gamma * (A - 0.5))  # 指数增强
    elif mode == 'linear':
        return gamma * A  # 线性增强
    else:
        return torch.exp(gamma * (A - 0.5))  # 默认指数增强

def get_weighted_perturbation(delta, R,
                            gamma=1.0, mode='linear',
                            epsilon_constraint=None):
    """生成加权扰动（保持原目的：高注意力区域获得更大扰动）"""
    # 兼容 R 为 tuple/list 或 tensor
    if isinstance(R, (tuple, list)):
        R = torch.stack(R, dim=0)
    elif not isinstance(R, torch.Tensor):
        raise TypeError(f"R must be tuple/list/tensor, got {type(R)}")

    R = R.to(delta.device, dtype=delta.dtype)

    # 维度验证
    assert R.shape[0] == delta.shape[0], \
        f"注意力矩阵的批次维度 {R.shape[0]} 应与扰动的批次维度 {delta.shape[0]} 匹配"
    assert R.shape[1:] == delta.shape[2:], \
        f"注意力矩阵的空间维度 {R.shape[1:]} 应与扰动的空间维度 {delta.shape[2:]} 匹配"

    # 每个样本独立归一化，避免 batch 内相互影响
    r_min = R.amin(dim=(1, 2), keepdim=True)
    r_max = R.amax(dim=(1, 2), keepdim=True)
    R_norm = (R - r_min) / (r_max - r_min + 1e-8)

    # 动态范围调整
    scaled_A = dynamic_scaling(R_norm, gamma, mode)

    # 扩展并广播注意力矩阵以匹配 delta 的维度
    A = scaled_A.unsqueeze(1)  # [batch, 1, h, w]
    A = A.expand_as(delta)     # [batch, c, h, w]

    # 应用注意力加权（不再额外放大 delta）
    weighted_delta = delta * A

    # 强度约束：按样本独立约束，避免一个样本影响整个 batch
    if epsilon_constraint is not None:
        current_max = weighted_delta.abs().amax(dim=(1, 2, 3), keepdim=True)
        weighted_delta = weighted_delta / (current_max + 1e-12)
        weighted_delta = weighted_delta * epsilon_constraint

    return weighted_delta





def KL(P,Q,mask=None):
    eps = 0.0000001
    d = (P+eps).log()-(Q+eps).log()
    d = P*d
    if mask !=None:
        d = d*mask
    return torch.sum(d)
def CE(P,Q,mask=None):
    return KL(P,Q,mask)+KL(1-P,1-Q,mask)

def umap(output_net, target_net, eps=0.0000001):
    # Normalize each vector by its norm
    (n, d) = output_net.shape
    output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
    output_net = output_net / (output_net_norm + eps)
    output_net[output_net != output_net] = 0
    target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0
    # Calculate the cosine similarity
    model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
    model_distance = 1-model_similarity #[0,2]
    model_distance[range(n), range(n)] = 3
    model_distance = model_distance - torch.min(model_distance, dim=1)[0].view(-1, 1)
    model_distance[range(n), range(n)] = 0
    model_similarity = 1-model_distance
    target_similarity = torch.mm(target_net, target_net.transpose(0, 1))
    target_distance = 1-target_similarity
    target_distance[range(n), range(n)] = 3
    target_distance = target_distance - torch.min(target_distance,dim=1)[0].view(-1,1)
    target_distance[range(n), range(n)] = 0
    target_similarity = 1 - target_distance
    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0
    # Transform them into probabilities
    model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)
    # Calculate the KL-divergence
    loss = CE(target_similarity,model_similarity)
    return loss

def umap_with_clusters(output_net, target_net, cluster_centers, top_k=4, eps=0.0000001, return_cluster_info=False):
    """
    使用聚类中心计算邻接概率矩阵，计算逻辑与原始 umap 函数一致
    
    Args:
        output_net: 当前 batch 的嵌入 [batch_size, embed_dim]
        target_net: 目标嵌入 [batch_size, embed_dim]
        cluster_centers: 聚类中心 [num_clusters, embed_dim]
        top_k: 选择最近的 k 个聚类中心
        eps: 数值稳定性参数
        return_cluster_info: 是否返回聚类信息（类别和最近的k个类别索引）
    
    Returns:
        loss: 拓扑损失
        cluster_info: (可选) 包含类别和最近k个类别索引的字典
    """
    device = output_net.device
    cluster_centers = cluster_centers.to(device).to(output_net.dtype)
    target_net = target_net.to(output_net.dtype)

    # 归一化，确保余弦相似度稳定
    output_net = F.normalize(output_net, p=2, dim=1, eps=eps)
    target_net = F.normalize(target_net, p=2, dim=1, eps=eps)
    cluster_centers = F.normalize(cluster_centers, p=2, dim=1, eps=eps)

    # 样本-聚类中心相似度矩阵（列即聚类中心）
    sample_cluster_sim = torch.mm(output_net, cluster_centers.t())  # [batch_size, num_clusters]
    top_k = min(top_k, sample_cluster_sim.shape[1])
    top_k_values, top_k_indices = torch.topk(sample_cluster_sim, top_k, dim=1)  # [batch_size, top_k]

    # model_similarity: adv 到 top-k 聚类中心的相似度
    model_similarity = top_k_values

    # target_similarity: clean 到同一组 top-k 聚类中心的相似度
    selected_centers = cluster_centers[top_k_indices.reshape(-1)].view(output_net.shape[0], top_k, -1)
    target_similarity = torch.bmm(
        target_net.unsqueeze(1),
        selected_centers.transpose(1, 2)
    ).squeeze(1)  # [batch_size, top_k]

    # 由相似度构造行概率分布，再计算分布差异
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0
    model_similarity = model_similarity / (torch.sum(model_similarity, dim=1, keepdim=True) + eps)
    target_similarity = target_similarity / (torch.sum(target_similarity, dim=1, keepdim=True) + eps)
    
    # Calculate the KL-divergence
    loss = CE(target_similarity, model_similarity)
    
    if return_cluster_info:
        cluster_info = {
            'cluster_id': top_k_indices[:, 0].cpu().tolist(),  # 最近的聚类中心索引（类别）
            'top_k_indices': top_k_indices.cpu().tolist(),  # 最近的k个聚类中心索引
            'top_k_similarities': top_k_values.cpu().tolist()  # 最近的k个聚类中心的相似度
        }
        return loss, cluster_info
    
    return loss

def precompute_cluster_centers(all_image_embeds, num_clusters=40, device='cpu'):
    """
    对整个数据集的图像嵌入进行 KMeans 聚类
    
    Args:
        all_image_embeds: 整个数据集的图像嵌入 [num_samples, embed_dim]
        num_clusters: 聚类数量，默认为 40
        device: 设备
    
    Returns:
        cluster_centers: 聚类中心 [num_clusters, embed_dim]
        cluster_labels: 每个样本的类别标签 [num_samples]
    """
    from sklearn.cluster import KMeans
    
    # 转换为 numpy 数组
    if isinstance(all_image_embeds, torch.Tensor):
        embeds_np = all_image_embeds.detach().cpu().numpy()
    else:
        embeds_np = all_image_embeds
    
    # 执行 KMeans 聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeds_np)
    cluster_centers = kmeans.cluster_centers_
    
    # 转换为 torch tensor
    cluster_centers = torch.from_numpy(cluster_centers).float().to(device)
    
    print(f"聚类完成: {num_clusters} 个聚类中心")
    print(f"总样本数: {len(embeds_np)}")
    
    # 统计每个聚类的样本数量
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print("每个聚类的样本数量:")
    for cluster_id, count in zip(unique, counts):
        print(f"  聚类 {cluster_id}: {count} 个样本")
    
    return cluster_centers, cluster_labels



class Attacker():
    def __init__(self, model, img_attacker, txt_attacker):
        self.model = model
        self.img_attacker = img_attacker
        self.txt_attacker = txt_attacker

    def attack(self, imgs, txts, txt2img, all_txt_supervisions,device='cpu', max_length=30, scales=None, attn_matrices=None, masks=None, return_cluster_info=False, **kwargs):
        with torch.no_grad():
            origin_img_output = self.model.inference_image(self.img_attacker.normalization(imgs))
            img_supervisions = origin_img_output['image_feat'][txt2img]
        adv_txts = self.txt_attacker.img_guided_attack(self.model, txts, img_embeds=img_supervisions)

        with torch.no_grad():
            txts_input = self.txt_attacker.tokenizer(adv_txts, padding='max_length', truncation=True,
                                                     max_length=max_length, return_tensors="pt").to(device)
            txts_output = self.model.inference_text(txts_input)
            txt_supervisions = txts_output['text_feat']
            # all_texts_input = self.txt_attacker.tokenizer(all_texts, padding='max_length', truncation=True,
            #                                          max_length=max_length, return_tensors="pt").to(device)
            # all_texts_output = self.model.inference_text(all_texts_input)
            
        start_time = time.time()
        attack_result = self.img_attacker.txt_guided_attack(self.model, imgs, txt2img,all_txt_supervisions, device,
                                                                      scales=scales, txt_embeds=txt_supervisions, attn_matrices=attn_matrices,
                                                                      return_cluster_info=return_cluster_info)
        if return_cluster_info:
            adv_imgs, last_adv_imgs, cluster_info = attack_result
        else:
            adv_imgs, last_adv_imgs = attack_result
        end_time = time.time()
        execuate_time = end_time - start_time

        with torch.no_grad():
            adv_imgs_outputs = self.model.inference_image(self.img_attacker.normalization(adv_imgs))
            adv_img_supervisions = adv_imgs_outputs['image_feat'][txt2img]
            last_adv_imgs_outputs = self.model.inference_image(self.img_attacker.normalization(last_adv_imgs))
            last_adv_img_supervisions = last_adv_imgs_outputs['image_feat'][txt2img]
        adv_txts = self.txt_attacker.img_guided_attack(self.model, txts, img_embeds=img_supervisions,
                                                       adv_img_embeds=adv_img_supervisions,
                                                       last_adv_img_embeds=last_adv_img_supervisions)
        
        if return_cluster_info:
            return adv_imgs, adv_txts, execuate_time, cluster_info
        
        return adv_imgs, adv_txts, execuate_time

def replicate_and_concatenate(adv_imgs_embeds, txts_embeds, txt2img):
    # 获取 k 的值（每个 adv_imgs_embeds 的张量需要复制的次数）
    k = len(txt2img) // len(adv_imgs_embeds)
    
    # 验证 txt2img 的长度和 txts_embeds 的长度是否一致
    assert len(txt2img) == len(txts_embeds), "txt2img 和 txts_embeds 的长度不一致"
    
    # 验证 txt2img 的值是否符合 k 的分布
    expected_txt2img = [i // k for i in range(len(txt2img))]
    assert txt2img == expected_txt2img, "txt2img 的格式不符合预期"
    
    # 复制每个 adv_imgs_embeds 的张量 k 次并拼接
    replicated_embeds = adv_imgs_embeds.repeat_interleave(k, dim=0)
    
    return replicated_embeds


def average_and_concat(txts_embeds, txt2img): 
    # 获取分组的数量
    num_groups = max(txt2img) + 1
    
    # 创建一个列表，用来存储每组的平均嵌入
    group_embeds = []
    
    # 遍历每个组
    for i in range(num_groups):
        # 找到属于该组的文本的索引
        group_indices = [idx for idx, group in enumerate(txt2img) if group == i]
        
        # 获取该组的 txts_embeds
        group_embed = txts_embeds[group_indices]
        
        # 对该组的嵌入取平均值
        group_avg_embed = group_embed.mean(dim=0)  # 沿着第0维取平均
        
        # 将该组的平均嵌入添加到结果列表中
        group_embeds.append(group_avg_embed)
    
    # 将每组的平均嵌入堆叠成一个张量
    group_embeds_tensor = torch.stack(group_embeds)
    
    # 返回合并后的张量
    return group_embeds_tensor



class ImageAttacker():
    def __init__(self, normalization, eps=2 / 255, steps=10, step_size=0.5 / 255, sample_numbers=5, cluster_centers=None, use_attention_weighted=True, use_topological_loss=True):
        self.normalization = normalization
        self.eps = eps
        self.steps = steps
        self.step_size = step_size
        self.sample_numbers = sample_numbers
        self.cluster_centers = cluster_centers
        self.use_attention_weighted = use_attention_weighted
        self.use_topological_loss = use_topological_loss
        self._it_labels_cache = {}

    def _get_it_labels(self, sim_shape, txt2img, device, dtype):
        cache_key = (sim_shape[0], sim_shape[1], tuple(txt2img), str(device), str(dtype))
        if cache_key in self._it_labels_cache:
            return self._it_labels_cache[cache_key]

        it_labels = torch.zeros(sim_shape, device=device, dtype=dtype)
        for i in range(len(txt2img)):
            it_labels[txt2img[i], i] = 1

        # 控制缓存规模，避免长时间评估时占用过多显存
        if len(self._it_labels_cache) > 16:
            self._it_labels_cache.clear()
        self._it_labels_cache[cache_key] = it_labels
        return it_labels

    def loss_func(self, adv_imgs_embeds, imgs_embs, txts_embeds, txt2img,projection_matrix, return_cluster_info=False):
        device = adv_imgs_embeds.device
        
        adv_imgs_embeds=adv_imgs_embeds.half()
        imgs_embs=imgs_embs.half()
        txts_embeds=txts_embeds.half()
        adv_imgs_embeds = adv_imgs_embeds @ projection_matrix
        imgs_embs = imgs_embs @ projection_matrix
        txts_embeds = txts_embeds @ projection_matrix

        it_sim_matrix = adv_imgs_embeds @ txts_embeds.T
        it_labels = self._get_it_labels(it_sim_matrix.shape, txt2img, device, it_sim_matrix.dtype)

        loss_IaTcpos = -(it_sim_matrix * it_labels).sum(-1).mean()

        average_txt_embeds = average_and_concat(txts_embeds, txt2img)
        
        cluster_info = None
        if self.use_topological_loss:
            if self.cluster_centers is not None:
                if return_cluster_info:
                    umap_loss_pos1, cluster_info = umap_with_clusters(adv_imgs_embeds, imgs_embs, self.cluster_centers, top_k=4, return_cluster_info=True)
                else:
                    umap_loss_pos1 = umap_with_clusters(adv_imgs_embeds, imgs_embs, self.cluster_centers, top_k=4)
            else:
                umap_loss_pos1 = umap(adv_imgs_embeds, imgs_embs)
            
            umap_loss=umap_loss_pos1
            print("loss_IaTcpos",loss_IaTcpos,"umap_loss",umap_loss)
            loss = loss_IaTcpos+5*umap_loss
        else:
            loss = loss_IaTcpos
        
        if return_cluster_info:
            return loss, cluster_info
        return loss
    
    def loss_func_old(self, adv_imgs_embeds, txts_embeds, txt2img):  
        device = adv_imgs_embeds.device    

        it_sim_matrix = adv_imgs_embeds @ txts_embeds.T
        it_labels = torch.zeros(it_sim_matrix.shape).to(device)
        
        for i in range(len(txt2img)):
            it_labels[txt2img[i], i]=1
        
        loss_IaTcpos = -(it_sim_matrix * it_labels).sum(-1).mean()

        loss = loss_IaTcpos
        
        return loss

    def get_cluster_info(self, imgs, image_embeds, eps=0.0000001):
        """
        获取图像的聚类信息（类别和最近的k个类别索引）
        
        Args:
            imgs: 图像张量 [batch_size, c, h, w]
            image_embeds: 图像嵌入 [batch_size, embed_dim]
            eps: 数值稳定性参数
        
        Returns:
            cluster_info: 包含类别和最近k个类别索引的字典
        """
        if self.cluster_centers is None:
            return None
        
        device = image_embeds.device
        cluster_centers = self.cluster_centers.to(device)
        
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
        top_k = 4
        top_k_values, top_k_indices = torch.topk(sample_cluster_sim, top_k, dim=1)  # [batch_size, top_k]
        
        cluster_info = {
            'cluster_id': top_k_indices[:, 0].cpu().tolist(),  # 最近的聚类中心索引（类别）
            'top_k_indices': top_k_indices.cpu().tolist(),  # 最近的k个聚类中心索引
            'top_k_similarities': top_k_values.cpu().tolist()  # 最近的k个聚类中心的相似度
        }
        
        return cluster_info

    def rand3Num(self): ### num1 -> adv num2-> clean num3->last
        while True:
            num1 = random.randint(1, 100)
            if 100 - num1 > 1:
                num2 = random.randint(1, 100 - num1)
            else:
                num1 = 98
                num2 = 1
            num3 = 100 - num1 - num2
            
            if 1 <= num3 <= 100 and num1 < num3 and num3 < num2:
                break

        return (num1, num2, num3)

    def txt_guided_attack(self, model, imgs, txt2img, projection_matrix,device, scales=None, txt_embeds=None, attn_matrices=None, return_cluster_info=False):

        model.eval()

        b, _, _, _ = imgs.shape

        if scales is None:
            scales_num = 1
        else:
            scales_num = len(scales) + 1
        # 加入高斯噪声
        adv_imgs = imgs.detach() + torch.from_numpy(np.random.uniform(-self.eps, self.eps, imgs.shape)).half().to(device)
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0) # 限制在0-1之间
        with torch.no_grad():
            imgs_output=model.inference_image(imgs)
            imgs_embeds=imgs_output['image_feat']
        last_adv_imgs = None

        start_time = time.time()
        ratio_list = []
        
        # 用于存储聚类信息
        cluster_info = None

        for step in range(self.steps):  # self.steps=10
            if last_adv_imgs != None:
                samples = []
                clone_adv_imgs = adv_imgs.clone()
                loss_list = []
                for k in range(self.sample_numbers):
                    samples.append(self.rand3Num()) # 生成三个随机数

                for sample in samples:
                    # 进行采样
                    adv_imgs = (sample[0] / 100) * clone_adv_imgs + (sample[1] / 100) * imgs + (
                                sample[2] / 100) * last_adv_imgs # sk = λ · xI + β ·  ̃xi−1I + γ ·  ̃xi I
                    adv_imgs.requires_grad_()

                    if self.normalization is not None:
                        adv_imgs_output = model.inference_image(self.normalization(adv_imgs))
                    else:
                        adv_imgs_output = model.inference_image(adv_imgs)

                    adv_imgs_embeds = adv_imgs_output['image_feat']
                    model.zero_grad()
                    with torch.enable_grad():
                        loss = torch.tensor(0.0, dtype=torch.float16).to(device)
                        loss = self.loss_func(adv_imgs_embeds, imgs_embeds, txt_embeds, txt2img,projection_matrix)
                    adv_imgs.retain_grad()
                    loss.backward()
                    grad = adv_imgs.grad
                    grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                    perturbation = self.step_size * grad.sign()
                    if self.use_attention_weighted and attn_matrices is not None:
                        perturbation = get_weighted_perturbation(
                            delta = perturbation,
                            R = attn_matrices,
                            gamma = 2.5,
                            mode = 'exponential',
                            epsilon_constraint = self.step_size
                        )

                    adv_imgs = clone_adv_imgs.detach() + perturbation
                    adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.eps), imgs + self.eps)
                    adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)



                    if self.normalization is not None:
                        adv_imgs_output = model.inference_image(self.normalization(adv_imgs))
                    else:
                        adv_imgs_output = model.inference_image(adv_imgs)
                    adv_imgs_embeds = adv_imgs_output['image_feat']
                    model.zero_grad()
                    with torch.enable_grad():
                        loss = torch.tensor(0.0, dtype=torch.float16).to(device)
                        loss = self.loss_func(adv_imgs_embeds, imgs_embeds, txt_embeds, txt2img,projection_matrix)
                    loss.backward()
                    loss_list.append(loss.item())
                #candidate_index = loss_list.index(max(loss_list))

                candidate_index = loss_list.index(max(loss_list))
                ratio_list.append(samples[candidate_index])

                adv_imgs = (samples[candidate_index][0] / 100) * clone_adv_imgs + (
                            samples[candidate_index][1] / 100) * imgs + (
                                    samples[candidate_index][2] / 100) * last_adv_imgs
                adv_imgs.requires_grad_()
                scaled_imgs = self.get_scaled_imgs(adv_imgs, [0.5, 0.75, 1.25, 1.5], device)

                if self.normalization is not None:
                    adv_imgs_output = model.inference_image(self.normalization(scaled_imgs))
                else:
                    adv_imgs_output = model.inference_image(scaled_imgs)

                adv_imgs_embeds = adv_imgs_output['image_feat']
                model.zero_grad()
                with torch.enable_grad():
                    loss = torch.tensor(0.0, dtype=torch.float16).to(device)
                    for i in range(5):
                        loss_item = self.loss_func(adv_imgs_embeds[i * b:i * b + b], imgs_embeds, txt_embeds, txt2img,projection_matrix)
                        loss += loss_item
                adv_imgs.retain_grad()
                print("loss", loss)
                loss.backward()

                grad = adv_imgs.grad
                grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                perturbation = self.step_size * grad.sign()
                if self.use_attention_weighted and attn_matrices is not None:
                    perturbation = get_weighted_perturbation(
                        delta = perturbation,
                        R = attn_matrices,
                        gamma = 2.5,
                        mode = 'exponential',
                        epsilon_constraint = self.step_size
                    )
                adv_imgs = clone_adv_imgs.detach() + perturbation
                adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.eps), imgs + self.eps)
                adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
                last_adv_imgs = clone_adv_imgs.clone()
            else:
                last_adv_imgs = adv_imgs.clone()
                adv_imgs.requires_grad_()
                scaled_imgs = self.get_scaled_imgs(adv_imgs, [0.5, 0.75, 1.25, 1.5], device)
                if self.normalization is not None:
                    adv_imgs_output = model.inference_image(self.normalization(scaled_imgs))
                else:
                    adv_imgs_output = model.inference_image(scaled_imgs)

                adv_imgs_embeds = adv_imgs_output['image_feat']
                model.zero_grad()
                with torch.enable_grad():
                    loss = torch.tensor(0.0, dtype=torch.float16).to(device)
                    for i in range(5):
                        loss_item = self.loss_func(adv_imgs_embeds[i * b:i * b + b], imgs_embeds, txt_embeds, txt2img,projection_matrix)
                        loss += loss_item
                loss.backward()
                print("loss",loss)
                grad = adv_imgs.grad
                grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                perturbation = self.step_size * grad.sign()
                if self.use_attention_weighted and attn_matrices is not None:
                    perturbation = get_weighted_perturbation(
                        delta = perturbation,
                        R = attn_matrices,
                        gamma = 2.5,
                        mode = 'exponential',
                        epsilon_constraint = self.step_size
                    )
                adv_imgs = adv_imgs.detach() + perturbation
                adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.eps), imgs + self.eps)
                adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"The function execution time: {elapsed_time} seconds")

        # 如果需要返回聚类信息，在最后一步计算
        cluster_info = None
        if return_cluster_info and self.cluster_centers is not None:
            with torch.no_grad():
                # 计算干净图像的聚类信息
                clean_cluster_info = self.get_cluster_info(imgs, imgs_embeds)
                # 计算对抗图像的聚类信息
                if self.normalization is not None:
                    adv_imgs_output = model.inference_image(self.normalization(adv_imgs))
                else:
                    adv_imgs_output = model.inference_image(adv_imgs)
                adv_cluster_info = self.get_cluster_info(adv_imgs, adv_imgs_output['image_feat'])
                cluster_info = {
                    'clean': clean_cluster_info,
                    'adversarial': adv_cluster_info
                }

        if return_cluster_info:
            return adv_imgs, last_adv_imgs, cluster_info
        return adv_imgs, last_adv_imgs

    def save_img(self, img_name, norm_img):
        pil_array = (norm_img * 255).to(torch.uint8).cpu().numpy()
        pil_img = Image.fromarray(np.transpose(pil_array, (1, 2, 0)))
        img_path = "./mscoco_imgs/"
        pil_img.save(img_path + img_name)

    def get_scaled_imgs(self, imgs, scales=None, device='cuda'):
        if scales is None:
            return imgs

        ori_shape = (imgs.shape[-2], imgs.shape[-1])

        reverse_transform = transforms.Resize(ori_shape,
                                              interpolation=transforms.InterpolationMode.BICUBIC)
        result = []
        for ratio in scales:
            scale_shape = (int(ratio * ori_shape[0]),
                           int(ratio * ori_shape[1]))
            scale_transform = transforms.Resize(scale_shape,
                                                interpolation=transforms.InterpolationMode.BICUBIC)
            scaled_imgs = imgs + torch.from_numpy(np.random.normal(0.0, 0.05, imgs.shape)).half().to(device)
            scaled_imgs = scale_transform(scaled_imgs)
            scaled_imgs = torch.clamp(scaled_imgs, 0.0, 1.0)

            reversed_imgs = reverse_transform(scaled_imgs)

            result.append(reversed_imgs)

        return torch.cat([imgs, ] + result, 0)


filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·']
filter_words = set(filter_words)


class TextAttacker():
    def __init__(self, ref_net, tokenizer, cls=True, max_length=30, number_perturbation=1, topk=10,
                 threshold_pred_score=0.3, batch_size=32, text_ratios=[0.6, 0.2, 0.2]):
        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.max_length = max_length
        # epsilon_txt
        self.num_perturbation = number_perturbation
        self.threshold_pred_score = threshold_pred_score
        self.topk = topk
        self.batch_size = batch_size
        self.cls = cls
        self.text_ratios = text_ratios

    def img_guided_attack(self, net, texts, img_embeds=None, adv_img_embeds=None, last_adv_img_embeds=None):
        device = self.ref_net.device

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length,
                                     return_tensors='pt').to(device)

        # substitutes
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)  # seq-len k

        # original state
        origin_output = net.inference_text(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_feat'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_feat'].flatten(1).detach()

        final_adverse = []
        for i, text in enumerate(texts):
            # word importance eval
            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in list_of_index:
                if change >= self.num_perturbation:
                    break

                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue

                substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
                word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                             self.threshold_pred_score)

                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue  # filter out original word
                    if '##' in substitute:
                        continue  # filter out sub-word

                    if substitute in filter_words:
                        continue
                    '''
                    # filter out atonyms
                    if substitute in w2i and tgt_word in w2i:
                        if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                            continue
                    '''
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))
                replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True,
                                                    max_length=self.max_length, return_tensors='pt').to(device)
                replace_output = net.inference_text(replace_text_input)
                if self.cls:
                    replace_embeds = replace_output['text_feat'][:, 0, :]
                else:
                    replace_embeds = replace_output['text_feat'].flatten(1)

                if adv_img_embeds == None:
                    loss = self.loss_func(replace_embeds, img_embeds, i)
                else:
                    loss = self.text_ratios[0] * self.loss_func(replace_embeds, img_embeds, i) + self.text_ratios[
                        1] * self.loss_func(replace_embeds, adv_img_embeds, i) + self.text_ratios[2] * self.loss_func(
                        replace_embeds, last_adv_img_embeds, i)
                candidate_idx = loss.argmax()

                final_words[top_index[0]] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1

            final_adverse.append(' '.join(final_words))

        return final_adverse

    def loss_func(self, txt_embeds, img_embeds, label):
        loss_TaIcpos = -txt_embeds.mul(img_embeds[label].repeat(len(txt_embeds), 1)).sum(-1)
        loss = loss_TaIcpos
        return loss

    def attack(self, net, texts):
        device = self.ref_net.device

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length,
                                     return_tensors='pt').to(device)

        # substitutes
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)  # seq-len k

        # original state
        origin_output = net.inference_text(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_embed'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_embed'].flatten(1).detach()

        criterion = torch.nn.KLDivLoss(reduction='none')
        final_adverse = []
        for i, text in enumerate(texts):
            # word importance eval
            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in list_of_index:
                if change >= self.num_perturbation:
                    break

                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue

                substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
                word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                             self.threshold_pred_score)

                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue  # filter out original word
                    if '##' in substitute:
                        continue  # filter out sub-word

                    if substitute in filter_words:
                        continue
                    '''
                    # filter out atonyms
                    if substitute in w2i and tgt_word in w2i:
                        if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                            continue
                    '''
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))
                replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True,
                                                    max_length=self.max_length, return_tensors='pt').to(device)
                replace_output = net.inference_text(replace_text_input)
                if self.cls:
                    replace_embeds = replace_output['text_embed'][:, 0, :]
                else:
                    replace_embeds = replace_output['text_embed'].flatten(1)

                loss = criterion(replace_embeds.log_softmax(dim=-1),
                                 origin_embeds[i].softmax(dim=-1).repeat(len(replace_embeds), 1))

                loss = loss.sum(dim=-1)
                candidate_idx = loss.argmax()

                final_words[top_index[0]] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1

            final_adverse.append(' '.join(final_words))

        return final_adverse

    def _tokenize(self, text):
        words = text.split(' ')

        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = self.tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, keys

    def _get_masked(self, text):
        words = text.split(' ')
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
        # list of words
        return masked_words

    def get_important_scores(self, text, net, origin_embeds, batch_size, max_length):
        device = origin_embeds.device

        masked_words = self._get_masked(text)
        masked_texts = [' '.join(words) for words in masked_words]  # list of text of masked words

        masked_embeds = []
        for i in range(0, len(masked_texts), batch_size):
            masked_text_input = self.tokenizer(masked_texts[i:i + batch_size], padding='max_length', truncation=True,
                                               max_length=max_length, return_tensors='pt').to(device)
            masked_output = net.inference_text(masked_text_input)
            if self.cls:
                masked_embed = masked_output['text_feat'][:, 0, :].detach()
            else:
                masked_embed = masked_output['text_feat'].flatten(1).detach()
            masked_embeds.append(masked_embed)
        masked_embeds = torch.cat(masked_embeds, dim=0)

        criterion = torch.nn.KLDivLoss(reduction='none')

        import_scores = criterion(masked_embeds.log_softmax(dim=-1),
                                  origin_embeds.softmax(dim=-1).repeat(len(masked_texts), 1))

        return import_scores.sum(dim=-1)


def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words

    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    #
    # print(words)
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    # substitutes L, k
    device = mlm_model.device
    substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates

    # find all possible candidates

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
    all_substitutes = all_substitutes[:24].to(device)
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words
