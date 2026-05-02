# SA-AET

Semantic-Aligned Adversarial Evolution Triangle（SA-AET）复现与使用说明（中文）

---

## 1. 项目简介

SA-AET 是一个面向视觉-语言预训练（VLP）模型的高迁移性多模态对抗攻击方法。

当前仓库主要用于 **Image-Text Retrieval** 场景下的迁移攻击评估，支持：

- 源模型：`ALBEF`、`TCL`、`CLIP_ViT`
- 目标模型：由 `--model_list` 指定（默认包含 `ALBEF/TCL/CLIP_ViT`）
- 消融设置：`main`、`wo_attn`、`wo_topo`

---

## 2. 环境安装

### 2.1 使用已导出的 Conda 环境（推荐）

仓库中已提供环境文件：`environment_AET.yml`

```bash
conda env create -f environment_AET.yml
conda activate AET
```

### 2.2 手动安装（备选）

```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 3. 数据与权重准备

### 3.1 数据集

请准备 Flickr30k / MSCOCO 数据集，并在配置文件中设置图像根目录：

- `configs/Retrieval_flickr.yaml`
- `configs/Retrieval_coco.yaml`

常见目录结构示例（可按你实际路径调整）：

```text
datasets/
  flickr30k-images/
  train2014/
  val2014/
```

### 3.2 模型权重（放到 `checkpoints/`）

先创建目录：

```bash
mkdir -p checkpoints
```

下载示例：

1) ALBEF Flickr30K
```bash
wget https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/flickr30k.pth -O checkpoints/albef_flickr.pth
```

2) ALBEF MSCOCO
```bash
wget https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/mscoco.pth -O checkpoints/albef_coco.pth
```

3) TCL 权重

原仓库部分链接可能失效，建议使用你已保存的权重，或从 Hugging Face：

- `Sensen02/VLPTransferAttackCheckpoints`

并放到：

- `checkpoints/tcl_flickr.pth`
- `checkpoints/tcl_coco.pth`

---

## 4. 运行方式

### 4.1 单次运行（示例：Flickr）

```bash
python eval_AET.py \
  --config ./configs/Retrieval_flickr.yaml \
  --cuda_id 0 \
  --source_model ALBEF \
  --experiment main \
  --save_cluster_info
```

> 注意：`--cuda_id` 是当前可见设备编号。若你设置了 `CUDA_VISIBLE_DEVICES=1`，则 `--cuda_id 0` 对应物理 `cuda:1`。

### 4.2 批量运行（推荐）

仓库中的 `retrieval.sh` 已配置为：

- 固定使用物理 `cuda:1`
- 依次遍历源模型：`ALBEF`、`TCL`、`CLIP_ViT`
- 每个源模型跑三种实验：`main`、`wo_attn`、`wo_topo`

直接运行：

```bash
bash retrieval.sh
```

---

## 5. 输出结果

结果默认保存在 `results/` 下，目录名格式类似：

```text
results/{timestamp}_{source_model}_{dataset}_{experiment}/
```

典型文件包括：

- `result.md`：该次实验汇总
- `adv_images/`：生成的对抗图像
- `cluster_info.csv`：聚类信息（开启 `--save_cluster_info` 时）

---
