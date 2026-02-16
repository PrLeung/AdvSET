import numpy as np
import matplotlib.pyplot as plt

# 假设有 4 种不同的方法
methods = ["DiSCo-SAP-PM-TAP", "DiSCo-SAP-PM", "DiSCo-SAP", 
           "DiSCo"]
# 每个子图有 2 个横坐标刻度
x_labels = ["TR R@1", "IR R@1"]

# 子图数量
num_subplots = 6  # 2行 × 3列 = 6个子图
num_x = len(x_labels)
num_methods = len(methods)

# 为每个子图设置不同的标题（使用 LaTeX 语法来正确显示下标）
titles = [
    "$ALBEF \ to \ CLIP_{ViT}$",
    "$ALBEF \ to \ TCL$",
    # "$TCL \  to \ ALBEF$",
    "$CLIP_{ViT} \ to \ ALBEF$",
    # "$CLIP_{ViT} \ to \ TCL$",
    "$TCL \ to \ CLIP_{ViT}$"
]

# 用虚拟数据替换随机数据
data = np.array([
    [[55.58, 56.2, 60.61, 67.24], [63.89, 64.66, 68.14, 71.62]],
    [[96.42, 96.5, 96.52, 96.94], [96.02, 96.29, 96.45, 96.98]],
    # [[98.85, 99.27, 99.33, 98.96], [98.5, 98.9, 98.95, 98.64]],
    [[36.6, 40.46, 52.03, 53.7], [50.44, 52.88, 61.11, 62.42]],
    # [[39.2, 41.2, 51.32, 52.48], [51.1, 53.24, 62.1, 61.98]],
    [[56.2, 55.83, 57.06, 60.86], [63.47, 63.37, 65.08, 68.78]]
])

# 指定每种方法的颜色
colors = [
    (224/255, 72/255, 50/255),     # 红色
    (251/255, 186/255, 115/255),   # 浅橙色
    (246/255, 240/255, 144/255),   # 黄色
    (184/255, 222/255, 236/255)    # 天蓝色
]

plt.rcParams.update({
    'font.size': 14,      # 基础字体大小
    'axes.labelsize': 15, # 坐标轴标签大小
    'xtick.labelsize': 15,# x轴刻度大小
    'ytick.labelsize': 15,# y轴刻度大小
    'legend.fontsize': 15 # 图例字体大小
})

# 创建 2×3 网格的子图
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(14, 8), sharey=False)

# 设置每个柱的宽度
bar_width = 0.12

for idx in range(num_subplots):
    row = idx // 3
    col = idx % 3
    ax = axs[row, col]
    x_positions = np.arange(num_x)
    
    for m in range(num_methods):
        offset = (m - (num_methods - 1)/2) * (bar_width + 0.02)
        ax.bar(x_positions + offset,
               data[idx, :, m],
               width=bar_width,
               label=methods[m],
               color=colors[m])
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.text(0.5, -0.2, titles[idx], transform=ax.transAxes, ha='center', va='center', fontsize=15)
        # 设置每个子图的 y 轴范围以放大对比
    if idx == 0:
        ax.set_ylim(50, 70)
    elif idx == 1:
        ax.set_ylim(95, 97)
    elif idx == 2:
        ax.set_ylim(98, 100)
    elif idx == 3:
        ax.set_ylim(30, 70)
    elif idx == 4:
        ax.set_ylim(30, 70)
    elif idx == 5:
        ax.set_ylim(50, 70)


# 在图表顶部添加图例
fig.legend(methods, loc='upper center', ncol=len(methods))

plt.tight_layout(rect=[0, 0, 1, 0.95], w_pad=3.0, h_pad=2.0)


# 保存图像并显示
# plt.savefig("ablation.png", format="png", bbox_inches="tight")
plt.savefig("ablation.eps", format="eps", bbox_inches="tight")
plt.show()
