import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import difflib

# 设置数据路径
clean_image_dir = "clean_images/"
adv_image_dir = "adversarial_images/"
perturbation_dir = "perturbations/val2014"

# 读取所有图像文件（假设所有目录下的文件名一致）
image_files = sorted([f for f in os.listdir(clean_image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

# 示例文本
clean_texts = [
    "A bathroom that has a shower curtain over a bathtub.",
    "a young boy barefoot holding an umbrella touching the horn of a cow.",
    "A person on a bike riding on a street.",
    "There are two sinks next to two mirrors.",
    "A bathroom with a gouged wall and wall socket with no panel.", 
    "A hallway leading into a white kitchen with appliances.",
    "A series of parking meters and cars are located next to each other.",
    "Girl with a yellow shirt holding a small cat.",

]

adversarial_texts = [
    "A bathroom that has a shower water over a bathtub.",
    "a young junior barefoot holding an umbrella touching the horn of a cow.",
    "A person on a each riding on a street.",
    "There are two couple next to two mirrors.",
    "A bathroom with a a be straight wall and wall socket with no panel.",
    "A hallway leading into a white kitchen with full.",
    "A series of parking space and cars are located next to each other.",
    "Girl with a yellow shirt holding a small bat.",
]

# 检查文本与图片数量是否匹配
if len(clean_texts) != len(image_files) or len(adversarial_texts) != len(image_files):
    raise ValueError("文本数据数量与图片数据数量不匹配，请检查数据！")

# --- 定义高亮函数 ---
def highlight_clean_tokens(clean_text, adv_text):
    s_words = clean_text.split()
    t_words = adv_text.split()
    tokens = []
    matcher = difflib.SequenceMatcher(None, s_words, t_words)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            tokens.extend([(word, "black") for word in s_words[i1:i2]])
        elif tag in ("replace", "delete"):
            tokens.extend([(word, "green") for word in s_words[i1:i2]])
        elif tag == "insert":
            # 干净文本中不显示对抗文本额外的单词
            pass
    return tokens

def highlight_adv_tokens(clean_text, adv_text):
    s_words = clean_text.split()
    t_words = adv_text.split()
    tokens = []
    matcher = difflib.SequenceMatcher(None, s_words, t_words)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            tokens.extend([(word, "black") for word in t_words[j1:j2]])
        elif tag in ("replace", "insert"):
            tokens.extend([(word, "red") for word in t_words[j1:j2]])
        elif tag == "delete":
            pass
    return tokens

# --- 自动换行绘制函数 ---
def draw_wrapped_tokens_fixed(ax, tokens, max_line_width=0.9, font_size=10, line_gap=0.15, start_y=0.9):
    fig = ax.figure
    fig.canvas.draw()  # 确保渲染器可用
    renderer = fig.canvas.get_renderer()
    
    def measure_text_width(token):
        txt = ax.text(0, 0, token, fontsize=font_size, fontfamily="monospace")
        extent = txt.get_window_extent(renderer=renderer)
        txt.remove()
        p0 = ax.transAxes.inverted().transform((0, 0))
        p1 = ax.transAxes.inverted().transform((extent.width, 0))
        return p1[0] - p0[0]
    
    space_width = measure_text_width(" ")
    lines = []
    current_line = []
    current_width = 0.0
    for token, color in tokens:
        token_width = measure_text_width(token)
        additional = token_width if not current_line else space_width + token_width
        if current_width + additional <= max_line_width:
            if current_line:
                current_line.append((" ", "black"))
                current_width += space_width
            current_line.append((token, color))
            current_width += token_width
        else:
            lines.append(current_line)
            current_line = [(token, color)]
            current_width = token_width
    if current_line:
        lines.append(current_line)
    
    for i, line in enumerate(lines):
        y = start_y - i * line_gap
        x = 0.0
        for token, color in line:
            ax.text(x, y, token, color=color, fontsize=font_size, fontfamily="monospace",
                    transform=ax.transAxes, verticalalignment='center', horizontalalignment='left')
            x += measure_text_width(token)

# --- 绘图 ---
row_labels = ["Adversarial Image", "Clean Image", "Perturbation", "Clean Text", "Adversarial Text"]

num_samples = 8

# 调整figsize为更大的尺寸，同时将左侧边距设为0.25，使整体向右移动
fig, axs = plt.subplots(5, num_samples, figsize=(24, 15),
                        gridspec_kw={'left': 0.2, 'right': 0.95})

fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i in range(num_samples):
    filename = image_files[i]
    clean_img = Image.open(os.path.join(clean_image_dir, filename))
    adv_img = Image.open(os.path.join(adv_image_dir, filename))
    
    perturbation_path = os.path.join(perturbation_dir, filename)
    if os.path.exists(perturbation_path):
        perturb_img = Image.open(perturbation_path)
    else:
        clean_arr = np.array(clean_img, dtype=np.float32)
        adv_arr = np.array(adv_img, dtype=np.float32)
        perturb_arr = np.clip((adv_arr - clean_arr) * 10 + 128, 0, 255).astype(np.uint8)
        perturb_img = Image.fromarray(perturb_arr)
    
    axs[0, i].imshow(adv_img)
    axs[0, i].axis("off")
    
    axs[1, i].imshow(clean_img)
    axs[1, i].axis("off")
    
    axs[2, i].imshow(perturb_img)
    axs[2, i].axis("off")
    
    clean_tokens = highlight_clean_tokens(clean_texts[i], adversarial_texts[i])
    draw_wrapped_tokens_fixed(axs[3, i], clean_tokens, max_line_width=0.9, font_size=18, line_gap=0.15, start_y=0.9)
    axs[3, i].axis("off")
    
    adv_tokens = highlight_adv_tokens(clean_texts[i], adversarial_texts[i])
    draw_wrapped_tokens_fixed(axs[4, i], adv_tokens, max_line_width=0.9, font_size=18, line_gap=0.15, start_y=0.9)
    axs[4, i].axis("off")

# 在每行的第一个子图左侧添加标签
for row in range(5):
    ax = axs[row, 0]
    ax.text(-0.1, 0.5, row_labels[row], rotation=0, 
            va='center', ha='right', transform=ax.transAxes, 
            fontsize=18, fontweight='bold')

plt.savefig("adversarial_attack_visualization.eps",bbox_inches='tight')
plt.show()
