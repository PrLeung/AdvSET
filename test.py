import torch

def average_and_concat(adv_imgs_embeds, txts_embeds, txt2img):
    # 验证 txts_embeds 的大小是 k 的整数倍
    k = len(txt2img) // len(adv_imgs_embeds)
    assert txts_embeds.shape[0] % k == 0, "txts_embeds 的第一维长度不是 k 的整数倍"
    
    # 计算 n 的大小
    n = txts_embeds.shape[0] // k
    
    # 将 txts_embeds 按 k 分组并取平均
    averaged_embeds = txts_embeds.view(n, k, -1).mean(dim=1)
    
    return averaged_embeds

# 示例张量
n, kn = 4, 20  # 假设 n = 4, k = 5
adv_imgs_embeds = torch.randn(n, 512)
txts_embeds = torch.randn(kn, 512)
txt2img = [i // (kn // n) for i in range(kn)]  # 生成示例 txt2img

# 调用函数
replicated_embeds = average_and_concat(adv_imgs_embeds, txts_embeds, txt2img)

# 检查结果
print("adv_imgs_embeds shape:", adv_imgs_embeds.shape)
print("replicated_embeds shape:", replicated_embeds.shape)