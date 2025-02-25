import torch
# from models import clip
from PIL import Image
import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from captum.attr import visualization

def interpret(image, texts, model, device, start_layer=-1, start_layer_text=-1):
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)
    logits_per_image, logits_per_text = model(images, texts)
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
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True,allow_unused=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]

    
    text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

    if start_layer_text == -1: 
      # calculate index of last layer 
      start_layer_text = len(text_attn_blocks) - 1

    num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
    R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
    R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(text_attn_blocks):
        if i < start_layer_text:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R_text = R_text + torch.bmm(cam, R_text)
    text_relevance = R_text
   
    return text_relevance, image_relevance

import cv2

# def show_image_relevance(image_relevance, image, orig_image):
#     # create heatmap from mask on image
#     def show_cam_on_image(img, mask):
#         heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
#         heatmap = np.float32(heatmap) / 255
#         cam = heatmap + np.float32(img)
#         cam = cam / np.max(cam)
#         return cam

#     fig, axs = plt.subplots(1, 2)
#     axs[0].imshow(orig_image)
#     axs[0].axis('off')

#     dim = int(image_relevance.numel() ** 0.5)
#     image_relevance = image_relevance.reshape(1, 1, dim, dim)
#     image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
#     image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
#     image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
#     image = image[0].permute(1, 2, 0).data.cpu().numpy()
#     image = (image - image.min()) / (image.max() - image.min())
#     vis = show_cam_on_image(image, image_relevance)
#     vis = np.uint8(255 * vis)
#     vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
#     axs[1].imshow(vis)
#     axs[1].axis('off')

def show_image_relevance(image_relevance, image, orig_image):
    # create heatmap from mask
    def create_heatmap(mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        return heatmap

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(orig_image)
    axs[0].axis('off')

    # Process image relevance
    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())

    # Create heatmap (no image blending)
    heatmap = create_heatmap(image_relevance)
    
    # Show heatmap in the second subplot
    axs[1].imshow(heatmap)
    axs[1].axis('off')


from models import clip
import matplotlib.pyplot as plt
img_path = "glasses.png"
device = "cuda:2" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device, jit=False)
# model, preprocess = clip.load("RN101", device=device, jit=False)

# model=model.half()
img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
texts = ["a man with glasses"]
text = clip.tokenize(texts).to(device)

R_text, R_image = interpret(model=model, image=img, texts=text, device=device)
# print(R_text.shape, R_image.shape)
batch_size = text.shape[0]
for i in range(batch_size):
  # show_heatmap_on_text(texts[i], text[i], R_text[i])
  show_image_relevance(R_image[i], img, orig_image=Image.open(img_path))
  plt.savefig("1.png")
  plt.show()