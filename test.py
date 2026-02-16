import cv2
import numpy as np
import matplotlib.pyplot as plt

def overlay_images(image1_path, image2_path, alpha=0.2):
    # 读取图像
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    # 确保两幅图像尺寸一致
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # 叠加图像，减少混合比例
    blended = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
    
    return blended

# 读取并叠加图像
result = overlay_images("example.jpg", "per.jpg", alpha=0.1)

# 使用 Matplotlib 显示结果
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# 保存结果
cv2.imwrite("blended_result.jpg", result)