import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.linalg import svd
import os

root=os.getcwd()
image_path = os.path.join(root,'images', '2.png')

def load_and_convert_image(image_path):
    # 加载图片并转换为灰度图
    img = Image.open(image_path).convert('L')
    img_matrix = np.array(img)
    return img_matrix

def reconstruct_image(img_matrix, k):
    # 对图片进行SVD分解
    U, s, VT = svd(img_matrix)
    # 保留前k个奇异值进行重构
    S = np.zeros((U.shape[0], VT.shape[0]))
    S[:k, :k] = np.diag(s[:k])
    img_reconstructed = np.dot(U[:, :k], np.dot(S[:k, :k], VT[:k, :]))
    return img_reconstructed

def compare_reconstructions(image_path, k_values):
    img_matrix = load_and_convert_image(image_path)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, len(k_values) + 1, 1)
    plt.imshow(img_matrix, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    for i, k in enumerate(k_values, start=2):
        img_reconstructed = reconstruct_image(img_matrix, k)
        plt.subplot(1, len(k_values) + 1, i)
        plt.imshow(img_reconstructed, cmap='gray')
        plt.title(f'k={k}')
        plt.axis('off')
    
    plt.show()

# 示例：比较保留不同数量的奇异值进行还原的差别
image_path = image_path  # 替换为你的图片路径
k_values = [5, 20, 50]  # 保留的奇异值数量
compare_reconstructions(image_path, k_values)
