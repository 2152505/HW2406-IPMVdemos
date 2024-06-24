import cv2
from matplotlib import pyplot as plt
import os
# 读取图片

root=os.getcwd()
image_path=os.path.join(root,'images','2.png')
# 假设 image_path 已经被定义
img = cv2.imread(image_path)

# 定义更大的高斯核大小
kernel_sizes = [(11, 11), (15, 15), (31, 31), (51, 51)]

# 创建一个matplotlib的subplot，用于显示原图和不同核大小的模糊效果
plt.figure(figsize=(10, 8))
plt.subplot(2, 3, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original')

# 对每个高斯核大小进行模糊处理并显示，同时指定较大的标准差
for i, kernel_size in enumerate(kernel_sizes, 2):
    # 这里我们不指定标准差（sigmaX），让OpenCV自动从核大小计算。也可以手动指定较大的标准差。
    blurred_img = cv2.GaussianBlur(img, kernel_size, 0)
    plt.subplot(2, 3, i), plt.imshow(cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB)), plt.title(f'Gaussian {kernel_size}')

plt.tight_layout()
plt.show()