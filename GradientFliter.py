import cv2
import numpy as np
from matplotlib import pyplot as plt

import os
# 读取图片

root=os.getcwd()
image_path=os.path.join(root,'images','2.png')
        
# 读取图片
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# Sobel算子的卷积核
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Prewitt算子的卷积核
prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

# Scharr算子的卷积核
scharr_kernel_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
scharr_kernel_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])

# box算子的卷积核
box_kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 1000

# 应用卷积核
sobel_x = cv2.filter2D(img, -1, sobel_kernel_x)
sobel_y = cv2.filter2D(img, -1, sobel_kernel_y)
sobel_gradient = np.sqrt(sobel_x**2 + sobel_y**2)

prewitt_x = cv2.filter2D(img, -1, prewitt_kernel_x)
prewitt_y = cv2.filter2D(img, -1, prewitt_kernel_y)
prewitt_gradient = np.sqrt(prewitt_x**2 + prewitt_y**2)

scharr_x = cv2.filter2D(img, -1, scharr_kernel_x)
scharr_y = cv2.filter2D(img, -1, scharr_kernel_y)
scharr_gradient = np.sqrt(scharr_x**2 + scharr_y**2)

box = cv2.filter2D(img, -1, box_kernel)
# 显示结果
plt.figure(figsize=(10, 8))
plt.subplot(3, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(3, 2, 2), plt.imshow(sobel_gradient, cmap='gray'), plt.title('Sobel Gradient')
plt.subplot(3, 2, 3), plt.imshow(prewitt_gradient, cmap='gray'), plt.title('Prewitt Gradient')
plt.subplot(3, 2, 4), plt.imshow(scharr_gradient, cmap='gray'), plt.title('Scharr Gradient')
plt.subplot(3, 2, 5), plt.imshow(box, cmap='gray'), plt.title('Box Gradient')
plt.tight_layout()
plt.show()


if 0:
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    import os
        # 读取图片

    root=os.getcwd()
    image_path=os.path.join(root,'images','2.png')
        
    # 读取图片
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Sobel算子
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_gradient = np.sqrt(sobelx**2 + sobely**2)

    # Prewitt算子
    prewittx = cv2.filter2D(img, -1, np.array([[-1,0,1],[-1,0,1],[-1,0,1]]))
    prewitty = cv2.filter2D(img, -1, np.array([[-1,-1,-1],[0,0,0],[1,1,1]]))
    prewitt_gradient = np.sqrt(prewittx**2 + prewitty**2)

    # Scharr算子
    scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    scharr_gradient = np.sqrt(scharrx**2 + scharry**2)

    # 显示结果
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.subplot(2, 2, 2), plt.imshow(sobel_gradient, cmap='gray'), plt.title('Sobel Gradient')
    plt.subplot(2, 2, 3), plt.imshow(prewitt_gradient, cmap='gray'), plt.title('Prewitt Gradient')
    plt.subplot(2, 2, 4), plt.imshow(scharr_gradient, cmap='gray'), plt.title('Scharr Gradient')
    plt.tight_layout()
    plt.show()

if 0:
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    import os
    # 读取图片

    root=os.getcwd()
    image_path=os.path.join(root,'images','2.png')
    # 读取图片
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 使用Sobel算子计算x和y方向上的梯度
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    # 计算梯度幅度
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

    # 将梯度幅度映射到0-255范围
    gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))

    # 显示原图、x方向梯度、y方向梯度和梯度幅度
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.subplot(2, 2, 2), plt.imshow(sobelx, cmap='gray'), plt.title('Sobel X')
    plt.subplot(2, 2, 3), plt.imshow(sobely, cmap='gray'), plt.title('Sobel Y')
    plt.subplot(2, 2, 4), plt.imshow(gradient_magnitude, cmap='gray'), plt.title('Gradient Magnitude')
    plt.tight_layout()
    plt.show()