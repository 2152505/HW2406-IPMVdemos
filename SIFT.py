import cv2
import matplotlib.pyplot as plt
import os
# 读取图片

root = os.getcwd()
image1_path=os.path.join(root,'images', '2.png')
image2_path=os.path.join(root,'images', '3.png')

img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)  # 查询图片
img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)  # 训练图片

# 创建SIFT检测器
sift = cv2.SIFT_create()

# 检测关键点和提取描述符
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# 创建BFMatcher对象
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# 进行匹配
matches = bf.match(descriptors1, descriptors2)

# 根据距离排序
matches = sorted(matches, key=lambda x:x.distance)

# 绘制前10个匹配项
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 显示结果
plt.imshow(img_matches)
plt.show()