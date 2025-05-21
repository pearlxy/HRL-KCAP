import cv2
import numpy as np

# 创建一个空白图像
canvas = np.zeros((500, 500, 3), dtype=np.uint8)

# 定义莫兰迪绿色色系的六种颜色（由浅到深）
morandi_green_colors = [
    (213, 228, 213),  # 浅绿色
    (183, 204, 183),  # 稍深的浅绿色
    (153, 179, 153),  # 中浅绿色
    (123, 155, 123),  # 中等绿色
    (93, 130, 93),    # 稍深的中等绿色
    (63, 106, 63)     # 深绿色
]

morandi_blue_colors = [
    (238, 228, 213),  # 浅蓝色
    (214, 204, 183),  # 稍深的浅蓝色
    (191, 179, 153),  # 中浅蓝色
    (167, 155, 123),  # 中等蓝色
    (144, 130, 93),   # 稍深的中等蓝色
    (120, 106, 63)    # 深蓝色
]

# 绘制六个不同颜色的矩形
for i, color in enumerate(morandi_blue_colors):
    cv2.rectangle(canvas, (50, 50 + i*60), (450, 100 + i*60), color, -1)

# 显示结果
cv2.imshow("Morandi Green Colors", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
