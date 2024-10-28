This is an implementation of Canny and HoughTransform

# Canny alg
用于检测边缘
1. 使用高斯核对图像进行平滑操作
2. 求梯度及梯度方向
3. 根据梯度方向及梯度值进行非最大化抑制(nms)
4. Hysteresis thresholding对非连续边进行填充(根据梯度方向旋转90度，对两个方向延伸)

# Hough alg
用于参数估计
1. 设定step大小，对参数遍历
2. 根据投票结果进行nms