'''

'''
import cv2
import numpy as np


class Canny:

    def __init__(self, Guassian_kernal_size, img, HT_high_threshold, HT_low_threshold):
        '''
        :param Guassian_kernal_size: 高斯滤波器尺寸
        :param img: 输入的图片，在算法过程中改变
        :param HT_high_threshold: 滞后阈值法中的高阈值
        :param HT_low_threshold: 滞后阈值法中的低阈值
        '''
        self.Guassian_kernal_size = Guassian_kernal_size
        self.img = img
        self.y, self.x = img.shape[0:2]
        self.angle = np.zeros([self.y, self.x])
        self.img_origin = None
        self.x_kernal = np.array([[-1, 1]])
        self.y_kernal = np.array([[-1], [1]])
        self.HT_high_threshold = HT_high_threshold
        self.HT_low_threshold = HT_low_threshold

    def Get_gradient_img(self):
        '''
        计算梯度图和梯度方向矩阵。
        :return: 生成的梯度图
        '''
        print ('Get_gradient_img')
        # ------------- write your code bellow ----------------
        # horizon_kernel = np.array([[-1,1]])
        # vertical_kernel = np.array([[1], [-1]])
        horizon = cv2.filter2D(self.img, ddepth = cv2.CV_32F, kernel = self.x_kernal)
        vertical = cv2.filter2D(self.img, ddepth = cv2.CV_32F, kernel = self.y_kernal)

        horizon[:, 0] = 1
        vertical[0, :] = 1
        # gradient = (horizon ** 2 + vertical ** 2) ** 0.5
        # theta_ = np.arctan(vertical / horizon)
        magnitude, theta = cv2.cartToPolar(horizon, vertical)
        self.angle = np.tan(theta)
        self.img = magnitude.astype(np.uint8)
        
        # ------------- write your code above ----------------        
        return self.img

    def Non_maximum_suppression (self):
        '''
        对生成的梯度图进行非极大化抑制，将tan值的大小与正负结合，确定离散中梯度的方向。
        :return: 生成的非极大化抑制结果图
        '''
        print ('Non_maximum_suppression')
        # ------------- write your code bellow ----------------
        thresh = 4
        res = np.zeros([self.y, self.x])
        for i in range(1, self.y-1):
            for j in range(1, self.x-1):
                if abs(self.img[i][j]) <= thresh:
                    res[i][j] = 0
                    continue
                elif abs(self.angle[i][j]) > 1:
                    gradient1 = self.img[i+1][j]
                    gradient2 = self.img[i-1][j]
                    if self.angle[i][j] > 0:
                        # g4 g2
                        #    c
                        #    g1 g3
                        gradient3 = self.img[i+1][j+1]
                        gradient4 = self.img[i-1][j-1]
                    else:
                        #    g2 g4
                        #    c
                        # g3 g1
                        gradient3 = self.img[i+1][j-1]
                        gradient4 = self.img[i-1][j+1]
                else:
                    gradient1 = self.img[i][j+1]
                    gradient2 = self.img[i][j-1]
                    if self.angle[i][j] > 0:
                        # g4
                        # g2 c g1
                        #      g3
                        gradient3 = self.img[i+1][j+1]
                        gradient4 = self.img[i-1][j-1]
                    else:
                        #      g3
                        # g2 c g1
                        # g4
                        gradient3 = self.img[i-1][j+1]
                        gradient4 = self.img[i+1][j-1]
                # weight = |dy| / |dx|
                weight = abs(self.angle[i][j])
                tmp1 = (1 - weight) * gradient1 + weight * gradient3
                tmp2 = (1 - weight) * gradient2 + weight * gradient4
                if self.img[i][j] >= tmp1 and self.img[i][j] >= tmp2:
                    res[i][j] = self.img[i][j]
        # print(res[6][757])
        self.img = res
        # ------------- write your code above ----------------        
        return self.img

    def Hysteresis_thresholding(self):
        '''
        对生成的非极大化抑制结果图进行滞后阈值法，用强边延伸弱边，这里的延伸方向为梯度的垂直方向，
        将比低阈值大比高阈值小的点置为高阈值大小，方向在离散点上的确定与非极大化抑制相似。
        :return: 滞后阈值法结果图
        '''
        print ('Hysteresis_thresholding')
        # ------------- write your code bellow ----------------

        res = np.copy(self.img)
        
        def convert(index_row, index_col):
            if self.img_origin[index_row][index_col] > self.HT_low_threshold:
                res[index_row][index_col] = self.HT_high_threshold

        # for i,j in zip(*np.where(self.img >= self.HT_high_threshold)):
        for i in range(1, self.y - 1):
            for j in range(1, self.x - 1):
                if self.img[i][j] >= self.HT_high_threshold:
            # res[i][j] = self.HT_high_threshold
                    if abs(self.angle[i][j]) < 1:
                        #  g1
                        #  c
                        #  g2
                        convert(i-1, j)
                        convert(i+1, j)

                        if self.angle[i][j] < 0:
                            # g3 g1
                            #    c
                            #    g2 g4
                            convert(i-1,j-1)
                            convert(i+1,j+1)
                        else:
                            #    g1 g3
                            #    c
                            # g4 g2
                            convert(i-1,j+1)
                            convert(i+1,j-1)
                    else:
                        #
                        # g1 c g2
                        #
                        convert(i,j+1)
                        convert(i,j-1)
                        if self.angle[i][j] < 0:
                            # g3
                            # g1 c g2
                            #      g4
                            convert(i-1,j-1)
                            convert(i+1,j+1)
                        else:
                            convert(i-1,j+1)
                            convert(i+1,j-1)
        # res[res < self.HT_high_threshold] = 0
        # print(np.sum(self.img >= self.HT_high_threshold))
        self.img = res
        # np.save("hys.npy", self.img)
        # ------------- write your code above ---------------- 
        
        return self.img

    def canny_algorithm(self):
        '''
        按照顺序和步骤调用以上所有成员函数。
        :return: Canny 算法的结果
        '''
        self.img = cv2.GaussianBlur(self.img, (self.Guassian_kernal_size, self.Guassian_kernal_size), 0)
        self.Get_gradient_img()

        self.img_origin = self.img.copy()
        self.Non_maximum_suppression()
        self.Hysteresis_thresholding()
        return self.img