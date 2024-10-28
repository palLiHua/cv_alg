'''

'''

import numpy as np
import math

class Hough_transform:
    def __init__(self, img, angle, step=5, threshold=135):
        '''

        :param img: 输入的图像
        :param angle: 输入的梯度方向矩阵
        :param step: Hough 变换步长大小
        :param threshold: 筛选单元的阈值
        '''
        self.img = img
        self.angle = angle
        self.y, self.x = img.shape[0:2]
        self.radius = math.ceil(math.sqrt(self.y**2 + self.x**2))
        self.step = step
        self.vote_matrix = np.zeros([math.ceil(self.y / self.step), math.ceil(self.x / self.step), math.ceil(self.radius / self.step)])
        self.threshold = threshold
        self.circles = []

    def Hough_transform_algorithm(self):
        '''
        按照 x,y,radius 建立三维空间，根据图片中边上的点沿梯度方向对空间中的所有单
        元进行投票。每个点投出来结果为一折线。
        :return:  投票矩阵
        '''
        print ('Hough_transform_algorithm')
        # ------------- write your code bellow ----------------
        for i,j in zip(*np.where(self.img > 0)):
            tmp_y = i
            tmp_x = j
            r = 0
            while tmp_y < self.y and tmp_x < self.x and tmp_y >= 0 and tmp_x >= 0:
                self.vote_matrix[math.floor(tmp_y / self.step)][math.floor(tmp_x / self.step) ][math.floor(r / self.step)] += 1
                tmp_x += self.step
                tmp_y += self.step * self.angle[i][j]
                r += math.sqrt(self.step ** 2 + (self.step * self.angle[i][j]) ** 2)
            
            tmp_y = i - self.step * self.angle[i][j]
            tmp_x = j - self.step
            r = math.sqrt((self.step * self.angle[i][j])**2 + self.step**2)
            while tmp_y < self.y and tmp_x < self.x and tmp_y >= 0 and tmp_x >= 0:
                self.vote_matrix[math.floor(tmp_y / self.step)][math.floor(tmp_x / self.step) ][math.floor(r / self.step)] += 1
                tmp_x -= self.step
                tmp_y -= self.step * self.angle[i][j]
                r += math.sqrt(self.step ** 2 + (self.step * self.angle[i][j]) ** 2)

        # ------------- write your code above ----------------
        # np.save("vote_my.npy", self.vote_matrix)
        return self.vote_matrix

    def Select_Circle(self):
        '''
        按照阈值从投票矩阵中筛选出合适的圆，并作极大化抑制。
        :return: None
        '''
        print ('Select_Circle')
        # ------------- write your code bellow ----------------
        suspend = []
        for i,j,r in zip(*np.where(self.vote_matrix >= self.threshold)):
            y = i * self.step + self.step / 2
            x = j * self.step + self.step / 2
            radius = r * self.step + self.step / 2
            suspend.append([y,x,radius])
        
        if len(suspend) == 0:
            print("No Circle in this threshold.")
            return

        center_y, center_x, center_r = suspend[0]
        except_ = []
        possible = []
        thresh = 15
        for index, (y, x, r) in enumerate(suspend):
            if abs(center_y - y) <= thresh and abs(center_x - x) <= thresh:
                possible.append([y,x,r])
            else:
                except_.append(index)
        
        res = []
        while len(except_) > 0:
            y_, x_, r_ = np.array(possible).mean(axis = 0)
            res.append([x_, y_, r_])
            print(f"Circle y = {res[-1][0]}, x = {res[-1][1]}, r = {res[-1][2]}")
            possible = []
            center_y, center_x, center_r = suspend[except_[0]]
            except_new = []
            for index in except_:
                y, x, r = suspend[index]
                if abs(center_y - y) <= thresh and abs(center_x - x) <= thresh:
                    possible.append([y,x,r])
                else:
                    except_new.append(index)
            except_ = except_new
        y_, x_, r_ = np.array(possible).mean(axis = 0)
        res.append([x_, y_, r_])
        print(f"Circle y = {res[-1][0]}, x = {res[-1][1]}, r = {res[-1][2]}")
    
        self.circles = res
        # ------------- write your code above ----------------


    def Calculate(self):
        '''
        按照算法顺序调用以上成员函数
        :return: 圆形拟合结果图，圆的坐标及半径集合
        '''
        self.Hough_transform_algorithm()

        self.Select_Circle()
        return self.circles