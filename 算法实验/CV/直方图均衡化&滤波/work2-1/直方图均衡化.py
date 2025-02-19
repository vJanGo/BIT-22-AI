from collections import defaultdict

import matplotlib.image as mpimg    # 用于读取
import matplotlib.pyplot as plt     # 用于显示
import logging


logging.basicConfig(level=logging.INFO)

# 直方图均衡化
class Histogram(object):
    def __init__(self):
        pass
    # 显示直方图
    def show(self,img,*,title = ''):
        # 只取单通道
        img = img[:,:,0]
        size_h,size_w= img.shape
        logging.info(f' size_h :{size_h} size_w:{size_w} MN:{size_w*size_h}')

        hist = defaultdict(lambda: 0)

        for i in range(size_h):
            for j in range(size_w):
                hist[img[i,j]] += 1
        
        # self.showByHistogram(hist,title)
        x_axis = range(0,256)
        y_axis = list(map(lambda x : hist[x] ,x_axis))

        fig = plt.figure(num = title)
        plot_1 = fig.add_subplot(111)
        plot_1.set_title(title)
        plot_1.bar(x_axis,y_axis)
        plt.show()
        del fig
    # 获取直方图，参数Normalized 决定是否归一化
    def get_histogram(img ,*,Normalized = False):
        # 只取 0 通道，若本身只有一个通道不会有影响
        img = img[:,:,0]
        size_h,size_w = img.shape
        logging.info(f' size_h :{size_h} size_w:{size_w} MN:{size_w*size_h}')

        hist = defaultdict(lambda: 0)
        
        for i in range(size_h):
            for j in range(size_w):
                hist[img[i,j]] += 1
                
        # 根据 Normalized 参数决定是否进行归一化
        if Normalized == True:
            sum = 0
            MN = size_h * size_w
            for pixel_value in hist: # Key 迭代
                hist[pixel_value] = hist[pixel_value]/MN
                sum += hist[pixel_value]
            logging.info(f'归一化后加和为：{sum}')
            del sum
        return hist
    # 直方图均衡化函数
    def equalization(img):
        size_h,size_w,size_c = img.shape
        hist = Histogram.get_histogram(img)
        MN = size_h * size_w
        new_hist = defaultdict(lambda: 0)
        # 计算 S_k
        for i in range(0,256):
            for pixel in range(i+1):
                new_hist[i] += hist[pixel]
            new_hist[i] = new_hist[i] * 255/MN
        
        for key in new_hist:
            new_hist[key] = round(new_hist[key])
        new_img = img.copy()
        
        for i in range(new_img.shape[0]):
            for j in range(new_img.shape[1]):
                new_img[i,j] = new_hist[new_img[i,j,0]]

        return new_img
    pass

if __name__ == '__main__':

    pic1 = mpimg.imread('原始图像.jpg')  # 读取图像

    
    plt.rcParams['font.sans-serif']=['SimHei']
    new_pic = Histogram.equalization(pic1)
    plt.imshow(pic1)
    plt.title('未处理前的图像')
    plt.axis('off')
    plt.show()
    plt.imshow(new_pic)
    plt.axis('off')
    plt.title('直方图均衡化后的图像')
    plt.show()
    histogram = Histogram()
    histogram.show(img=pic1,title='原始直方图')
    histogram.show(img=new_pic,title='均衡化后的直方图')
    mpimg.imsave(r'.\\output1.jpg',new_pic)
    
    plt.imshow
