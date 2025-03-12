import numpy as np
import cv2
import argparse
import math

ITERATION = 4  # 多尺度分解的迭代次数
RADIUS = 9  # 引导滤波的半径
EPS = 10**3 / (256**2)  # 引导滤波的正则化参数，防止除以零


def guidedFilter(img_i, img_p, r, eps):
    """
    引导滤波函数。

    Args:
        img_i: 引导图像 (灰度图).
        img_p: 待滤波的图像 (灰度图).
        r: 滤波窗口的半径.
        eps: 正则化参数.

    Returns:
        q: 滤波后的图像.
    """
    wsize = int(2*r)+1 # 滤波窗口的尺寸
    meanI = cv2.boxFilter(img_i, ksize=(wsize, wsize), ddepth=-1, normalize=True) # 计算引导图像的均值
    meanP = cv2.boxFilter(img_p, ksize=(wsize, wsize), ddepth=-1, normalize=True) # 计算待滤波图像的均值
    corr_I = cv2.boxFilter(img_i*img_i, ksize=(wsize,wsize), ddepth=-1, normalize=True) # 计算引导图像的方差
    corrIP = cv2.boxFilter(img_i*img_p, ksize=(wsize,wsize), ddepth=-1, normalize=True) # 计算引导图像和待滤波图像的协方差
    varI = corr_I - meanI*meanI # 计算引导图像的方差
    covIP = corrIP - meanI*meanP # 计算引导图像和待滤波图像的协方差
    a = covIP / (varI+eps) # 计算线性系数 a
    b = meanP - a*meanI # 计算线性系数 b
    meanA = cv2.boxFilter(a, ksize=(wsize, wsize), ddepth=-1, normalize=True) # 计算 a 的均值
    meanB = cv2.boxFilter(b, ksize=(wsize, wsize), ddepth=-1, normalize=True) # 计算 b 的均值
    q = meanA * img_i + meanB # 计算滤波后的图像
    return q

def MGFF_GRAY(img_r, img_v):
    """
    多尺度梯度融合 (MGFF) 算法，用于融合两张灰度图像。

    Args:
        img_r: 红外图像 (灰度图).
        img_v: 可见光图像 (灰度图).

    Returns:
        fused_img: 融合后的图像 (灰度图).
    """
    img_r_base_pre = img_r.astype(np.float32)/ 255 # 将红外图像归一化到 [0, 1] 范围
    img_v_base_pre = img_v.astype(np.float32)/ 255 # 将可见光图像归一化到 [0, 1] 范围
    img_r_detail = [] # 存储红外图像的细节层
    img_v_detail = [] # 存储可见光图像的细节层
    for i in range(ITERATION): # 进行多尺度分解
        #统一图像尺寸
        min_height = min(img_v_base_pre.shape[0],img_r_base_pre.shape[0])
        min_width = min(img_v_base_pre.shape[1],img_r_base_pre.shape[1])
        img_v_base_pre_resized = cv2.resize(img_v_base_pre,(min_width,min_height))
        img_r_base_pre_resized = cv2.resize(img_r_base_pre,(min_width,min_height))
        img_r_base_cur = guidedFilter(img_v_base_pre_resized, img_r_base_pre_resized, RADIUS, EPS) # 使用引导滤波提取红外图像的基础层
        img_v_base_cur = guidedFilter(img_r_base_pre_resized, img_v_base_pre_resized, RADIUS, EPS) # 使用引导滤波提取可见光图像的基础层
        img_r_detail.append(img_r_base_pre_resized - img_r_base_cur) # 计算红外图像的细节层
        img_v_detail.append(img_v_base_pre_resized - img_v_base_cur) # 计算可见光图像的细节层
        img_r_base_pre = img_r_base_cur # 更新红外图像的基础层
        img_v_base_pre = img_v_base_cur # 更新可见光图像的基础层
    img_base_fused = (img_r_base_pre + img_v_base_pre) / 2 # 融合基础层
    fused_img = img_base_fused # 初始化融合图像
    for  i in range(ITERATION-1, -1, -1): # 从最粗的尺度开始融合细节层
        weights = np.abs(img_r_detail[i]) / (np.abs(img_r_detail[i]) +np.abs(img_v_detail[i]) + 1e-8) # 计算权重，基于细节层梯度的绝对值, 防止除0
        fused_img += weights * img_r_detail[i] + (1-weights)* img_v_detail[i] # 加权融合细节层
    fused_img = cv2.normalize(fused_img, None, 0., 255., cv2.NORM_MINMAX, cv2.CV_8U) # 将融合后的图像归一化到 [0, 255] 范围
    #fused_img = cv2.convertScaleAbs(fused_img) # 将图像转换为 uint8 类型 ,cv2.normalize 已经做了，这一步可以省略
    return fused_img

def MGFF_RGB(img_r, img_v):
    """
    多尺度梯度融合 (MGFF) 算法，用于融合两张彩色图像。

    Args:
        img_r: 红外图像 (彩色图).
        img_v: 可见光图像 (彩色图).

    Returns:
        fused_img: 融合后的图像 (彩色图).
    """

    #统一图像尺寸
    min_height = min(img_v.shape[0],img_r.shape[0])
    min_width = min(img_v.shape[1],img_r.shape[1])
    img_v_resized = cv2.resize(img_v,(min_width,min_height))
    img_r_resized = cv2.resize(img_r,(min_width,min_height))

    r_R = img_r_resized[:,:,2] # 提取红外图像的红色通道
    r_G = img_r_resized[:,:,1] # 提取绿色通道
    r_B = img_r_resized[:,:,0] # 提取蓝色通道
    v_R = img_v_resized[:,:,2] # 提取可见光图像的红色通道
    v_G = img_v_resized[:,:,1] # 提取绿色通道
    v_B = img_v_resized[:,:,0] # 提取蓝色通道
    fused_R= MGFF_GRAY(r_R, v_R) # 融合红色通道
    fused_G= MGFF_GRAY(r_G, v_G) # 融合绿色通道
    fused_B= MGFF_GRAY(r_B, v_B) # 融合蓝色通道
    fused_img = cv2.merge((fused_B,fused_G,fused_R)) # 将融合后的通道堆叠成彩色图像
    return fused_img

def MGFF(r_path, v_path):
    """
    多尺度梯度融合 (MGFF) 算法的主函数。

    Args:
        r_path: 红外图像的路径.
        v_path: 可见光图像的路径.
    """
    img_r = cv2.imread(r_path) # 读取红外图像
    img_v = cv2.imread(v_path) # 读取可见光图像
    if not isinstance(img_r, np.ndarray): # 检查红外图像是否成功读取
        print("img_r is not an image")
        return
    if not isinstance(img_v, np.ndarray): # 检查可见光图像是否成功读取
        print("img_v is not an image")
        return

    #统一图像尺寸
    min_height = min(img_v.shape[0],img_r.shape[0])
    min_width = min(img_v.shape[1],img_r.shape[1])
    img_v_resized = cv2.resize(img_v,(min_width,min_height))
    img_r_resized = cv2.resize(img_r,(min_width,min_height))

    fused_img = None # 初始化融合图像
    if len(img_r_resized.shape)==2 or img_r_resized.shape[-1] ==1: # 如果红外图像是灰度图
        if len(img_v_resized.shape)==3:
            img_v_resized = cv2.cvtColor(img_v_resized, cv2.COLOR_BGR2GRAY) # 将可见光图像转换为灰度图
        fused_img = MGFF_GRAY(img_r_resized, img_v_resized) # 融合灰度图像
    else: # 如果红外图像是彩色图
         if len(img_v_resized.shape)==2:
            img_r_resized = cv2.cvtColor(img_r_resized, cv2.COLOR_BGR2GRAY) # 将红外图像转换为灰度图
            fused_img = MGFF_GRAY(img_r_resized, img_v_resized) # 融合灰度图像
         else:
            fused_img = MGFF_RGB(img_r_resized, img_v_resized) # 融合彩色图像

    cv2.imshow("fused image", fused_img) # 显示融合后的图像
    cv2.imwrite("fused_image_mgff.jpg", fused_img) # 保存融合后的图像
    cv2.waitKey(0) # 等待按键按下

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--IR",  default='IR.jpg', help="path to IR image", required=False) # 添加红外图像路径参数
    parser.add_argument("--VIS",  default='VIS.jpg', help="path to IR image", required=False) # 添加可见光图像路径参数
    a = parser.parse_args() # 解析命令行参数
    MGFF(a.IR, a.VIS) # 调用 MGFF 函数进行图像融合