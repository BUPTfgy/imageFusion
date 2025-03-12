import numpy as np
import cv2
import argparse
import math
from scipy.sparse import csr_matrix  # 导入但未使用

# 常量定义
ITERATION = 4        # 迭代次数
SIGMA_S = 2.         # 空间域标准差
SIGMA_R = 0.05       # 值域标准差
WLS_LAMBDA = 0.01    # WLS平滑的lambda参数


def guidedFilter(img_i, img_p, r, eps):
    """
    引导滤波器.

    Args:
        img_i: 引导图像.
        img_p: 待滤波图像.
        r: 滤波半径.
        eps: 正则化参数，防止除以零.

    Returns:
        滤波后的图像.
    """
    wsize = int(2*r)+1  # 窗口大小
    meanI = cv2.boxFilter(img_i, ksize=(wsize,wsize), ddepth=-1, normalize=True)  # 计算引导图像的均值
    meanP = cv2.boxFilter(img_p, ksize=(wsize,wsize), ddepth=-1, normalize=True)  # 计算待滤波图像的均值
    corrI = cv2.boxFilter(img_i*img_i, ksize=(wsize,wsize), ddepth=-1, normalize=True) # 计算引导图像的平方的均值
    corrIP = cv2.boxFilter(img_i*img_p, ksize=(wsize,wsize), ddepth=-1, normalize=True) # 计算引导图像和待滤波图像乘积的均值
    varI = corrI-meanI*meanI    # 计算引导图像的方差
    covIP = corrIP-meanI*meanP  # 计算引导图像和待滤波图像的协方差
    a = covIP/(varI+eps)        # 计算系数 a
    b = meanP-a*meanI          # 计算系数 b
    meanA = cv2.boxFilter(a, ksize=(wsize,wsize), ddepth=-1, normalize=True)   # 计算 a 的均值
    meanB = cv2.boxFilter(b, ksize=(wsize,wsize), ddepth=-1, normalize=True)   # 计算 b 的均值
    q = meanA*img_i+meanB      # 计算滤波后的图像
    return q

def RGF(img_r, img_v, sigma_s, sigma_r):
    """
    递归引导滤波 (Recursive Guided Filter).

    Args:
        img_r: 红外图像.
        img_v: 可见光图像.
        sigma_s: 空间域标准差.
        sigma_r: 值域标准差.

    Returns:
        滤波后的红外图像和可见光图像.
    """
    img_r_gs = cv2.GaussianBlur(img_r, (int(3*sigma_s)*2+1, int(3*sigma_s)*2+1), sigma_s)  # 高斯模糊
    img_v_gs = cv2.GaussianBlur(img_v, (int(3*sigma_s)*2+1, int(3*sigma_s)*2+1), sigma_s)  # 高斯模糊
    for i in range(ITERATION): # 迭代进行引导滤波
        img_r_gf = guidedFilter(img_r_gs, img_r, sigma_s, sigma_r*sigma_r) # 引导滤波
        img_v_gf = guidedFilter(img_v_gs, img_v, sigma_s, sigma_r*sigma_r) # 引导滤波
        img_r_gs = img_r_gf # 更新引导图像
        img_v_gs = img_v_gf # 更新引导图像
    return img_r_gf, img_v_gf # 返回滤波后的图像

def SOLVE_OPTIMAL(M, img_r_d, img_v_d, wls_lambda):
    """
    求解最优融合系数.  使用WLS优化细节层融合.

    Args:
        M: 初始融合系数.
        img_r_d: 红外图像的细节层.
        img_v_d: 可见光图像的细节层.
        wls_lambda: WLS平滑的lambda参数.

    Returns:
        优化后的融合系数.
    """
    m, n = img_r_d.shape  # 图像尺寸
    small_number = 0.0001  # 一个很小的数，防止除以零
    img_r_d_blur = np.abs(cv2.blur(img_r_d, (7,7))) # 对细节层进行模糊
    img_r_d_blur = 1 / (img_r_d_blur + small_number) # 计算模糊后细节层的倒数
    _row = np.arange(0, m*n)  # 行索引
    _col = np.arange(0, m*n)  # 列索引
    _data_A = img_r_d_blur.reshape(m*n)  # 重塑为一维数组
    _data_M = M.reshape(m*n)              # 重塑为一维数组
    _data_d2 = img_v_d.reshape(m*n)            # 重塑为一维数组
    _data_I = np.array([1]*(m*n))         # 创建全1数组
    # 计算优化后的融合系数 (使用了广播机制)
    D = (_data_M + wls_lambda*_data_A*_data_d2) / (_data_I + wls_lambda*_data_A)
    return D.reshape([m,n])  # 重塑为原始尺寸

def VSMWLS_GRAY(img_r, img_v):
    """
    VSMWLS (Visual Saliency Map and Weighted Least Squares) 灰度图像融合.
    多尺度分解，利用视觉显著性图计算权重，并使用WLS优化融合.

    Args:
        img_r: 红外图像 (灰度图).
        img_v: 可见光图像 (灰度图).

    Returns:
        融合后的灰度图像.
    """
    bases_r = []       # 红外图像的基础层
    bases_v = []       # 可见光图像的基础层
    details_r = []     # 红外图像的细节层
    details_v = []     # 可见光图像的细节层
    img_r_copy = img_r.astype(np.float64)/255  # 归一化到[0,1]并确保数据类型正确
    img_v_copy  = img_v.astype(np.float64)/255  # 归一化到[0,1]并确保数据类型正确
    bases_r.append(img_r_copy)  # 将原始图像加入基础层
    bases_v.append(img_v_copy)  # 将原始图像加入基础层
    sigma_s = SIGMA_S          # 空间域标准差初始化
    img_r_rgf = None         # 递归引导滤波后的红外图像
    img_v_rgf = None         # 递归引导滤波后的可见光图像
    for i in range(ITERATION-1): # 迭代进行多尺度分解
        img_r_rgf, img_v_rgf = RGF(img_r_copy, img_v_copy, sigma_s, SIGMA_R) # 递归引导滤波
        bases_r.append(img_r_rgf)  # 将滤波后的图像加入基础层
        bases_v.append(img_v_rgf)  # 将滤波后的图像加入基础层
        details_r.append(bases_r[i] - bases_r[i+1]) # 计算细节层
        details_v.append(bases_v[i] - bases_v[i+1]) # 计算细节层
        sigma_s *= 2         # 空间域标准差翻倍
    sigma_s = 2 # 重新设置空间域标准差
    # 对最后一层基础层进行高斯模糊
    img_r_base = cv2.GaussianBlur(bases_r[ITERATION-1], (int(3*sigma_s)*2+1, int(3*sigma_s)*2+1), sigma_s)
    img_v_base = cv2.GaussianBlur(bases_v[ITERATION-1], (int(3*sigma_s)*2+1, int(3*sigma_s)*2+1), sigma_s)
    details_r.append(bases_r[ITERATION-1]- img_r_base)  # 计算最后一层细节层
    details_v.append( bases_v[ITERATION-1]-img_v_base) # 计算最后一层细节层

    # 计算直方图
    img_r_hist = cv2.calcHist([img_r.astype(np.uint8)],[0],None,[256],[0,256])  #确保输入是uint8
    img_v_hist = cv2.calcHist([img_v.astype(np.uint8)],[0],None,[256],[0,256])  #确保输入是uint8
    img_r_hist_tab = np.zeros(256, np.float64)  # 红外图像直方图表
    img_v_hist_tab = np.zeros(256, np.float64)  # 可见光图像直方图表

    # 根据直方图计算视觉显著性
    for i in range(256):
        for j in range(256):
            img_r_hist_tab[i] = img_r_hist_tab[i]  + img_r_hist[j].item() *math.fabs(i-j) # 计算红外图像视觉显著性
            img_v_hist_tab[i] = img_v_hist_tab[i]  + img_v_hist[j].item() *math.fabs(i-j) # 计算可见光图像视觉显著性

    img_r_base_weights = img_r.astype(np.float64)  # 红外图像的基础层权重
    img_v_base_weights = img_v.astype(np.float64)  # 可见光图像的基础层权重

    # 将视觉显著性值赋予图像像素
    for i in range(256):
        img_r_base_weights[img_r == i] = img_r_hist_tab[i]
        img_v_base_weights[img_v == i] = img_v_hist_tab[i]

    # 归一化基础层权重
    img_r_b_weights = cv2.normalize(img_r_base_weights,  None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_64F)
    img_v_b_weights = cv2.normalize(img_v_base_weights, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_64F)

    # 融合基础层 (根据视觉显著性权重进行加权平均)
    img_fused_base = img_r_base * (0.5 +(img_r_b_weights - img_v_b_weights) /2) + img_v_base * (0.5 +(img_v_b_weights - img_r_b_weights) /2)

    w = int(3*sigma_s) # 高斯模糊的窗口大小
    #  根据细节层的绝对值大小来分配权重，进行融合
    C_0 = (details_v[0] < details_r[0]).astype(np.float64)
    C_0 = cv2.GaussianBlur(C_0, (2*w+1, 2*w+1), sigma_s) # 对权重进行高斯模糊
    img_fused_detail = C_0*details_r[0]+(1-C_0)*details_v[0] # 融合细节层
    # 迭代优化融合细节层 (使用WLS)
    for i in range(ITERATION-1, 0, -1):
        C_0 = (details_v[i] < details_r[i]).astype(np.float64)
        C_0 = cv2.GaussianBlur(C_0, (2*w+1, 2*w+1), sigma_s)  # 对权重进行高斯模糊
        M = C_0*details_r[i]+(1-C_0)*details_v[i]     # 融合细节层
        fused_di = SOLVE_OPTIMAL(M, details_r[i], details_v[i], WLS_LAMBDA)   # 使用WLS优化
        img_fused_detail += fused_di     # 累加细节层

    img_fused = img_fused_base + img_fused_detail  # 融合基础层和细节层
    img_fused = cv2.normalize(img_fused, None, 0, 255, cv2.NORM_MINMAX,dtype=cv2.CV_8U) # 归一化到 [0, 255]
    return img_fused # 转换为 uint8 类型

def VSMWLS_RGB(img_r, img_v):
    """
    VSMWLS RGB图像融合.  对RGB图像的每个通道分别进行灰度图像融合.

    Args:
        img_r: 红外图像 (RGB).
        img_v: 可见光图像 (RGB).

    Returns:
        融合后的RGB图像.
    """
    # 使用与输入图像相同的数据类型创建融合图像
    fused_img = np.zeros_like(img_r)

    r_R = img_r[:,:,2]  # 提取红外图像的R通道
    v_R = img_v[:,:,2]  # 提取可见光图像的R通道
    r_G = img_r[:,:,1]  # 提取红外图像的G通道
    v_G = img_v[:,:,1]  # 提取可见光图像的G通道
    r_B = img_r[:,:,0]  # 提取红外图像的B通道
    v_B = img_v[:,:,0]  # 提取可见光图像的B通道
    fused_R = VSMWLS_GRAY(r_R, v_R) # 融合R通道
    fused_G = VSMWLS_GRAY(r_G, v_G) # 融合G通道
    fused_B = VSMWLS_GRAY(r_B, v_B) # 融合B通道
    fused_img[:,:,2] = fused_R # 将融合后的R通道赋值给融合图像
    fused_img[:,:,1] = fused_G # G
    fused_img[:,:,0] = fused_B # B
    return fused_img # 返回融合后的图像

def resize_image(img, target_size):
    """等比例缩放图像到目标尺寸（保持宽高比）"""
    h, w = img.shape[:2]
    target_h, target_w = target_size

    # 计算缩放比例
    scale = min(target_h/h, target_w/w)
    new_h, new_w = int(h*scale), int(w*scale)

    # 使用高质量插值方法
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 创建目标尺寸画布并居中放置
    canvas = np.zeros((target_h, target_w, 3) if len(img.shape) == 3 else (target_h, target_w),
                      dtype=img.dtype)
    y_start = (target_h - new_h) // 2
    x_start = (target_w - new_w) // 2
    canvas[y_start:y_start+new_h, x_start:x_start+new_w] = resized
    return canvas

def VSMWLS(_rpath, _vpath):
    """
    主融合函数.  读取图像并根据图像类型调用相应的融合函数.

    Args:
        _rpath: 红外图像路径.
        _vpath: 可见光图像路径.
    """
    img_r = cv2.imread(_rpath,cv2.IMREAD_ANYDEPTH)  # 读取红外图像,保持原深度
    img_v = cv2.imread(_vpath,cv2.IMREAD_ANYDEPTH)  # 读取可见光图像,保持原深度
    # 检查图像是否成功读取
    if not isinstance(img_r, np.ndarray) :
        print('img_r is null')
        return
    if not isinstance(img_v, np.ndarray) :
        print('img_v is null')
        return
    # 检查图像大小是否相同
    if img_r.shape[0] != img_v.shape[0]  or img_r.shape[1] != img_v.shape[1]:
        print('size is not equal')
        # 统一尺寸处理
        target_size = (min(img_r.shape[0], img_v.shape[0]),
                       min(img_r.shape[1], img_v.shape[0]))

     # 等比例缩放
        img_r = resize_image(img_r, target_size)
        img_v = resize_image(img_v, target_size)
        print('统一尺寸处理完成')

    fused_img = None
    # 判断图像是灰度图像还是RGB图像，调用相应的融合函数
    if len(img_r.shape)  < 3 or img_r.shape[2] ==1:  # 红外图像是灰度图
        if len(img_v.shape)  < 3 or img_v.shape[-1] ==1:  # 可见光图像也是灰度图
            fused_img = VSMWLS_GRAY(img_r, img_v)
        else:  # 可见光图像是RGB图
            img_v_gray = cv2.cvtColor(img_v, cv2.COLOR_BGR2GRAY)  # 将可见光图转换为灰度图
            fused_img = VSMWLS_GRAY(img_r, img_v_gray)
    else:  # 红外图像是RGB图
        if len(img_v.shape)  < 3 or img_v.shape[-1] ==1: # 可见光图像是灰度图
            # 如果红外图像是RGB，可见光是灰度，将两个图像都转为RGB再融合
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
            img_v = cv2.cvtColor(img_v, cv2.COLOR_GRAY2BGR)
            fused_img = VSMWLS_RGB(img_r, img_v)
        else:  # 可见光图像也是RGB图
            fused_img = VSMWLS_RGB(img_r, img_v)
    # 显示融合后的图像
    cv2.imshow('fused image', fused_img)
    # 保存融合后的图像
    cv2.imwrite("fused_image_vsmwls.jpg", fused_img)
    # 等待按键
    cv2.waitKey(0)


if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 红外图像路径
    parser.add_argument('-r', type=str, default='IR.jpg' ,help='input IR image path', required=False)
    # 可见光图像路径
    parser.add_argument('-v', type=str, default= 'VIS.jpg',help='input Visible image path', required=False)
    # 解析参数
    args = parser.parse_args()
    # 调用融合函数
    VSMWLS(args.r, args.v)