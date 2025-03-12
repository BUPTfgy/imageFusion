import numpy as np
import cv2
import argparse
 
# 定义全局常量，用于指导滤波器的参数
R_G = 5  # 全局半径参数，用于高斯滤波和权重计算
D_G = 5  # 细节半径参数（未使用）
 
# 定义引导滤波器函数
def guidedFilter(img_i, img_p, r, eps):
    """
    引导滤波器实现
    参数：
        img_i: 引导图像 (I)
        img_p: 输入图像 (P)
        r: 滤波器半径
        eps: 正则化参数，防止除零
    返回：
        滤波后的图像
    """
    wsize = int(2 * r) + 1  # 根据半径计算窗口大小
    # 计算均值
    meanI = cv2.boxFilter(img_i, ksize=(wsize, wsize), ddepth=-1, normalize=True)  # 引导图像的均值
    meanP = cv2.boxFilter(img_p, ksize=(wsize, wsize), ddepth=-1, normalize=True)  # 输入图像的均值
    # 计算相关
    corrI = cv2.boxFilter(img_i * img_i, ksize=(wsize, wsize), ddepth=-1, normalize=True)  # I的平方的均值
    corrIP = cv2.boxFilter(img_i * img_p, ksize=(wsize, wsize), ddepth=-1, normalize=True)  # I和P的乘积的均值
    # 计算方差和协方差
    varI = corrI - meanI * meanI  # 引导图像的方差
    covIP = corrIP - meanI * meanP  # 引导图像和输入图像的协方差
    # 计算系数 a 和 b
    a = covIP / (varI + eps)  # 线性系数
    b = meanP - a * meanI  # 偏移量
    # 对系数 a 和 b 进行均值滤波
    meanA = cv2.boxFilter(a, ksize=(wsize, wsize), ddepth=-1, normalize=True)
    meanB = cv2.boxFilter(b, ksize=(wsize, wsize), ddepth=-1, normalize=True)
    # 计算输出结果
    q = meanA * img_i + meanB
    return q
 
# 灰度图像融合函数
def GFF_GRAY(img_r, img_v):
    """
    灰度图像融合
    参数：
        img_r: 红外图像 (灰度)
        img_v: 可见光图像 (灰度)
    返回：
        融合后的灰度图像
    """
    img_r = img_r * 1. / 255  # 归一化
    img_v = img_v * 1. / 255  # 归一化
    # 对图像进行均值模糊，得到基础亮度分量
    img_r_blur = cv2.blur(img_r, (31, 31))
    img_v_blur = cv2.blur(img_v, (31, 31))
    # 提取细节分量
    img_r_detail = img_r.astype(float) - img_r_blur.astype(float)
    img_v_detail = img_v.astype(float) - img_v_blur.astype(float)
    # 计算拉普拉斯金字塔（边缘特征提取）
    img_r_lap = cv2.Laplacian(img_r.astype(float), -1, ksize=3)
    img_v_lap = cv2.Laplacian(img_v.astype(float), -1, ksize=3)
    # 计算权重图
    win_size = 2 * R_G + 1
    s1 = cv2.GaussianBlur(np.abs(img_r_lap), (win_size, win_size), R_G)  # 红外图像边缘特征的模糊
    s2 = cv2.GaussianBlur(np.abs(img_v_lap), (win_size, win_size), R_G)  # 可见光图像边缘特征的模糊
    p1 = np.zeros_like(img_r)  # 红外图像的权重掩膜
    p2 = np.zeros_like(img_r)  # 可见光图像的权重掩膜
    p1[s1 > s2] = 1  # 红外图像边缘强度更大的区域赋值为1
    p2[s1 <= s2] = 1  # 可见光图像边缘强度更大的区域赋值为1
    # 使用引导滤波器计算权重
    w1_b = guidedFilter(p1, img_r.astype(float), 45, 0.3)  # 红外图像基础分量的权重
    w2_b = guidedFilter(p2, img_v.astype(float), 45, 0.3)  # 可见光图像基础分量的权重
    w1_d = guidedFilter(p1, img_r.astype(float), 7, 0.000001)  # 红外图像细节分量的权重
    w2_d = guidedFilter(p2, img_v.astype(float), 7, 0.000001)  # 可见光图像细节分量的权重
    # 权重归一化
    w1_b_w = w1_b / (w1_b + w2_b + 1e-8) # 添加一个极小值，防止除零
    w2_b_w = w2_b / (w1_b + w2_b + 1e-8)
    w1_d_w = w1_d / (w1_d + w2_d + 1e-8)
    w2_d_w = w2_d / (w1_d + w2_d + 1e-8)
    # 融合图像
    fused_b = w1_b_w * img_r_blur + w2_b_w * img_v_blur  # 融合基础分量
    fused_d = w1_d_w * img_r_detail + w2_d_w * img_v_detail  # 融合细节分量
    img_fused = fused_b + fused_d  # 合并基础分量和细节分量
    img_fused = cv2.normalize(img_fused, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # 图像归一化到0-255, 并转换为uint8
    return img_fused  # 返回类型为 uint8
 
# RGB图像融合函数
def GFF_RGB(img_r, img_v):
    """
    RGB图像融合
    参数：
        img_r: 红外图像 (RGB)
        img_v: 可见光图像 (RGB)
    返回：
        融合后的RGB图像
    """
    # 统一图像尺寸
    min_height = min(img_v.shape[0], img_r.shape[0])
    min_width = min(img_v.shape[1], img_r.shape[1])
    img_v_resized = cv2.resize(img_v, (min_width, min_height))
    img_r_resized = cv2.resize(img_r, (min_width, min_height))

    fused_img = np.zeros_like(img_r_resized, dtype=np.uint8)  # 初始化融合图像, 确保数据类型为 uint8
    # 分别对R、G、B三个通道进行灰度融合
    r_R = img_r_resized[:, :, 2]
    v_R = img_v_resized[:, :, 2]
    r_G = img_r_resized[:, :, 1]
    v_G = img_v_resized[:, :, 1]
    r_B = img_r_resized[:, :, 0]
    v_B = img_v_resized[:, :, 0]
    fused_R = GFF_GRAY(r_R, v_R)  # 融合红色通道
    fused_G = GFF_GRAY(r_G, v_G)  # 融合绿色通道
    fused_B = GFF_GRAY(r_B, v_B)  # 融合蓝色通道
    # 合并三个通道
    fused_img[:, :, 2] = fused_R
    fused_img[:, :, 1] = fused_G
    fused_img[:, :, 0] = fused_B
    return fused_img
 
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

# 主图像融合函数
def GFF(_rpath, _vpath):
    """
    主函数，读取两幅图像并进行融合
    参数：
        _rpath: 红外图像路径
        _vpath: 可见光图像路径
    """
    img_r = cv2.imread(_rpath)  # 读取红外图像
    img_v = cv2.imread(_vpath)  # 读取可见光图像
    # 检查图像是否读取成功
    if not isinstance(img_r, np.ndarray):
        print('img_r is null')
        return
    if not isinstance(img_v, np.ndarray):
        print('img_v is null')
        return
    # 检查图像尺寸是否一致
    if img_r.shape[0] != img_v.shape[0]  or img_r.shape[1] != img_v.shape[1]:
        print('size is not equal')
        # 统一尺寸处理
        target_size = (min(img_r.shape[0], img_v.shape[0]),
                       min(img_r.shape[1], img_v.shape[1]))

        # 等比例缩放
        img_r = resize_image(img_r, target_size)
        img_v = resize_image(img_v, target_size)
        print('统一尺寸处理完成')

    fused_img = None
    if len(img_r.shape)  < 3 or img_r.shape[2] ==1:
        if len(img_v.shape)  < 3 or img_v.shape[-1] ==1:
            fused_img = GFF_GRAY(img_r, img_v)
        else:
            img_v_gray = cv2.cvtColor(img_v, cv2.COLOR_BGR2GRAY)
            fused_img = GFF_GRAY(img_r, img_v_gray)
    else:
        if len(img_v.shape)  < 3 or img_v.shape[-1] ==1:
            img_r_gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            fused_img = GFF_GRAY(img_r_gray, img_v)
        else:
            fused_img = GFF_RGB(img_r, img_v)
    cv2.imshow('fused image', fused_img)
    cv2.imwrite("fused_image_gff.jpg", fused_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', type=str, default='IR.jpg' ,help='input IR image path', required=False)
    parser.add_argument('-v', type=str, default= 'VIS.jpg',help='input Visible image path', required=False)
    args = parser.parse_args()
    GFF(args.r, args.v)