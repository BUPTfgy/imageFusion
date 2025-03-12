import numpy as np
import cv2
import argparse

def TIF_GRAY(img_r, img_v):
    """
    灰度图像融合函数。

    Args:
        img_r: 红外图像 (灰度图).
        img_v: 可见光图像 (灰度图).

    Returns:
        融合后的灰度图像.
    """
    # 对红外图像和可见光图像进行均值模糊，平滑图像
    img_r_blur = cv2.blur(img_r, (35,35))  # 35x35的均值模糊
    img_v_blur = cv2.blur(img_v, (35,35))  # 35x35的均值模糊
    # 对红外图像和可见光图像进行中值滤波，去除噪声
    img_r_median = cv2.medianBlur(img_r, 3) # 3x3的中值滤波
    img_v_median = cv2.medianBlur(img_v, 3) # 3x3的中值滤波
    # 计算红外图像和可见光图像的细节层 (原图 - 模糊图)
    img_r_detail = img_r*1. - img_r_blur*1.
    img_v_detail = img_v*1. - img_v_blur*1.
    # 计算红外图像和可见光图像的显著性图 (中值滤波图与均值滤波图的差异的平方)
    img_r_the = cv2.pow(cv2.absdiff(img_r_median,img_r_blur), 2)
    img_v_the = cv2.pow(cv2.absdiff(img_v_median ,img_v_blur),2)
    # 计算红外图像和可见光图像的权重 (基于显著性图)
    img_r_weight = cv2.divide(img_r_the*1.,img_r_the*1.+img_v_the*1.+0.000001) # 避免除以零
    img_v_weight = 1- img_r_weight # 可见光权重 = 1 - 红外权重
    # 融合基础层 (均值模糊图的平均)
    img_base_fused = (img_r_blur*1.  + img_v_blur*1.) / 2
    # 融合细节层 (加权平均)
    img_detail_fused = img_r_weight * img_r_detail + img_v_weight * img_v_detail
    # 融合图像 (基础层 + 细节层)
    img_fused_tmp = (img_base_fused  + img_detail_fused).astype(np.int32) # 将数据类型转换为int32
    # 第一种方法：将小于0的值设置为0，大于255的值设置为255
    img_fused_tmp[img_fused_tmp<0] = 0
    img_fused_tmp[img_fused_tmp>255]=255
    # 第二种方法：使用minmax方法将值更改为[0,255]
    #cv2.normalize(img_fused_tmp,img_fused_tmp,0,255,cv2.NORM_MINMAX)
    # 将图像数据类型转换回uint8
    img_fused = cv2.convertScaleAbs(img_fused_tmp)
    return img_fused
    

def TIF_RGB(img_r, img_v):
    """
    RGB图像融合函数。 对RGB图像的每个通道分别进行灰度图像融合.

    Args:
        img_r: 红外图像 (RGB).
        img_v: 可见光图像 (RGB).

    Returns:
        融合后的RGB图像.
    """
    fused_img = np.ones_like(img_r) # 创建一个与输入图像大小相同的数组
    # 分别提取红外图像和可见光图像的R, G, B通道
    r_R = img_r[:,:,2]
    v_R = img_v[:,:,2]
    r_G = img_r[:,:,1]
    v_G = img_v[:,:,1]
    r_B = img_r[:,:,0]
    v_B = img_v[:,:,0]
    # 对每个通道进行灰度图像融合
    fused_R = TIF_GRAY(r_R, v_R)
    fused_G = TIF_GRAY(r_G, v_G)
    fused_B = TIF_GRAY(r_B, v_B)
    # 将融合后的通道组合成RGB图像
    fused_img[:,:,2] = fused_R
    fused_img[:,:,1] = fused_G
    fused_img[:,:,0] = fused_B
    return fused_img



def TIF(_rpath, _vpath):
    """
    主融合函数。 读取图像并根据图像类型调用相应的融合函数.

    Args:
        _rpath: 红外图像路径.
        _vpath: 可见光图像路径.
    """
    # 读取红外图像和可见光图像
    img_r = cv2.imread(_rpath)
    img_v = cv2.imread(_vpath)
    # 检查图像是否成功读取
    if not isinstance(img_r, np.ndarray) :
        print('img_r is null')
        return
    if not isinstance(img_v, np.ndarray) :
        print('img_v is null')
        return
    # 检查图像大小是否相同
    if img_r.shape[0] != img_v.shape[0]  or img_r.shape[1] != img_v.shape[1]:
        print('size is not equal, resizing images')
        # Resize images to the same size
        img_v = cv2.resize(img_v, (img_r.shape[1], img_r.shape[0]), interpolation=cv2.INTER_AREA)
        

    fused_img = None
    # 判断图像是灰度图像还是RGB图像，并调用相应的融合函数
    if len(img_r.shape)  < 3 or img_r.shape[2] ==1: # 红外图像是灰度图
        if len(img_v.shape)  < 3 or img_v.shape[-1] ==1: # 可见光图像也是灰度图
            fused_img = TIF_GRAY(img_r, img_v)
        else: # 可见光图像是RGB图
            img_v_gray = cv2.cvtColor(img_v, cv2.COLOR_BGR2GRAY) # 将可见光图转换为灰度图
            fused_img = TIF_GRAY(img_r, img_v_gray)
    else: # 红外图像是RGB图
        if len(img_v.shape)  < 3 or img_v.shape[-1] ==1: # 可见光图像是灰度图
            img_r_gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY) # 将红外图转换为灰度图
            fused_img = TIF_GRAY(img_r_gray, img_v)
        else: # 可见光图像也是RGB图
            fused_img = TIF_RGB(img_r, img_v)
    # 显示融合后的图像
    cv2.imshow('fused image', fused_img)
    # 保存融合后的图像
    cv2.imwrite("fused_image_tif.jpg", fused_img)
    # 等待按键
    cv2.waitKey(0)


if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加参数：红外图像路径
    parser.add_argument('-r', type=str, default='IR.jpg' ,help='input IR image path', required=False)
    # 添加参数：可见光图像路径
    parser.add_argument('-v', type=str, default= 'VIS.jpg',help='input Visible image path', required=False)
    # 解析参数
    args = parser.parse_args()
    # 调用融合函数
    TIF(args.r, args.v)