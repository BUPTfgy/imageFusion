import numpy as np
import cv2
import argparse
import math

# 引入ADF_ANISO 函数在 ADF.py 中定义
from ADF import ADF_ANISO

# 控制迭代次数的常量，可根据需要进行调整
LEVEL = 5

def GFCE_GRAY(img_r, img_v):
    # 图像转换成浮点数并归一化到 [0, 1]
    img_r = img_r.astype(np.float64) / 255
    img_v = img_v.astype(np.float64) / 255
    
    # 创建基础图像副本
    img_r_base = img_r[:, :]
    img_v_base = img_v[:, :]
    
    # 迭代应用 ADF_ANISO 函数
    for i in range(LEVEL):
        img_r_base = ADF_ANISO(img_r_base)
        img_v_base = ADF_ANISO(img_v_base)
    
    # 计算细节图像
    img_r_detail = img_r - img_r_base
    img_v_detail = img_v - img_v_base
    
    # 融合基础图像
    fused_base = (img_r_base + img_v_base) / 2
    
    # 将细节图像展平
    img_r_detail_fla = img_r_detail.flatten(order='F')
    img_v_detail_fla = img_v_detail.flatten(order='F')
    
    # 计算细节图像的均值
    img_r_mean = np.mean(img_r_detail_fla)
    img_v_mean = np.mean(img_v_detail_fla)
    
    # 构建细节矩阵并去除均值
    img_detail_mat = np.stack((img_r_detail_fla, img_v_detail_fla), axis=-1)
    img_detail_mat = img_detail_mat - np.array((img_r_mean, img_v_mean))
    
    # 计算细节矩阵的相关矩阵
    img_detail_corr = np.matmul(img_detail_mat.transpose(), img_detail_mat)
    
    # 计算特征值和特征向量
    eig_v, eig_vec = np.linalg.eig(img_detail_corr)
    
    # 对特征值排序并选择主成分
    sorted_indices = np.argsort(eig_v)
    eig_vec_ch = eig_vec[:, sorted_indices[:-1-1:-1]]
    
    # 融合细节图像
    fused_detail = (img_r_detail * eig_vec_ch[0][0] / (eig_vec_ch[0][0] + eig_vec_ch[1][0]) +
                    img_v_detail * eig_vec_ch[1][0] / (eig_vec_ch[0][0] + eig_vec_ch[1][0]))
    
    # 生成最终融合图像
    fused_img = fused_detail + fused_base
    fused_img = cv2.normalize(fused_img, None, 0., 255., cv2.NORM_MINMAX)
    fused_img = cv2.convertScaleAbs(fused_img)
    
    return fused_img

def GFCE_RGB(img_r, img_v):
    # 分别处理 RGB 通道
    r_R = img_r[:, :, 2]
    r_G = img_r[:, :, 1]
    r_B = img_r[:, :, 0]
    v_R = img_v[:, :, 2]
    v_G = img_v[:, :, 1]
    v_B = img_v[:, :, 0]
    
    # 融合每个通道
    fused_R = GFCE_GRAY(r_R, v_R)
    fused_G = GFCE_GRAY(r_G, v_G)
    fused_B = GFCE_GRAY(r_B, v_B)
    
    # 合并融合后的通道
    fused_img = np.stack((fused_B, fused_G, fused_R), axis=-1)
    
    return fused_img

def GFCE(r_path, v_path):
    # 读取图像
    img_r = cv2.imread(r_path)
    img_v = cv2.imread(v_path)
    
    # 检查图像是否读取成功
    if not isinstance(img_r, np.ndarray):
        print("img_r is not an image")
        return
    if not isinstance(img_v, np.ndarray):
        print("img_v is not an image")
        return
    
    fused_img = None
    
    # 根据图像类型选择处理方式
    if len(img_r.shape) == 2 or img_r.shape[-1] == 1:
        if img_r.shape[-1] == 3:
            img_v = cv2.cvtColor(img_v, cv2.COLOR_BGR2GRAY)
        fused_img = GFCE_GRAY(img_r, img_v)
    else:
        if img_r.shape[-1] == 3:
            #图像resize到相同大小
            img_v = cv2.resize(img_v, (img_r.shape[1], img_r.shape[0]), interpolation=cv2.INTER_AREA)
            fused_img = GFCE_RGB(img_r, img_v)
        else:
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            fused_img = GFCE_GRAY(img_r, img_v)
    
    # 显示和保存融合后的图像
    cv2.imshow("fused image", fused_img)
    cv2.imwrite("fused_image_gfce.jpg", fused_img)
    cv2.waitKey(0)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--IR", default='IR.jpg', help="path to IR image", required=False)
    parser.add_argument("--VIS", default='VIS.jpg', help="path to VIS image", required=False)
    a = parser.parse_args()
    GFCE(a.IR, a.VIS)