import numpy as np  # 用于数值计算
import cv2  # 用于图像处理
import argparse  # 用于命令行参数解析
import math  # 用于数学函数（当前代码中未直接使用）

# 全局参数定义
LEVEL = 3        # 各向异性扩散滤波的迭代次数
LAMBDA = 1/8     # 扩散速率系数，控制扩散强度
K = 4            # 扩散系数中的梯度阈值参数，影响边缘保留程度

#各向异性扩散滤波（ADF）图像融合,优势是能在降低噪声的同时有效保留图像的边界和线条等特征信息

def ADF_ANISO(img):
    """各向异性扩散滤波函数，用于平滑图像并保留边缘"""
    # 使用反射填充处理图像边界（避免边缘效应）
    img_pad = np.pad(img, ((1, 1), (1, 1)), mode='reflect')
    # 创建与输入图像同形的零矩阵，存储处理结果
    img_aniso = np.zeros_like(img)
    h, w = img_pad.shape

    # 遍历每个像素（排除填充的边界）
    for i in range(1, h-1, 1):
        for j in range(1, w-1, 1):
            # 计算四个方向的梯度差
            cn_d = img_pad[i-1, j] - img_pad[i, j]  # 北方向梯度
            cs_d = img_pad[i+1, j] - img_pad[i, j]  # 南方向梯度
            ce_d = img_pad[i, j+1] - img_pad[i, j]  # 东方向梯度
            cw_d = img_pad[i, j-1] - img_pad[i, j]  # 西方向梯度

            # 计算各向异性扩散系数（Perona-Malik模型）
            c_n = np.exp(-pow(cn_d/K, 2)) * cn_d  # 北方向传导系数
            c_s = np.exp(-pow(cs_d/K, 2)) * cs_d  # 南方向传导系数
            c_e = np.exp(-pow(ce_d/K, 2)) * ce_d  # 东方向传导系数
            c_w = np.exp(-pow(cw_d/K, 2)) * cw_d  # 西方向传导系数

            # 更新当前像素值，结合各方向传导量
            img_aniso[i-1, j-1] = img_pad[i, j] + \
                LAMBDA * (c_n + c_s + c_e + c_w)
    return img_aniso


def ADF_GRAY(img_r, img_v):
    """灰度图像融合函数，处理两幅输入图像"""
    # 归一化图像到[0,1]范围
    img_r = img_r.astype(np.float32) / 255
    img_v = img_v.astype(np.float32) / 255

    # 初始化基图像（Base Layer）
    img_r_base = img_r.copy()
    img_v_base = img_v.copy()

    # 进行LEVEL次各向异性扩散滤波，提取基图像
    for i in range(LEVEL):
        img_r_base = ADF_ANISO(img_r_base)
        img_v_base = ADF_ANISO(img_v_base)

    # 计算细节层（原图 - 基图像）
    img_r_detail = img_r - img_r_base
    img_v_detail = img_v - img_v_base

    # 基图像融合（简单平均）
    fused_base = (img_r_base + img_v_base) / 2

    # 将细节层展平为一维数组（按列优先顺序）
    img_r_detail_fla = img_r_detail.flatten(order='F')
    img_v_detail_fla = img_v_detail.flatten(order='F')

    # 计算细节层的均值
    img_r_mean = np.mean(img_r_detail_fla)
    img_v_mean = np.mean(img_v_detail_fla)

    # 构建中心化的细节矩阵
    img_detail_mat = np.stack((img_r_detail_fla, img_v_detail_fla), axis=-1)
    img_detail_mat = img_detail_mat - np.array([img_r_mean, img_v_mean])

    # 计算协方差矩阵并进行特征分解
    img_detail_corr = np.matmul(img_detail_mat.T, img_detail_mat)
    eig_v, eig_vec = np.linalg.eig(img_detail_corr)

    # 按特征值大小降序排列特征向量
    sorted_indices = np.argsort(eig_v)
    eig_vec_ch = eig_vec[:, sorted_indices[::-1]]  # 取最大特征值对应的特征向量

    # 基于主成分分析的细节融合（权重按特征向量比例分配）
    weight_r = eig_vec_ch[0, 0] / (eig_vec_ch[0, 0] + eig_vec_ch[1, 0])
    weight_v = eig_vec_ch[1, 0] / (eig_vec_ch[0, 0] + eig_vec_ch[1, 0])
    fused_detail = img_r_detail * weight_r + img_v_detail * weight_v

    # 合并基图像和细节层
    fused_img = fused_detail + fused_base

    # 归一化到[0,255]并转换为uint8格式
    fused_img = cv2.normalize(fused_img, None, 0., 255., cv2.NORM_MINMAX)
    fused_img = cv2.convertScaleAbs(fused_img)
    return fused_img


def ADF_RGB(img_r, img_v):
    """RGB彩色图像融合函数，分通道处理"""
    # 分离各颜色通道（OpenCV使用BGR顺序）
    r_R = img_r[:, :, 2]
    r_G = img_r[:, :, 1]
    r_B = img_r[:, :, 0]
    v_R = img_v[:, :, 2]
    v_G = img_v[:, :, 1]
    v_B = img_v[:, :, 0]

    # 对各通道分别进行灰度融合
    fused_R = ADF_GRAY(r_R, v_R)
    fused_G = ADF_GRAY(r_G, v_G)
    fused_B = ADF_GRAY(r_B, v_B)

    # 合并通道并恢复BGR顺序
    fused_img = np.stack((fused_B, fused_G, fused_R), axis=-1)
    return fused_img


def resize_image(img, target_shape):  # 调整图像尺寸
    """Resize image to target shape."""
    return cv2.resize(img, (target_shape[1], target_shape[0]))


def ADF(r_path, v_path) -> np.ndarray:
    import os
    """主处理函数：适配Web服务的接口"""
    # 规范化路径
    r_path = os.path.normpath(r_path)
    v_path = os.path.normpath(v_path)
    
    print(f"尝试读取文件:\n红外: {r_path}\n可见光: {v_path}")
    
    # 增强文件存在检查
    if not os.path.exists(r_path):
        raise FileNotFoundError(f"红外图像路径不存在: {r_path}")
    if not os.path.exists(v_path):
        raise FileNotFoundError(f"可见光图像路径不存在: {v_path}")
    
    # 读取时指定flag确保正确读取
    img_r = cv2.imread(r_path, cv2.IMREAD_ANYCOLOR)
    img_v = cv2.imread(v_path, cv2.IMREAD_ANYCOLOR)


    # 读取输入图像
    img_r = cv2.imread(r_path)
    img_v = cv2.imread(v_path)

    # 增强错误处理
    if img_r is None:
        raise ValueError(f"无法读取可见光图像: {r_path}")
    if img_v is None:
        raise ValueError(f"无法读取红外图像: {v_path}")

    # 统一图像尺寸处理逻辑
    if img_r.shape != img_v.shape:
        # 自动选择最大尺寸作为目标尺寸
        target_shape = (max(img_r.shape[0], img_v.shape[0]),
                       max(img_r.shape[1], img_v.shape[1]))
        img_r = resize_image(img_r, target_shape)
        img_v = resize_image(img_v, target_shape)

    # 优化通道处理逻辑
    if len(img_r.shape) == 2:  # 灰度图像
        img_r = cv2.cvtColor(img_r, cv2.COLOR_GRAY2BGR)
    if len(img_v.shape) == 2:  # 灰度图像
        img_v = cv2.cvtColor(img_v, cv2.COLOR_GRAY2BGR)

    # 统一转换为BGR格式处理
    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)  # 统一色彩空间
    img_v = cv2.cvtColor(img_v, cv2.COLOR_BGR2RGB)

    # 执行融合处理
    try:
        if img_r.shape[2] == 3:  # 彩色图像
            fused_img = ADF_RGB(img_r, img_v)
        else:  # 其他情况
            fused_img = ADF_GRAY(cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(img_v, cv2.COLOR_BGR2GRAY))
    except Exception as e:
        raise RuntimeError(f"图像融合失败: {str(e)}")

    # 转换回BGR格式用于保存
    fused_img = cv2.cvtColor(fused_img, cv2.COLOR_RGB2BGR)
    return fused_img
