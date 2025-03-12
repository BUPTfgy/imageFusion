# 导入依赖库
import numpy as np
import cv2

# 全局参数配置
cov_wsize = 5    # 协方差窗口尺寸
sigmas = 1.8     # 空间高斯核标准差
sigmar = 25      # 值域高斯核标准差
ksize = 11       # 高斯核尺寸

def gaussian_kernel_2d_opencv(kernel_size=11, sigma=1.8):
    """生成二维高斯核（保持不变）"""
    kx = cv2.getGaussianKernel(kernel_size, sigma)
    ky = cv2.getGaussianKernel(kernel_size, sigma)
    return np.multiply(kx, np.transpose(ky))

def bilateralFilterEx(img_r, img_v):
    """交叉双边滤波（优化性能）"""
    win_size = ksize // 2
    # 使用统一填充处理
    img_r_pad = cv2.copyMakeBorder(img_r, win_size, win_size, win_size, win_size, cv2.BORDER_REFLECT)
    img_v_pad = cv2.copyMakeBorder(img_v, win_size, win_size, win_size, win_size, cv2.BORDER_REFLECT)
    
    # 预计算高斯核
    gk = gaussian_kernel_2d_opencv()
    
    # 向量化处理提升性能
    img_r_cbf = np.zeros_like(img_r, dtype=np.float32)
    img_v_cbf = np.zeros_like(img_v, dtype=np.float32)
    
    for i in range(win_size, img_r.shape[0]+win_size):
        for j in range(win_size, img_r.shape[1]+win_size):
            # 提取ROI区域
            roi_r = img_r_pad[i-win_size:i+win_size+1, j-win_size:j+win_size+1]
            roi_v = img_v_pad[i-win_size:i+win_size+1, j-win_size:j+win_size+1]
            
            # 优化权重计算
            delta_v = roi_v - roi_v[win_size, win_size]
            weight_r = np.exp(-(delta_v**2)/(2*sigmar**2)) * gk
            
            # 避免重复计算
            sumr1 = np.sum(weight_r)
            sumr2 = np.sum(roi_r.astype(np.float32) * weight_r)
            
            img_r_cbf[i-win_size, j-win_size] = sumr2 / (sumr1 + 1e-6)
            img_v_cbf[i-win_size, j-win_size] = roi_v[win_size, win_size]
    
    return (img_r.astype(np.float32) - img_r_cbf, 
            img_v.astype(np.float32) - img_v_cbf)

def CBF_WEIGHTS(img_r_d, img_v_d):
    """权重计算（优化矩阵运算）"""
    win_size = cov_wsize // 2
    pad_size = [(win_size, win_size), (win_size, win_size)]
    
    # 使用统一填充方式
    img_r_d_pad = np.pad(img_r_d, pad_size, mode='reflect')
    img_v_d_pad = np.pad(img_v_d, pad_size, mode='reflect')
    
    # 预计算窗口索引
    weights_r = np.zeros_like(img_r_d)
    weights_v = np.zeros_like(img_v_d)
    
    # 向量化计算
    for i in range(win_size, img_r_d.shape[0]+win_size):
        for j in range(win_size, img_r_d.shape[1]+win_size):
            # 提取局部窗口
            npt_r = img_r_d_pad[i-win_size:i+win_size+1, j-win_size:j+win_size+1]
            npt_v = img_v_d_pad[i-win_size:i+win_size+1, j-win_size:j+win_size+1]
            
            # 优化协方差计算
            cov_r = np.cov(npt_r, rowvar=False)
            cov_v = np.cov(npt_v, rowvar=False)
            
            weights_r[i-win_size, j-win_size] = np.trace(cov_r)
            weights_v[i-win_size, j-win_size] = np.trace(cov_v)
    
    return weights_r, weights_v

def resize_image(img, target_shape):  # 调整图像尺寸
    """Resize image to target shape."""
    return cv2.resize(img, (target_shape[1], target_shape[0]))

def CBF_GRAY(img_r, img_v):
    """灰度融合（增加类型检查）"""
    # 输入验证
    if img_r.dtype != np.uint8 or img_v.dtype != np.uint8:
        raise ValueError("输入图像必须为uint8格式")
    
    # 统一转换为float32
    img_r = img_r.astype(np.float32)
    img_v = img_v.astype(np.float32)
    
    # 执行融合
    img_r_d, img_v_d = bilateralFilterEx(img_r, img_v)
    weights_r, weights_v = CBF_WEIGHTS(img_r_d, img_v_d)
    
    # 加权融合
    fused = (img_r * weights_r + img_v * weights_v) / (weights_r + weights_v + 1e-6)
    return cv2.convertScaleAbs(fused)

def CBF_RGB(img_r, img_v):
    """改进的彩色图像融合"""
    # 分通道处理
    fused_channels = []
    for ch in range(3):
        # 提取单通道并保持二维结构
        r_ch = img_r[:, :, ch][..., np.newaxis]
        v_ch = img_v[:, :, ch][..., np.newaxis]
        
        # 执行单通道融合
        fused_ch = CBF_GRAY(r_ch.squeeze(), v_ch.squeeze())
        fused_channels.append(fused_ch)
    
    # 合并通道
    return np.stack(fused_channels, axis=-1)

def CBF(r_path, v_path) -> np.ndarray:
    import os
    """主接口函数（适配Web服务）"""
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

    # 执行融合处理
    try:
        if img_r.shape[2] == 3:  # 彩色图像
            fused_img = CBF_RGB(img_r, img_v)
        else:  # 其他情况
            fused_img = CBF_GRAY(cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(img_v, cv2.COLOR_BGR2GRAY))
    except Exception as e:
        raise RuntimeError(f"图像融合失败: {str(e)}")

    # 转换回BGR格式用于保存
    return fused_img  # 不需要normalize了，CBF_RGB已经输出了uint8格式，确保返回值的正确性