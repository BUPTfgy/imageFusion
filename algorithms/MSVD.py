import numpy as np
import cv2
import argparse
import math

LEVEL = 1  # 分解的层数，默认为1

def MSVD_SVD(img, h, w):
    '''
    使用MSVD分解图像。

    Args:
        img: 输入图像（灰度图）.
        h: 图像的高度.
        w: 图像的宽度.

    Returns:
        t_d[0]: 分解后的第一个子带.
        t_d[1]: 分解后的第二个子带.
        t_d[2]: 分解后的第三个子带.
        t_d[3]: 分解后的第四个子带.
        eig_vec_ch: 特征向量.
    '''
    '''
    以下注释掉的部分使用了直接索引，效率较低。
    tl = img[0:h//2, 0:w//2]             #top left
    tr = img[0:h//2, w//2:]             #top right
    bl = img[h//2:, 0:w//2]             #bottom left
    br = img[h//2:, w//2:]            #bottom right
    tl_r = tl.flatten(order="F")
    tr_r = tr.flatten(order="F")
    bl_r = bl.flatten(order="F")
    br_r = br.flatten(order="F")
    t_x = np.stack((tl_r, bl_r, tr_r , br_r), axis=0)
    '''
    nh = h // 2  # 计算子带的高度
    nw = w // 2  # 计算子带的宽度
    t_x = np.zeros((4, nh*nw)) # 初始化矩阵，用于存储分解后的子带
    # 循环遍历每个2x2的块，将其展平并存储到t_x中
    for j in range(nw):
        for i in range(nh):
            t_x[:, j*nh+i] = img[2*i:2*(i+1), 2*j:2*(j+1)].flatten(order="F")
    x1 = np.matmul(t_x, t_x.transpose())  # 计算t_x和其转置的乘积
    eig_v, eig_vec = np.linalg.eig(x1)  # 计算x1的特征值和特征向量
    sorted_indices = np.argsort(eig_v)  # 对特征值进行排序，获取排序后的索引
    eig_vec_ch = eig_vec[:, sorted_indices[:-4-1:-1]]  # 选择前4个最大的特征值对应的特征向量
    eig_vec_ch_t = eig_vec_ch.transpose()  # 对特征向量进行转置
    t_d = np.matmul(eig_vec_ch_t, t_x)  # 计算分解后的子带
    return t_d[0], t_d[1], t_d[2], t_d[3], eig_vec_ch # 返回分解后的子带和特征向量


def MSVD_GRAY(img_r, img_v):
    '''
    多尺度奇异值分解（MSVD）算法，用于融合两张灰度图像。

    Args:
        img_r: 红外图像 (灰度图).
        img_v: 可见光图像 (灰度图).

    Returns:
        fused_img: 融合后的图像 (灰度图).
    '''
    img_r = img_r.astype(np.float64)/255  # 将红外图像转换为float类型并归一化到[0,1]
    img_v = img_v.astype(np.float64)/255  # 将可见光图像转换为float类型并归一化到[0,1]
    h, w = img_r.shape  # 获取图像的形状
    nh = (h // int(pow(2, LEVEL)) )* int(pow(2, LEVEL))  # 计算调整后的高度，使其可以被2的LEVEL次方整除
    nw = (w // int(pow(2, LEVEL)) )* int(pow(2, LEVEL))  # 计算调整后的宽度，使其可以被2的LEVEL次方整除
    h, w = nh, nw  # 更新高度和宽度
    img_r = cv2.resize(img_r, (w,h))  # 调整红外图像的大小
    img_v = cv2.resize(img_v, (w,h))  # 调整可见光图像的大小
    fer_rl = None  # 初始化分解后的红外图像
    fer_vl = None  # 初始化分解后的可见光图像
    theta_rl = []  # 初始化红外图像分解后的子带
    theta_vl = []  # 初始化可见光图像分解后的子带
    u_rl = []  # 初始化红外图像的特征向量
    u_vl = []  # 初始化可见光图像的特征向量
    img_r_copy = img_r[:,:]  # 复制红外图像
    img_v_copy = img_v[:,:]  # 复制可见光图像
    for i in range(LEVEL): # 进行多尺度分解
        fei_r, theta_r_1, theta_r_2, theta_r_3, u_r = MSVD_SVD(img_r_copy, h, w) # 对红外图像进行MSVD分解
        fei_v, theta_v_1, theta_v_2, theta_v_3, u_v = MSVD_SVD(img_v_copy,  h, w) # 对可见光图像进行MSVD分解
        h = h // 2  # 降低高度
        w = w // 2  # 降低宽度
        img_r_copy = fei_r.reshape((w,h)).transpose() # 调整红外图像的大小
        img_v_copy = fei_v.reshape((w,h)).transpose() # 调整可见光图像的大小
        fer_rl = fei_r[:]  # 存储分解后的红外图像
        fer_vl = fei_v[:]  # 存储分解后的可见光图像
        theta_rl.append((theta_r_1, theta_r_2, theta_r_3))  # 存储红外图像分解后的子带
        theta_vl.append((theta_v_1, theta_v_2, theta_v_3))  # 存储可见光图像分解后的子带
        u_rl.append(u_r)  # 存储红外图像的特征向量
        u_vl.append(u_v)  # 存储可见光图像的特征向量
    fused_fei = None # 初始化融合后的图像
    for i in range(LEVEL-1,-1,-1): # 从最低层开始融合
        if i == LEVEL-1: # 如果是最低层
            fused_fei = (fer_rl+fer_vl)/ 2 # 直接取平均
        theta_0_effi = (np.abs(theta_vl[i][0])>np.abs(theta_rl[i][0])).astype(np.float64) # 计算第一个子带的融合系数
        theta_1_effi = (np.abs(theta_vl[i][1])>np.abs(theta_rl[i][1])).astype(np.float64) # 计算第二个子带的融合系数
        theta_2_effi = (np.abs(theta_vl[i][2])>np.abs(theta_rl[i][2])).astype(np.float64) # 计算第三个子带的融合系数
        fused_theta_0=theta_0_effi*theta_vl[i][0]+(1-theta_0_effi)*theta_rl[i][0] # 融合第一个子带
        fused_theta_1=theta_1_effi*theta_vl[i][1]+(1-theta_1_effi)*theta_rl[i][1] # 融合第二个子带
        fused_theta_2=theta_2_effi*theta_vl[i][2]+(1-theta_2_effi)*theta_rl[i][2] # 融合第三个子带
        fused_u = (u_rl[i]+u_vl[i]) / 2 # 融合特征向量
        fused_fei = np.stack((fused_fei, fused_theta_0, fused_theta_1, fused_theta_2), axis = 0) # 将融合后的子带堆叠在一起
        fused_fei = np.matmul(fused_u, fused_fei) # 计算融合后的图像
        th = h # 存储当前高度
        tw = w # 存储当前宽度
        h *= 2 # 恢复高度
        w *= 2 # 恢复宽度
        fused_img = np.zeros([h,w]) # 初始化融合后的图像
        # 恢复图像大小
        for k in range(tw):
            for l in range(th):
                fused_img[l*2, k*2] = fused_fei[0][l+k*th]
                fused_img[l*2+1, k*2] = fused_fei[1][l+k*th]
                fused_img[l*2,k*2+1] = fused_fei[2][l+k*th]
                fused_img[l*2+1, k*2+1] = fused_fei[3][l+k*th]
        fused_fei = fused_img.flatten(order="F") # 将图像展平

    fused_img = cv2.normalize(fused_img, None, 0., 255., cv2.NORM_MINMAX) # 将融合后的图像归一化到[0,255]
    fused_img = cv2.convertScaleAbs(fused_img) # 将融合后的图像转换为uint8类型
    return fused_img


def MSVD_RGB(img_r, img_v):
    '''
    多尺度奇异值分解（MSVD）算法，用于融合两张彩色图像。

    Args:
        img_r: 红外图像 (彩色图).
        img_v: 可见光图像 (彩色图).

    Returns:
        fused_img: 融合后的图像 (彩色图).
    '''
    r_R = img_r[:,:,2]  # 提取红外图像的红色通道
    r_G = img_r[:,:,1]  # ~~绿色通道
    r_B = img_r[:,:,0]  # ~~蓝色通道
    v_R = img_v[:,:,2]  # 提取可见光图像的红色通道
    v_G = img_v[:,:,1]  # ~~绿色通道
    v_B = img_v[:,:,0]  # ~~蓝色通道
    fused_R = MSVD_GRAY(r_R, v_R)  # 融合红色通道
    fused_G = MSVD_GRAY(r_G, v_G)  # 融合绿色通道
    fused_B = MSVD_GRAY(r_B, v_B)  # 融合蓝色通道
    fused_img = np.stack((fused_B, fused_G, fused_R), axis=-1)  # 将融合后的通道堆叠在一起
    return fused_img

def MSVD(r_path, v_path):
    '''
    多尺度奇异值分解（MSVD）算法的主函数。

    Args:
        r_path: 红外图像的路径.
        v_path: 可见光图像的路径.
    '''
    img_r = cv2.imread(r_path)  # 读取红外图像
    img_v = cv2.imread(v_path)  # 读取可见光图像
    if not isinstance(img_r, np.ndarray):  # 检查红外图像是否读取成功
        print("img_r is not an image")
        return
    if not isinstance(img_v, np.ndarray):  # 检查可见光图像是否读取成功
        print("img_v is not an image")
        return

    # 统一图像大小
    img_v = cv2.resize(img_v, (img_r.shape[1], img_r.shape[0]), interpolation=cv2.INTER_AREA)


    fused_img = None  # 初始化融合后的图像
    if len(img_r.shape)==2 or img_r.shape[-1] ==1:  # 如果红外图像是灰度图
        if img_r.shape[-1] ==3:
            img_v = cv2.cvtColor(img_v, cv2.COLOR_BGR2GRAY) # 将可见光图像转换为灰度图
        fused_img = MSVD_GRAY(img_r, img_v)  # 融合灰度图像
    else:  # 如果红外图像是彩色图
        if img_r.shape[-1] ==3:
            fused_img = MSVD_RGB(img_r, img_v) # 融合彩色图像
        else:
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY) # 将红外图像转换为灰度图
            fused_img = MSVD_GRAY(img_r, img_v)  # 融合灰度图像
    cv2.imshow("fused image", fused_img)  # 显示融合后的图像
    cv2.imwrite("fused_image_msvd.jpg", fused_img)  # 保存融合后的图像
    cv2.waitKey(0)  # 等待按键按下
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--IR",  default='IR.jpg', help="path to IR image", required=False) # 添加红外图像路径参数
    parser.add_argument("--VIS",  default='VIS.jpg', help="path to IR image", required=False) # 添加可见光图像路径参数
    a = parser.parse_args() # 解析命令行参数
    MSVD(a.IR, a.VIS) # 调用MSVD函数进行图像融合