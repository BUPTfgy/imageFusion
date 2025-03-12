import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
import argparse

# 基于深度学习的特征融合
# 定义VGG19特征提取器
class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG19FeatureExtractor, self).__init__()
        vgg19 = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg19[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg19[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg19[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg19[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg19[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_4 = h
        return h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_4

# 图像融合策略
def fusion_strategy(source_a, source_b):
    weight_ave = 0.5  # 设置平均权重
    weight_spatial = 0.5  # 设置空间权重
    weight_detail = 0.5  # 设置细节权重

    source_a = source_a.astype(np.float32) / 255.0  # 归一化
    source_b = source_b.astype(np.float32) / 255.0  # 归一化

    weight_ave_temp = source_a * weight_ave + source_b * (1 - weight_ave)  # 平均权重
    weight_spatial_temp = np.abs(source_a - source_b)  # 绝对值空间权重
    weight_detail_temp = (source_a + source_b) / 2  # 细节权重

    # 空间权重融合
    weight_spatial_blur = cv2.GaussianBlur(weight_spatial_temp, (5, 5), 0)  # 高斯模糊
    weight_ave_temp1 = source_a * weight_spatial_blur + source_b * (1 - weight_spatial_blur)  # 加权图像A

    # 细节权重融合
    weight_detail_blur = cv2.GaussianBlur(weight_detail_temp, (5, 5), 0)  # 高斯模糊
    weight_ave_temp2 = source_a * weight_detail_blur + source_b * (1 - weight_detail_blur)  # 加权图像B

    # 调整 weight_ave_temp2 的形状以匹配 source_b
    weight_ave_temp2 = cv2.resize(weight_ave_temp2, (source_b.shape[1], source_b.shape[0]))
    if len(weight_ave_temp2.shape) == 2:
      weight_ave_temp2 = np.expand_dims(weight_ave_temp2, axis=2)

    # 确保权重在 0 到 1 之间
    weight_ave_temp2 = np.clip(weight_ave_temp2, 0, 1)

    source_b_fuse = source_b * weight_ave_temp2  # 加权图像B
    #调整 weight_ave_temp1 的形状以匹配 source_a
    weight_ave_temp1 = cv2.resize(weight_ave_temp1, (source_a.shape[1], source_a.shape[0]))
    if len(weight_ave_temp1.shape) == 2:
      weight_ave_temp1 = np.expand_dims(weight_ave_temp1, axis=2)

    # 确保权重在 0 到 1 之间
    weight_ave_temp1 = np.clip(weight_ave_temp1, 0, 1)
    source_a_fuse = source_a * weight_ave_temp1  # 加权图像A

    # 最终融合
    saliency_map = source_a_fuse + source_b_fuse
    saliency_map = np.clip(saliency_map * 255.0, 0, 255).astype(np.uint8)  # 反归一化

    return saliency_map

#图像尺寸调整函数
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


# 主函数
def DLF(img_path_a, img_path_b):
    # 读取图像
    source_a = cv2.imread(img_path_a)
    source_b = cv2.imread(img_path_b)

     # 检查图像是否成功读取
    if not isinstance(source_a, np.ndarray) :
        print('source_a is null')
        return
    if not isinstance(source_b, np.ndarray) :
        print('source_b is null')
        return

     # 检查图像大小是否相同
    if source_a.shape[0] != source_b.shape[0]  or source_a.shape[1] != source_b.shape[1]:
        print('size is not equal')
        # 统一尺寸处理
        target_size = (min(source_a.shape[0], source_b.shape[0]),
                       min(source_a.shape[1], source_b.shape[1]))

     # 等比例缩放
        source_a = resize_image(source_a, target_size)
        source_b = resize_image(source_b, target_size)
        print('统一尺寸处理完成')


    # 图像转换为RGB
    source_a = cv2.cvtColor(source_a, cv2.COLOR_BGR2RGB)
    source_b = cv2.cvtColor(source_b, cv2.COLOR_BGR2RGB)
    # 转换为torch张量
    source_a_tensor = torch.from_numpy(source_a).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    source_b_tensor = torch.from_numpy(source_b).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # 加载VGG19特征提取器
    model = VGG19FeatureExtractor()
    #  如果CUDA可用，则使用CUDA，否则使用CPU
    if torch.cuda.is_available():
        model = model.cuda()
        source_a_tensor = source_a_tensor.cuda()
        source_b_tensor = source_b_tensor.cuda()
    model.eval()  # 切换到评估模式

    # 提取VGG19特征
    feature_a = model(source_a_tensor)
    feature_b = model(source_b_tensor)

    # 计算显著图
    saliency_current = fusion_strategy(
        source_a, source_b
    )  # 使用改进后的策略函数

    # 保存结果
    cv2.imwrite("fused_image_dlf.jpg", cv2.cvtColor(saliency_current, cv2.COLOR_RGB2BGR))
    cv2.imshow("fused_image", cv2.cvtColor(saliency_current, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--IR", default='IR.jpg', help="path to IR image", required=False)
    parser.add_argument("--VIS", default='VIS.jpg', help="path to visible image", required=False)
    args = parser.parse_args()
    DLF(args.IR, args.VIS)