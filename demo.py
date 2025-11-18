from load_data import load_data
import tensorflow as tf
import numpy as np
import os
import tifffile as tiff
from loss import combined_weighted_loss

# 数据文件夹的路径
data_dir = r'data\1'
image_data, label_data = load_data(data_dir)  # 加载数据
print("Image data shape:", image_data.shape)  # 打印图像数据的形状
print("Label data shape:", label_data.shape)  # 打印标签数据的形状

# 加载已经训练好的模型，并指定自定义损失函数
model = tf.keras.models.load_model(r'weights\model.h5',
                                  custom_objects={'loss': combined_weighted_loss(dice_weight=0.6,
                                                                                cldice_weight=0.1,
                                                                                bce_weight=0.3, weight=1.0,
                                                                                iters=20, alpha=0.5)})
# 进行预测
pred = model.predict(image_data)
# 打印预测结果的形状
print(pred.shape)

# 保存为小体——通道数为1
def save_predictions(pred, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 去除维度为 1 的维度
    pred = np.squeeze(pred)
    # 循环保存预测数据到 tiff 文件
    for i, volume_data in enumerate(pred):
        # 将概率值映射到二值标签
        binary_data = (volume_data > 0.3).astype(np.uint8)
        # 构建文件路径
        file_path = os.path.join(output_folder, f'prediction_{i}.tif')
        # 将体数据保存到 tiff 文件
        tiff.imwrite(file_path, binary_data * 255)  # 将二值标签数据进行压缩并保存

# 设置输出文件夹路径
output_folder = r'predict\demo'

# 保存预测数据到 tiff 文件
save_predictions(pred, output_folder)
