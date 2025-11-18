from load_data import load_data
import tensorflow as tf
import keras
from loss import combined_weighted_loss
from model import ours_model

# 导入文件夹数据
data_dir = r'train_data'  # 数据文件夹的路径
image_data, label_data = load_data(data_dir)  # 加载数据
print("Image data shape:", image_data.shape)  # 打印图像数据的形状
print("Label data shape:", label_data.shape)  # 打印标签数据的形状

model = ours_model(32, 128, 128, 2)
model.summary()

callbacks_list=[
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath='weights/model/model.h5',
        verbose=1,
        monitor='val_loss',
        mode='auto',
        save_best_only=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        verbose=1,
        patience=4,
    ),
    keras.callbacks.CSVLogger(filename='weights/model/training_log.csv', append=True)
]

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=combined_weighted_loss(dice_weight=0.6, cldice_weight=0.1, bce_weight=0.3, weight=1.0, iters=25, alpha=0.5), metrics=['accuracy'])

history = model.fit(image_data, label_data, batch_size=4, epochs=30, shuffle=True, validation_split=0.2,
                 callbacks=callbacks_list)