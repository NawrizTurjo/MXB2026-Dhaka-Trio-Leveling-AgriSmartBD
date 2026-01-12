import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 参数设置
IMAGE_SIZE = (380, 380)  # EfficientNetB4的推荐输入尺寸
BATCH_SIZE = 8
EPOCHS = 15 # 20/10/5
NUM_CLASSES = 38  # PlantVillage数据集有38个类别（包含健康叶片）
DATA_DIR = "./PlantVillage-Dataset-master/raw/color"  # 替换为你的数据集路径

# 数据增强和预处理
def create_data_generator():
    # 使用EfficientNet的专用预处理方法
    return ImageDataGenerator(
        preprocessing_function=applications.efficientnet.preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.05  # 保留5%数据作为验证集
    )

# 创建数据生成器
train_datagen = create_data_generator()

# 训练数据流
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

# 验证数据流
val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# 构建模型
def build_model():
    # 加载预训练基模型
    base_model = applications.EfficientNetB4(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3)
    )
    
    # 冻结预训练层（初始训练阶段）
    base_model.trainable = False
    
    # 自定义顶层
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    
    model = models.Model(inputs, outputs)
    return model

model = build_model()

# 编译模型
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 回调函数
callbacks_list = [
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    callbacks.ModelCheckpoint(
        "best_model_initial",  # 去后缀或使用.keras
        save_best_only=True,
        monitor="val_accuracy",
        save_format="tf"  # 显式指定保存格式
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3
    )
]

# 初始训练（仅训练自定义顶层）
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_list
)

# 解冻部分层进行微调
def fine_tune_model(model):
    # 解冻顶层卷积块
    model.get_layer("efficientnetb4").trainable = True
    for layer in model.layers[1].layers[:-10]:  # 保留最后10层可训练
        layer.trainable = False
    
    # 重新编译模型（使用更小的学习率）
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = fine_tune_model(model)

# 微调训练
fine_tune_history = model.fit(
    train_generator,
    initial_epoch=history.epoch[-1],
    epochs=history.epoch[-1] + 10,  # 再训练10个epoch
    validation_data=val_generator,
    callbacks=[
        callbacks.ModelCheckpoint(
            "best_model_finetuned.h5",
            save_best_only=True,
            monitor="val_accuracy"
        )
    ]
)

# 保存最终模型
model.save("plant_disease_efficientnetb4.h5")

# 可视化训练过程
def plot_history(history, title):
    plt.figure(figsize=(12, 4))
    
    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{title} Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')  # 与第一个文件一致
    
    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{title} Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')  # 统一图例位置
    
    plt.tight_layout()
    plt.show()
# 修改调用方式（替换最后两行plot_training调用）
plot_history(history, "Initial Training")
plot_history(fine_tune_history, "Fine-tuning")

# 评估模型
def evaluate_model(model_path):
    model = models.load_model(model_path)
    loss, acc = model.evaluate(val_generator)
    print(f"Validation accuracy: {acc*100:.2f}%")
    print(f"Validation loss: {loss:.4f}")

print("Initial model evaluation:")
evaluate_model("best_model_initial.h5")

print("\nFine-tuned model evaluation:")
evaluate_model("best_model_finetuned.h5")