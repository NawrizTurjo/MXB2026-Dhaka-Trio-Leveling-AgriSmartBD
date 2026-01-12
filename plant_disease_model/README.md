---
license: mit
datasets:
- GVJahnavi/PlantVillage_dataset
metrics:
- accuracy
base_model:
- google/efficientnet-b4
pipeline_tag: image-classification
tags:
- biology
- agriculture
library_name: keras
---

# Plant Disease Classification with EfficientNetB4
## Overview
This project leverages EfficientNetB4, a state-of-the-art convolutional neural network (CNN), to classify plant diseases using the PlantVillage dataset. The trained model achieves 97.66% accuracy on the test set, demonstrating robust performance in identifying crop diseases from leaf images.
## Key Features
High Accuracy: 97.66% test accuracy, outperforming many traditional CNN architectures.
Efficient Architecture: EfficientNetB4 balances computational efficiency and model capacity, making it suitable for deployment.
Scalability: Adaptable to other plant disease datasets or extended to new disease classes.
## Dataset
The model is trained on the PlantVillage dataset, which contains 54,305 images of healthy and diseased leaves across 14 crop species and 38 disease classes. Images are resized to 380x380 pixels to match EfficientNetB4’s input dimensions.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66b71eca22ca88908202ebab/iPRWmcQ-9FJcCscoeBLBB.png)
![image/png](https://cdn-uploads.huggingface.co/production/uploads/66b71eca22ca88908202ebab/agN6UcriJox5j0GDNDGJR.png)