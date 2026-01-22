## HP-3DNSM

This repository contains the code for the paper "HP-3DNSM: A dedicated 3D deep learning model for neuron segmentation based on Hessian prompt".

### Environment configuration

pip install -r requirements.txt

### Data Architecture Example

```
data/
├── 1/
│   ├── image  #Original image
│   ├── hessian_prompt  #Hessian Prompt
│   └── label  #Label
```

### Model training

python train.py

### Pretrained Model

The pretrained model weights will be made publicly available soon.

### Sample Inference

This example performs prediction using a subset of the data and outputs a generated prediction image.

python demo.py
