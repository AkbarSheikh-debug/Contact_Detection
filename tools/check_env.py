import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
try:
    import torchvision
    print("torchvision", torchvision.__version__)
except ImportError:
    print("torchvision MISSING")
import cv2, numpy, sklearn
print("cv2", cv2.__version__, "numpy", numpy.__version__, "sklearn", sklearn.__version__)
