import tensorflow as tf
print("TF version:", tf.__version__)

from cancernet import CancerNet

model = CancerNet.build(width=48, height=48, depth=3, classes=2)
model.summary()
