import matplotlib
matplotlib.use("Agg")

import tensorflow as tf

# GPU setup: list devices and enable memory growth to avoid grabbing all GPU memory.
gpus = tf.config.list_physical_devices('GPU')
print("Detected GPUs:", gpus)
if gpus:
	try:
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
	except Exception as e:
		print("Could not set memory growth:", e)

# Use a mirrored strategy so the model runs on available GPU(s).
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices for MirroredStrategy: {strategy.num_replicas_in_sync}")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
#from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from cancernet.cancernet import CancerNet
from cancernet import config
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

NUM_EPOCHS=40; INIT_LR=1e-2; BS=128

trainPaths=list(paths.list_images(config.TRAIN_PATH))
lenTrain=len(trainPaths)
lenVal=len(list(paths.list_images(config.VAL_PATH)))
lenTest=len(list(paths.list_images(config.TEST_PATH)))

trainLabels=[int(p.split(os.path.sep)[-2]) for p in trainPaths]
trainLabels=to_categorical(trainLabels)
classTotals=trainLabels.sum(axis=0)
classWeight=classTotals.max()/classTotals
classWeight={0:classWeight[0],1:classWeight[1]}

trainAug = ImageDataGenerator(
	rescale=1/255.0,
	rotation_range=20,
	zoom_range=0.05,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.05,
	horizontal_flip=True,
	vertical_flip=True,
	fill_mode="nearest")

valAug=ImageDataGenerator(rescale=1 / 255.0)

trainGen = trainAug.flow_from_directory(
	config.TRAIN_PATH,
	class_mode="categorical",
	target_size=(48,48),
	color_mode="rgb",
	shuffle=True,
	batch_size=BS)
valGen = valAug.flow_from_directory(
	config.VAL_PATH,
	class_mode="categorical",
	target_size=(48,48),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)
testGen = valAug.flow_from_directory(
	config.TEST_PATH,
	class_mode="categorical",
	target_size=(48,48),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

with strategy.scope():
    model = CancerNet.build(width=48, height=48, depth=3, classes=2)
    opt = Adam(learning_rate=1e-3)  # lower LR, Adam instead of Adagrad
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
    )

callbacks = [
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
]


M = model.fit(
    trainGen,
    steps_per_epoch=lenTrain // BS,
    validation_data=valGen,
    validation_steps=lenVal // BS,
    class_weight=classWeight,
    epochs=NUM_EPOCHS,
    callbacks=callbacks
)


#M = model.fit(
#	trainGen,
#	steps_per_epoch=lenTrain//BS,
#	validation_data=valGen,
#	validation_steps=lenVal//BS,
#	class_weight=classWeight,
#	epochs=NUM_EPOCHS)

model.save("cancernet2.h5")

print("Now evaluating the model")
testGen.reset()

# 1) Get probabilities
probs = model.predict(testGen, steps=(lenTest // BS) + 1)  # shape (N, 2)
y_true = testGen.classes                                  # shape (N,)
pos_probs = probs[:, 1]                                   # P(class 1)

# 2) Hard predictions via argmax (same as before)
y_pred = np.argmax(probs, axis=1)

# 3) Classification report (as before)
print(classification_report(
    y_true,
    y_pred,
    target_names=list(testGen.class_indices.keys())
))

# 4) Confusion matrix (numeric)
cm = confusion_matrix(y_true, y_pred)
print(cm)
tn, fp, fn, tp = cm.ravel()
total = cm.sum()
accuracy = (tn + tp) / total
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
print(f'Accuracy: {accuracy}')
print(f'Specificity: {specificity}')
print(f'Sensitivity: {sensitivity}')

# ----- Confusion matrix figure -----
plt.figure(figsize=(5, 4))
im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar(im, fraction=0.046, pad=0.04)

classes = list(testGen.class_indices.keys())
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# Print numbers inside the boxes
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, str(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

from sklearn.metrics import roc_curve, auc

# ----- ROC & AUC -----
fpr, tpr, thresholds = roc_curve(y_true, pos_probs)
roc_auc = auc(fpr, tpr)
print("ROC AUC:", roc_auc)

plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
plt.close()


history = M.history
epochs = range(1, len(history["loss"]) + 1)

plt.style.use("ggplot")
plt.figure(figsize=(10, 6))

plt.plot(epochs, history["loss"], label="train_loss")
plt.plot(epochs, history["val_loss"], label="val_loss")
plt.plot(epochs, history["accuracy"], label="train_acc")
plt.plot(epochs, history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy on the IDC Dataset")
plt.xlabel("Epoch")
plt.ylabel("Loss / Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("plot.png")

