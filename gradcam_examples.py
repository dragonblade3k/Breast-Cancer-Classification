import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from cancernet.cancernet import CancerNet
from cancernet import config

import cv2



# --------- Load model (same as training) ----------
model = CancerNet.build(width=48, height=48, depth=3, classes=2)
model.load_weights("cancernet2.h5")  # or your final weights filename

model.summary()

# Set this to the name of your LAST conv layer
# Check model.summary() to confirm the exact name.
LAST_CONV_LAYER_NAME = "separable_conv2d_4"



# --------- Grad-CAM utility ----------

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    img_array: (48, 48, 3) or (1, 48, 48, 3)
    """

    # Ensure a batch dimension: (1, H, W, C)
    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype("float32")

    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]  # (H, W, C)

    heatmap = tf.einsum("hwc,c->hw", conv_outputs, pooled_grads)
    heatmap = tf.nn.relu(heatmap)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_gradcam(img, heatmap, alpha=0.4):
    # img: original RGB image (H, W, 3) in [0, 255]
    # heatmap: (H, W) in [0, 1]
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.stack([heatmap] * 3, axis=-1)

    heatmap = tf.image.resize(heatmap, (img.shape[0], img.shape[1])).numpy()
    overlay = np.uint8(alpha * heatmap + (1 - alpha) * img)
    return overlay


# --------- Pick some example test images ----------
def get_example_paths(n_benign=2, n_malignant=2):
    """
    Assumes:
      C:\Major Project\idc\testing\0\...
      C:\Major Project\idc\testing\1\...
    """
    benign_dir = os.path.join(config.TEST_PATH, "0")
    malig_dir = os.path.join(config.TEST_PATH, "1")

    benign_files = [os.path.join(benign_dir, f) for f in os.listdir(benign_dir)][:n_benign]
    malig_files  = [os.path.join(malig_dir, f)  for f in os.listdir(malig_dir)][:n_malignant]

    return benign_files, malig_files


def process_and_save_gradcam(img_path, out_prefix):
    import matplotlib.pyplot as plt

    # Load and preprocess
    img = load_img(img_path, target_size=(48, 48))
    img_array = img_to_array(img) / 255.0

    heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)

    # Resize heatmap
    import numpy as np
    heatmap_resized = cv2.resize(heatmap, (48, 48))

    # Plot image + heatmap
    plt.figure(figsize=(4, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_array)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_array)
    plt.imshow(heatmap_resized, cmap='jet', alpha=0.5)
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_gradcam.png", dpi=300)
    plt.close()

    print("Saved:", f"{out_prefix}_gradcam.png")



if __name__ == "__main__":
    benign_files, malig_files = get_example_paths()

    # 2 benign examples
    for i, path in enumerate(benign_files):
        process_and_save_gradcam(path, f"gradcam_benign_{i}")


    # 2 malignant examples
    for i, path in enumerate(malig_files):
        process_and_save_gradcam(path, f"gradcam_malignant_{i}")
