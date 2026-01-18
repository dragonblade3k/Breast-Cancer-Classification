import tensorflow as tf

model = tf.keras.models.load_model("cancernet1.h5")   # if saved with model.save()

model.summary()
