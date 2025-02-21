import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Load dataset from folders
dataset_path = "./hackshop project``````````````"
batch_size = 32
img_size = (128, 128)

train_ds = image_dataset_from_directory(
    dataset_path,
    labels="inferred",
    batch_size=batch_size,
    image_size=img_size
)
