import zipfile
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import layers, models
from sklearn.model_selection import train_test_split

zip_file_path = 'TASK4/working/leapGestRecog.zip'
extract_dir = 'TASK4/working/leapGestRecog'

if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Contents extracted to: {extract_dir}")

print("Listing contents of the extracted directory:")
print(os.listdir(extract_dir))

def create_label_dicts(base_dir):
    label_map = {}  
    reverse_label_map = {}  
    index = 0  

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    for folder_name in os.listdir(base_dir):
        if not folder_name.startswith('.'):
            label_map[folder_name] = index
            reverse_label_map[index] = folder_name
            index += 1

    return label_map, reverse_label_map

def load_images(base_dir, label_map, img_size=(320, 120)):
    images = []
    labels = []
    total_images = 0

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    for folder_index in range(10):
        folder_path = f"{base_dir}/0{folder_index}/"
        if not os.path.exists(folder_path):
            continue
        for gesture_folder in os.listdir(folder_path):
            if not gesture_folder.startswith('.'):
                gesture_path = f"{folder_path}/{gesture_folder}/"
                if not os.path.exists(gesture_path):
                    continue
                image_count = 0
                for img_file in os.listdir(gesture_path):
                    img_path = f"{gesture_path}/{img_file}"
                    if os.path.exists(img_path):
                        img = Image.open(img_path).convert('L').resize(img_size)
                        images.append(np.array(img))
                        image_count += 1
                labels.append(np.full((image_count, 1), label_map[gesture_folder]))
                total_images += image_count

    images = np.array(images, dtype='float32')
    labels = np.concatenate(labels).reshape(total_images, 1)
    return images, labels

def display_sample_images(images, labels, reverse_label_map):
    sample_indices = [i * 200 for i in range(10)]
    for idx in sample_indices:
        plt.imshow(images[idx], cmap='gray')
        plt.title(reverse_label_map[labels[idx, 0]])
        plt.show()

dataset_dir = 'TASK4/working/leapGestRecog/leapGestRecog'  

if os.path.exists(dataset_dir):
    print("Dataset directory exists")
    if os.path.exists(f"{dataset_dir}/00"):
        print("Subdirectory '00' exists")
    else:
        raise FileNotFoundError(f"Subdirectory '00' does not exist under {dataset_dir}")
else:
    raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

label_dict, reverse_label_dict = create_label_dicts(f"{dataset_dir}/00")

X, y = load_images(dataset_dir, label_dict)

display_sample_images(X, y, reverse_label_dict)

X = X.reshape((X.shape[0], 120, 320, 1)) / 255.0

y = to_categorical(y)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

gesture_model = models.Sequential([
    layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(120, 320, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

gesture_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

gesture_model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val), verbose=1)

test_loss, test_accuracy = gesture_model.evaluate(X_test, y_test, verbose=1)

print(f"Test Accuracy: {test_accuracy:.4f}")

