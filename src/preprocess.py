import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Set path to extracted training/dataset folder
DATASET_DIR = "data/raw/Data" 
TARGET_SIZE = (224, 224)
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

images = []
labels = []
skipped_files_count = 0

# Detect class folders automatically
class_names = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
print(f"Detected Classes ({len(class_names)}): {class_names}\n")

for class_name in class_names:
    class_folder = os.path.join(DATASET_DIR, class_name)
    file_list = os.listdir(class_folder)
    print(f"Processing class '{class_name}' ({len(file_list)} files)...")
    
    for file_name in file_list:
        file_path = os.path.join(class_folder, file_name)
        
        # 1. Filter out invalid file extensions
        if not file_path.lower().endswith(VALID_EXTENSIONS):
            skipped_files_count += 1
            continue
            
        # 2. Defensive loading with try-except
        try:
            img = cv2.imread(file_path)
            
            # Check for unreadable/corrupt image
            if img is None:
                skipped_files_count += 1
                continue
                
            # Convert BGR (OpenCV default) to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image to standard dimensions (224x224)
            img = cv2.resize(img, TARGET_SIZE)
            
            # Store image and corresponding text label
            images.append(img)
            labels.append(class_name)
            
        except Exception as e:
            skipped_files_count += 1
            print(f"Skipping corrupt file {file_name}: {e}")

# Convert lists to NumPy arrays for Keras/CNN compatibility
X = np.array(images, dtype='float32')
y = np.array(labels)

print("\n" + "="*50)
print("DATASET PREPROCESSING COMPLETE")
print("="*50)
print(f"Total valid images loaded: {X.shape[0]}")
print(f"Total skipped/corrupt files: {skipped_files_count}")
print(f"Feature matrix (X) shape: {X.shape}  -> (Samples, Height, Width, Channels)")
print(f"Label vector (y) shape:   {y.shape}")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.20, random_state=42, stratify=y_encoded
)

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# ---------------------------------------------------------
# 4. SAVE PROCESSED ARRAYS TO DISK (.npy format)
# ---------------------------------------------------------
np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/y_test.npy", y_test)

# Save the encoder classes as a text file for future inference
np.save("models/classes.npy", encoder.classes_)

print("\n" + "="*50)
print("SUCCESS: Local structure created & preprocessed data saved!")
print("="*50)
print("Saved files in 'data/processed/':")
print(" - X_train.npy | X_test.npy")
print(" - y_train.npy | y_test.npy")
print("Saved class list to 'models/classes.npy':", encoder.classes_)