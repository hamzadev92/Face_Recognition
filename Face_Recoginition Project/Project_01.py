import os
import numpy as np
from PIL import Image

# --- YOUR MLP CLASS ---
class MLPFaceRecognizer:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize weights with He initialization for ReLU
        np.random.seed(42) # Ensure consistent random initialization
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2./input_size)
        # Creates a matrix with 128 rows and 4096 columns, filled with random values from the normal distribution.
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2./hidden_size)
        self.b2 = np.zeros((output_size, 1))
        self.lr = learning_rate

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_deriv(self, Z):
        return Z > 0

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z)) # Stability shift
        return expZ / expZ.sum(axis=0, keepdims=True)


# prediction engine
    def forward_propagation(self, X):  
        self.Z1 = self.W1.dot(X) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2
    

# training engine
    def backward_propagation(self, X, Y, m):
        # Y must be one-hot encoded
        dZ2 = self.A2 - Y
        dW2 = (1/m) * dZ2.dot(self.A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        
        dZ1 = self.W2.T.dot(dZ2) * self.relu_deriv(self.Z1)
        dW1 = (1/m) * dZ1.dot(X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
        
        # Update Weights
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, Y, epochs=1000):
        m = X.shape[1]
        for i in range(epochs):
            output = self.forward_propagation(X)
            self.backward_propagation(X, Y, m)
            
            if i % 100 == 0:
                loss = -np.mean(Y * np.log(output + 1e-8))
                print(f"Epoch {i}, Loss: {loss:.4f}")

# --- DATASET LOADER FUNCTION ---
def load_face_data(data_dir, img_size=(64, 64)):
    X = []
    Y_labels = []
    classes = sorted(os.listdir(data_dir)) # Ensure consistent indexing
    
    for idx, folder in enumerate(classes):
        path = os.path.join(data_dir, folder)
        if not os.path.isdir(path):
            continue
        for img_name in os.listdir(path):
            # Load, grayscale, resize, and convert to numpy
            img = Image.open(os.path.join(path, img_name)).convert('L')
            img = img.resize(img_size)
            img_array = np.array(img).flatten() / 255.0 # Normalize 0-1
            X.append(img_array)
            Y_labels.append(idx)
            augmented = img.transpose(Image.FLIP_LEFT_RIGHT)
            aug_array = np.array(augmented).flatten() / 255.0
            X.append(aug_array)
            Y_labels.append(idx)
            
    # Convert to Numpy Arrays
    X = np.array(X).T  # Shape: (pixels, samples)
    
    # One-hot encoding for Y
    num_samples = len(Y_labels)
    num_classes = len(classes)
    Y_onehot = np.zeros((num_classes, num_samples))
    for i, label in enumerate(Y_labels):
        Y_onehot[label, i] = 1
        
    return X, Y_onehot, classes

def predict_single_face(recognizer, image_path, class_names, img_size=(64, 64), threshold=85):
    img = Image.open(image_path).convert('L')
    img = img.resize(img_size)
    img_array = np.array(img).flatten() / 255.0
    X_test = img_array.reshape(-1, 1)

    prediction = recognizer.forward_propagation(X_test)
    predicted_idx = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)
    predicted_class = class_names[predicted_idx]

    if predicted_class.lower() == "unknown" or confidence < threshold:
        return "Unknown person", confidence
    else:
        return predicted_class, confidence



# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset")

    IMG_WIDTH, IMG_HEIGHT = 64, 64
    INPUT_SIZE = IMG_WIDTH * IMG_HEIGHT
    HIDDEN_SIZE = 128
    
    try:
        X_train, Y_train, class_names = load_face_data(DATASET_PATH)
        OUTPUT_SIZE = len(class_names)
        
        print(f"Found classes: {class_names}")
        print(f"Data loaded with shape: {X_train.shape[1]} images found.")
        
        # Initialize model
        recognizer = MLPFaceRecognizer(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, learning_rate=0.05)
        
        # Train model
        print("\n--- Starting Training ---")
        recognizer.train(X_train, Y_train, epochs=500)
        
        # Test the prediction on the first image
        prediction = recognizer.forward_propagation(X_train[:, 1].reshape(-1, 1))
        predicted_idx = np.argmax(prediction)
        actual_idx = np.argmax(Y_train[:, 1])
        
        print("\n--- Test Prediction ---")
        print(f"Predicted class: {class_names[predicted_idx]}")
        print(f"Actual class:    {class_names[actual_idx]}")

    except Exception as e:
        print(f"\nError: {e}")
        print("Please ensure your 'dataset/' folder exists and contains subfolders with images!")

    # --- TEST A NEW IMAGE ---
    test_image_path = os.path.join(os.path.dirname(__file__), 'test_face.jpg')
    
    if os.path.exists(test_image_path):
        print(f"\n--- Live Prediction ---")
        predicted_name, conf = predict_single_face(recognizer, test_image_path, class_names)
        print(f"Recognized Person: {predicted_name}")
        print(f"Confidence: {conf:.2f}%")
    else:

        print(f"\nPlace an image at '{test_image_path}' to test live predictions.")