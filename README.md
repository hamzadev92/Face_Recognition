This project implements a Face Recognition System using a custom Multi‑Layer Perceptron (MLP) built entirely from scratch with NumPy and SciPy.
No external deep learning frameworks (TensorFlow, PyTorch, Keras, etc.) were used only the neural network logic, forward propagation, backpropagation, and training loop are all hand‑crafted.
The goal was to understand the fundamentals of neural networks and apply them to a practical task: recognizing faces from a small dataset.

Face_Recognition Project/
│── dataset/                # Training images (ignored in GitHub via .gitignore)
│   ├── Hamza_Shabbir/
│   ├── Salman_Khan/
│   ├── Sharukh_Khan/

│── Project_01.py            # Main implementation of the MLP
│── test_face.jpg            # Example test image (ignored in GitHub)
│── test_images/             # Additional test samples 
│── Unknown/                 # Images not belonging to known classes (ignored in GitHub)
