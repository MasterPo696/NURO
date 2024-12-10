# **NumberNet: MNIST Digit Recognition**

NumberNet is a neural network designed to recognize handwritten digits from the MNIST dataset. It uses a simple feedforward architecture with fully connected (dense) layers. The project demonstrates basic machine learning concepts, including data preprocessing, model training, evaluation, and image prediction.

---

## **Features**
- **MNIST Dataset Integration**: Uses the MNIST handwritten digits dataset for training and testing.
- **Custom Image Recognition**: Load and test custom images of digits in `.jpg` format.
- **Fully Connected Neural Network**: Built using Keras with three dense layers.
- **Visualization**: Displays processed input images for better understanding of the pipeline.
- **Accuracy Tracking**: Measures the model's accuracy on custom images.

---

## **Dependencies**
To use this project, the following Python libraries are required:
- `keras` (TensorFlow backend)
- `numpy`
- `matplotlib`
- `Pillow`

Install dependencies via pip:

```bash
pip install numpy keras matplotlib pillow
```

---

## **How It Works**

### **1. Data Preprocessing**
- The MNIST dataset is loaded from `keras.datasets`.
- Images are flattened from 28x28 matrices into vectors of size 784.
- Input values are normalized to the range [0, 1].
- Labels are converted to one-hot encoding for multiclass classification.

### **2. Model Architecture**
The model uses three dense layers:
1. **Input Layer**: 784 input neurons (one per pixel), activation: `ReLU`.
2. **Hidden Layer**: 400 neurons, activation: `ReLU`.
3. **Output Layer**: 10 neurons (one per digit), activation: `Softmax`.

### **3. Training**
The model is compiled with:
- **Loss Function**: Categorical cross-entropy.
- **Optimizer**: Adam.
- **Metric**: Accuracy.

It trains for 15 epochs with a batch size of 128.

### **4. Testing**
- Pre-trained model weights are loaded.
- Custom images are resized to 28x28 pixels, converted to grayscale, inverted, normalized, and reshaped to match the input format.
- The model predicts the digit, and accuracy is calculated.

---

## **Usage**

### **1. Training the Model**
Run the following code to train the model on the MNIST dataset:

```python
model.fit(x_train, y_train, batch_size=128, epochs=15, verbose=1)
model.save_weights('model.weights.h5')  # Save model weights
```

### **2. Testing with Custom Images**
Place your custom digit images in the `nums` folder (e.g., `nums/2/0.jpg` for digit 0). Ensure the images are:
- Grayscale
- Sized 28x28 pixels

Run the testing script to evaluate predictions:

```python
python net.py
```

The script:
1. Loads and preprocesses each image.
2. Passes it to the neural network.
3. Outputs the predicted digit and displays the processed image.

### **3. Output Example**
After processing, you'll see results like:

```
Checking image: nums/2/0.jpg
Expected: 0, Predicted: 0

Checking image: nums/2/1.jpg
Expected: 1, Predicted: 1
...
Out of 10 attempts, the model predicted correctly 9 times.
```

---

## **Folder Structure**
```
NumberNet/
├── data/
│   ├── x_train.npy
│   ├── y_train.npy
│   ├── x_test.npy
│   ├── y_test.npy
├── nums/
│   └── 2/
│       ├── 0.jpg
│       ├── 1.jpg
│       ├── ...
├── model.weights.h5
├── net.py
└── README.md
```

- `data/`: Contains preprocessed MNIST data in `.npy` format.
- `nums/`: Folder for custom digit images.
- `model.weights.h5`: Pre-trained model weights.
- `net.py`: Main script for loading, testing, and evaluating the model.

---

## **Customization**
### **Adjust Model Layers**
You can modify the number of neurons, layers, or activation functions in the model to experiment with performance.

### **Train on Custom Data**
Replace the MNIST dataset with your own data by following the same preprocessing steps.

---

## **Future Improvements**
- Support for rotated or noisy images.
- Improve preprocessing for better recognition of real-world digits.
- Extend the model to work with colored or larger images.

---

## **Author**
This project is a basic implementation of digit recognition to demonstrate neural network concepts in Python.