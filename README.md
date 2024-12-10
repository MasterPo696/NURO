# Machine Learning and Computer Vision Projects

This repository contains a collection of projects that utilize machine learning and computer vision techniques for various tasks, including image recognition, object detection, and classification.

## Projects Overview

### 1. **Upper - CcontrolL**
A project focused on license plate recognition and character detection using preprocessed image datasets and Haar cascade classifiers.

- **Key Features**:
  - Detection of license plates and individual characters (letters, numbers).
  - Utilizes Haar cascades for object detection.
  - Organized datasets for training and testing, including characters, numbers, and plates.
- **Folder Structure**:
  - `main.py`: Core script for recognition.
  - `test.py`: Script for testing detection and recognition functionality.
  - `data/`: Contains datasets:
    - `x_data/letters`: Images of individual letters (e.g., `A.png`, `B.png`).
    - `x_data/nums`: Images of digits (`0.png`, `1.png`).
    - `x_data/plates`: Full license plates (`plate.png`).
    - `x_data/samples`: Sample training images.
    - `haarcascade/`: XML classifiers for face, car, and object detection.

---

### 2. **Basic - NumberNet**
A neural network-based project for digit classification.

- **Key Features**:
  - Custom neural network architecture for recognizing handwritten digits.
  - Pretrained model weights available (`model.weights.h5`).
  - Includes training and testing datasets in `.npy` format.
- **Folder Structure**:
  - `net.py`: Core implementation of the neural network.
  - `data/`: Contains training and testing data (`x_train.npy`, `y_train.npy`).
  - `nums/`: Subfolders with image datasets grouped by numbers (e.g., `1/8.jpg`, `2/3.jpg`).
  - `README.md`: Documentation specific to this project.

---

### 3. **Basic - Bussanger**
A minimal project with a placeholder for processing or classification tasks.

- **Key Features**:
  - Includes a single script (`net.py`) with foundational logic.
  - Suitable as a starting point for new projects or rapid prototyping.

---

### 4. **Basic - WhichCar**
A sophisticated project for car image classification and preprocessing.

- **Key Features**:
  - Modular design for augmentation, preprocessing, and neural network inference.
  - Focuses on analyzing car images with high accuracy.
  - Uses various utility scripts for data preparation and augmentation.
- **Folder Structure**:
  - `app/`:
    - `main.py`: Entry point for the project.
    - `config.py`: Configuration settings.
    - `augmentation.py`: Tools for augmenting image data.
    - `preproccessing.py`: Scripts for preparing data for model input.
    - `net.py`: Neural network model.

---

## How to Use

### Prerequisites
- Python 3.x
- Required libraries: OpenCV, NumPy, TensorFlow/Keras (if applicable).

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/your_repo_name.git
   ```
2. Navigate to the desired project folder:
   ```bash
   cd Upper/CcontrolL
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Run
- Use `main.py` or equivalent scripts to start processing images or executing models:
  ```bash
  python main.py
  ```

## Contribution
Contributions are welcome! Feel free to open issues or submit pull requests for improvements, bug fixes, or new features.

---

## License
This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
