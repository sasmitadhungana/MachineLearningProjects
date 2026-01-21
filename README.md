This repository contains multiple Machine Learning projects implemented in Python using TensorFlow, Keras, Scikit-learn, and OpenCV.
Each project is designed as a standalone application, with fully functional notebooks, scripts, datasets (small samples), and results. These projects are suitable for learning, portfolio showcase, and real-world demonstration.
Projects Included:

1️⃣ CNN Image Classifier
Problem: Classify images from the CIFAR-10 dataset into 10 different categories.
Solution: Implemented a Convolutional Neural Network (CNN) using TensorFlow/Keras to automatically extract features and classify images.
Tech Stack: Python, TensorFlow/Keras, NumPy, Matplotlib
Dataset: CIFAR-10 (60,000 32x32 color images in 10 classes)
Evaluation: Accuracy, loss curves, confusion matrix
Folder: CNN_Image_Classifier/

How to Run:
pip install -r CNN_Image_Classifier/requirements.txt
cd CNN_Image_Classifier
python scripts/train_model.py
python scripts/predict.py --image path_to_image.jpg

2️⃣ Breast Cancer Classifier (SVM)
Problem: Predict whether a tumor is malignant or benign from patient features.
Solution: Built an SVM-based classifier with preprocessing, feature scaling, and hyperparameter tuning.
Tech Stack: Python, Scikit-learn, Pandas, Matplotlib
Dataset: Breast Cancer Wisconsin (Diagnostic) dataset
Evaluation: Accuracy, precision, recall, F1-score, ROC curve
Folder: BreastCancerSVM/

How to Run:
pip install -r BreastCancerSVM/requirements.txt
cd BreastCancerSVM
python scripts/train_model.py
python scripts/predict.py --sample path_to_sample.csv

3️⃣ Linear Regression Projects
Problem: Predict numeric outcomes based on input features.
Solution: Implemented Simple and Multiple Linear Regression models with data preprocessing, feature analysis, and performance evaluation.
Tech Stack: Python, Pandas, Scikit-learn, Matplotlib, Seaborn
Datasets: Provided as CSV files (synthetic or sample datasets)
Evaluation: R² score, Mean Squared Error (MSE)
Folders: Simplelinearregression/, Multiplelinearregression/

How to Run:

pip install -r <project_folder>/requirements.txt
cd <project_folder>
jupyter notebook <notebook_name>.ipynb


Structure:
MachineLearningProjects/
│
├─ CNN_Image_Classifier/
├─ BreastCancerSVM/
├─ Simplelinearregression/
├─ Multiplelinearregression/
└─ README.md
