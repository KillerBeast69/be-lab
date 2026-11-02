"""
====================================================================
ASSIGNMENT: Handwritten Digit Classification using SVM
====================================================================

1. ASSIGNMENT DETAILS
---------------------
Objective: Implement a Support Vector Machine (SVM) algorithm to classify 
images of handwritten digits (0 through 9) into their respective numerical 
classes. The dataset used is the standard "Digits" dataset, which consists 
of 8x8 pixel images of handwritten digits.

2. VIVA THEORY
--------------
- What is an SVM (Support Vector Machine)?
  It is a supervised machine learning algorithm used for both classification 
  and regression. It works by finding the hyperplane that best separates 
  different classes in the feature space.
- What is a Hyperplane and Margin?
  The hyperplane is the decision boundary separating classes. The margin is 
  the distance between the hyperplane and the closest data points from any 
  class. SVM maximizes this margin.
- What are Support Vectors?
  These are the data points that lie closest to the decision boundary (hyperplane). 
  They are the critical elements that dictate the position and orientation 
  of the hyperplane.
- What is the Kernel Trick?
  When data is not linearly separable, SVM uses a "kernel" (like RBF, Polynomial) 
  to implicitly transform the data into a higher-dimensional space where a 
  linear separator can be found.
- Why is SVM good for image classification?
  SVMs are highly effective in high-dimensional spaces (like images, where 
  every pixel is a feature) and are memory efficient because they only use a 
  subset of training points (support vectors) in the decision function.

3. CODE DETAILS
---------------
- datasets.load_digits(): Loads the Scikit-Learn built-in digits dataset 
  (1797 images of 8x8 pixels).
- Data Flattening: 8x8 images (2D arrays) are flattened into 64-element 
  1D arrays (vectors) so they can be fed into the SVM model.
- train_test_split: Splits the dataset into a training set (50%) and a 
  testing set (50%).
- svm.SVC(gamma=0.001): Initializes the Support Vector Classifier. The 'gamma' 
  parameter defines the influence of a single training example (RBF kernel is 
  used by default).
- fit() & predict(): Trains the model on the training data and predicts 
  classes for the testing data.
- classification_report & confusion_matrix: Used to evaluate precision, 
  recall, f1-score, and visualize how many digits were misclassified.

4. IMPLEMENTATION & EXECUTION
-----------------------------
Prerequisites (Necessary Packages):
- scikit-learn
- matplotlib
- seaborn

How to run:
1. Open your terminal or command prompt.
2. Install the required packages by running:
   pip install scikit-learn matplotlib seaborn
3. Save this code in a file named `svm_digits.py`.
4. Execute the script by running:
   python svm_digits.py
====================================================================
"""

import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import seaborn as sns

def main():
    print("1. Loading the Digits dataset...")
    # The digits dataset consists of 8x8 pixel images of digits
    digits = datasets.load_digits()

    # 2. Pre-process the dataset
    # To apply a classifier on this data, we need to flatten the images
    # from 8x8 2D arrays into 64-feature 1D arrays.
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    
    print(f"Dataset loaded. Total samples: {n_samples}")
    print(f"Original image shape: {digits.images[0].shape}, Flattened shape: {data[0].shape}")

    # 3. Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False
    )

    print("\n2. Training the SVM Classifier...")
    # Create a Support Vector Classifier
    # We use gamma=0.001 (scaling parameter for the RBF kernel)
    clf = svm.SVC(gamma=0.001)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    print("\n3. Predicting on the Test subset...")
    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)

    print("\n4. Evaluating the Model...")
    # Print classification report
    print(f"Classification report for classifier {clf}:\n")
    print(metrics.classification_report(y_test, predicted))

    # Calculate and print accuracy
    accuracy = metrics.accuracy_score(y_test, predicted)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")

    # 5. Visualizing Results
    print("Generating visualizations (Confusion Matrix & Sample Predictions)...")
    
    # Plotting Confusion Matrix
    cm = metrics.confusion_matrix(y_test, predicted)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=digits.target_names, yticklabels=digits.target_names)
    plt.title("Confusion Matrix for SVM Digit Classification")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('svm_confusion_matrix.png')
    print("-> Saved 'svm_confusion_matrix.png'")

    # Plotting some test images with their predicted labels
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        # Reshape the flattened 64-element array back to 8x8 for visualization
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
    
    plt.tight_layout()
    plt.savefig('svm_sample_predictions.png')
    print("-> Saved 'svm_sample_predictions.png'")
    
    print("\nExecution Complete. Please check the directory for generated image plots.")

if __name__ == "__main__":
    main()
