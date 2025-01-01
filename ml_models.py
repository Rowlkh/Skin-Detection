# %% ml_models.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import cv2


#%% Feature Extraction Function
def extract_features(image):
    # Convert to HSV 
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate histograms 
    hsv_hist = [cv2.normalize(cv2.calcHist([hsv_image], [i], None, [256], [0, 256]), None).flatten() for i in range(3)]

    # Calculate mean and standard deviation 
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)

    # Use Canny edge detection for edge count
    edges = cv2.Canny(image, 100, 200)
    edge_count = np.sum(edges)

    return np.concatenate(([mean_intensity, std_intensity, edge_count], *hsv_hist))


# %% Model Training and Evaluation
def train_and_evaluate(X, y):
    # Split into Train --> 70% Test --> 30%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    # Random Forest Model
    rf_model = RandomForestClassifier(n_estimators=10, random_state=40)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_probabilities = rf_model.predict_proba(X_test)[:, 1]
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    
    print("Random Forest Accuracy:", rf_accuracy)
    print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions))

    # Logistic Regression Model
    lr_model = LogisticRegression(max_iter=100) 
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    
    print("Logistic Regression Accuracy:", lr_accuracy)
    print("Logistic Regression Classification Report:\n", classification_report(y_test, lr_predictions))

    return rf_model, rf_accuracy, lr_model, lr_accuracy, X_train, X_test, y_train, y_test, rf_predictions, lr_predictions

#%% Evaluation Function (Confusion matrix + ROC + classification report)
def evaluate_model(y_test, y_pred, y_prob):
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Skin", "Skin"]).plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC: {auc:.3f}")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()







#%%
# Model Training and Evaluation
""" def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    rf_model = RandomForestClassifier(n_estimators=10, random_state=40)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_probabilities = rf_model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, rf_predictions)
    print("Random Forest Accuracy:", accuracy)
    print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions))
    evaluate_model(y_test, rf_predictions, rf_probabilities)
    return rf_model,accuracy, X_train, X_test, y_train, y_test, rf_predictions 
 """
#%%
# Evaluation Function
""" def evaluate_model(y_test, y_pred, y_prob):
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Skin", "Skin"]).plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC: {auc:.3f}")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()
 """