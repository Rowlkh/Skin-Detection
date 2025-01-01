# %% main.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_acquisition import load_images_from_folder
from preprocessing import restore_image, enhance_image
from segmentation import segment_image
from ml_models import extract_features, train_and_evaluate
from sklearn.metrics import classification_report
import joblib # saving model
import random

#%% Paths to the datasets
skin_folder = r"D:\\MIU\\3.1\\Image Processing Project\\Dataset1\\skin"
non_skin_folder = r"D:\\MIU\\3.1\\Image Processing Project\\Dataset1\\non_skin"

#%% Step 1: Load and resize images
skin_images, _ = load_images_from_folder(skin_folder, resize_dim=(800, 600))
non_skin_images, _ = load_images_from_folder(non_skin_folder, resize_dim=(800, 600))

#%% Validate image loading
print(f"Number of skin images loaded: {len(skin_images)}")
print(f"Number of non-skin images loaded: {len(non_skin_images)}")
assert len(skin_images) == 1134, "Skin images count mismatch!"
assert len(non_skin_images) == 1128, "Non-skin images count mismatch!"

# %% Visualize 3 Randomized Skin & Non-Skin Images  (Original images)
random_skin_images = random.sample(skin_images, 3)
random_non_skin_images = random.sample(non_skin_images, 3)

plt.figure(figsize=(15, 10))
# Skin
for i, img in enumerate(random_skin_images):
    plt.subplot(2, 3, i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Random Skin Image {i + 1}")
    plt.axis("off")
# non-skin 
for i, img in enumerate(random_non_skin_images):
    plt.subplot(2, 3, i + 4)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Random Non-Skin Image {i + 1}")
    plt.axis("off")
plt.tight_layout()
plt.show()

""" Preprocessing """
#%% Step 2: Preprocess images
preprocessed_skin_images = [restore_image(enhance_image(img)) for img in skin_images]

#%% Visualize 3 Randomized Skin (Processed images)
random_indices = random.sample(range(len(skin_images)), 3)
plt.figure(figsize=(15, 10))
for i, idx in enumerate(random_indices):
    original_image = skin_images[idx]
    preprocessed_image = restore_image(enhance_image(original_image))
    # Original image
    plt.subplot(3, 2, 2 * i + 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Original Skin Image {i + 1}")
    plt.axis("off")
    # Preprocessed image
    plt.subplot(3, 2, 2 * i + 2)
    plt.imshow(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Preprocessed Skin Image {i + 1}")
    plt.axis("off")
plt.tight_layout()
plt.show()

""" Segmentation """
#%% Step 3: Segment skin images only 
segmented_skin_images = [segment_image(img) for img in preprocessed_skin_images]

#%% Ensure all lists have the same length or select the minimum length
num_images = min(len(skin_images), len(preprocessed_skin_images), len(segmented_skin_images))

#%% Visualize 3 Randomized Skin (Segmented images)
random_indices = random.sample(range(len(skin_images)), 3)
plt.figure(figsize=(15, 10))
for i, idx in enumerate(random_indices):
    original_image = skin_images[idx]
    segmented_image = segmented_skin_images[idx]
    # Original image
    plt.subplot(3, 2, 2 * i + 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Original Skin Image {i + 1}")
    plt.axis("off")
    # Segmented image
    plt.subplot(3, 2, 2 * i + 2)
    plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Segmented Skin Image {i + 1}")
    plt.axis("off")
plt.tight_layout()
plt.show()

""" Apply labels """
#%% Step 4: Assign labels
skin_labels = [1] * len(segmented_skin_images)
non_skin_labels = [0] * len(non_skin_images)

# Visualize dataset balance
label_counts = [len(skin_labels), len(non_skin_labels)]
label_names = ['Skin', 'Non-Skin']
plt.figure(figsize=(8, 5))
plt.bar(label_names, label_counts, color=['blue', 'orange'])
plt.xlabel('Label Type')
plt.ylabel('Count')
plt.title('Dataset Balance')
plt.show()

# Print dataset counts
print(f"Number of skin labels: {len(skin_labels)}")
print(f"Number of non-skin labels: {len(non_skin_labels)}")

#%% Step 5: Combine datasets
all_images = segmented_skin_images + non_skin_images
all_labels = skin_labels + non_skin_labels

""" Feature Extraction """
#%% Step 6: Extract features
X = np.array([extract_features(img) for img in all_images])
y = np.array(all_labels)

""" ML """
#%% Step 7: Train and evaluate models
rf_model, rf_accuracy, lr_model, lr_accuracy, X_train, X_test, y_train, y_test, rf_predictions, lr_predictions = train_and_evaluate(X, y)

#%% Step 8: Save models & accuracy (Logistic regression & random forest)
# save model
joblib.dump(rf_model, 'skin_classification_rf_model.pkl')
joblib.dump(lr_model, 'skin_classification_lr_model.pkl')
# Save accuracy
with open('model_performance.txt', 'w') as f:
    f.write(f'Random Forest Accuracy: {rf_accuracy:.2f}\n')
    f.write(f'Logistic Regression Accuracy: {lr_accuracy:.2f}\n')



#%%
print(f"\nRandom Forest Model Accuracy: {rf_accuracy:.2f}")
print(f"Logistic Regression Model Accuracy: {lr_accuracy:.2f}")

""" Display prediction written """
#%% Step 9: Display training and testing split details

# Validate train-test split
print(f"Total images: {len(X)}")
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Display predictions vs actual labels for Random Forest
print("\nRandom Forest Predictions vs Actual Labels:")
for actual, predicted in zip(y_test, rf_predictions):
    status = "Correct" if actual == predicted else "Incorrect"
    print(f"Actual: {actual}, Predicted: {predicted}, Status: {status}")

# Display classification report for Random Forest
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_predictions))

# Display predictions vs actual labels for Logistic Regression
print("\nLogistic Regression Predictions vs Actual Labels:")
for actual, predicted in zip(y_test, lr_predictions):
    status = "Correct" if actual == predicted else "Incorrect"
    print(f"Actual: {actual}, Predicted: {predicted}, Status: {status}")

# Display classification report for Logistic Regression
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_predictions))

""" Predictions Visualizations """
#%% Step 10: Visualize test predictions
print("Testing on new images...")
test_images = skin_images[:5] + non_skin_images[:5]
test_labels = [1] * 5 + [0] * 5  

for i, test_image in enumerate(test_images):
    # Preprocess the test image
    enhanced = enhance_image(test_image)
    restored = restore_image(enhanced)
    segmented = segment_image(restored)
    features = extract_features(segmented)
    
    # Predictions from both models
    rf_prediction = rf_model.predict([features])[0]
    lr_prediction = lr_model.predict([features])[0]

    # Display the test image with predictions and actual label
    plt.figure()
    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    plt.title(f"RF Prediction: {'Skin' if rf_prediction == 1 else 'Non-Skin'} | "
              f"LR Prediction: {'Skin' if lr_prediction == 1 else 'Non-Skin'} | "
              f"Actual: {'Skin' if test_labels[i] == 1 else 'Non-Skin'}")
    plt.axis("off")
    plt.show()
# %%
