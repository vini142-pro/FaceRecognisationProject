import os
import cv2
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------- STEP 0: Unzip dataset --------------------
zip_path = r"C:\Users\HP\OneDrive\Documents\face reco\dataset.zip"
extract_path = r"C:\Users\HP\OneDrive\Documents\face reco\dataset"

if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ Dataset extracted successfully!")
else:
    print("✅ Dataset already extracted.")

# Adjust this path based on your dataset structure
dataset_path = os.path.join(extract_path, "dataset", "faces")  # may vary based on unzip

# -------------------- STEP 1: Load images --------------------
def load_dataset(root_dir, size=(100,100)):
    images = []
    labels = []
    for person in sorted(os.listdir(root_dir)):
        person_path = os.path.join(root_dir, person)
        if not os.path.isdir(person_path):
            continue
        for fname in sorted(os.listdir(person_path)):
            img_path = os.path.join(person_path, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, size)
            images.append(img.flatten().astype(np.float32))
            labels.append(person)
    X = np.array(images).T   # shape = (mn, p)
    labels = np.array(labels)
    return X, labels

X, labels = load_dataset(dataset_path, size=(100,100))
print("Shape of X:", X.shape)
print("Number of labels:", len(labels))

# -------------------- STEP 2: Mean Face --------------------
mean_face = np.mean(X, axis=1, keepdims=True)
Delta = X - mean_face
print("Mean face shape:", mean_face.shape)
print("Delta shape:", Delta.shape)

plt.imshow(mean_face.reshape(100,100), cmap='gray')
plt.title("Mean Face")
plt.axis('off')
plt.show()

# -------------------- STEP 3: PCA / Eigenfaces --------------------
print("\nPerforming PCA...")
U, S, Vt = np.linalg.svd(Delta, full_matrices=False)
print("U shape:", U.shape, "S shape:", S.shape, "Vt shape:", Vt.shape)

k = 50
eigenfaces = U[:, :k]
eigenfaces /= np.linalg.norm(eigenfaces, axis=0, keepdims=True)

fig, axes = plt.subplots(1, 5, figsize=(12,3))
for i, ax in enumerate(axes):
    ax.imshow(eigenfaces[:, i].reshape(100,100), cmap='gray')
    ax.axis('off')
plt.suptitle("First 5 Eigenfaces")
plt.show()

# -------------------- STEP 4: Project faces → signatures --------------------
signatures = eigenfaces.T @ Delta
X_features = signatures.T
print("Feature matrix shape:", X_features.shape)

# -------------------- STEP 5: Train ANN --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_features, labels, test_size=0.4, random_state=42, stratify=labels
)

model = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                  max_iter=500, random_state=42)
)
model.fit(X_train, y_train)
print("✅ ANN training completed!")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy: {:.2f}%".format(accuracy*100))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------- STEP 6: Accuracy vs k --------------------
ks = [10, 20, 30, 50, 80, 100]
accuracies = []

for k_val in ks:
    print(f"\nRunning for k = {k_val} ...")
    eigenfaces_k = U[:, :k_val]
    eigenfaces_k /= np.linalg.norm(eigenfaces_k, axis=0, keepdims=True)
    signatures_k = eigenfaces_k.T @ Delta
    X_features_k = signatures_k.T

    X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(
        X_features_k, labels, test_size=0.4, stratify=labels, random_state=42
    )

    model_k = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    )
    model_k.fit(X_train_k, y_train_k)
    acc = model_k.score(X_test_k, y_test_k)
    accuracies.append(acc)
    print(f"Accuracy for k={k_val}: {acc*100:.2f}%")

plt.figure(figsize=(7,5))
plt.plot(ks, [a*100 for a in accuracies], marker='o', linewidth=2)
plt.xlabel('k (Number of Eigenfaces)')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Number of Eigenfaces (k)')
plt.grid(True)
plt.show()

# -------------------- STEP 7: Imposter Detection --------------------
# Compute centroids
unique_labels = np.unique(labels)
centroids = {}
for person in unique_labels:
    idx = np.where(labels == person)[0]
    centroids[person] = np.mean(X_features[idx], axis=0)

def detect_imposter(test_feat, centroids, threshold=50):
    distances = {person: np.linalg.norm(test_feat - centroids[person]) for person in centroids}
    min_person = min(distances, key=distances.get)
    if distances[min_person] > threshold:
        return "unknown"
    else:
        return min_person

# Example on test set
print("\nSample imposter detection on test set:")
for i in range(5):
    test_feat = X_test[i]
    true_label = y_test[i]
    pred_label = detect_imposter(test_feat, centroids, threshold=50)
    print(f"True: {true_label}, Predicted: {pred_label}")
# Load unknown face
imp_img = cv2.imread("imp_face.jpg", cv2.IMREAD_GRAYSCALE)
imp_img = cv2.resize(imp_img, (100,100)).flatten().astype(np.float32)
imp_feat = (imp_img - mean_face.flatten()) @ eigenfaces  # project to eigenfaces

pred_label = detect_imposter(imp_feat, centroids, threshold=50)
print("Imposter prediction:", pred_label)  # should print "unknown"

