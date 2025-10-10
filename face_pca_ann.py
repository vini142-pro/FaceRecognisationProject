import os, cv2, zipfile
import numpy as np

# 1️⃣ Unzip the dataset if not extracted
zip_path = r"C:\Users\HP\OneDrive\Documents\face reco\dataset.zip"
extract_path = r"C:\Users\HP\OneDrive\Documents\face reco\dataset"

if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ Dataset extracted successfully!")
else:
    print("✅ Dataset already extracted.")

# 2️⃣ Define the function that loads images
def load_dataset(root_dir, size=(100, 100)):
    images = []
    labels = []
    for person in sorted(os.listdir(root_dir)):
        person_path = os.path.join(root_dir, person)
        if not os.path.isdir(person_path):
            continue
        for filename in sorted(os.listdir(person_path)):
            img_path = os.path.join(person_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, size)
            images.append(img.flatten().astype(np.float32))
            labels.append(person)
    X = np.array(images).T
    labels = np.array(labels)
    return X, labels

# 3️⃣ Load the extracted dataset
X, labels = load_dataset(r"C:\Users\HP\OneDrive\Documents\face reco\dataset\dataset\faces", size=(100, 100))
print("Shape of X:", X.shape)
print("Number of labels:", len(labels))
import matplotlib.pyplot as plt

# Compute the mean face (average of all images)
mean_face = np.mean(X, axis=1, keepdims=True)

# Subtract mean from each image → mean-zero data (Δ)
Delta = X - mean_face

print("Mean face shape:", mean_face.shape)
print("Delta shape:", Delta.shape)

# Show the mean face
plt.imshow(mean_face.reshape(100, 100), cmap='gray')
plt.title("Mean Face")
plt.axis('off')
plt.show()

# -------------------- STEP 4: PCA / Eigenfaces --------------------
print("\nPerforming PCA... this may take a few seconds...")

# Compute SVD of the mean-zero matrix Δ
U, S, Vt = np.linalg.svd(Delta, full_matrices=False)

print("U shape:", U.shape)    # (mn, p)
print("S shape:", S.shape)    # (p,)
print("Vt shape:", Vt.shape)  # (p, p)

# Select top-k eigenfaces
k = 50   # you can try 10, 20, 50, 100 later
eigenfaces = U[:, :k]

# Normalize each eigenface (unit length)
eigenfaces /= np.linalg.norm(eigenfaces, axis=0, keepdims=True)

print(f"Top {k} eigenfaces computed!")

# Visualize the first 5 eigenfaces
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i, ax in enumerate(axes):
    ax.imshow(eigenfaces[:, i].reshape(100, 100), cmap='gray')
    ax.axis('off')
plt.suptitle("First 5 Eigenfaces")
plt.show()
# -------------------- STEP 5: Generate Signatures --------------------
print("\nProjecting faces onto Eigenfaces...")

# Project all faces onto the eigenfaces to get their feature vectors (signatures)
signatures = eigenfaces.T @ Delta    # shape (k, p)

print("Signatures shape:", signatures.shape)  # expected: (50, 450)

# For sklearn we need samples as rows -> transpose it
X_features = signatures.T   # shape (p, k)
print("Feature matrix shape for ANN:", X_features.shape)
# -------------------- STEP 6: Train ANN Classifier --------------------
print("\nTraining Artificial Neural Network...")

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Split data → 60% training, 40% testing
X_train, X_test, y_train, y_test = train_test_split(
    X_features, labels, test_size=0.4, random_state=42, stratify=labels
)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Build the ANN (1 hidden layer with 100 neurons)
model = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(100,),
                  activation='relu',
                  solver='adam',
                  max_iter=500,
                  random_state=42)
)

# Train the model
model.fit(X_train, y_train)
print("✅ ANN training completed!")

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# -------------------- STEP 7: Save the Model --------------------
# -------------------- STEP 7: Accuracy vs k --------------------
import matplotlib.pyplot as plt

ks = [10, 20, 30, 50, 80, 100]   # number of eigenfaces to test
accuracies = []

for k in ks:
    print(f"\nRunning for k = {k} ...")
    
    # Select top-k eigenfaces
    eigenfaces_k = U[:, :k]
    eigenfaces_k /= np.linalg.norm(eigenfaces_k, axis=0, keepdims=True)
    
    # Project faces onto new eigenfaces
    signatures_k = eigenfaces_k.T @ Delta
    X_features_k = signatures_k.T
    
    # Split and train ANN
    X_train, X_test, y_train, y_test = train_test_split(
        X_features_k, labels, test_size=0.4, random_state=42, stratify=labels
    )
    
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(100,),
                      activation='relu',
                      solver='adam',
                      max_iter=500,
                      random_state=42)
    )
    
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    accuracies.append(acc)
    print(f"Accuracy for k={k}: {acc*100:.2f}%")

# Plot Accuracy vs k
plt.figure(figsize=(7,5))
plt.plot(ks, [a*100 for a in accuracies], marker='o', linewidth=2)
plt.xlabel('k (Number of Eigenfaces)')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Number of Eigenfaces (k)')
plt.grid(True)
plt.show()

# Create dictionary of centroids
unique_labels = np.unique(labels)
centroids = {}

for person in unique_labels:
    idx = np.where(labels == person)[0]        # indices of this person
    centroids[person] = np.mean(X_features[idx], axis=0)  # average signature

def detect_imposter(test_feat, centroids, threshold=50):
    """
    test_feat: 1D numpy array (signature of test image)
    centroids: dict of person_name → mean signature
    threshold: distance above which it is considered unknown
    """
    distances = {person: np.linalg.norm(test_feat - centroids[person]) 
                 for person in centroids}
    min_person = min(distances, key=distances.get)
    min_dist = distances[min_person]
    
    if min_dist > threshold:
        return "unknown"
    else:
        return min_person
# Example: pick some test samples
for i in range(5):
    test_feat = X_test[i]     # features of test image
    true_label = y_test[i]
    pred_label = detect_imposter(test_feat, centroids, threshold=50)
    print(f"True: {true_label}, Predicted: {pred_label}")
# Load unknown face
imp_img = cv2.imread("imp_face.jpg", cv2.IMREAD_GRAYSCALE)
imp_img = cv2.resize(imp_img, (100,100)).flatten().astype(np.float32)
imp_feat = (imp_img - mean_face.flatten()) @ eigenfaces  # project to eigenfaces

pred_label = detect_imposter(imp_feat, centroids, threshold=50)
print("Imposter prediction:", pred_label)  # should print "unknown"











