# used to test normalization and denormalization consistency between training and inference scripts 
import cv2
import numpy as np

# Load an image
img = cv2.imread("test_samples/Places365_val_00036331.jpg")
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# --- Normalize (same as training) ---
L = img_lab[:, :, 0] / 255.0
ab = (img_lab[:, :, 1:] - 128) / 128.0

print("Normalized L range:", L.min(), L.max())
print("Normalized ab range:", ab.min(), ab.max())

# --- Denormalize (reverse of training) ---
L_denorm = (L * 255.0).astype(np.uint8)
ab_denorm = (ab * 128.0 + 128).astype(np.uint8)

# --- Reconstruct LAB image ---
lab_reconstructed = np.stack([L_denorm, ab_denorm[:, :, 0], ab_denorm[:, :, 1]], axis=2)

# Convert back to BGR
bgr_reconstructed = cv2.cvtColor(lab_reconstructed, cv2.COLOR_LAB2BGR)

# Save and compare
cv2.imwrite("reconstructed.png", bgr_reconstructed)