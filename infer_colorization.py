import torch
import cv2
import numpy as np
from train_colorization import ColorizationNetWithResNet 
import argparse

# load model - assumes model class is defined in train_colorization.py
def load_model(weights_path):
    model = ColorizationNetWithResNet()  # Use new class
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

# Optional sharpening functions 
def sharpen_lab(ab_img, amount=1.2):
    """
    Optional color sharpening by increasing the contrast of ab channels.
    """
    lab = ab_img.astype(np.float32)
    lab[:, :, 1:] = (lab[:, :, 1:] - 128) * amount + 128
    return np.clip(lab, 0, 255)

# Optional unsharp mask in BGR space
def unsharp_mask(bgr_img, strength=1.0):
    """
    High-quality sharpening in BGR space.
    """
    blurred = cv2.GaussianBlur(bgr_img, (0, 0), sigmaX=3)
    sharp = cv2.addWeighted(bgr_img, 1 + strength, blurred, -strength, 0)
    return sharp

# Main colorization function
def colorize(
    model,
    img_path,
    out_path="output.png",
    sharpen=False,
    method="lab_boost",
):
    """
    model: trained ColorizationNet
    img_path: path to input image
    sharpen: True/False (optional sharpening)
    method: "lab_boost" | "unsharp"
    """
    
    # load the image 
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")

    orig_size = (img.shape[1], img.shape[0])

    # preprocess image
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_rs = cv2.resize(lab, (256, 256))

    L = lab_rs[:, :, 0] / 255.0  # correct training normalization

    # L tensor shape (1,1,H,W)
    L_tensor = (
        torch.tensor(L)
        .unsqueeze(0).unsqueeze(0)
        .float()
    )

    # predict ab channels
    with torch.no_grad():
        ab_pred = model(L_tensor).cpu().numpy()[0]  # (2,H,W)


    # debugging info - print stats
    print(f"Model output range: [{ab_pred.min():.3f}, {ab_pred.max():.3f}]")
    print(f"Model output mean: {ab_pred.mean():.3f}")


    ab_pred = ab_pred.transpose(1, 2, 0)  # (H,W,2)

    # denormalize ab channels
    ab_pred = (ab_pred * 128 + 128).clip(0, 255).astype("uint8")

    # reassemble the LAB image
    L_uint8 = (L * 255).astype("uint8")

    lab_out = np.zeros((256, 256, 3), dtype="uint8")
    lab_out[:, :, 0] = L_uint8
    lab_out[:, :, 1:] = ab_pred

    # convert to color using LAB2BGR which handles out-of-gamut colors
    bgr = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    # optional sharpening - either in LAB or BGR space
    if sharpen:
        if method == "lab_boost":
            # boost only chroma channels
            lab_out_boosted = sharpen_lab(lab_out, amount=1.25)
            bgr = cv2.cvtColor(lab_out_boosted.astype("uint8"), cv2.COLOR_LAB2BGR)
        elif method == "unsharp":
            bgr = unsharp_mask(bgr, strength=0.6)

    # upscale back to original size
    bgr = cv2.resize(bgr, orig_size)

    cv2.imwrite(out_path, bgr)
    print(f"Saved colorized output to {out_path}")

    return bgr


if __name__ == "__main__":
    # for consistency: use Places365_val_00000034

    parser = argparse.ArgumentParser(description="Colorize grayscale images")
    parser.add_argument("--input", "-i", type=str, default="test_samples/Places365_val_000000104.jpg",
                       help="Input image path")
    parser.add_argument("--output", "-o", type=str, default="colorized_example.png",
                       help="Output image path")
    parser.add_argument("--sharpen", action="store_true",
                       help="Apply sharpening")
    parser.add_argument("--method", type=str, default="lab_boost", choices=["lab_boost", "unsharp"],
                       help="Sharpening method")
    parser.add_argument("--model", "-m", type=str, default="colorization_model.pth",
                       help="Model weights path")
    args = parser.parse_args()
    
    model = load_model(args.model)
    colorize(model,
             args.input,
             args.output,
             sharpen=args.sharpen,
             method=args.method)
    
