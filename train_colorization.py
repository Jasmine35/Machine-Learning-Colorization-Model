import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm
import argparse
import glob
import random
import shutil

# define perceptual loss using VGG19 to help with color realism
# this class computes perceptual loss between two images
class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Load pretrained VGG19 model - we only need features
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        # Select intermediate layers
        self.layers = nn.Sequential(
            vgg[:4].eval(),   # relu1_2
            vgg[4:9].eval(),  # relu2_2
            vgg[9:18].eval(), # relu3_3
        )

        for p in self.layers.parameters():
            p.requires_grad = False

        self.layers.to(device)

        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

        self.crit = nn.L1Loss()

    def forward(self, pred_rgb, target_rgb):
        # Normalize for VGG
        pred_norm = (pred_rgb - self.mean) / self.std
        tgt_norm  = (target_rgb - self.mean) / self.std

        loss = 0.0
        x = pred_norm
        y = tgt_norm

        # removed torch.no_grad to allow gradients to flow for perceptual loss
        for layer in self.layers:
            x = layer(x)
            y = layer(y)
            loss += self.crit(x, y)

        return loss

# Create small subset (for testing training model) - copies random images to new folder
def create_subset(src_folder, dst_folder, num_samples):
    os.makedirs(dst_folder, exist_ok=True)
    # Gather all images recursively
    imgs = [f for f in glob.glob(os.path.join(src_folder, '**', '*.*'), recursive=True)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    selected = random.sample(imgs, min(num_samples, len(imgs)))
    for img in selected:
        shutil.copy(img, os.path.join(dst_folder, os.path.basename(img)))
    print(f"Subset created: {dst_folder} ({len(selected)} images)")

# Dataset class for colorization - loads images, converts to LAB, normalizes, and returns tensors
class ColorizationDataset(Dataset):
    def __init__(self, root_folder, img_size=256):
        self.root_folder = os.path.expanduser(root_folder)
        # Recursively gather all image paths
        self.fnames = [f for f in glob.glob(os.path.join(self.root_folder, '**', '*.*'), recursive=True)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print("Found", len(self.fnames), "images in", self.root_folder)
        self.img_size = img_size

        self.debug_printed = False  # Add this to print only once


    def __len__(self):
        return len(self.fnames)

    # this function loads and preprocesses an image
    def __getitem__(self, idx):
        img_path = self.fnames[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")

        # Convert to LAB and resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype("float32") # convert to float32 so that normalization works correctly
        
        img = cv2.resize(img, (self.img_size, self.img_size))


        # Normalize L channel to [0,1]
        L = img[:, :, 0] / 255.0      
        
        # Normalize a and b channels to [-1, 1]
        ab = (img[:, :, 1:] - 128) / 128.0 # changed because neutral LAB a/b = 128

        # add debugging info - print raw ranges so we can verify normalization - prints only once
        if not self.debug_printed and idx == 0:
            print(f"Raw LAB ranges - L: [{img[:,:,0].min()}, {img[:,:,0].max()}], "
                f"AB: [{img[:,:,1:].min()}, {img[:,:,1:].max()}]")
            print(f"Normalized AB - Min: {ab.min():.3f}, Max: {ab.max():.3f}, Mean: {ab.mean():.3f}")
            self.debug_printed = True


        # Convert to tensors
        L = torch.tensor(L).unsqueeze(0).float()         # (1, H, W)
        ab = torch.tensor(ab).permute(2, 0, 1).float()   # (2, H, W)


        return L, ab

# LAB to RGB conversion using torch and cv2 - for the loss function
def lab_to_rgb_torch(L, ab):
    """
    L:  (B,1,H,W) in [0,1]
    ab: (B,2,H,W) in [-1,1]

    Returns RGB in [0,1], pure PyTorch implementation.
    """

    # Undo normalization 
    L = L * 255.0
    a = ab[:, 0:1, :, :] * 128.0
    b = ab[:, 1:2, :, :] * 128.0

    # Convert L,a,b → X,Y,Z (CIE 1931)
    # LAB → XYZ conversion constants
    fy = (L + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)

    # inverse f
    def finv(t):
        delta = 6.0/29.0
        return torch.where(
            t > delta,
            t ** 3,
            3 * (delta ** 2) * (t - 4/29.0)
        )

    X = 0.95047 * finv(fx)
    Y = 1.00000 * finv(fy)
    Z = 1.08883 * finv(fz)

    # XYZ → linear RGB 
    r =  3.2406*X - 1.5372*Y - 0.4986*Z
    g = -0.9689*X + 1.8758*Y + 0.0415*Z
    b =  0.0557*X - 0.2040*Y + 1.0570*Z

    rgb = torch.cat([r, g, b], dim=1)
    rgb = rgb.clamp(min=0) # ensure no negative values to avoid issues in gamma correction

    # Gamma correction (sRGB) 
    rgb = torch.where(
        rgb > 0.0031308,
        1.055 * torch.pow(rgb.clamp(min=0), 1/2.4) - 0.055,
        12.92 * rgb.clamp(min=0)
    )

    return rgb.clamp(0, 1)

# Model definition using ResNet34 encoder - improved performance than simply using just U-Net
class ColorizationNetWithResNet(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # Modify input conv to accept 1 channel
        self.enc_input = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.enc_input.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.enc2 = resnet.layer1    # Output: 64 channels
        self.enc3 = resnet.layer2    # Output: 128 channels  
        self.enc4 = resnet.layer3    # Output: 256 channels

        # Decoder with corrected channel dimensions
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)  # 16x16 -> 32x32
        self.dec4 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),  # 128 (up4) + 128 (e3) = 256
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)   # 32x32 -> 64x64
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),   # 64 (up3) + 64 (e2) = 128
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(64, 64, 2, stride=2)    # 64x64 -> 128x128
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),   # 64 (up2) + 64 (e1) = 128
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)    # 128x128 -> 256x256
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder with proper ResNet structure
        e1 = self.relu(self.bn1(self.enc_input(x)))  # (64, 128, 128)
        e1_pool = self.maxpool(e1)                   # (64, 64, 64)
        
        e2 = self.enc2(e1_pool)                      # (64, 64, 64)
        e3 = self.enc3(e2)                           # (128, 32, 32)
        e4 = self.enc4(e3)                           # (256, 16, 16)
        
        # Decoder with skip connections
        d4 = self.up4(e4)                            # (128, 32, 32)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))   # (128, 32, 32)
        
        d3 = self.up3(d4)                            # (64, 64, 64)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))   # (64, 64, 64)
        
        d2 = self.up2(d3)                            # (64, 128, 128)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))   # (64, 128, 128) - Use e1 (before pool) for skip
        
        d1 = self.up1(d2)                            # (32, 256, 256)
        d1 = self.dec1(d1)                           # (32, 256, 256)
        
        out = self.out_conv(d1)                      # (2, 256, 256)
        return out
    
# Function to resume training from epoch checkpoint - loads model and optimizer states, continues training
def resume_training(ckpt_path, train_path, val_path, total_epochs=50, batch_size=8, img_size=256, lr=1e-5, clip_norm=1.0):
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # Debug: Check checkpoint
    print(f"Checkpoint keys: {list(ckpt.keys())}")
    print(f"Checkpoint was at epoch: {ckpt['epoch']}")
    print(f"Best validation L1: {ckpt.get('best_val_l1', 'Not found')}")
    
    # Define device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Resuming training on device: {device}")
    
    # Create model and move to device
    model = ColorizationNetWithResNet().to(device)
    
    # Check state dict match - debugging
    model_dict = model.state_dict()
    checkpoint_dict = ckpt["model_state"]
    
    missing = [k for k in model_dict.keys() if k not in checkpoint_dict]
    unexpected = [k for k in checkpoint_dict.keys() if k not in model_dict]
    
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    
    # Load model weights with strict=False to handle potential mismatches
    model.load_state_dict(checkpoint_dict, strict=False)
    print("Model weights loaded successfully")
    
    # Test forward pass with dummy data - debugging
    with torch.no_grad():
        dummy_input = torch.randn(1, 1, img_size, img_size).to(device)
        dummy_output = model(dummy_input)
        print(f"Dummy test - Output range: [{dummy_output.min():.3f}, {dummy_output.max():.3f}], Std: {dummy_output.std().item():.3f}")
    
    # Create optimizer and load its state
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    
    # Check if optimizer state exists in checkpoint
    if "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
        print("Optimizer state loaded")
        
        # Update learning rate to the new value (if different)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            print(f"Set learning rate to: {lr}")
    else:
        print("Warning: No optimizer state in checkpoint")
    
    # Get starting epoch
    start_epoch = ckpt["epoch"] + 1
    print(f"Resuming training from epoch {start_epoch}")
    
    # Calculate remaining epochs
    remaining_epochs = total_epochs - start_epoch
    if remaining_epochs <= 0:
        print(f"Training already completed. Checkpoint is at epoch {ckpt['epoch']}, requested total is {total_epochs}")
        return
    
    print(f"Will train for {remaining_epochs} more epochs (epochs {start_epoch} to {total_epochs-1})")
    
    # Get best validation loss
    best_val_l1 = ckpt.get("best_val_l1", float("inf"))
    print(f"Previous best validation L1: {best_val_l1:.4f}")
    
    # Call train_model with all parameters
    train_model(train_path, val_path,
                epochs=total_epochs, 
                batch_size=batch_size,
                lr=lr,
                img_size=img_size,
                start_epoch=start_epoch,
                model=model,
                optimizer=optimizer,
                resume=True,
                best_val_l1=best_val_l1, 
                clip_norm=clip_norm)

# training function - this main training loop handles data loading, loss computation, backpropagation, and validation
def train_model(train_path, val_path=None, epochs=20, batch_size=8, lr=1e-5, img_size=256, start_epoch=0, model=None, optimizer=None, resume=False, best_val_l1=float("inf"), clip_norm=1.0):

    # for better performance, want to use MPS if available on Mac
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on {device} ...")

    # Add debug info: make sure device is mps so it has better performance on Mac
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    # train_dataset - load training data
    # create DataLoader for training and validation
    train_dataset = ColorizationDataset(train_path, img_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False) # added num_workers and pin_memory for performance

    # validation dataset - if val_path is provided then create val_loader which is used for validation after each epoch
    if val_path:
        val_dataset = ColorizationDataset(val_path, img_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False) # fewer workers for validation
    else:
        val_loader = None
    
    if model is None:
        model = ColorizationNetWithResNet().to(device)  
    
    # L1 is a good loss function for colorization - computes loss of predicted ab channels (want this to be minimized)
    l1_loss = nn.L1Loss()

    # to improve color realism, add perceptual loss
    perceptual = VGGPerceptualLoss(device)

    # lower learning weight for pretrained layers
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # prevents LR from shrinking too low - set a minimum LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, min_lr=1e-7)


    for epoch in range(start_epoch, epochs):
        
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        # Freeze encoder layers for first few epochs when not resuming
        if not resume:
            for name, param in model.named_parameters():
                if "enc" in name:
                    param.requires_grad = False

        # Unfreeze after 5 epochs - helps stabilize training
        if epoch == 5 and not resume:
            for param in model.parameters():
                param.requires_grad = True
            print("Unfroze encoder")

        for batch_idx, (L, ab) in enumerate(loop):
            L, ab = L.to(device), ab.to(device)
            pred_ab = model(L)

            # Base L1 loss
            loss_l1 = l1_loss(pred_ab, ab)
            
            # Debug: Monitor output statistics
            if random.random() < 0.01:
                print(f"Output range: [{pred_ab.min():.3f}, {pred_ab.max():.3f}], Std: {pred_ab.std():.3f}")
            
            # Safer perceptual loss when resuming
            '''warmup_epochs = 10
            if resume:
                # Delay and reduce perceptual loss when resuming
                if epoch >= max(warmup_epochs, start_epoch + 5):  # Wait at least 5 epochs after resume
                    max_w = 0.001  # Much smaller when resuming
                    w = max_w * min(1.0, (epoch - max(warmup_epochs, start_epoch + 5)) / 10)
                else:
                    w = 0.0
            elif epoch >= warmup_epochs:
                max_w = 0.001  # Reduced from 0.01
                w = max_w * min(1.0, (epoch - warmup_epochs) / 10)
            else:
                w = 0.0'''
            
            w = 0.0  # No perceptual loss at all - simplified for stability
            
            if w > 0:
                pred_ab_clamped = pred_ab.clamp(-0.95, 0.95)

                pred_rgb = lab_to_rgb_torch(L, pred_ab_clamped).to(device)
                tgt_rgb = lab_to_rgb_torch(L, ab).to(device)
                loss_perc = perceptual(pred_rgb, tgt_rgb)
            else:
                loss_perc = torch.tensor(0.0).to(device)
            
            # Combined loss - no regularization
            loss = loss_l1 + w * loss_perc
            
            # Check for NaN/inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss: {loss.item()}, skipping batch")
                optimizer.zero_grad()
                continue
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient monitoring and clipping - USE clip_norm parameter
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            
            # Log gradient norm occasionally
            if random.random() < 0.01:
                print(f"Grad norm: {total_norm:.4f}, Loss L1: {loss_l1.item():.4f}")
            
            if total_norm > 50:
                print(f"⚠️ Warning: High gradient norm: {total_norm:.2f}")
            
            optimizer.step()
            
            # Update progress bar
            loop.set_postfix(
                l1=loss_l1.item(), 
                perc=loss_perc.item() if w > 0 else 0.0, 
                grad_norm=total_norm.item()
            )
        
        # Validation - check performance on validation set after each epoch
        if val_loader:
            model.eval()
            val_loss_l1 = 0
            with torch.no_grad():
                for L, ab in val_loader:
                    L, ab = L.to(device), ab.to(device)
                    pred_ab = model(L)
                    val_loss_l1 += l1_loss(pred_ab, ab).item()
            
            val_loss_l1 /= len(val_loader)
            scheduler.step(val_loss_l1)  # Update learning rate
            
            print(f"Epoch {epoch+1} Val L1: {val_loss_l1:.4f}")
            
            # Save best checkpoint
            if val_loss_l1 < best_val_l1:
                best_val_l1 = val_loss_l1
                ckpt = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_l1": best_val_l1
                }
                torch.save(ckpt, "best_checkpoint.pth")
                print(f"Saved best model at epoch {epoch+1} (Val L1: {val_loss_l1:.4f})")

    # Save the final model 
    torch.save(model.state_dict(), "colorization_model.pth")
    print("Model saved as colorization_model.pth")

# command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train colorization model")
    parser.add_argument("--train", type=str, default="train_samples", help="Training dataset folder")
    parser.add_argument("--val", type=str, default="valid_samples", help="Validation dataset folder")
    parser.add_argument("--subset", type=int, default=None, help="Number of images to use for small subset (optional)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--img_size", type=int, default=256, help="Resize image to this size")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--clip_norm", type=float, default=0.5, help="Gradient clipping norm")  # gradient clipping parameter to prevent explosion
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")  # resume from checkpoint to continue training
    args = parser.parse_args()
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        resume_training(
            ckpt_path=args.resume,
            train_path=args.train,
            val_path=args.val,
            total_epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            lr=args.lr,
            clip_norm=args.clip_norm
        )

    else:
        # If subset is requested (for testing), create temp folders
        if args.subset:
            temp_train = "train_subset"
            temp_val   = "val_subset"
            create_subset(args.train, temp_train, args.subset)
            if args.val:
                create_subset(args.val, temp_val, args.subset)
            train_model(temp_train, val_path=temp_val if args.val else None,
                        epochs=args.epochs, batch_size=args.batch_size, img_size=args.img_size)
        else:
            train_model(args.train, val_path=args.val, epochs=args.epochs, batch_size=args.batch_size, img_size=args.img_size)