import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# =========================================================================
# 1. Baseline Model: Custom Convolutional U-Net from Scratch
# =========================================================================

class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class BaselineUNet(nn.Module):
    """
    Standard U-Net architecture trained from scratch.
    Used as an experimental baseline to compare against transfer learning.
    """
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super(BaselineUNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downscaling Encoder
        curr_in = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in, feature))
            curr_in = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Upscaling Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final 1x1 classification head
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            # Handle possible odd spatial dimensions padding
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])

            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_x)

        return self.final_conv(x)


# =========================================================================
# 2. Proposed Model: Transfer Learning U-Net (Pretrained ResNet34)
# =========================================================================

def build_transfer_unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1):
    """
    Builds a U-Net with an ImageNet pre-trained encoder (e.g., ResNet-34 or MobileNetV2).
    Pretrained features enable rapid convergence and high sensitivity on edge cases.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None  # We output raw logits for numerical stability with BCEWithLogitsLoss
    )
    return model


# =========================================================================
# 3. Complementary Approach: Convolutional Autoencoder (Reconstruction)
# =========================================================================

class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder trained to reconstruct normal road scene patterns.
    At inference, anomalous hazards cause higher reconstruction errors:
    Anomaly Heatmap = |Image - Reconstructed_Image|
    """
    def __init__(self, in_channels=3):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # /2
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),          # /4
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),         # /8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),        # /16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # *2
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # *4
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # *8
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1), # *16
            nn.Sigmoid()  # Reconstruct normalized image pixels [0, 1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def compute_anomaly_map(self, x):
        """
        Calculates per-pixel reconstruction error (L1 difference averaged over channels).
        """
        reconstruction = self.forward(x)
        error = torch.mean(torch.abs(x - reconstruction), dim=1, keepdim=True)
        return error

if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 384, 512)
    
    print("Testing BaselineUNet...")
    baseline = BaselineUNet()
    out_b = baseline(dummy_input)
    print("  Baseline output shape:", out_b.shape)
    
    print("\nTesting TransferLearningUNet (ResNet-34)...")
    tl_unet = build_transfer_unet(encoder_name="resnet34")
    out_tl = tl_unet(dummy_input)
    print("  Transfer U-Net output shape:", out_tl.shape)
    
    print("\nTesting ConvAutoencoder...")
    cae = ConvAutoencoder()
    recon = cae(dummy_input)
    err = cae.compute_anomaly_map(dummy_input)
    print("  Autoencoder recon shape:", recon.shape)
    print("  Autoencoder anomaly map shape:", err.shape)
    print("\nAll model architectures initialized and verified successfully!")
