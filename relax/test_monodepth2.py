"""Test Monodepth2 with pretrained weights."""

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2


def load_monodepth2_models(model_path="model/pth"):
    """Load pretrained encoder and decoder models."""
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_path = os.path.join(model_path, "depth.pth")

    print(f"Loading encoder from: {encoder_path}")
    print(f"Loading depth decoder from: {depth_path}")

    # Load the state dictionaries
    encoder_dict = torch.load(encoder_path, map_location='cpu')
    depth_dict = torch.load(depth_path, map_location='cpu')

    return encoder_dict, depth_dict


def build_encoder_decoder():
    """Build encoder and decoder from monodepth2 repository structure."""
    # Import monodepth2 networks
    # Note: You'll need the original monodepth2 repository
    # Clone it from: https://github.com/nianticlabs/monodepth2

    try:
        import sys
        sys.path.append(".")

        # Import from monodepth2 (if available)
        from networks import ResnetEncoder, DepthDecoder

        # Build models
        encoder = ResnetEncoder(num_layers=18, pretrained=False)
        decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc)

        return encoder, decoder
    except ImportError:
        print("Warning: Could not import monodepth2 networks.")
        print("Please clone the monodepth2 repository:")
        print("  git clone https://github.com/nianticlabs/monodepth2.git")
        print("  cd monodepth2")
        return None, None


def load_image(image_path, height=192, width=640):
    """Load and preprocess an image."""
    # Load image
    img = Image.open(image_path).convert('RGB')

    # Resize to model input size
    original_width, original_height = img.size
    img_resized = img.resize((width, height), Image.LANCZOS)

    # Convert to tensor and normalize
    transform = transforms.ToTensor()
    img_tensor = transform(img_resized)

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor, (original_height, original_width)


def disp_to_depth(disp, min_depth=0.1, max_depth=100.0):
    """Convert disparity to depth."""
    min_disp = 1.0 / max_depth
    max_disp = 1.0 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1.0 / scaled_disp
    return scaled_disp, depth


def visualize_depth(depth_map, save_path=None):
    """Visualize depth map."""
    # Normalize depth for visualization
    depth_np = depth_map.squeeze().cpu().numpy()

    # Apply colormap
    vmax = np.percentile(depth_np, 95)
    normalizer = plt.Normalize(vmin=depth_np.min(), vmax=vmax)
    mapper = plt.cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_depth = (mapper.to_rgba(depth_np)[:, :, :3] * 255).astype(np.uint8)

    # Display
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(depth_np, cmap='magma')
    plt.title('Depth Map')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(colormapped_depth)
    plt.title('Depth Map (Colored)')
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to: {save_path}")

    plt.tight_layout()
    plt.show()

    return colormapped_depth


def test_with_pytorch_models(image_path, model_path="model/pth"):
    """Test monodepth2 using PyTorch models directly."""
    print("="*60)
    print("Testing Monodepth2 with PyTorch Models")
    print("="*60)

    # Load models
    encoder_dict, depth_dict = load_monodepth2_models(model_path)
    encoder, decoder = build_encoder_decoder()

    if encoder is None or decoder is None:
        print("\nFalling back to weight inspection...")
        print("\nEncoder weights structure:")
        print(f"Number of parameters: {len(encoder_dict)}")
        for key in list(encoder_dict.keys())[:10]:
            print(f"  {key}: {encoder_dict[key].shape}")
        print("  ...")

        print("\nDepth decoder weights structure:")
        print(f"Number of parameters: {len(depth_dict)}")
        for key in list(depth_dict.keys())[:10]:
            print(f"  {key}: {depth_dict[key].shape}")
        print("  ...")
        return

    # Load weights
    encoder.load_state_dict(encoder_dict)
    decoder.load_state_dict(depth_dict)

    # Set to evaluation mode
    encoder.eval()
    decoder.eval()

    # Load and preprocess image
    print(f"\nLoading image from: {image_path}")
    input_image, original_size = load_image(image_path)

    print(f"Input image shape: {input_image.shape}")
    print(f"Original image size: {original_size}")

    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        # Encoder forward pass
        features = encoder(input_image)

        # Decoder forward pass
        outputs = decoder(features)

        # Get disparity at scale 0 (finest scale)
        disp = outputs[("disp", 0)]

        # Convert to depth
        _, depth = disp_to_depth(disp)

    print(f"Output disparity shape: {disp.shape}")
    print(f"Output depth shape: {depth.shape}")
    print(f"Depth range: [{depth.min().item():.2f}, {depth.max().item():.2f}]")

    # Visualize
    print("\nVisualizing results...")
    output_path = "output_depth.png"
    visualize_depth(depth, save_path=output_path)

    return depth


def simple_weight_inspection(model_path="model/pth"):
    """Simple inspection of the weight files."""
    print("="*60)
    print("Inspecting Monodepth2 Weight Files")
    print("="*60)

    encoder_dict, depth_dict = load_monodepth2_models(model_path)

    print("\n" + "="*60)
    print("ENCODER WEIGHTS")
    print("="*60)
    print(f"Total parameters: {len(encoder_dict)}")

    # Group by layer type
    conv_layers = [k for k in encoder_dict.keys() if 'conv' in k.lower()]
    bn_layers = [k for k in encoder_dict.keys() if 'bn' in k.lower()]

    print(f"\nConvolution layers: {len(conv_layers)}")
    for layer in conv_layers[:5]:
        print(f"  {layer}: {encoder_dict[layer].shape}")
    if len(conv_layers) > 5:
        print(f"  ... and {len(conv_layers) - 5} more")

    print(f"\nBatch normalization layers: {len(bn_layers)}")
    for layer in bn_layers[:5]:
        print(f"  {layer}: {encoder_dict[layer].shape}")
    if len(bn_layers) > 5:
        print(f"  ... and {len(bn_layers) - 5} more")

    print("\n" + "="*60)
    print("DEPTH DECODER WEIGHTS")
    print("="*60)
    print(f"Total parameters: {len(depth_dict)}")

    # Group by decoder layers
    upconv_layers = [k for k in depth_dict.keys() if 'upconv' in k.lower()]
    dispconv_layers = [k for k in depth_dict.keys() if 'dispconv' in k.lower()]

    print(f"\nUpsampling convolution layers: {len(upconv_layers)}")
    for layer in upconv_layers[:10]:
        print(f"  {layer}: {depth_dict[layer].shape}")
    if len(upconv_layers) > 10:
        print(f"  ... and {len(upconv_layers) - 10} more")

    print(f"\nDisparity convolution layers: {len(dispconv_layers)}")
    for layer in dispconv_layers:
        print(f"  {layer}: {depth_dict[layer].shape}")


def main():
    """Main testing function."""
    import argparse

    parser = argparse.ArgumentParser(description='Test Monodepth2')
    parser.add_argument('--image', type=str, default='data/0000000000.png',
                        help='Path to test image')
    parser.add_argument('--model_path', type=str, default='model/pth',
                        help='Path to model weights directory')
    parser.add_argument('--inspect_only', action='store_true',
                        help='Only inspect weights without running inference')

    args = parser.parse_args()

    if args.inspect_only:
        simple_weight_inspection(args.model_path)
    else:
        # Check if image exists
        if not os.path.exists(args.image):
            print(f"Error: Image not found at {args.image}")
            print("Available images:")
            data_dir = os.path.dirname(args.image)
            if os.path.exists(data_dir):
                images = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])
                for img in images[:10]:
                    print(f"  {os.path.join(data_dir, img)}")
            return

        try:
            test_with_pytorch_models(args.image, args.model_path)
        except Exception as e:
            print(f"\nError during inference: {e}")
            print("\nFalling back to weight inspection...")
            simple_weight_inspection(args.model_path)


if __name__ == "__main__":
    main()
