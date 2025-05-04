import torch
import torch.nn as nn
import torchvision.models as models
import clip


def get_feature_extractor(model_name, pretrained=True, device="cpu"):
    """
    Get a feature extractor model by name with minimal code.

    Args:
        model_name: One of 'resnet18', 'mobilenet', or 'clip'
        pretrained: Whether to use pretrained weights
        device: Device to load the model on

    Returns:
        model: The model for feature extraction
        feature_dim: Dimension of the extracted features
    """
    weights = 'DEFAULT' if pretrained else None

    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Identity()  # Remove classification layer
        feature_dim = 512

    elif model_name == "mobilenet":
        model = models.mobilenet_v2(weights=weights)
        model.classifier = nn.Identity()  # Remove classification layer
        feature_dim = 1280

    elif model_name == "clip":
        clip_model, _ = clip.load("ViT-B/32", device=device)
        model = clip_model.visual

        # by default for efficiency reason clip is loaded in fp16
        # to make it fp32
        #model = model.float()
        # OR, return float32 for all

        feature_dim = 512

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # by default for efficiency reason clip is loaded in fp16
    # to make it fp32
    return model.to(device,dtype=torch.float32).eval(), feature_dim