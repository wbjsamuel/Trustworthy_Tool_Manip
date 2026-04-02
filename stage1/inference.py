import os
from typing import Iterable

import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image

from stage1.model.stage1_transformer import Stage1Transformer


def load_config() -> dict:
    with open("stage1/config/stage1.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model_from_config(config: dict, checkpoint_path: str, device: torch.device):
    model = Stage1Transformer.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        embed_dim=config["model"]["embed_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        image_feature_dim=config["model"]["image_feature_dim"],
        language_feature_dim=config["model"]["language_feature_dim"],
        pose_dim=config["model"]["pose_dim"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 1e-4),
        dino_repo=config["model"].get("dino_repo", "facebookresearch/dinov2"),
        dino_model_name=config["model"].get("dino_model_name", "dinov2_vitb14_reg"),
        siglip_model_name=config["model"].get("siglip_model_name", "google/siglip-base-patch16-224"),
        text_model_name=config["model"].get("text_model_name", "t5-base"),
        freeze_backbones=config["model"].get("freeze_backbones", True),
    )
    model.eval()
    model.to(device)
    return model


def infer(image_path: str, instruction: str, current_tool_pose: Iterable[float]) -> np.ndarray:
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = config["inference"]["checkpoint_path"]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Train the model or update the config."
        )

    model = build_model_from_config(config, checkpoint_path, device)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    pose_tensor = torch.as_tensor(current_tool_pose, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_pose = model(image_tensor, pose_tensor, [instruction])

    return predicted_pose.cpu().numpy()


if __name__ == "__main__":
    sample_image = "data/stage1_data/parsed_taco_data/dummy_task/seq_0/rgb/000000.png"
    sample_pose = np.zeros(7, dtype=np.float32)
    if os.path.exists(sample_image):
        print(infer(sample_image, "pick up the tool", sample_pose))
    else:
        print(f"Sample image not found: {sample_image}")
