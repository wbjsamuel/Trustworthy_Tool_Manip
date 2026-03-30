import torch
import yaml
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
from stage1.model.stage1_transformer import Stage1Transformer

def infer(image_path, instruction, current_tool_pose):
    # Load configuration
    with open('stage1/config/stage1.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model from checkpoint
    checkpoint_path = config['inference']['checkpoint_path']
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please train the model first and update the path in the config.")
        
    model = Stage1Transformer.load_from_checkpoint(checkpoint_path).to(device)
    model.eval()

    # Prepare inputs
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    instruction = [instruction] # list of strings
    
    current_tool_pose = torch.from_numpy(current_tool_pose).float().unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        predicted_pose = model(image, current_tool_pose, instruction)

    return predicted_pose.cpu().numpy()

if __name__ == '__main__':
    # Example usage
    
    # Create a dummy checkpoint for testing if it doesn't exist
    checkpoint_path = "checkpoints/last.ckpt"
    if not os.path.exists(checkpoint_path):
        print("Creating dummy checkpoint for testing...")
        os.makedirs("checkpoints", exist_ok=True)
        # We need a model to save a checkpoint
        model = Stage1Transformer()
        trainer = pl.Trainer(max_epochs=1)
        # A dummy dataloader is needed to save a checkpoint
        dummy_loader = torch.utils.data.DataLoader(
            [{
                "image": torch.randn(3, 224, 224),
                "instruction": "dummy",
                "current_tool_pose": torch.randn(7),
                "target_pose": torch.randn(7)
            }]
        )
        try:
            # We need to run a dummy training step to create a checkpoint
            trainer.fit(model, train_dataloader=dummy_loader)
            trainer.save_checkpoint(checkpoint_path)
            print("Dummy checkpoint created.")
        except Exception as e:
            print(f"Could not create a dummy checkpoint due to: {e}")
            print("Please train the model to generate a valid checkpoint.")


    # Create dummy data for inference if it doesn't exist
    data_dir = "data/stage1_data/parsed_taco_data/episode_001"
    image_path = os.path.join(data_dir, "image_000.png")
    if not os.path.exists(image_path):
        os.makedirs(data_dir, exist_ok=True)
        Image.new('RGB', (224, 224)).save(image_path)


    instruction = "Pick up the tool."
    current_tool_pose = np.random.rand(7)
    
    if os.path.exists(checkpoint_path):
        predicted_pose = infer(image_path, instruction, current_tool_pose)
        print("Predicted Pose:", predicted_pose)
    else:
        print("Could not run inference because a valid checkpoint was not found.")

