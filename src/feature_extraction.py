import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torchvision.models as models
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from src.config import load_config

class FeatureExtractionDataset(Dataset):
    """
    Custom PyTorch Dataset to handle loading and trasforming images.
    """
    def __init__(self, folder_map, valid_extensions, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for folder_path, label in folder_map.items():
            folder = Path(folder_path)
            if not folder.exists():
                print(f"Warning: Folder '{folder}' does not exist. Skipping!")
                continue
            for file_path in folder.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                    self.image_paths.append(str(file_path))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.labels[index]

        try:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label, img_path
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None

def custom_collate(batch):
    """
    Filter out any images that failed to load
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.Tensor(), [], []
    imgs, labels, paths = zip(*batch)
    return torch.stack(imgs), list(labels), list(paths)

def extract_features(batch_size=32, num_workers=4):
    """
    Extract features from images using a pre-trained trochvision model
    """
    config = load_config()
    extract_featrues = config.get("features_extractor", {})
    gen_config = config.get("general_configuration", {})

    model_name = extract_featrues.get("backbone", "wide_resnet_50")
    weights = extract_featrues.get("weights", "DEFAULT")
    folder_map = extract_featrues.get("folder_map", {})

    try:
        model_builder = getattr(models, model_name)
        model = model_builder(weights=weights)
    except AttributeError:
        raise ValueError(f"Model '{model_name}' not found in torchvision.models.")
    
    if hasattr(model, "fc"):
        model.fc = torch.nn.Identity()
    elif hasattr(model, "classifier"):
        model.classifier = torch.nn.Identity()
    elif hasattr(model, "head"):
        model.head = torch.nn.Identity()
    else:
        print("Warning: Could not automatically detect the classification head.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    model.to(device)
    model.eval()
    
    img_size = gen_config.get("img_size", [256, 256])
    valid_extensions = gen_config.get("valid_extensions", [".bmp", ".BMP"])

    transform = v2.Compose([
        v2.Resize(img_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = FeatureExtractionDataset(folder_map=folder_map, valid_extensions=valid_extensions, transform=transform)
    datloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate,
        pin_memory=True if torch.cuda.is_available() else False
    )

    data = []

    print(f"Extracting features on {device}...")
    with torch.no_grad():
        for imgs, labels, paths in tqdm(datloader, desc="Processing batches"):
            if imgs.numel() == 0:
                continue
            
            imgs = imgs.to(device)
            feats = model(imgs)
            feats_np = feats.cpu().numpy()

            for path, label, feat in zip(paths, labels, feats_np):
                record = {
                    'filename': path,
                    'label': label,
                    'features': feat
                }
                data.append(record)
    df = pd.DataFrame(data)
    return df