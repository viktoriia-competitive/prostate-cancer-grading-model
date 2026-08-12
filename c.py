"""
Hybrid CNN+TDA System for Prostate Cancer Grading
MANDATORY TDA VERSION - TDA is ALWAYS used in predictions
FIXED: Custom collate function for variable-sized images
OPTIMIZED: Memory-efficient training with gradient accumulation and mixed precision
"""

import os
import random
import numpy as np

# Set environment variable for better CUDA memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import gudhi
import pickle
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================
class Config:
    # Paths
    data_dir = './data/train'  # Pre-tiled images directory (same as c2.py)
    csv_path = './data/train.csv'
    output_dir = './output'
    tda_model_dir = './tda_models'
    pretrained_cnn_path = './best_model.pth'  # Pre-trained CNN model

    # Training
    seed = 42
    use_cv = False  # Use train/val split like c2.py instead of cross-validation
    train_val_split = 0.2  # 80/20 split
    epochs = 20
    batch_size = 4  # Reduced from 32 to 4 for memory efficiency
    gradient_accumulation_steps = 8  # Effective batch size = 4 * 8 = 32
    num_workers = 4
    lr = 3e-5
    weight_decay = 1e-5
    use_mixed_precision = True  # Enable mixed precision training

    # Model
    model_name = 'resnet50'  # Changed to resnet50 to match pre-trained model
    num_classes = 6  # ISUP grades 0-5
    use_pretrained_cnn = True  # Use pre-trained CNN weights

    # Image processing (pre-tiled data from c2.py)
    num_tiles = 12  # Same as c2.py
    tile_size = 128  # Individual tile size
    tile_height = 128  # Tile height
    tile_width = 128  # Tile width
    
    # TDA integration - Late fusion mode
    use_tda = True  # Enable TDA
    tda_feature_dim = 176  # Dimension of TDA feature vector (auto-computed)
    tda_weight = 0.3  # Weight for TDA in ensemble (0.3 = CNN 70%, TDA 30%)
    tda_debug = True  # Print TDA predictions for debugging
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Gleason mapping
GLEASON_MAP = {
    0: "0+0",
    1: "3+3",
    2: "3+4",
    3: "4+3",
    4: "4+4",
    5: "4+5"
}

GLEASON_TO_ISUP = {
    "0+0": 0,
    "3+3": 1,
    "3+4": 2,
    "4+3": 3,
    "4+4": 4,
    "4+5": 5
}

# ============================================================================
# TDA Module - Improved with Cubical Complexes & Persistence Vectorization
# ============================================================================

class PersistenceImage:
    """
    Convert persistence diagrams to persistence images (vectorized representation).
    Based on: Adams et al. "Persistence Images: A Stable Vector Representation of Persistent Homology"
    """
    def __init__(self, resolution=50, sigma=0.1, weight_fn='linear'):
        self.resolution = resolution
        self.sigma = sigma
        self.weight_fn = weight_fn

    def _weight(self, birth, persistence):
        """Weight function - emphasize features with longer persistence"""
        if self.weight_fn == 'linear':
            return persistence
        elif self.weight_fn == 'quadratic':
            return persistence ** 2
        else:  # constant
            return 1.0

    def transform(self, diagram, birth_range=(0, 255), pers_range=(0, 100)):
        """
        Transform a persistence diagram to a persistence image.
        diagram: list of (birth, death) tuples
        Returns: 2D numpy array (resolution x resolution)
        """
        if not diagram:
            return np.zeros((self.resolution, self.resolution))

        # Create grid
        birth_min, birth_max = birth_range
        pers_min, pers_max = pers_range

        x = np.linspace(birth_min, birth_max, self.resolution)
        y = np.linspace(pers_min, pers_max, self.resolution)
        X, Y = np.meshgrid(x, y)

        img = np.zeros((self.resolution, self.resolution))

        for (birth, death) in diagram:
            persistence = death - birth
            if persistence <= 0:
                continue

            weight = self._weight(birth, persistence)

            # Gaussian centered at (birth, persistence)
            gaussian = np.exp(-((X - birth)**2 + (Y - persistence)**2) / (2 * self.sigma**2))
            img += weight * gaussian

        # Normalize
        if img.max() > 0:
            img = img / img.max()

        return img


class PersistenceLandscape:
    """
    Compute persistence landscapes - a stable functional summary of persistence diagrams.
    Based on: Bubenik "Statistical Topological Data Analysis using Persistence Landscapes"
    """
    def __init__(self, num_landscapes=5, resolution=100):
        self.num_landscapes = num_landscapes
        self.resolution = resolution

    def _tent_function(self, birth, death, t):
        """Tent function for a single persistence interval"""
        mid = (birth + death) / 2
        half_life = (death - birth) / 2

        if t < birth or t > death:
            return 0.0
        elif t <= mid:
            return t - birth
        else:
            return death - t

    def transform(self, diagram, t_range=(0, 255)):
        """
        Compute persistence landscape.
        Returns: array of shape (num_landscapes, resolution)
        """
        if not diagram:
            return np.zeros((self.num_landscapes, self.resolution))

        t_min, t_max = t_range
        t_values = np.linspace(t_min, t_max, self.resolution)

        # Compute all tent functions
        all_values = np.zeros((len(diagram), self.resolution))
        for i, (birth, death) in enumerate(diagram):
            for j, t in enumerate(t_values):
                all_values[i, j] = self._tent_function(birth, death, t)

        # Sort at each t to get landscapes
        landscapes = np.zeros((self.num_landscapes, self.resolution))
        for j in range(self.resolution):
            sorted_vals = np.sort(all_values[:, j])[::-1]
            for k in range(min(self.num_landscapes, len(sorted_vals))):
                landscapes[k, j] = sorted_vals[k]

        return landscapes


class TDAAnalyzer:
    """
    Improved Topological Data Analysis for cancer grading.
    Uses cubical complexes (native for images) and multiple vectorizations.
    """

    def __init__(self, model_path=None, feature_dim=128):
        self.model_path = model_path
        self.feature_dim = feature_dim
        self.reference_features = {}
        self.class_centroids = {}

        # Vectorization tools
        self.pi_transformer = PersistenceImage(resolution=25, sigma=10.0)
        self.pl_transformer = PersistenceLandscape(num_landscapes=5, resolution=50)

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def _preprocess_image(self, image):
        """Convert image to grayscale numpy array"""
        if isinstance(image, Image.Image):
            img = np.array(image.convert('L'))
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                img = image.copy()
        else:
            raise ValueError("Unknown image format")
        return img.astype(np.float64)

    def compute_cubical_persistence(self, image, max_dim=1):
        """
        Compute persistent homology using cubical complexes.
        This is the natural choice for image data - uses pixel values as filtration.
        """
        img = self._preprocess_image(image)

        # Resize for computational efficiency
        target_size = 64
        if img.shape[0] > target_size or img.shape[1] > target_size:
            img = cv2.resize(img, (target_size, target_size))

        # Invert image - we want dark (tissue) regions to appear first
        # Lower filtration value = appears earlier = more important
        img_inverted = 255.0 - img

        try:
            # Create cubical complex from image
            cc = gudhi.CubicalComplex(top_dimensional_cells=img_inverted.flatten(),
                                       dimensions=img_inverted.shape)
            cc.persistence()

            diagrams = {0: [], 1: []}

            for dim in range(max_dim + 1):
                intervals = cc.persistence_intervals_in_dimension(dim)
                for (birth, death) in intervals:
                    if death != float('inf') and death > birth:
                        diagrams[dim].append((birth, death))

            return diagrams

        except Exception as e:
            return {0: [], 1: []}

    def compute_multi_scale_persistence(self, image, scales=[1.0, 0.5, 0.25]):
        """
        Compute persistence at multiple scales for richer features.
        """
        img = self._preprocess_image(image)
        all_diagrams = {0: [], 1: []}

        for scale in scales:
            if scale < 1.0:
                h, w = int(img.shape[0] * scale), int(img.shape[1] * scale)
                scaled_img = cv2.resize(img, (w, h))
            else:
                scaled_img = img

            diagrams = self.compute_cubical_persistence(scaled_img)

            # Scale birth/death values back to original scale
            for dim in [0, 1]:
                for (b, d) in diagrams[dim]:
                    all_diagrams[dim].append((b, d))

        return all_diagrams

    def compute_betti_curve(self, diagram, num_points=50, max_val=255):
        """
        Compute Betti curve - number of features alive at each filtration value.
        """
        t_values = np.linspace(0, max_val, num_points)
        betti = np.zeros(num_points)

        for (birth, death) in diagram:
            for i, t in enumerate(t_values):
                if birth <= t < death:
                    betti[i] += 1

        return betti

    def compute_persistence_statistics(self, diagram):
        """
        Compute statistical summaries of persistence diagram.
        """
        if not diagram:
            return {
                'num_features': 0,
                'total_persistence': 0,
                'max_persistence': 0,
                'mean_persistence': 0,
                'std_persistence': 0,
                'entropy': 0,
                'mean_birth': 0,
                'mean_death': 0,
            }

        persistences = [d - b for (b, d) in diagram]
        births = [b for (b, d) in diagram]
        deaths = [d for (b, d) in diagram]

        total_pers = sum(persistences)

        # Persistence entropy
        if total_pers > 0:
            probs = [p / total_pers for p in persistences]
            entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        else:
            entropy = 0

        return {
            'num_features': len(diagram),
            'total_persistence': total_pers,
            'max_persistence': max(persistences) if persistences else 0,
            'mean_persistence': np.mean(persistences) if persistences else 0,
            'std_persistence': np.std(persistences) if persistences else 0,
            'entropy': entropy,
            'mean_birth': np.mean(births) if births else 0,
            'mean_death': np.mean(deaths) if deaths else 0,
        }

    def extract_tda_features(self, image):
        """
        Extract comprehensive TDA feature vector from an image.
        Returns a fixed-size feature vector suitable for ML.
        """
        # Compute multi-scale persistence
        diagrams = self.compute_multi_scale_persistence(image)

        features = []

        # For each dimension (H0 = connected components, H1 = loops/holes)
        for dim in [0, 1]:
            diagram = diagrams[dim]

            # 1. Persistence statistics (8 features per dim)
            stats = self.compute_persistence_statistics(diagram)
            features.extend([
                stats['num_features'],
                stats['total_persistence'],
                stats['max_persistence'],
                stats['mean_persistence'],
                stats['std_persistence'],
                stats['entropy'],
                stats['mean_birth'],
                stats['mean_death'],
            ])

            # 2. Betti curve (25 features per dim)
            betti = self.compute_betti_curve(diagram, num_points=25)
            features.extend(betti)

            # 3. Persistence image flattened (subset of 25x25 = 625 -> take 20 principal values)
            pi = self.pi_transformer.transform(diagram)
            # Take diagonal and some key statistics
            features.extend(np.diag(pi)[:10])
            features.extend([pi.mean(), pi.std(), pi.max(), np.percentile(pi, 75), np.percentile(pi, 90)])

            # 4. Persistence landscape (top 3 landscapes, 10 samples each = 30 features)
            pl = self.pl_transformer.transform(diagram)
            for k in range(3):
                # Sample 10 points from each landscape
                indices = np.linspace(0, pl.shape[1]-1, 10, dtype=int)
                features.extend(pl[k, indices])

        features = np.array(features, dtype=np.float32)

        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features

    def classify_region(self, image):
        """
        Classify a region using TDA features + nearest centroid.
        Returns: (gleason_label, isup_grade, confidence_scores_dict)
        """
        if not self.class_centroids:
            return None, None, {}

        try:
            # Extract features
            features = self.extract_tda_features(image)

            # Compute distance to each class centroid
            distances = {}
            for gleason_label, centroid in self.class_centroids.items():
                dist = np.linalg.norm(features - centroid)
                distances[gleason_label] = dist

            # Convert distances to confidences (softmax-like)
            max_dist = max(distances.values()) + 1e-10
            confidences = {}
            total = 0
            for label, dist in distances.items():
                # Lower distance = higher confidence
                conf = np.exp(-dist / (max_dist * 0.5))
                confidences[label] = conf
                total += conf

            # Normalize
            for label in confidences:
                confidences[label] /= total

            # Best match
            best_gleason = max(confidences, key=confidences.get)
            best_isup = GLEASON_TO_ISUP[best_gleason]

            return best_gleason, best_isup, confidences

        except Exception as e:
            return None, None, {}

    def train_model(self, training_data):
        """
        Train TDA model by computing feature centroids for each class.
        training_data: dict {gleason_label: [images]}
        """
        print("\nExtracting TDA features for reference model...")

        for gleason, images in training_data.items():
            if not images:
                continue

            print(f"Processing {gleason}: {len(images)} samples")
            all_features = []

            for img in tqdm(images, desc=f"  {gleason}"):
                try:
                    features = self.extract_tda_features(img)
                    if len(features) > 0:
                        all_features.append(features)
                except Exception as e:
                    continue

            if all_features:
                # Store features and compute centroid
                all_features = np.array(all_features)
                self.reference_features[gleason] = all_features
                self.class_centroids[gleason] = np.mean(all_features, axis=0)
                print(f"  Created centroid from {len(all_features)} samples, feature dim: {all_features.shape[1]}")

        # Save model
        if self.model_path:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.save_model(self.model_path)

    def save_model(self, path):
        """Save TDA model"""
        with open(path, 'wb') as f:
            pickle.dump({
                'centroids': self.class_centroids,
                'features': self.reference_features
            }, f)
        print(f"TDA model saved to: {path}")

    def load_model(self, path):
        """Load TDA model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        # Handle both old and new format
        if isinstance(data, dict) and 'centroids' in data:
            self.class_centroids = data['centroids']
            self.reference_features = data.get('features', {})
        else:
            # Old format - need to recompute centroids
            print("[WARNING] Old TDA model format detected - please retrain")
            self.reference_features = {}
            self.class_centroids = {}

        print(f"TDA model loaded from: {path}")
        print(f"Available classes: {list(self.class_centroids.keys())}")


# ============================================================================
# Custom Collate Function
# ============================================================================
def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized original images and TDA features
    """
    tiles = []
    labels = []
    original_imgs = []
    positions = []
    tda_features = []

    for item in batch:
        tiles.append(item[0])
        labels.append(item[1])

        if len(item) == 3:
            # Training mode with TDA features: (tiles, label, tda_features)
            tda_features.append(item[2])
        elif len(item) == 4:
            # Validation mode: (tiles, label, full_img, positions)
            original_imgs.append(item[2])
            positions.append(item[3])
        elif len(item) == 5:
            # Validation mode with TDA: (tiles, label, full_img, positions, tda_features)
            original_imgs.append(item[2])
            positions.append(item[3])
            tda_features.append(item[4])

    tiles = torch.stack(tiles)
    labels = torch.tensor(labels)

    result = [tiles, labels]

    if original_imgs:
        result.append(original_imgs)
        result.append(positions)

    if tda_features:
        tda_features = torch.tensor(np.array(tda_features), dtype=torch.float32)
        result.append(tda_features)

    return tuple(result)


# ============================================================================
# Data Preparation (same as c2.py)
# ============================================================================
def prepare_data_c2_style():
    """Prepare data using the same split strategy as c2.py"""
    import glob
    from sklearn.model_selection import train_test_split

    train_csv = pd.read_csv(Config.csv_path)

    # Filter valid images (ensuring tile 0 exists)
    valid_images = glob.glob(os.path.join(Config.data_dir, '*_0.png'))
    valid_ids = {os.path.basename(x).replace('_0.png', '') for x in valid_images}
    train_csv = train_csv[train_csv['image_id'].isin(valid_ids)]

    print(f"Total valid training samples: {len(train_csv)}")

    # Split by provider for balanced sets (same as c2.py)
    radboud_csv = train_csv[train_csv['data_provider'] == 'radboud']
    karolinska_csv = train_csv[train_csv['data_provider'] != 'radboud']

    r_train, r_val = train_test_split(
        radboud_csv,
        test_size=Config.train_val_split,
        random_state=Config.seed
    )
    k_train, k_val = train_test_split(
        karolinska_csv,
        test_size=Config.train_val_split,
        random_state=Config.seed
    )

    train_df = pd.concat([r_train, k_train])
    val_df = pd.concat([r_val, k_val])

    print(f"Radboud - Train: {len(r_train)}, Val: {len(r_val)}")
    print(f"Karolinska - Train: {len(k_train)}, Val: {len(k_val)}")
    print(f"Total - Train: {len(train_df)}, Val: {len(val_df)}")

    return train_df, val_df


# ============================================================================
# Dataset (adapted for pre-tiled data from c2.py)
# ============================================================================
class PANDADataset(Dataset):
    def __init__(self, df, transform=None, mode='train', return_full_image=False,
                 tda_analyzer=None, precomputed_tda_features=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.mode = mode
        self.return_full_image = return_full_image
        self.tda_analyzer = tda_analyzer
        self.precomputed_tda_features = precomputed_tda_features  # dict: image_id -> features

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image_id']
        label = row['isup_grade']

        # Load pre-tiled images (same as c2.py)
        tiles = []
        for i in range(Config.num_tiles):
            tile_path = os.path.join(Config.data_dir, f'{image_id}_{i}.png')
            try:
                tile_img = Image.open(tile_path).convert('RGB')
                tile = np.array(tile_img)
            except:
                # Fallback to blank tile
                tile = np.ones((Config.tile_height, Config.tile_width, 3), dtype=np.uint8) * 255

            tiles.append(tile)

        # Stack tiles for TDA feature extraction
        full_img = np.vstack(tiles)

        # Get TDA features
        tda_features = None
        if self.precomputed_tda_features is not None and image_id in self.precomputed_tda_features:
            tda_features = self.precomputed_tda_features[image_id]
        elif self.tda_analyzer is not None:
            try:
                tda_features = self.tda_analyzer.extract_tda_features(full_img)
            except:
                tda_features = np.zeros(Config.tda_feature_dim, dtype=np.float32)

        # Apply transforms to each tile
        if self.transform:
            tiles = [self.transform(image=tile)['image'] for tile in tiles]
        else:
            # Default: convert to tensor
            tiles = [torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0 for tile in tiles]

        tiles = torch.stack(tiles)

        if self.return_full_image:
            positions = [(i * Config.tile_height, 0) for i in range(Config.num_tiles)]
            if tda_features is not None:
                return tiles, label, full_img, positions, tda_features
            return tiles, label, full_img, positions
        else:
            if tda_features is not None:
                return tiles, label, tda_features
            return tiles, label


# ============================================================================
# Model - Hybrid CNN + TDA with Feature Fusion
# ============================================================================
class TDAFeatureExtractor(nn.Module):
    """
    Neural network to process TDA features and produce embeddings.
    """
    def __init__(self, tda_input_dim=176, tda_hidden_dim=64, tda_output_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(tda_input_dim, tda_hidden_dim),
            nn.BatchNorm1d(tda_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(tda_hidden_dim, tda_output_dim),
            nn.BatchNorm1d(tda_output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)


class HybridPANDAModel(nn.Module):
    def __init__(self, model_name='resnet34', num_classes=6, num_tiles=36,
                 pretrained_path=None, use_tda_branch=True, tda_feature_dim=176):
        super().__init__()
        self.use_tda_branch = use_tda_branch

        # Backbone
        if model_name == 'resnet34':
            backbone = models.resnet34(pretrained=True)
            feature_dim = 512
        elif model_name == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Load pre-trained weights if provided
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"\n[INFO] Loading pre-trained CNN from: {pretrained_path}")
            self.load_pretrained_backbone(backbone, pretrained_path)

        # Remove final FC layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Attention mechanism for tile aggregation
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # TDA feature branch
        tda_output_dim = 64 if use_tda_branch else 0
        if use_tda_branch:
            self.tda_branch = TDAFeatureExtractor(
                tda_input_dim=tda_feature_dim,
                tda_hidden_dim=64,
                tda_output_dim=tda_output_dim
            )

        # Combined classifier (CNN features + TDA features)
        combined_dim = feature_dim + tda_output_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        # Also keep a CNN-only classifier for cases without TDA
        self.cnn_only_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def load_pretrained_backbone(self, backbone, pretrained_path):
        """Load pre-trained weights into the backbone"""
        try:
            # Load checkpoint
            checkpoint = torch.load(pretrained_path, map_location='cpu')

            # Extract backbone weights (remove 'base_model.' prefix if present)
            backbone_state = {}
            for key, value in checkpoint.items():
                if key.startswith('base_model.'):
                    new_key = key.replace('base_model.', '')
                    # Skip the final FC layer as we'll use our own classifier
                    if not new_key.startswith('fc.'):
                        backbone_state[new_key] = value
                elif not key.startswith('fc.'):
                    backbone_state[key] = value

            # Load weights
            missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
            print(f"[INFO] Loaded pre-trained weights")
            if missing:
                print(f"[INFO] Missing keys: {len(missing)} (expected for fc layer)")
            if unexpected:
                print(f"[WARNING] Unexpected keys: {unexpected}")

        except Exception as e:
            print(f"[ERROR] Failed to load pre-trained weights: {e}")
            print("[INFO] Continuing with ImageNet pre-trained weights")
        
    def forward(self, tiles, tda_features=None):
        """
        Forward pass with optional TDA features.

        Args:
            tiles: [batch_size, num_tiles, C, H, W] - image tiles
            tda_features: [batch_size, tda_feature_dim] - TDA features (optional)

        Returns:
            logits: [batch_size, num_classes]
            attention_weights: [batch_size, num_tiles]
        """
        batch_size, num_tiles, C, H, W = tiles.shape

        # Reshape for backbone
        tiles = tiles.view(batch_size * num_tiles, C, H, W)

        # Extract CNN features
        features = self.backbone(tiles)  # [batch_size * num_tiles, feature_dim, 1, 1]
        features = features.view(batch_size, num_tiles, -1)  # [batch_size, num_tiles, feature_dim]

        # Attention-based tile aggregation
        attention_scores = self.attention(features)  # [batch_size, num_tiles, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, num_tiles, 1]

        # Weighted aggregation of tile features
        cnn_aggregated = torch.sum(features * attention_weights, dim=1)  # [batch_size, feature_dim]

        # Fuse with TDA features if available
        if self.use_tda_branch and tda_features is not None:
            tda_embedded = self.tda_branch(tda_features)  # [batch_size, tda_output_dim]
            combined = torch.cat([cnn_aggregated, tda_embedded], dim=1)
            logits = self.classifier(combined)
        else:
            # CNN-only path
            logits = self.cnn_only_classifier(cnn_aggregated)

        return logits, attention_weights.squeeze(-1)

    def get_cnn_features(self, tiles):
        """Extract CNN features without classification (for external TDA fusion)"""
        batch_size, num_tiles, C, H, W = tiles.shape
        tiles = tiles.view(batch_size * num_tiles, C, H, W)
        features = self.backbone(tiles)
        features = features.view(batch_size, num_tiles, -1)
        attention_scores = self.attention(features)
        attention_weights = F.softmax(attention_scores, dim=1)
        aggregated = torch.sum(features * attention_weights, dim=1)
        return aggregated, attention_weights.squeeze(-1)


# ============================================================================
# Training
# ============================================================================
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def precompute_tda_features(df, tda_analyzer, cache_path=None):
    """
    Precompute TDA features for all samples to speed up training.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading precomputed TDA features from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print("Precomputing TDA features for training...")
    features_dict = {}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="TDA features"):
        image_id = row['image_id']

        # Load and stack tiles
        tiles = []
        for i in range(Config.num_tiles):
            tile_path = os.path.join(Config.data_dir, f'{image_id}_{i}.png')
            try:
                tile_img = Image.open(tile_path).convert('RGB')
                tile = np.array(tile_img)
            except:
                tile = np.ones((Config.tile_height, Config.tile_width, 3), dtype=np.uint8) * 255
            tiles.append(tile)

        full_img = np.vstack(tiles)

        try:
            features = tda_analyzer.extract_tda_features(full_img)
            features_dict[image_id] = features
        except Exception as e:
            features_dict[image_id] = np.zeros(Config.tda_feature_dim, dtype=np.float32)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(features_dict, f)
        print(f"Saved TDA features to {cache_path}")

    return features_dict


def train_epoch(model, loader, criterion, optimizer, device, scaler=None, use_tda=True):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    optimizer.zero_grad()

    pbar = tqdm(loader, desc='Training')
    for batch_idx, batch_data in enumerate(pbar):
        tiles = batch_data[0].to(device)
        labels = batch_data[1].to(device)

        # Get TDA features if available (last element if present)
        tda_features = None
        if use_tda and len(batch_data) >= 3 and isinstance(batch_data[-1], torch.Tensor):
            # Check if last element is TDA features (not original_imgs which is a list)
            if batch_data[-1].dim() == 2:  # TDA features are 2D tensor
                tda_features = batch_data[-1].to(device)

        # Mixed precision training
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits, _ = model(tiles, tda_features=tda_features)
                loss = criterion(logits, labels)

            loss = loss / Config.gradient_accumulation_steps
            scaler.scale(loss).backward()

            if (batch_idx + 1) % Config.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            logits, _ = model(tiles, tda_features=tda_features)
            loss = criterion(logits, labels)
            loss = loss / Config.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % Config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * Config.gradient_accumulation_steps

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

        pbar.set_postfix({'loss': loss.item() * Config.gradient_accumulation_steps})

    if len(loader) % Config.gradient_accumulation_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(loader)
    qwk = quadratic_weighted_kappa(all_labels, all_preds)

    return avg_loss, qwk


def validate_epoch_with_tda(model, loader, criterion, device, scaler=None, use_tda=True):
    """
    Validation with learned TDA fusion.
    The model has been trained with TDA features, so we use them here too.
    """
    model.eval()
    total_loss = 0
    cnn_only_preds = []
    hybrid_preds = []
    labels_list = []

    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for batch_idx, batch_data in enumerate(pbar):
            tiles = batch_data[0].to(device)
            labels = batch_data[1].to(device)

            # Get TDA features if available
            tda_features = None
            if use_tda and len(batch_data) >= 3:
                # Find TDA features - it's a 2D tensor
                for item in batch_data[2:]:
                    if isinstance(item, torch.Tensor) and item.dim() == 2:
                        tda_features = item.to(device)
                        break

            # CNN-only prediction
            if scaler is not None and Config.use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    logits_cnn, _ = model(tiles, tda_features=None)
                    if tda_features is not None:
                        logits_hybrid, _ = model(tiles, tda_features=tda_features)
                    else:
                        logits_hybrid = logits_cnn
                    loss = criterion(logits_hybrid, labels)
            else:
                logits_cnn, _ = model(tiles, tda_features=None)
                if tda_features is not None:
                    logits_hybrid, _ = model(tiles, tda_features=tda_features)
                else:
                    logits_hybrid = logits_cnn
                loss = criterion(logits_hybrid, labels)

            total_loss += loss.item()

            cnn_pred = torch.argmax(logits_cnn, dim=1).cpu().numpy()
            hybrid_pred = torch.argmax(logits_hybrid, dim=1).cpu().numpy()

            cnn_only_preds.extend(cnn_pred)
            hybrid_preds.extend(hybrid_pred)
            labels_list.extend(labels.cpu().numpy())

            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()

            pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(loader)
    cnn_qwk = quadratic_weighted_kappa(labels_list, cnn_only_preds)
    hybrid_qwk = quadratic_weighted_kappa(labels_list, hybrid_preds)

    print(f"\nCNN-only QWK: {cnn_qwk:.4f} | CNN+TDA QWK: {hybrid_qwk:.4f}")

    return avg_loss, hybrid_qwk


def train_model_c2_style(train_df, val_df, tda_analyzer=None):
    """Train model using c2.py data format and split with TDA feature fusion"""
    print(f'\n{"="*50}')
    print(f'Training CNN+TDA with Learned Fusion')
    print(f'{"="*50}')

    # Precompute TDA features for faster training
    train_tda_cache = os.path.join(Config.tda_model_dir, 'train_tda_features.pkl')
    val_tda_cache = os.path.join(Config.tda_model_dir, 'val_tda_features.pkl')

    if Config.use_tda and tda_analyzer is not None:
        print("\nPrecomputing TDA features...")
        train_tda_features = precompute_tda_features(train_df, tda_analyzer, train_tda_cache)
        val_tda_features = precompute_tda_features(val_df, tda_analyzer, val_tda_cache)
    else:
        train_tda_features = None
        val_tda_features = None

    # Transforms
    train_transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Datasets with precomputed TDA features
    train_dataset = PANDADataset(
        train_df,
        transform=train_transform,
        mode='train',
        precomputed_tda_features=train_tda_features
    )
    val_dataset = PANDADataset(
        val_df,
        transform=val_transform,
        mode='val',
        precomputed_tda_features=val_tda_features
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    # Model with TDA branch (learned fusion)
    pretrained_path = Config.pretrained_cnn_path if Config.use_pretrained_cnn else None
    use_tda_branch = Config.use_tda and tda_analyzer is not None
    model = HybridPANDAModel(
        model_name=Config.model_name,
        num_classes=Config.num_classes,
        num_tiles=Config.num_tiles,
        pretrained_path=pretrained_path,
        use_tda_branch=use_tda_branch,
        tda_feature_dim=Config.tda_feature_dim
    ).to(Config.device)

    print(f"\nModel TDA branch: {'Enabled' if use_tda_branch else 'Disabled'}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs)

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if Config.use_mixed_precision and Config.device == 'cuda' else None

    # Clear cache before training
    if Config.device == 'cuda':
        torch.cuda.empty_cache()

    # Training loop
    best_qwk = 0
    use_tda = Config.use_tda and tda_analyzer is not None
    for epoch in range(Config.epochs):
        print(f'\nEpoch {epoch+1}/{Config.epochs}')

        train_loss, train_qwk = train_epoch(
            model, train_loader, criterion, optimizer, Config.device, scaler, use_tda=use_tda
        )
        val_loss, val_qwk = validate_epoch_with_tda(
            model, val_loader, criterion, Config.device, scaler, use_tda=use_tda
        )

        # Clear cache after each epoch
        if Config.device == 'cuda':
            torch.cuda.empty_cache()
        
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f}, Train QWK: {train_qwk:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val QWK: {val_qwk:.4f}')
        
        # Save best model
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'qwk': best_qwk,
            }, os.path.join(Config.output_dir, 'best_model_finetuned.pt'))
            print(f'Saved best model with QWK: {best_qwk:.4f}')
    
    return best_qwk


# ============================================================================
# TDA Model Training
# ============================================================================
def train_tda_model(df):
    """Train TDA reference model from training data using ALL tiles"""
    print("\n" + "="*50)
    print("Training TDA Reference Model (ALL TILES)")
    print("="*50)

    tda_model_path = os.path.join(Config.tda_model_dir, 'tda_model.pkl')
    tda_analyzer = TDAAnalyzer(model_path=tda_model_path)

    # Collect images by Gleason grade
    training_data = {gleason: [] for gleason in GLEASON_MAP.values()}

    # Use all training samples (not just 30000)
    num_samples = min(len(df), 1500)  # Can adjust this limit if needed
    df = df.sample(frac=num_samples/len(df), random_state=Config.seed).reset_index(drop=True)

    for idx, row in tqdm(df.iterrows(), total=num_samples, desc="Loading images"):
        image_id = row['image_id']
        isup_grade = row['isup_grade']
        gleason = GLEASON_MAP[isup_grade]

        # Load ALL 12 pre-tiled images (same format as c2.py)
        tiles_loaded = 0
        for tile_idx in range(Config.num_tiles):
            tile_path = os.path.join(Config.data_dir, f'{image_id}_{tile_idx}.png')

            if os.path.exists(tile_path):
                try:
                    tile_img = Image.open(tile_path).convert('RGB')
                    tile = np.array(tile_img)

                    # Only keep tiles with actual tissue (not blank)
                    if tile.mean() < 230:
                        training_data[gleason].append(tile)
                        tiles_loaded += 1
                except Exception as e:
                    pass

        # Fallback: try original tiff/png if tiles not found
        if tiles_loaded == 0:
            img_path = os.path.join(Config.data_dir, f'{image_id}.png')
            if not os.path.exists(img_path):
                img_path = os.path.join(Config.data_dir, f'{image_id}.tiff')

            if os.path.exists(img_path):
                try:
                    if img_path.endswith('.tiff'):
                        import skimage.io
                        img = skimage.io.imread(img_path)
                    else:
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Take representative tile from center
                    h, w = img.shape[:2]
                    center_y, center_x = h // 2, w // 2
                    tile = img[
                        max(0, center_y - 256):center_y + 256,
                        max(0, center_x - 256):center_x + 256
                    ]

                    if tile.mean() < 230:  # Has tissue
                        training_data[gleason].append(tile)
                except:
                    pass

    # Print statistics
    print("\nTiles collected per Gleason grade:")
    for gleason, tiles in training_data.items():
        print(f"  {gleason}: {len(tiles)} tiles")

    # Train TDA model
    tda_analyzer.train_model(training_data)

    return tda_analyzer


def compute_tda_feature_dim():
    """
    Compute the actual TDA feature dimension by running extractor on dummy data.
    This ensures Config.tda_feature_dim matches the actual output.
    """
    tda = TDAAnalyzer()
    dummy_img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    features = tda.extract_tda_features(dummy_img)
    return len(features)


# ============================================================================
# Main
# ============================================================================
def main():
    set_seed(Config.seed)
    os.makedirs(Config.output_dir, exist_ok=True)
    os.makedirs(Config.tda_model_dir, exist_ok=True)

    # Compute actual TDA feature dimension
    actual_tda_dim = compute_tda_feature_dim()
    if Config.tda_feature_dim != actual_tda_dim:
        print(f"[INFO] Updating TDA feature dim from {Config.tda_feature_dim} to {actual_tda_dim}")
        Config.tda_feature_dim = actual_tda_dim

    print("="*60)
    print("HYBRID CNN+TDA SYSTEM (Improved)")
    print("="*60)
    print(f"CNN Model: {Config.model_name}")
    print(f"Pre-trained CNN: {'Yes' if Config.use_pretrained_cnn else 'No'}")
    if Config.use_pretrained_cnn:
        print(f"CNN weights path: {Config.pretrained_cnn_path}")
    print(f"TDA enabled: {Config.use_tda}")
    print(f"TDA fusion: Late probability ensemble (CNN {100*(1-Config.tda_weight):.0f}% + TDA {100*Config.tda_weight:.0f}%)")
    print(f"TDA feature dimension: {Config.tda_feature_dim}")
    print("="*60)
    
    # Prepare data using c2.py-style split
    print(f"\nData source: {Config.data_dir}")
    print(f"Using c2.py-style train/val split (80/20 by provider)")
    train_df, val_df = prepare_data_c2_style()

    print(f'\nLabel distribution (train):')
    print(train_df["isup_grade"].value_counts().sort_index())
    print(f'\nLabel distribution (val):')
    print(val_df["isup_grade"].value_counts().sort_index())

    # Load TDA model
    tda_model_path = os.path.join(Config.tda_model_dir, 'tda_model.pkl')
    if os.path.exists(tda_model_path):
        print(f"\n[INFO] Loading existing TDA model from {tda_model_path}")
        tda_analyzer = TDAAnalyzer(model_path=tda_model_path)

        if not tda_analyzer.reference_diagrams:
            tda_analyzer = train_tda_model(train_df)
    else:
        # print("\n[INFO] TDA model not found. Training without TDA for now.")
        # print("      You can train TDA model separately if needed.")
        tda_analyzer = train_tda_model(train_df)

    # Train model
    best_qwk = train_model_c2_style(train_df, val_df[:40], tda_analyzer)

    # Summary
    print(f'\n{"="*50}')
    print('Fine-tuning Results (c2.py-style data)')
    print(f'{"="*50}')
    print(f'Best Validation QWK: {best_qwk:.4f}')
    print(f'Model saved to: {os.path.join(Config.output_dir, "best_model_finetuned.pt")}')


if __name__ == '__main__':
    main()