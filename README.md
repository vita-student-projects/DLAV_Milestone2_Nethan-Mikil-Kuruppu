# Perception-Aware Planning

This project implements an end-to-end deep learning model for trajectory prediction in autonomous driving. The model takes camera images and vehicle motion history as input to predict future vehicle trajectories.

## Project Overview

The autonomous driving trajectory planner:
- Takes dashboard camera images and vehicle history as input
- Predicts the vehicle's future trajectory (60 time steps)
- Uses a ResNet18 backbone for visual feature extraction
- Employs a transformer-based decoder for sequential trajectory prediction
- Implements data augmentation techniques to improve generalization
- Uses specialized loss functions to minimize trajectory errors

## Repository Structure

The main components of the implementation are:

- `DrivingDataset`: Data loading and augmentation
- `DrivingPlanner`: Main model architecture 
- `TransformerTrajectoryDecoder`: Specialized decoder for trajectory prediction
- `TemporalWeightedLoss`: Custom loss function for trajectory prediction
- Training and inference code

## Setup Instructions

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- torchvision
- matplotlib
- numpy
- pandas

### Google Colab Setup

The easiest way to run this code is using Google Colab with GPU acceleration:

1. Upload the notebook to Google Colab
2. Set the runtime type to use GPU:
   - Go to "Runtime" > "Change runtime type" > Select "GPU" as Hardware accelerator
3. Upload your data files or use the provided cells to download the datasets

### Data Setup

The code expects data in the following directory structure:
```
├── train/            # Training data directory (.pkl files)
├── val/              # Validation data directory (.pkl files)
└── test_public/      # Test data directory (.pkl files)
```

Each pickle file contains:
- `camera`: Dashboard camera image (RGB)
- `sdc_history_feature`: Vehicle's past trajectory coordinates (21 time steps)
- `sdc_future_feature`: Vehicle's future trajectory coordinates (60 time steps)

## Running Training

To train the model:

```python
# Setup data directories
train_data_dir = "train"
val_data_dir = "val"

# Get file paths
train_files = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir) if f.endswith('.pkl')]
val_files = [os.path.join(val_data_dir, f) for f in os.listdir(val_data_dir) if f.endswith('.pkl')]

# Create datasets with augmentation for training
train_dataset = DrivingDataset(train_files, augment=True)
val_dataset = DrivingDataset(val_files)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4)

# Create model
model = DrivingPlanner()

# Initialize optimizer
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

# Create logger
logger = Logger()

# Train the model
train(model, train_loader, val_loader, optimizer, logger, num_epochs=100)
```

The training function includes:
- Learning rate scheduling
- Early stopping based on validation ADE
- Model checkpointing (saves the best model)
- ADE (Average Displacement Error) and FDE (Final Displacement Error) metrics

## Running Inference

To run inference and generate predictions:

```python
# Setup test data
test_data_dir = "test_public"
test_files = [os.path.join(test_data_dir, fn) for fn in sorted([f for f in os.listdir(test_data_dir) if f.endswith(".pkl")], key=lambda fn: int(os.path.splitext(fn)[0]))]
test_dataset = DrivingDataset(test_files, test=True)
test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4)

# Load best model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DrivingPlanner()
model.load_state_dict(torch.load("best_phase1_model.pth"))
model.to(device)
model.eval()

# Generate predictions
all_plans = []
with torch.no_grad():
    for batch in test_loader:
        camera = batch['camera'].to(device)
        history = batch['history'].to(device)

        pred_future = model(camera, history)
        all_plans.append(pred_future.cpu().numpy()[..., :2])
all_plans = np.concatenate(all_plans, axis=0)

# Create submission file
pred_xy = all_plans
total_samples, T, D = pred_xy.shape
pred_xy_flat = pred_xy.reshape(total_samples, T * D)

# Format as DataFrame
ids = np.arange(total_samples)
df_xy = pd.DataFrame(pred_xy_flat)
df_xy.insert(0, "id", ids)

# Set column names
new_col_names = ["id"]
for t in range(1, T + 1):
    new_col_names.append(f"x_{t}")
    new_col_names.append(f"y_{t}")
df_xy.columns = new_col_names

# Save to CSV
df_xy.to_csv("submission_phase1.csv", index=False)
```

This generates a CSV file with trajectory predictions in the required format for submission.

## Model Architecture Details

### DrivingPlanner
The main model consists of:
- ResNet18 backbone for image feature extraction
- Spatial attention module for focusing on important visual features
- History encoder for processing past trajectory
- Feature fusion module for combining visual and trajectory features
- Transformer-based decoder for autoregressive trajectory prediction

### TransformerTrajectoryDecoder
The trajectory decoder:
- Uses transformer architecture for sequential prediction
- Implements positional encoding for time steps
- Uses causal masking to ensure autoregressive generation
- Predicts coordinates one time step at a time

### Augmentation Techniques
- Image color jittering (brightness, contrast, saturation, hue)
- Horizontal flipping (for 'forward' driving)
- Trajectory rotation (with heading adjustment)
- Random jitter in trajectory positions
- Speed variation (stretching/compressing trajectories)

## Evaluation Metrics
- ADE (Average Displacement Error): Average L2 distance between predicted and ground truth positions
- FDE (Final Displacement Error): L2 distance at the final prediction point

## Troubleshooting

### Common Issues:
- Out of memory errors: Reduce batch size or model complexity
- Slow training: Ensure you're using GPU acceleration
- Poor predictions: Try increasing augmentation or adjusting learning rate

### Performance Tips:
- Increase training epochs for better convergence
- Adjust learning rate and weight decay
- Try different transformer configurations (layers, heads)
- Experiment with different loss weighting schemes
