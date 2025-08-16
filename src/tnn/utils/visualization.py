import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Optional, List, Tuple, Union
import matplotlib.patches as patches
import os
from pathlib import Path
import pickle

def ensure_viz_dir(save_path: Optional[Union[str, Path]] = None, file_type: str = "image") -> Path:
    """
    Ensure appropriate directory exists and return the full path
    
    Args:
        save_path: Optional path for the file
        file_type: Type of file ('image' for .png, 'attribution' for .pkl, 'other' for general)
        
    Returns:
        Path object for the file location
    """
    # Determine the base directory structure
    if file_type == "image":
        base_dir = Path("info/images")
    elif file_type == "attribution":
        base_dir = Path("info/attributions")
    else:
        base_dir = Path("info")
    
    # Create directories if they don't exist
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # If save_path is provided, use it within the appropriate directory
    if save_path:
        save_path = Path(save_path)
        if save_path.parent.name not in ["images", "attributions", "info"]:
            # If the save_path doesn't already specify the organized structure,
            # place it in the appropriate subdirectory
            full_path = base_dir / save_path.name
        else:
            # If the path already has the structure, create the full directory tree
            full_path = Path(save_path)
            full_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        full_path = base_dir
    
    return full_path

def plot_decision_boundary(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    title: str = "Decision Boundary",
    save_path: Optional[Union[str, Path]] = None,
    resolution: int = 100
):
    """
    Plot decision boundary for 2D classification problems
    
    Args:
        model: Trained model
        X: Input features of shape (n_samples, 2)
        y: Labels of shape (n_samples,) or (n_samples, 1)
        title: Plot title
        save_path: Path to save the plot (will be saved in info/images/)
        resolution: Resolution of the decision boundary mesh
    """
    model.eval()
    
    # Determine save path for image
    if save_path:
        image_path = ensure_viz_dir(save_path, "image")
    else:
        image_path = ensure_viz_dir("decision_boundary.png", "image")
    
    # Ensure y is 1D
    if y.dim() > 1:
        y = y.squeeze()
    
    # Move data to CPU for mesh creation (numpy operations)
    X_cpu = X.cpu()
    y_cpu = y.cpu()
    
    # Create mesh
    x_min, x_max = X_cpu[:, 0].min() - 0.5, X_cpu[:, 0].max() + 0.5
    y_min, y_max = X_cpu[:, 1].min() - 0.5, X_cpu[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    # Get predictions on mesh
    mesh_points = torch.tensor(
        np.c_[xx.ravel(), yy.ravel()], 
        dtype=torch.float32,
        device=X.device
    )
    
    with torch.no_grad():
        mesh_pred = model(mesh_points)
        if mesh_pred.shape[1] == 1:  # Binary classification
            Z = torch.sigmoid(mesh_pred).cpu().numpy()
        else:  # Multi-class
            Z = torch.softmax(mesh_pred, dim=1)[:, 1].cpu().numpy()
    
    Z = Z.reshape(xx.shape)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot decision boundary
    contour = plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    plt.colorbar(contour, label='Prediction Probability')
    
    # Plot data points
    X_np, y_np = X_cpu.numpy(), y_cpu.numpy()
    unique_labels = np.unique(y_np)
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, label in enumerate(unique_labels):
        mask = y_np == label
        plt.scatter(
            X_np[mask, 0], X_np[mask, 1],
            c=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            label=f'Class {int(label)}',
            s=60,
            edgecolors='black',
            alpha=0.8
        )
    
    plt.xlabel('X1', fontsize=12)
    plt.ylabel('X2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save PNG in info/images/
    plt.savefig(image_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {image_path}")
    
    # Save numerical data as pickle in info/attributions/
    pickle_name = image_path.stem + '.pkl'  # Same base name but .pkl extension
    pickle_path = ensure_viz_dir(pickle_name, "attribution")
    numerical_data = {
        'mesh_xx': xx,
        'mesh_yy': yy,
        'decision_values': Z,
        'data_points_X': X_np,
        'data_points_y': y_np,
        'mesh_resolution': resolution
    }
    with open(pickle_path, 'wb') as f:
        pickle.dump(numerical_data, f)
    print(f"Saved numerical data: {pickle_path}")
    
    plt.show()

def plot_prototypes(
    prototypes: torch.Tensor,
    feature_bank: torch.Tensor,
    title: str = "Learned Prototypes",
    save_path: Optional[Union[str, Path]] = None
):
    """
    Visualize learned prototypes and feature bank
    
    Args:
        prototypes: Prototype tensor of shape (num_prototypes, feature_dim)
        feature_bank: Feature bank tensor of shape (feature_dim,)
        title: Plot title
        save_path: Path to save the plot (will be saved in info/images/)
    """
    # Determine save path for image
    if save_path:
        image_path = ensure_viz_dir(save_path, "image")
    else:
        image_path = ensure_viz_dir("prototypes.png", "image")
    
    prototypes_np = prototypes.cpu().numpy()
    feature_bank_np = feature_bank.cpu().numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot prototypes
    im1 = ax1.imshow(prototypes_np, cmap='viridis', aspect='auto')
    ax1.set_title('Learned Prototypes (Π)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Feature Dimension')
    ax1.set_ylabel('Prototype Index')
    plt.colorbar(im1, ax=ax1)
    
    # Plot feature bank
    ax2.bar(range(len(feature_bank_np)), feature_bank_np, color='skyblue', alpha=0.7)
    ax2.set_title('Learned Feature Bank (Ω)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('Feature Weight')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save PNG in info/images/
    plt.savefig(image_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {image_path}")
    
    # Save numerical data as pickle in info/attributions/
    pickle_name = image_path.stem + '.pkl'  # Same base name but .pkl extension
    pickle_path = ensure_viz_dir(pickle_name, "attribution")
    numerical_data = {
        'prototypes': prototypes_np,
        'feature_bank': feature_bank_np
    }
    with open(pickle_path, 'wb') as f:
        pickle.dump(numerical_data, f)
    print(f"Saved numerical data: {pickle_path}")
    
    plt.show()

def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_accuracies: Optional[List[float]] = None,
    val_accuracies: Optional[List[float]] = None,
    title: str = "Training Curves",
    save_path: Optional[Union[str, Path]] = None
):
    """
    Plot training curves for loss and accuracy
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accuracies: List of training accuracies
        val_accuracies: List of validation accuracies
        title: Plot title
        save_path: Path to save the plot (will be saved in info/images/)
    """
    # Determine save path for image
    if save_path:
        image_path = ensure_viz_dir(save_path, "image")
    else:
        image_path = ensure_viz_dir("training_curves.png", "image")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if val_losses:
        val_epochs = range(1, len(val_losses) + 1)
        axes[0].plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracies
    if train_accuracies:
        axes[1].plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    if val_accuracies:
        val_epochs = range(1, len(val_accuracies) + 1)
        axes[1].plot(val_epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    if train_accuracies or val_accuracies:
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No accuracy data', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Accuracy (No Data)')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save PNG in info/images/
    plt.savefig(image_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {image_path}")
    
    # Save numerical data as pickle in info/attributions/
    pickle_name = image_path.stem + '.pkl'  # Same base name but .pkl extension
    pickle_path = ensure_viz_dir(pickle_name, "attribution")
    numerical_data = {
        'train_losses': train_losses,
        'val_losses': val_losses if val_losses else [],
        'train_accuracies': train_accuracies if train_accuracies else [],
        'val_accuracies': val_accuracies if val_accuracies else []
    }
    with open(pickle_path, 'wb') as f:
        pickle.dump(numerical_data, f)
    print(f"Saved numerical data: {pickle_path}")
    
    plt.show()

def plot_similarity_heatmap(
    similarities: torch.Tensor,
    prototype_labels: Optional[List[str]] = None,
    sample_labels: Optional[List[str]] = None,
    title: str = "Tversky Similarities",
    save_path: Optional[Union[str, Path]] = None
):
    """
    Plot heatmap of Tversky similarities between samples and prototypes
    
    Args:
        similarities: Similarity matrix of shape (n_samples, n_prototypes)
        prototype_labels: Labels for prototypes
        sample_labels: Labels for samples
        title: Plot title
        save_path: Path to save the plot
    """
    similarities_np = similarities.cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    
    # Convert ranges to lists for proper type handling
    x_labels = prototype_labels if prototype_labels else [str(i) for i in range(similarities.shape[1])]
    y_labels = sample_labels if sample_labels else [str(i) for i in range(similarities.shape[0])]
    
    # Create heatmap
    sns.heatmap(
        similarities_np,
        annot=True,
        fmt='.3f',
        cmap='viridis',
        xticklabels=x_labels,
        yticklabels=y_labels,
        cbar_kws={'label': 'Tversky Similarity'}
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Prototypes', fontsize=12)
    plt.ylabel('Samples', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
