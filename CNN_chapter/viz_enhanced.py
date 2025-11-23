#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Visualization Suite for ResNet-18 CIFAR-10
Comprehensive analysis with all text in English (no Chinese font required)
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns

from models.resnet_cifar import ResNet18_CIFAR

# Use colorblind-friendly palettes (avoiding red-blue as per user preference)
COLORS = {
    'train': '#FFA500',      # Orange
    'test': '#9370DB',       # Purple
    'accent1': '#2E8B57',    # SeaGreen
    'accent2': '#FFD700',    # Gold
    'accent3': '#8B4789',    # DarkOrchid
}

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def setup_plot_style():
    """Setup matplotlib style for publication-quality figures"""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 13,
        'font.family': 'DejaVu Sans',
    })

def get_test_loader(data_dir, batch_size=256, num_workers=2):
    tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=tfms)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)

def load_model(ckpt_path, device):
    model = ResNet18_CIFAR(num_classes=10).to(device)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt

@torch.no_grad()
def collect_predictions_and_features(model, loader, device):
    """Collect all predictions, probabilities, features and targets"""
    logits_list, targets_list, probs_list = [], [], []
    feats_list = []
    
    def hook_fn(module, inp, out):
        x = out.view(out.size(0), -1).detach()
        feats_list.append(x)
    
    handle = model.avgpool.register_forward_hook(hook_fn)
    
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        out = model(images)
        probs = F.softmax(out, dim=1)
        logits_list.append(out.cpu())
        targets_list.append(targets.cpu())
        probs_list.append(probs.cpu())
    
    handle.remove()
    
    logits = torch.cat(logits_list)
    targets = torch.cat(targets_list)
    probs = torch.cat(probs_list)
    feats = torch.cat(feats_list).cpu()
    
    return logits, probs, targets, feats

# ==================== Training Dynamics ====================

def plot_training_curves(history_file, save_path):
    """Plot loss and accuracy vs epoch for train/test"""
    if not os.path.exists(history_file):
        print(f"Warning: {history_file} not found, skipping training curves")
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = history['epochs']
    train_loss = history['train_loss']
    train_acc = history['train_acc']
    val_loss = history['val_loss']
    val_acc = history['val_acc']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(epochs, train_loss, label='Train', color=COLORS['train'], linewidth=2)
    ax1.plot(epochs, val_loss, label='Test', color=COLORS['test'], linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_acc, label='Train', color=COLORS['train'], linewidth=2)
    ax2.plot(epochs, val_acc, label='Test', color=COLORS['test'], linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_learning_rate_schedule(history_file, save_path):
    """Plot learning rate vs epoch"""
    if not os.path.exists(history_file):
        print(f"Warning: {history_file} not found, skipping LR schedule")
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = history['epochs']
    lrs = history.get('learning_rates', [])
    
    if not lrs:
        print("No learning rate history found")
        return
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, lrs, color=COLORS['accent1'], linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# ==================== Performance Analysis ====================

def plot_confusion_matrix(y_true, y_pred, save_path, normalize=True):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    
    if normalize:
        cm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, aspect='auto', cmap='YlOrBr')
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_yticklabels(CLASS_NAMES)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Proportion' if normalize else 'Count', rotation=270, labelpad=15)
    
    # Add text annotations
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            text = ax.text(j, i, f'{cm[i, j]:.2f}' if normalize else f'{int(cm[i, j])}',
                          ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black",
                          fontsize=8)
    
    ax.set_title('Confusion Matrix (Normalized)' if normalize else 'Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_per_class_metrics(y_true, y_pred, save_path):
    """Plot precision, recall, F1 for each class"""
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, 
                                   output_dict=True, digits=4, zero_division=0)
    
    classes = CLASS_NAMES
    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1 = [report[c]['f1-score'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, precision, width, label='Precision', color='#FFA500', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', color='#9370DB', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', color='#2E8B57', alpha=0.8)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    
    # Also save as CSV
    csv_path = save_path.replace('.png', '.csv')
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
        for c in classes:
            writer.writerow([c, report[c]['precision'], report[c]['recall'], 
                           report[c]['f1-score'], report[c]['support']])
        avg = report['weighted avg']
        writer.writerow(['Weighted Avg', avg['precision'], avg['recall'], 
                        avg['f1-score'], avg['support']])
    print(f"Saved: {csv_path}")

def plot_topk_accuracy(probs, y_true, save_path, k_values=[1, 3, 5]):
    """Plot top-k accuracy bar chart"""
    topk_accs = []
    
    for k in k_values:
        topk_pred = torch.topk(torch.from_numpy(probs), k, dim=1)[1].numpy()
        correct = np.any(topk_pred == y_true[:, None], axis=1)
        acc = correct.mean() * 100
        topk_accs.append(acc)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(k_values)), topk_accs, color=['#FFA500', '#9370DB', '#2E8B57'], 
                  alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Top-K')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Top-K Accuracy Performance')
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([f'Top-{k}' for k in k_values])
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, topk_accs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_misclassified_gallery(data_dir, y_true, y_pred, probs, save_path, n=16):
    """Gallery of misclassified samples with highest confidence"""
    # Load raw images
    raw_tfms = transforms.ToTensor()
    raw_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=raw_tfms)
    raw_loader = DataLoader(raw_set, batch_size=256, shuffle=False, num_workers=2)
    
    raw_images = []
    for x, _ in raw_loader:
        raw_images.append(x.numpy())
    raw_images = np.concatenate(raw_images, axis=0)
    
    wrong_idx = np.where(y_true != y_pred)[0]
    conf = probs[wrong_idx, y_pred[wrong_idx]]
    order = np.argsort(-conf)[:n]
    pick = wrong_idx[order]
    
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4))
    axes = axes.flatten()
    
    for i, idx in enumerate(pick):
        img = raw_images[idx].transpose(1, 2, 0)
        ax = axes[i]
        ax.imshow(img)
        ax.axis('off')
        pred_conf = probs[idx, y_pred[idx]]
        ax.set_title(f"Pred: {CLASS_NAMES[y_pred[idx]]}\n"
                    f"True: {CLASS_NAMES[y_true[idx]]}\n"
                    f"Conf: {pred_conf:.3f}", fontsize=7)
    
    for j in range(i+1, rows*cols):
        axes[j].axis('off')
    
    plt.suptitle('Most Confident Misclassifications', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# ==================== Calibration & Confidence ====================

def plot_reliability_diagram(probs, y_true, y_pred, save_path, n_bins=10):
    """Reliability diagram and ECE calculation"""
    max_probs = probs.max(axis=1)
    correct = (y_true == y_pred).astype(np.float32)
    
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(max_probs, bins) - 1
    
    bin_acc, bin_conf, bin_counts = [], [], []
    for b in range(n_bins):
        mask = inds == b
        if mask.sum() == 0:
            bin_acc.append(0.0)
            bin_conf.append(0.0)
            bin_counts.append(0)
        else:
            bin_acc.append(correct[mask].mean())
            bin_conf.append(max_probs[mask].mean())
            bin_counts.append(mask.sum())
    
    bin_acc = np.array(bin_acc)
    bin_conf = np.array(bin_conf)
    bin_counts = np.array(bin_counts)
    
    ece = np.sum(np.abs(bin_acc - bin_conf) * (bin_counts / max(1, bin_counts.sum())))
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    ax.bar((bins[:-1] + bins[1:]) / 2.0, bin_acc, width=1.0/n_bins, 
           alpha=0.7, color=COLORS['train'], edgecolor='black', label='Model Output')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Reliability Diagram\nECE = {ece:.4f}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path} (ECE={ece:.4f})")

def plot_confidence_distribution(probs, y_true, y_pred, save_path):
    """Confidence distribution histogram for correct vs incorrect"""
    max_probs = probs.max(axis=1)
    correct_conf = max_probs[y_true == y_pred]
    incorrect_conf = max_probs[y_true != y_pred]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(correct_conf, bins=30, alpha=0.6, label=f'Correct (n={len(correct_conf)})', 
            color=COLORS['accent1'], edgecolor='black')
    ax.hist(incorrect_conf, bins=30, alpha=0.6, label=f'Incorrect (n={len(incorrect_conf)})', 
            color=COLORS['train'], edgecolor='black')
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Confidence Distribution: Correct vs Incorrect Predictions', fontsize=13)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics in bottom right
    textstr = f'Correct: μ={correct_conf.mean():.3f}, σ={correct_conf.std():.3f}\n'
    textstr += f'Incorrect: μ={incorrect_conf.mean():.3f}, σ={incorrect_conf.std():.3f}'
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# ==================== Representation & Interpretability ====================

def plot_tsne(features, labels, save_path, max_points=5000, seed=42):
    """t-SNE visualization of penultimate layer features"""
    X = features
    y = labels
    
    if X.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        X = X[idx]
        y = y[idx]
    
    print(f"Running t-SNE on {X.shape[0]} samples...")
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', 
                init='pca', random_state=seed, verbose=1)
    emb = tsne.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use colorblind-friendly palette
    palette = sns.color_palette("tab10", 10)
    
    for c in range(10):
        mask = y == c
        ax.scatter(emb[mask, 0], emb[mask, 1], s=20, alpha=0.6, 
                  label=CLASS_NAMES[c], color=palette[c])
    
    ax.legend(markerscale=1.5, fontsize=9, ncol=2, loc='best')
    ax.set_title('t-SNE Visualization of Penultimate Layer Features', fontsize=13)
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def denormalize(img):
    """Denormalize image tensor"""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return (img.cpu() * std + mean).clamp(0, 1)

def gradcam_single(model, image_tensor, target_class=None):
    """Generate Grad-CAM for single image"""
    model.eval()
    conv_module = model.layer4[-1].conv2
    feats = None
    grads = None
    
    def fwd_hook(m, inp, out):
        nonlocal feats
        feats = out.detach()
    
    def bwd_hook(m, gin, gout):
        nonlocal grads
        grads = gout[0].detach()
    
    h1 = conv_module.register_forward_hook(fwd_hook)
    h2 = conv_module.register_full_backward_hook(bwd_hook)
    
    image_tensor.requires_grad_(True)
    logits = model(image_tensor)
    
    if target_class is None:
        target_class = logits.argmax(dim=1).item()
    
    loss = logits[0, target_class]
    model.zero_grad()
    loss.backward()
    
    h1.remove()
    h2.remove()
    
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feats).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam[0, 0].cpu().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    
    return cam

def overlay_cam_on_image(cam, image_np):
    """Overlay CAM heatmap on image"""
    import cv2
    h, w = image_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = plt.cm.jet(cam_resized)[..., :3]
    blended = 0.5 * image_np + 0.5 * heatmap
    return np.clip(blended, 0, 1)

def plot_gradcam_gallery(model, loader, device, save_path, n_samples=16):
    """Grad-CAM gallery showing model attention"""
    images_to_show = []
    labels = []
    preds = []
    
    with torch.no_grad():
        for imgs, t in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            p = out.argmax(dim=1)
            for i in range(imgs.size(0)):
                images_to_show.append(imgs[i].detach().cpu())
                labels.append(int(t[i]))
                preds.append(int(p[i]))
                if len(images_to_show) >= n_samples:
                    break
            if len(images_to_show) >= n_samples:
                break
    
    cols = 8
    rows = n_samples // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4))
    axes = axes.flatten()
    
    for idx in range(n_samples):
        img_t = images_to_show[idx]
        img_dn = denormalize(img_t).permute(1, 2, 0).numpy()
        cam = gradcam_single(model, img_t.unsqueeze(0).to(device), target_class=None)
        blend = overlay_cam_on_image(cam, img_dn)
        
        ax = axes[idx]
        ax.imshow(blend)
        ax.axis('off')
        is_correct = labels[idx] == preds[idx]
        title_color = 'green' if is_correct else 'red'
        ax.set_title(f"T:{CLASS_NAMES[labels[idx]]}\nP:{CLASS_NAMES[preds[idx]]}", 
                    fontsize=7, color=title_color)
    
    plt.suptitle('Grad-CAM Visualization (Green=Correct, Red=Incorrect)', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# ==================== Model Statistics ====================

def plot_model_statistics(model, save_path):
    """Plot model architecture statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count parameters by layer type
    layer_stats = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf module
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                layer_type = type(module).__name__
                layer_stats[layer_type] = layer_stats.get(layer_type, 0) + params
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall statistics
    categories = ['Total\nParameters', 'Trainable\nParameters']
    values = [total_params / 1e6, trainable_params / 1e6]
    bars1 = ax1.bar(categories, values, color=[COLORS['train'], COLORS['accent1']], 
                    alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Parameters (Millions)')
    ax1.set_title('Model Parameter Statistics')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}M', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Parameters by layer type
    layer_types = list(layer_stats.keys())
    layer_params = [layer_stats[lt] / 1e6 for lt in layer_types]
    sorted_idx = np.argsort(layer_params)[::-1]
    layer_types = [layer_types[i] for i in sorted_idx]
    layer_params = [layer_params[i] for i in sorted_idx]
    
    colors_palette = sns.color_palette("husl", len(layer_types))
    bars2 = ax2.barh(layer_types, layer_params, color=colors_palette, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Parameters (Millions)')
    ax2.set_title('Parameters by Layer Type')
    ax2.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars2, layer_params):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f'{val:.2f}M', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

# ==================== Main Function ====================

def main():
    parser = argparse.ArgumentParser(description="Enhanced Visualization for ResNet-18 CIFAR-10")
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--ckpt', type=str, default='checkpoints/best.pth')
    parser.add_argument('--history', type=str, default='checkpoints/history.json',
                       help='Training history JSON file')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--outdir', type=str, default='visualizations')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.outdir, exist_ok=True)
    setup_plot_style()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.ckpt}...")
    model, ckpt = load_model(args.ckpt, device)
    print(f"Model loaded. Best accuracy: {ckpt.get('best_acc', 'N/A'):.2f}%")
    
    # Load data
    print("\nLoading test data...")
    loader = get_test_loader(args.data_dir, batch_size=args.batch_size, 
                            num_workers=args.num_workers)
    
    # Collect predictions
    print("\nCollecting predictions and features...")
    logits, probs, targets, feats = collect_predictions_and_features(model, loader, device)
    y_true = targets.numpy()
    y_pred = logits.argmax(dim=1).numpy()
    prob_np = probs.numpy()
    feats_np = feats.numpy()
    
    # Calculate overall accuracy
    acc = (y_true == y_pred).mean() * 100
    print(f"\nTest Accuracy: {acc:.2f}%")
    
    # ========== Generate Visualizations ==========
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # 1. Training Dynamics
    print("\n[1] Training Dynamics...")
    plot_training_curves(args.history, os.path.join(args.outdir, '1_training_curves.png'))
    plot_learning_rate_schedule(args.history, os.path.join(args.outdir, '1_lr_schedule.png'))
    
    # 2. Performance Analysis
    print("\n[2] Performance Analysis...")
    plot_confusion_matrix(y_true, y_pred, os.path.join(args.outdir, '2_confusion_matrix.png'))
    plot_per_class_metrics(y_true, y_pred, os.path.join(args.outdir, '2_per_class_metrics.png'))
    plot_topk_accuracy(prob_np, y_true, os.path.join(args.outdir, '2_topk_accuracy.png'))
    plot_misclassified_gallery(args.data_dir, y_true, y_pred, prob_np, 
                               os.path.join(args.outdir, '2_misclassified.png'))
    
    # 3. Calibration & Confidence
    print("\n[3] Calibration & Confidence...")
    plot_reliability_diagram(prob_np, y_true, y_pred, 
                            os.path.join(args.outdir, '3_reliability_diagram.png'))
    plot_confidence_distribution(prob_np, y_true, y_pred, 
                                os.path.join(args.outdir, '3_confidence_distribution.png'))
    
    # 4. Representation & Interpretability
    print("\n[4] Representation & Interpretability...")
    plot_tsne(feats_np, y_true, os.path.join(args.outdir, '4_tsne.png'))
    plot_gradcam_gallery(model, loader, device, os.path.join(args.outdir, '4_gradcam_gallery.png'))
    
    # 5. Model Statistics
    print("\n[5] Model Statistics...")
    plot_model_statistics(model, os.path.join(args.outdir, '5_model_statistics.png'))
    
    print("\n" + "="*60)
    print(f"All visualizations saved to: {args.outdir}")
    print("="*60)

if __name__ == '__main__':
    main()
