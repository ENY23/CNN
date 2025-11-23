# viz.py
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from models.resnet_cifar import ResNet18_CIFAR

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
CLASS_NAMES = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

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
    return model

@torch.no_grad()
def collect_logits_features(model, loader, device):
    logits_list, targets_list, probs_list = [], [], []
    # register hook for penultimate features (before FC)
    feats_list = []
    handle = None

    def hook_fn(module, inp, out):
        # module is AdaptiveAvgPool2d -> output shape (B, C, 1, 1)
        x = out.view(out.size(0), -1).detach()
        feats_list.append(x)

    # hook on avgpool
    handle = model.avgpool.register_forward_hook(hook_fn)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        out = model(images)
        probs = F.softmax(out, dim=1)
        logits_list.append(out.cpu())
        targets_list.append(targets.cpu())
        probs_list.append(probs.cpu())

    if handle is not None:
        handle.remove()

    logits = torch.cat(logits_list)
    targets = torch.cat(targets_list)
    probs = torch.cat(probs_list)
    feats = torch.cat(feats_list)
    return logits, probs, targets, feats

def plot_confusion_matrix(cm, normalize=True, save_path='confusion_matrix.png'):
    if normalize:
        cm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(cm, aspect='auto')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_xticks(range(len(CLASS_NAMES))); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_yticks(range(len(CLASS_NAMES))); ax.set_yticklabels(CLASS_NAMES)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('Confusion Matrix (normalized)' if normalize else 'Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

def save_classification_report(y_true, y_pred, save_dir):
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, digits=4)
    # save as csv
    import csv
    csv_path = os.path.join(save_dir, 'classification_report.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['class','precision','recall','f1-score','support']
        writer.writerow(header)
        for cls in CLASS_NAMES:
            row = [cls, report[cls]['precision'], report[cls]['recall'], report[cls]['f1-score'], report[cls]['support']]
            writer.writerow(row)
        avg = report['weighted avg']
        writer.writerow(['weighted avg', avg['precision'], avg['recall'], avg['f1-score'], avg['support']])

def reliability_diagram_and_ece(max_probs, correct, n_bins=10, save_path='reliability_ece.png'):
    bins = np.linspace(0.0, 1.0, n_bins+1)
    inds = np.digitize(max_probs, bins) - 1
    bin_acc, bin_conf, bin_counts = [], [], []
    for b in range(n_bins):
        mask = inds == b
        if mask.sum() == 0:
            bin_acc.append(0.0); bin_conf.append(0.0); bin_counts.append(0)
        else:
            bin_acc.append(correct[mask].mean())
            bin_conf.append(max_probs[mask].mean())
            bin_counts.append(mask.sum())
    bin_acc = np.array(bin_acc)
    bin_conf = np.array(bin_conf)
    bin_counts = np.array(bin_counts)
    ece = np.sum((np.abs(bin_acc - bin_conf)) * (bin_counts / max(1, bin_counts.sum())))

    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot([0,1],[0,1])
    ax.bar((bins[:-1]+bins[1:])/2.0, bin_acc, width=1.0/n_bins, alpha=0.8, align='center', edgecolor='black')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy')
    ax.set_title(f'Reliability Diagram (ECE={ece:.3f})')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

def tsne_plot(features, labels, save_path='tsne.png', max_points=5000, seed=42):
    X = features
    y = labels
    if X.shape[0] > max_points:
        # subsample for speed
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        X = X[idx]; y = y[idx]
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=seed)
    emb = tsne.fit_transform(X)
    fig, ax = plt.subplots(figsize=(6,6))
    for c in range(10):
        m = y == c
        ax.scatter(emb[m,0], emb[m,1], s=6, alpha=0.7, label=CLASS_NAMES[c])
    ax.legend(markerscale=2, fontsize=8, ncol=2)
    ax.set_title('t-SNE of Penultimate Features')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

def gradcam_single(model, image_tensor, target_class=None):
    """
    image_tensor: (1,3,32,32) normalized
    returns: cam np.array (H,W) normalized 0..1
    """
    model.eval()
    conv_module = model.layer4[-1].conv2  # last conv
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

    h1.remove(); h2.remove()
    # weights: GAP over gradients
    weights = grads.mean(dim=(2,3), keepdim=True)  # (B,C,1,1)
    cam = (weights * feats).sum(dim=1, keepdim=True)  # (B,1,H,W)
    cam = F.relu(cam)
    cam = cam[0,0].cpu().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cam

def overlay_cam_on_image(cam, image_np):
    """
    image_np: (H,W,3), unnormalized [0,1]
    cam: (H,W) 0..1
    returns blended image (H,W,3)
    """
    h, w = cam.shape
    cam_rgb = plt.cm.jet(cam)[..., :3]  # 0..1
    blended = 0.5*image_np + 0.5*cam_rgb
    blended = np.clip(blended, 0, 1)
    return blended

def denormalize(img):
    # img: (3,H,W) tensor
    mean = torch.tensor(CIFAR10_MEAN).view(3,1,1)
    std = torch.tensor(CIFAR10_STD).view(3,1,1)
    return (img.cpu()*std + mean).clamp(0,1)

def gradcam_gallery(model, loader, device, save_path='gradcam.png', n_samples=8):
    images_to_show = []
    labels = []
    preds = []
    # collect some correct and incorrect examples
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

    fig, axes = plt.subplots(2, n_samples//2, figsize=(2.6*n_samples//2, 5.2))
    axes = axes.flatten()
    for idx in range(n_samples):
        img_t = images_to_show[idx]
        img_dn = denormalize(img_t).permute(1,2,0).numpy()
        cam = gradcam_single(model, img_t.unsqueeze(0).to(device), target_class=None)
        blend = overlay_cam_on_image(cam, img_dn)
        ax = axes[idx]
        ax.imshow(blend)
        ax.axis('off')
        ax.set_title(f"T:{CLASS_NAMES[labels[idx]]}\nP:{CLASS_NAMES[preds[idx]]}", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

def misclassified_gallery(images, y_true, y_pred, probs, save_path='misclassified.png', n=16):
    wrong = (y_true != y_pred).nonzero()[0]
    # sort by confidence descending (most confident wrong)
    conf = probs[wrong, y_pred[wrong]]
    order = np.argsort(-conf)
    pick = wrong[order[:n]]

    cols = 8
    rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2*cols, 2.2*rows))
    axes = axes.flatten()
    for i, idx in enumerate(pick):
        img = images[idx].transpose(1,2,0)  # HWC
        ax = axes[i]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"P:{CLASS_NAMES[y_pred[idx]]}\nT:{CLASS_NAMES[y_true[idx]]}\nconf:{probs[idx,y_pred[idx]]:.2f}", fontsize=8)
    for j in range(i+1, rows*cols):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--ckpt', type=str, default='checkpoints/best.pth')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--outdir', type=str, default='viz_outputs')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.ckpt, device)
    loader = get_test_loader(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    # 1) collect logits/probs/targets/features + also keep denormalized images for galleries
    logits, probs, targets, feats = collect_logits_features(model, loader, device)
    y_true = targets.numpy()
    y_pred = logits.argmax(dim=1).numpy()
    prob_np = probs.numpy()

    # We also need raw images for misclassified gallery
    # Rebuild a non-normalized loader to fetch raw pixels
    raw_tfms = transforms.ToTensor()
    raw_set = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=raw_tfms)
    raw_loader = DataLoader(raw_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    raw_images = []
    for x,_ in raw_loader:
        raw_images.append(x.numpy())
    raw_images = np.concatenate(raw_images, axis=0)  # (N,3,32,32)
    raw_images = np.transpose(raw_images, (0,2,3,1))  # (N,32,32,3)

    # 2) Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    plot_confusion_matrix(cm, normalize=True, save_path=os.path.join(args.outdir, 'confusion_matrix.png'))
    save_classification_report(y_true, y_pred, args.outdir)

    # 3) Reliability Diagram + ECE
    max_probs = prob_np.max(axis=1)
    correct = (y_true == y_pred).astype(np.float32)
    reliability_diagram_and_ece(max_probs, correct, n_bins=10,
                                save_path=os.path.join(args.outdir, 'reliability_ece.png'))

    # 4) t-SNE of penultimate features
    tsne_plot(feats.numpy(), y_true, save_path=os.path.join(args.outdir, 'tsne.png'), max_points=5000)

    # 5) Grad-CAM gallery
    gradcam_gallery(model, loader, device, save_path=os.path.join(args.outdir, 'gradcam.png'), n_samples=8)

    # 6) Misclassified gallery (most confident wrong predictions)
    misclassified_gallery(raw_images.transpose(0,3,1,2), y_true, y_pred, prob_np,
                          save_path=os.path.join(args.outdir, 'misclassified.png'), n=16)

    print(f"Saved visualizations to: {args.outdir}")

if __name__ == '__main__':
    main()
