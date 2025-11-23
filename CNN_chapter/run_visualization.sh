#!/bin/bash
# Complete Visualization Pipeline for ResNet-18 CIFAR-10

echo "=========================================="
echo "ResNet-18 CIFAR-10 Visualization Pipeline"
echo "=========================================="

# Configuration
CHECKPOINT="checkpoints/best.pth"
DATA_DIR="../data"
OUTPUT_DIR="visualizations"
BATCH_SIZE=256

# Step 1: Extract training history
echo ""
echo "[Step 1/2] Extracting training history from checkpoint..."
python3 extract_history.py --ckpt "$CHECKPOINT" --output checkpoints/history.json

# Step 2: Generate all visualizations
echo ""
echo "[Step 2/2] Generating comprehensive visualizations..."
python3 viz_enhanced.py \
    --data-dir "$DATA_DIR" \
    --ckpt "$CHECKPOINT" \
    --history checkpoints/history.json \
    --batch-size $BATCH_SIZE \
    --outdir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Visualization complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "Generated visualizations:"
echo "  [1] Training Dynamics:"
echo "      - 1_training_curves.png"
echo "      - 1_lr_schedule.png"
echo "  [2] Performance Analysis:"
echo "      - 2_confusion_matrix.png"
echo "      - 2_per_class_metrics.png (+ CSV)"
echo "      - 2_topk_accuracy.png"
echo "      - 2_misclassified.png"
echo "  [3] Calibration & Confidence:"
echo "      - 3_reliability_diagram.png"
echo "      - 3_confidence_distribution.png"
echo "  [4] Representation & Interpretability:"
echo "      - 4_tsne.png"
echo "      - 4_gradcam_gallery.png"
echo "  [5] Model Statistics:"
echo "      - 5_model_statistics.png"
echo ""
