import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.classification import UltrasoundBaselineCNN, UltrasoundTransferClassifier
from src.models.segmentation import UltrasoundUNet

def run_audit():
    ckpt_files = [
        ('checkpoints/classification/best_baseline.pth', UltrasoundBaselineCNN, {'num_classes': 3, 'in_channels': 3}),
        ('checkpoints/classification/best_transfer.pth', UltrasoundTransferClassifier, {'num_classes': 3, 'pretrained': False}),
        ('checkpoints/segmentation/best_unet.pth', UltrasoundUNet, {'in_channels': 1, 'out_channels': 1, 'channels': (16, 32, 64, 128), 'strides': (2, 2, 2), 'num_res_units': 1})
    ]

    print("=== CHECKPOINT INTEGRITY ===")
    for path, cls, kwargs in ckpt_files:
        assert os.path.exists(path), f"Missing: {path}"
        state = torch.load(path, map_location='cpu', weights_only=False)
        model = cls(**kwargs)
        model.load_state_dict(state['model_state_dict'])
        metric_val = state.get('val_macro_f1', state.get('val_dice', 0.0))
        print(f"OK: {path} | Epoch {state.get('epoch')} | Val Metric: {metric_val:.4f}")

    print("\n=== DATASET SPLIT & LEAKAGE AUDIT ===")
    with open('data/processed/dataset_metadata.json', 'r') as f:
        meta = json.load(f)

    train_cases = [c for c in meta['cases'] if c['split'] == 'train']
    val_cases = [c for c in meta['cases'] if c['split'] == 'val']
    test_cases = [c for c in meta['cases'] if c['split'] == 'test']

    print(f"Train cases: {len(train_cases)}")
    print(f"Val cases: {len(val_cases)}")
    print(f"Test cases: {len(test_cases)}")
    assert len(test_cases) == 116, f"Expected 116 test cases, got {len(test_cases)}"

    train_ids = set(c['case_id'] for c in train_cases)
    val_ids = set(c['case_id'] for c in val_cases)
    test_ids = set(c['case_id'] for c in test_cases)

    train_val_overlap = train_ids.intersection(val_ids)
    train_test_overlap = train_ids.intersection(test_ids)
    val_test_overlap = val_ids.intersection(test_ids)

    print(f"Train-Val Overlap: {len(train_val_overlap)}")
    print(f"Train-Test Overlap: {len(train_test_overlap)}")
    print(f"Val-Test Overlap: {len(val_test_overlap)}")
    assert len(train_test_overlap) == 0, "LEAKAGE DETECTED between Train and Test!"
    assert len(train_val_overlap) == 0, "LEAKAGE DETECTED between Train and Val!"
    assert len(val_test_overlap) == 0, "LEAKAGE DETECTED between Val and Test!"

    test_class_counts = {}
    for c in test_cases:
        cl = c.get('class') or c.get('label_name')
        test_class_counts[cl] = test_class_counts.get(cl, 0) + 1
    print("Test set class counts:", test_class_counts)

    print("\n=== METRIC VERIFICATION AGAINST RAW JSON ===")
    with open('outputs/test_evaluation_report.json', 'r') as f:
        eval_rep = json.load(f)

    print("Baseline Accuracy:", eval_rep['classification']['baseline_cnn']['accuracy'])
    print("Baseline Macro F1:", eval_rep['classification']['baseline_cnn']['f1_macro'])
    print("ResNet18 Accuracy:", eval_rep['classification']['resnet18_transfer']['accuracy'])
    print("ResNet18 Macro F1:", eval_rep['classification']['resnet18_transfer']['f1_macro'])
    print("MONAI U-Net Overall Dice:", eval_rep['segmentation']['monai_unet']['overall_dice'])
    print("MONAI U-Net Overall IoU:", eval_rep['segmentation']['monai_unet']['overall_iou'])

if __name__ == '__main__':
    run_audit()
