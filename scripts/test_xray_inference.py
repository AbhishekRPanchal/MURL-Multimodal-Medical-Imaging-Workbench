import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.platform.adapters.chest_xray_adapter import ChestXRayAdapter

def main():
    img_path = "data/samples/nih_chest_xray_00000001.png"
    print(f"Loading authentic NIH radiograph: {img_path} ({os.path.getsize(img_path)} bytes)")
    
    adapter = ChestXRayAdapter({
        "model_id": "chexpert_chest_xray_classifier",
        "name": "TorchXRayVision DenseNet-121",
        "output_task": "classification"
    })
    
    res = adapter.predict(img_path)
    fmt = res.formatted_results
    
    print("\n=== REAL MEDICAL RADIOGRAPH INFERENCE EVIDENCE ===")
    print(f"Model: {fmt['model']}")
    print(f"Hardware: {fmt['execution_hardware']}")
    print(f"Inference Duration: {res.inference_time_ms} ms")
    print(f"Pathologies Evaluated: {fmt['pathologies_evaluated']}")
    print(f"Flagged Pathologies Exceeding Thresholds: {fmt['flagged_pathologies_count']}")
    
    print("\nTop 5 Radiological Findings on Authentic NIH Frontal Radiograph:")
    for f in fmt["top_findings"]:
        print(f"  - {f['pathology']:<28}: {f['probability_percent']:>5.2f}% (operating thresh: {f['operating_threshold']}, alert: {f['threshold_exceeded']})")

if __name__ == "__main__":
    main()
