"""
Operator UI Audit Script
Verifies that all technical noise, implementation jargon, internal badges,
and unneeded elements have been removed from the default operator view in index.html,
and confirms that the required clean elements and PDF workflow are in place.
"""

with open("api/static/index.html", "r", encoding="utf-8") as f:
    content = f.read()

# Assert '18 evaluated' and '18 alerts' are gone
assert "18 Evaluated" not in content, "18 Evaluated found in UI"
assert "18 evaluated" not in content, "18 evaluated found in UI"
assert "18 Alerts" not in content, "18 Alerts found in UI"
assert "18 alerts" not in content, "18 alerts found in UI"

# Assert Print Summary is gone
assert "Print Summary" not in content, "Print Summary found in UI"

# Assert Download Result exists
assert "Download Result" in content, "Download Result missing from UI"

# Assert technical details is a collapsed details element
assert 'id="tech-details"' in content, "Tech details element missing"
assert "Technical details" in content, "Technical details summary missing"

# Assert PDF support is present in dropzone and input
assert ".pdf" in content, "PDF extension missing from input accept"
assert "PDF" in content, "PDF missing from supported text"
assert "pdf-container-box" in content, "PDF container workflow missing"

# Assert header badges only has Ready, no CPU Pipeline or version
assert "CPU Pipeline" not in content, "CPU Pipeline badge found in header"
assert "Level 5" not in content, "Level 5 badge found in UI"
assert "E2E tested" not in content, "E2E tested badge found in UI"
assert "Auto-Route" not in content, "Auto-Route badge found in UI"

# Assert MRI clean operator status
assert "AI segmentation unavailable." in content, "AI segmentation unavailable text missing"
assert "MRI viewing is available." in content, "MRI viewing is available text missing"
assert "No AI segmentation was generated." in content, "No AI segmentation was generated text missing"
assert "CHECKPOINT REQUIRED" not in content, "CHECKPOINT REQUIRED found in UI"

# Assert Thyroid clean operator status
assert "Ultrasound Analysis" in content, "Ultrasound Analysis missing"
assert "Image-based measurement" in content, "Image-based measurement missing"
assert "Acoustic Heuristic" not in content, "Acoustic Heuristic found in UI"
assert "TI-RADS Morphology" not in content, "TI-RADS Morphology found in UI"

# Assert MedSAM clean operator status
assert "Region segmentation unavailable." in content, "Region segmentation unavailable missing"
assert "ViT-B Universal Segmentor" not in content, "ViT-B Universal Segmentor found in UI"

print("============================================================")
print(" ALL OPERATOR UI ASSERTIONS PASSED 100%!")
print("============================================================")
