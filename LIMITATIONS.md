# Scientific Limitations, Failure Modes & Ethical Considerations

## 1. Clinical & Regulatory Disclaimers

> [!IMPORTANT]
> **RESEARCH USE ONLY:** The models, pipelines, and web endpoints implemented in this repository are developed strictly for academic evaluation and engineering benchmarking. This system is **NOT** approved by CDSCO, US FDA, or CE for clinical diagnosis, patient triage, or therapy planning. No algorithmic output should be used to override clinical judgment.

---

## 2. Imaging Modality & Clinical Acquisition Constraints

1. **Operator Dependency of B-Mode Ultrasound:**
   Unlike computed tomography (CT) or magnetic resonance imaging (MRI), medical ultrasound is an acoustic, real-time, highly operator-dependent modality. Image quality varies markedly based on:
   - Acoustic coupling gel thickness, transducer contact pressure, and rib acoustic shadowing.
   - Transducer probe frequency (e.g. 1–5 MHz convex vs. 7–15 MHz high-frequency linear probes), elevation plane focal distance, and patient respiratory motion.
   - Manual machine gain controls (overall gain, dynamic range, and time gain compensation / TGC) adjusted arbitrarily at the point of care.
   These real-world acquisition variations cannot be fully corrected from static, post-processed B-mode raster images.

2. **Acoustic Physics & Raw Signal Scope:**
   - **Beamforming and raw RF acquisition are outside the current scope.** The platform operates on reconstructed, envelope-detected B-mode images and DICOM ultrasound studies. It does not perform digital beamforming, synthetic aperture processing, or raw radiofrequency (RF) channel data reconstruction.
   - **Speckle Noise & Multi-path Reverberation:** Coherent acoustic interference produces characteristic multiplicative speckle noise and acoustic clutter that degrade lesion boundary contrast.

3. **Absence of 3D / Doppler / Elastography Features:**
   In clinical practice, breast ultrasound evaluation utilizes color/power Doppler (vascularity quantification) and shear-wave elastography (tissue stiffness mapping). The BUSI cohort provides exclusively static 2D grayscale B-mode slices.

3. **Lack of Histopathological Subtyping:**
   The dataset categorizes scans into coarse categories (*benign*, *malignant*, *normal*) without clinical pathological staging (e.g., Invasive Ductal Carcinoma vs Lobular Carcinoma, Grade I–III, or ER/PR/HER2 receptor status).

---

## 3. Dataset Constraints & Provenance Artifacts

1. **Single-Center Retrospective Cohort:**
   The BUSI benchmark was collected at a single regional hospital in Cairo, Egypt (2018). Models trained solely on this cohort risk domain shift when deployed on ultrasound equipment from other vendors (e.g., Philips, Siemens, Mindray) or across ethnically diverse demographic populations.

2. **Severe Class Imbalance:**
   The cohort consists of 437 benign (56.0%), 210 malignant (26.9%), and 133 normal (17.0%) scans. While inverse class frequency weighting was applied during training, benign patterns dominate gradient updates.

3. **Upstream Annotation Inconsistency:**
   Our automated SHA-256 byte audit discovered that `benign (433).png` is identical to `malignant (145).png`. This confirms an upstream labeling collision present in the original Mendeley Data release. In a formal clinical study, such samples must be adjudicated by an independent consensus panel of certified radiologists.

4. **Multi-Focal Lesion Boundaries:**
   In 17 cases, patients presented with multiple discrete lesion foci. While our pipeline fuses them deterministically via logical OR, the proximity of adjacent lesions can create ambiguity for distance metrics and bounding box extraction.

---

## 4. Object Detection Investigation Finding

As established in our Phase 5 audit (`src/detection_investigation.py`), the BUSI dataset contains **zero** native bounding-box annotations. 

> [!WARNING]
> While minimum bounding rectangles $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ can be derived algorithmically from the connected components of ground truth segmentation masks, **they must never be misrepresented as native radiologist bounding boxes**. Radiologists typically draw bounding boxes to capture margin spiculation, halo effects, and adjacent architectural distortion that extend beyond the core segmented lesion body.

---

## 5. Ethical & Patient Safety Implications

1. **Asymmetric Clinical Costs (False Negatives vs False Positives):**
   - **False Negative (Malignant classified as Benign or Normal):** Carries catastrophic clinical consequences: missed or delayed cancer diagnosis, leading to tumor progression and reduced survival rates.
   - **False Positive (Benign classified as Malignant):** Causes acute patient anxiety and prompts unnecessary invasive core needle biopsies.
   In a production medical AI deployment, the operating decision threshold must be tuned toward high sensitivity (recall) to minimize false negatives.

2. **Explainability & Automation Bias:**
   Clinicians must not place unquestioning faith in AI predictions (*automation bias*). Our implementation provides both segmentation heatmaps and confidence distributions rather than a single black-box classification label, allowing the clinician to cross-reference the visual region of interest against standard BI-RADS (Breast Imaging Reporting and Data System) lexical descriptors.
