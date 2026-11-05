# Medical Software Quality & Regulatory Awareness Guide

> [!IMPORTANT]
> **Regulatory Status & Explicit Non-Certification Declaration:**  
> **MURL is not certified or claimed to be compliant with ISO 13485, IEC 62304, HIPAA or GDPR. The project demonstrates awareness of relevant software quality, privacy and regulatory concepts in a research/educational context.**  
> Outputs are strictly experimental and intended solely for algorithm development, bench testing, and academic research. They do not constitute medical advice or clinical diagnoses.

---

## 1. Distinction: Engineering Awareness vs. Certified Compliance

In medical imaging engineering, a critical distinction exists between:
1. **Engineering Quality Awareness:** Adopting disciplined software engineering principles, rigorous verification, design traceability, defensive input handling, and privacy filters inspired by international standards.
2. **Regulatory Compliance / Certification:** Formal, audited adherence to statutory medical device quality management systems (e.g., FDA 21 CFR Part 820, EU MDR 2017/745, CDSCO Medical Device Rules) evaluated by accredited Notified Bodies or regulatory authorities.

The MURL platform operates firmly within category (1) as an open-source academic workbench.

---

## 2. ISO 13485: Medical Devices — Quality Management Systems Awareness

ISO 13485 specifies requirements for a quality management system (QMS) where an organization needs to demonstrate its ability to provide medical devices and related services that consistently meet customer and applicable regulatory requirements.

### Key Conceptual Pillars & MURL Engineering Practices:

- **Requirements Management:**  
  Clearly defined software functional requirements (e.g., supported input modalities, DICOM metadata ingestion bounds, rejection thresholds).
- **Design Documentation & Architecture Specifications:**  
  Detailed component-level architectural documentation ([`MURL_ARCHITECTURE.md`](../MURL_ARCHITECTURE.md), [`PROJECT_ARCHITECTURE.md`](../PROJECT_ARCHITECTURE.md)), data flow graphs, and class interaction models ensuring design transparency.
- **Verification & Testing (Design Verification):**  
  Comprehensive automated regression test suites covering unit logic, integration routes, end-to-end multi-modal inference, and cross-modality clinical safety rejection.
- **Configuration Management:**  
  Strict semantic versioning, reproducible virtual environment specifications (`requirements.txt`), deterministic random seeds, and isolation of third-party dependencies through thin adapters.
- **Traceability Matrix:**  
  Tracing input imaging modalities through schema validators, compatibility matrices, inference adapters, and multi-format exporters.
- **Risk Management Concepts (aligned with ISO 14971 principles):**  
  Identifying failure modes (e.g., operator selecting Breast Ultrasound analysis on a Chest X-ray), assessing hazards (misleading output), and implementing risk control measures (hard program safety rejection blocking execution).

---

## 3. IEC 62304: Medical Device Software — Software Life Cycle Processes

IEC 62304 is the global benchmark standard for the software development life cycle (SDLC) of medical device software.

### SDLC Phases & MURL Academic Implementations:

- **Software Development Planning:**  
  Modular separation of concerns between ingestion, metadata parsing, neural execution, UI rendering, and export serialization.
- **Software Requirements Analysis:**  
  Formal model registry specifications ([`model_registry/registry.yaml`](../model_registry/registry.yaml)) enforcing 17 mandatory schema fields per registered model before execution is permitted.
- **Software Architectural & Detailed Design:**  
  Extensible base class contracts ([`src/platform/base_model.py`](../src/platform/base_model.py)) preventing arbitrary runtime extensions and isolating neural weights from web microservices.
- **Software Unit Implementation & Verification:**  
  Automated pytest suites ([`tests/test_platform.py`](../tests/test_platform.py), [`tests/test_auto_manual_selection.py`](../tests/test_auto_manual_selection.py), [`tests/test_export_system.py`](../tests/test_export_system.py)) verifying array shape assertions, dtype preservation, and boundary condition handling.
- **Software Integration & Integration Testing:**  
  Full end-to-end integration tests ([`test_demo_dataset_e2e.py`](../test_demo_dataset_e2e.py), [`verify_all_paths.py`](../verify_all_paths.py)) running against live server endpoints.
- **Software Release & Configuration Control:**  
  Clean public repository hygiene, exclusion of heavyweight serialized checkpoints from git history, and explicit provenance tracking ([`MODEL_PROVENANCE.md`](../MODEL_PROVENANCE.md)).
- **Anomaly / Issue Tracking:**  
  Documented failure analysis ([`SEGMENTATION_FAILURE_ANALYSIS.md`](../SEGMENTATION_FAILURE_ANALYSIS.md)) and transparent technical constraints ([`LIMITATIONS.md`](../LIMITATIONS.md)).

---

## 4. HIPAA & Health Data Privacy Awareness

The Health Insurance Portability and Accountability Act (HIPAA) Privacy Rule establishes national standards for the protection of individually identifiable health information (Protected Health Information - PHI).

### Privacy Engineering in MURL:

- **DICOM Metadata Privacy Filtering:**  
  When medical DICOM files are ingested by [`src/platform/dicom_parser.py`](../src/platform/dicom_parser.py), an automated whitelist filter strips **16 direct Protected Health Information (PHI) elements** (`PatientName`, `PatientID`, `PatientBirthDate`, `PatientAddress`, `InstitutionName`, `ReferringPhysicianName`, etc.) before data is returned or cached.
- **Research Data Handling:**  
  Patient IDs are replaced by ephemeral random session UUIDs. Direct patient identifiers are never recorded in application logs, database caches, or error stack traces.
- **Zero Real Patient Data in Public Repository:**  
  The repository demo dataset (`MURL_DEMO_DATA/`) contains exclusively open-access benchmark samples (BUSI, NIH/Open-I, TotalSegmentator demo CT, BraTS demo MRI, 3D Slicer head sample) and synthetic documents with no real patient PHI.
- **Preservation of Non-Identifying Physics Metadata:**  
  While identifying personal data is scrubbed, essential physical acquisition parameters (pixel spacing, slice thickness, modality, rescale slope/intercept) are preserved to allow legitimate AI inference.

---

## 5. GDPR & Data Protection Principles Awareness

The General Data Protection Regulation (GDPR) regulates data protection and privacy in the European Union and sets global standards for handling personal data.

### Principles Reflected in MURL:

- **Data Minimization (Article 5(1)(c)):**  
  Only imaging pixels and non-identifying physical metadata required for image reconstruction and tensor conversion are processed.
- **Privacy by Design & by Default (Article 25):**  
  De-identification occurs at the earliest ingestion layer (`DICOMMetadataParser`) before model routing, rather than as an afterthought prior to export.
- **Purpose Limitation (Article 5(1)(b)):**  
  Uploaded files are processed strictly for the active session's inference task and are not harvested for covert model training.
- **No Third-Party Tracking or External Telemetry:**  
  The platform operates entirely locally on localhost with zero external analytics beacons, tracking cookies, or remote logging services.

---

## 6. Summary of Regulatory & Quality Declarations

| Standard / Regulation | Scope / Purpose | MURL Status | Practical Implementation in MURL |
|:---|:---|:---:|:---|
| **ISO 13485** | Medical QMS | **Awareness / Non-Certified** | Requirements tracing, regression test harness, configuration management |
| **IEC 62304** | Medical Software Lifecycle | **Awareness / Non-Certified** | Modular architecture, unit/integration verification, failure analysis |
| **ISO 14971** | Risk Management | **Awareness / Non-Certified** | Cross-modality safety rejection, input dimension bounds, magic header checks |
| **HIPAA** | Health Information Privacy | **Awareness / Non-Certified** | Automated 16-tag DICOM PHI scrubbing, session UUID replacement |
| **GDPR** | Data Protection & Privacy | **Awareness / Non-Certified** | Data minimization, privacy by design, zero external telemetry |
