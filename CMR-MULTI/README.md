---
license: cc-by-nc-4.0
task_categories:
- image-segmentation
tags:
- medical
size_categories:
- n<1K
extra_gated_fields:
  Affiliation:
    type: text
    label: "Your institution/affiliation"
    required: true
  Intended Use:
    type: text
    label: "Intended use of this dataset"
    required: true
extra_gated_prompt: "Please provide your affiliation and intended use to apply for access."
---
# CMR-MULTI DATASET

## Universal Multi-Sequence, Multi-Center and Multi-View CMR Segmentation Challenge
https://mwm2026.github.io/cmr-multi

## Dataset Details

### Dataset Description

This dataset consists of paired CMR images and pixel-level segmentation labels, covering two core clinical sequences: CINE (steady-state free precession, SSFP) for functional analysis and LGE (Late Gadolinium Enhancement) for scar tissue detection. It is structured into two main tasks to facilitate multi-task learning and cross-sequence validation. The annotations cover comprehensive cardiac structures including all four chambers and the myocardium, as well as pathological regions (scar).

- **Curated by:** Beijing Academy of Artificial Intelligence (BAAI)
- **Funded by:** BAAI, Beijing Anzhen Hospital
- **Shared by:** BAAI · Beijing Anzhen Hospital
- **Language(s) (NLP):** English
- **Language(s) (NLP):** Not applicable (imaging dataset; metadata is in English)
- **License:** CC BY-SA 4.0 (Creative Commons Attribution-ShareAlike 4.0 International License)

### Dataset Sources [optional]

<!-- Provide the basic links for the dataset. -->

- **Repository:** https://huggingface.co/datasets/TaipingQu/CMR-MULTI
- **Paper [optional]:** [More Information Needed]

## 🧪 JOIN CMR-MULTI Challenge (MICCAI 2026)  
*1st Workshop on Medical World Models*

This repository supports the **JOIN CMR-MULTI Challenge**, focusing on:

- 📐 Training and benchmarking **universal cardiac MRI segmentation models**  
- 🌐 **Multi-sequence** (Cine + LGE), **multi-view**, and **multi-center** medical foundation model research  
- 📊 **Cardiac function quantification** and **myocardial scar analysis**

---

### ⚠️ Out-of-Scope Use

The provided models, data, and code **MUST NOT** be used for:

- ❌ **Clinical diagnosis** or **direct patient treatment**  
- ❌ **Commercial purposes** without explicit written permission from the organizers  
- ❌ Any use **outside of academic research** and **official challenge evaluation**

All usage must comply with applicable ethical guidelines, data licenses, and institutional review board (IRB) requirements.

## Dataset Structure

Task 1: CINE_MULTI (Functional Sequence)
This task focuses on dynamic CINE scans, which are used to assess cardiac motion and function.
```
CINE_MULTI/
├── SAX_TR/
│   ├── image/ (Short-axis CINE images, .nii.gz)
│   └── anno/ (Segmentation labels, .nii.gz)
├── 2CH_TR/
│   ├── image/ (2-chamber view CINE images, .nii.gz)
│   └── anno/ (Segmentation labels, .nii.gz)
├── 4CH_TR/
│   ├── image/ (4-chamber view CINE images, .nii.gz)
│   └── anno/ (Segmentation labels, .nii.gz)
└── dataset.xlsx (Metadata file with 3 sheets: SAX, 2CH, 4CH)
```
Label Definition (CINE Sequence)

| View | Label ID | Description |
| --- | --- | --- |
| SAX_TR | 1 | Left Ventricle Myocardium |
| | 2 | Left Ventricle Cavity |
| | 3 | Right Ventricle Cavity |
| 2CH_TR | 1 | Left Ventricle Cavity |
| | 2 | Left Ventricle Myocardium |
| 4CH_TR | 1 | Left Ventricle Cavity |
| | 2 | Left Ventricle Myocardium |
| | 3 | Right Ventricle Cavity |
| | 4 | Right Atrium |
| | 5 | Left Atrium |

Metadata (dataset.xlsx - CINE)
The Excel file contains image/label path mapping and clinical metrics.
Sheets: SAX, 2CH, 4CH
Common Columns: image_path, anno_path
Special Columns (SAX Sheet only): LVEF (Left Ventricular Ejection Fraction, a key clinical metric for cardiac function)
Task 2: LGE_MULTI (Pathological Sequence)
This task focuses on Late Gadolinium Enhancement scans, which highlight myocardial scar tissue.
```
LGE_MULTI/
├── SAX_TR/
│   ├── image/ (Short-axis LGE images, .nii.gz)
│   └── anno/ (Segmentation labels, .nii.gz)
├── 2CH_TR/
│   ├── image/ (2-chamber view LGE images, .nii.gz)
│   └── anno/ (Segmentation labels, .nii.gz)
├── 4CH_TR/
│   ├── image/ (4-chamber view LGE images, .nii.gz)
│   └── anno/ (Segmentation labels, .nii.gz)
├── RAS_TR/
│   ├── image/ (Right Atrium - SAX/long-axis LGE images, .nii.gz)
│   └── anno/ (Segmentation labels, .nii.gz)
└── dataset.xlsx (Metadata file with 4 sheets: SAX, 2CH, 4CH, RAS)
```
Label Definition (LGE Sequence)

| View | Label ID | Description |
| --- | --- | --- |
| SAX_TR | 1 | Left Ventricle Cavity |
| | 2 | Left Ventricle Myocardium |
| | 3 | Myocardial Scar (Scar) |
| | 4 | Right Ventricle Cavity |
| 2CH_TR | 1 | Left Ventricle Cavity |
| | 2 | Left Ventricle Myocardium |
| | 3 | Myocardial Scar (Scar) |
| 4CH_TR | 1 | Left Ventricle Cavity |
| | 2 | Left Ventricle Myocardium |
| | 3 | Myocardial Scar (Scar) |
| | 4 | Right Ventricle Cavity |
| RAS_TR | 1 | Right Atrium |

Metadata (dataset.xlsx - LGE)
The Excel file contains image/label path mapping and pathological metadata.
Sheets: SAX, 2CH, 4CH, RAS
Common Columns: image_path, anno_path
Special Columns (SAX Sheet only): Scar_Quality (Quantitative metric assessing the extent/quality of scar tissue annotation)

The original images of the RAS_TR dataset are sourced from https://www.cardiacatlas.org/atriaseg2018-challenge/atria-seg-data/, and the annotations are derived from https://zenodo.org/records/15524472.

## Baseline Code
https://github.com/qutaiping/CMR_multi_baseline/

## Dataset Creation

### Curation Rationale

Cardiac MRI analysis requires robust segmentation across both functional (CINE) and pathological (LGE) sequences to enable comprehensive clinical assessment, including function quantification and scar detection. This dataset unifies multi-view (SAX, 2CH, 4CH) annotations for both sequences, providing a valuable resource for developing multi-task AI models that can integrate structural, functional, and pathological information.

### Source Data

- **Data Collection & Processing: Multi-center, IRB-approved, de-identified CMR data**
- **Views: SAX · 2CH · 4CH**
- **Total Annotated Cases: 750+**
- **Format: NIfTI .nii.gz**
- **Annotations: Pixel-level segmentation by professional cardiologists & radiologists**


### Annotations

Standardized cardiac segmentation protocol; reviewed by senior cardiac imaging experts.


#### Who are the annotators?

Cardiologists and radiologists from Beijing Anzhen Hospital & collaborative clinical centers.


#### Personal and Sensitive Information

All data fully de-identified; IRB approved; no patient privacy information included.

## Bias, Risks, and Limitations

- **For research & challenge only; not for clinical use** 
- **Data from 3 scanner vendors; may have domain shift between centers**
- **Scar annotation affected by LGE image quality**


## Citation

**BibTeX:**

```bibtex
@misc{qu2026baaicardiacagentintelligent,
      title={BAAI Cardiac Agent: An intelligent multimodal agent for automated reasoning and diagnosis of cardiovascular diseases from cardiac magnetic resonance imaging}, 
      author={Taiping Qu and Hongkai Zhang and Lantian Zhang and Can Zhao and Nan Zhang and Hui Wang and Zhen Zhou and Mingye Zou and Kairui Bo and Pengfei Zhao and Xingxing Jin and Zixian Su and Kun Jiang and Huan Liu and Yu Du and Maozhou Wang and Ruifang Yan and Zhongyuan Wang and Tiejun Huang and Lei Xu and Henggui Zhang},
      year={2026},
      eprint={2604.04078},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2604.04078}, 
}
```


## Dataset Card Contact

For questions, feedback, or to report issues, please contact:
tpqu@baai.ac.cn

## Access & License
This dataset is available **for non-commercial research and challenge purposes only**.
Access is granted **only after manual approval** by the dataset authors.

By requesting access, you agree to:
- Use the data solely for academic research.
- Not redistribute or share the data.
- Not use the data for clinical purposes.
- Cite the dataset in any publications