<div align="center">    
 
# ClinicalGait-CrossAttention_ASD_PyTorch

<!--
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)   -->
<!--
ARXIV
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->

<!--  
Conference   
-->
</div>


Official PyTorch implementation of our research:  
**Cross-Attentive Temporal Fusion with Clinical Priors for Adult Spinal Deformity Classification**

This work proposes a clinically informed attention framework that integrates orthopedic knowledge into video-based gait analysis, aiming to enhance the interpretability and accuracy of automated ASD diagnosis.

## 🧠 Key Highlights

- 🎯 **Clinical Knowledge Integration**: Region-of-interest (ROI) priors from orthopedic experts guide attention to pathological joints (lumbar, pelvis, head, shoulder).
- 🔄 **Cross-Attentive Temporal Fusion**: Enhances the model’s ability to capture periodic gait dynamics and inter-joint correlations.
- 📹 **Monocular Video Input**: Works on 2D pose sequences extracted from standard RGB video.
- 🔍 **Explainability**: Generates interpretable attention maps aligned with clinical heuristics.


## 🗂️ Project Structure

```

ClinicalGait-CrossAttention\_ASD\_PyTorch/
├── configs/                # YAML configs for training/evaluation
├── data/                   # Dataset loading and preprocessing
├── models/                 # Backbone (CNN/ViT), Cross-Attention, Fusion
├── trainer/                # Training & evaluation scripts
├── visualization/          # Attention heatmap generation and demo tools
├── scripts/                # Utility bash scripts
├── docs/                   # Figures and documentation
├── requirements.txt        # Python package requirements
└── README.md

```


## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your_username/ClinicalGait-CrossAttention_ASD_PyTorch.git
cd ClinicalGait-CrossAttention_ASD_PyTorch
pip install -r requirements.txt
```

### Training

```bash
python trainer/train.py --config configs/asd_crossattention.yaml
```

### Evaluation

```bash
python trainer/evaluate.py --config configs/asd_crossattention.yaml
```

---

## 📊 Performance

| Model                          | Accuracy  | F1-Score | AUC      |
| ------------------------------ | --------- | -------- | -------- |
| CNN Baseline                   | 78.5%     | 0.76     | 0.81     |
| ViT + PhaseMix                 | 82.7%     | 0.80     | 0.85     |
| **Ours (CK + CrossAttention)** | **86.1%** | **0.84** | **0.90** |

---

## 📁 Dataset

Our dataset consists of video clips of individuals with or without Adult Spinal Deformity (ASD), recorded at 30 FPS in 1920x1080 resolution, from side-view walking trials.
Each clip is annotated by spine surgeons based on clinical assessments.

- ✅ 81 patients
- 📹 1,957 gait video clips (2–10 seconds)
- 🩻 Diagnosis from full-spine radiographs
- ⚠️ _Due to ethical constraints, dataset is not publicly released. Contact for collaboration._

---

## 📄 Citation

If you find this project helpful, please cite our work:

```bibtex
@article{chen2025crossattention,
  title={Cross-Attentive Temporal Fusion with Clinical Priors for Adult Spinal Deformity Classification},
  author={Chen, Kaixu and ...},
  journal={TBA},
  year={2025}
}
```


## 🔗 Related Works

- \[Chen et al., 2023] Two-stage gait classification with CNNs \~\cite{chen2023two}
- \[Chen et al., 2024] PhaseMix for periodic motion fusion \~\cite{chen2024phasemix}

