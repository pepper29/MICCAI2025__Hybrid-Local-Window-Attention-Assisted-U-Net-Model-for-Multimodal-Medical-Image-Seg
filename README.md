# Feature Representation-Aware Knowledge Distillation for Incomplete Local Information in Federated Learning

> **🚀 PyTorch implementation of the paper presented at [MICCAI 2025](https://conferences.miccai.org/2025/).**

> **🚀 Paper Link: [https:conferences.miccai.org/2025/](https://link.springer.com/chapter/10.1007/978-3-032-04937-7_22).**
### 👋 Introduction

Hi there! I'm **Seyong Jin**, a Master's student at the Department of Artificial Intelligence, **Sejong University**.

I am excited to share my research paper, which I personally presented at **MICCAI 2025** 🏛️. This project addresses a critical challenge in medical image analysis: **Federated Learning with missing modalities**.

In real-world clinical settings, not all hospitals have the same complete set of MRI sequences (e.g., T1, T2, FLAIR, T1ce). My work introduces a novel **Feature Representation-Aware Knowledge Distillation** framework to handle this incomplete local information effectively.

### ✨ Key Contributions

To solve the performance degradation caused by missing modalities, I proposed two core mechanisms:

* **🧩 Disentangled Representation Learning (DRL):**
* A module that decomposes complex features into independent latent representations.
* This allows the model to learn robust structural features even when specific modalities are missing.


* **🎯 Region-aware Contrastive Learning (RCL):**
* Unlike standard contrastive losses that look at the global image, **RCL** focuses on specific regions (e.g., tumor core vs. background).
* It maximizes the similarity of features within the same region (using masks) while pushing away irrelevant background noise, ensuring high-quality segmentation.



### 🏆 Achievement

* **Conference:** MICCAI 2025 (Medical Image Computing and Computer Assisted Intervention)
* **Presenter:** Seyong Jin
* **Task:** Brain Tumor Segmentation (BraTS) in Federated Learning

---

# Feature Representation-Aware Knowledge Distillation for Incomplete Local Information in Federated Learning

> **🚀 [MICCAI 2025](https://conferences.miccai.org/2025/)에서 발표된 논문의 저장소입니다.**

### 👋 소개

안녕하세요! **세종대학교 인공지능학과 석사 과정 진세용**입니다.

제가 **MICCAI 2025** 🏛️에서 직접 발표한 연구 논문을 공개하게 되어 기쁩니다. 이 프로젝트는 의료 영상 분석 분야의 중요한 난제 중 하나인 **'모달리티가 누락된 상황에서의 연합 학습(Federated Learning)'** 문제를 다루고 있습니다.

실제 의료 현장에서는 모든 병원이 동일한 MRI 시퀀스(T1, T2, FLAIR, T1ce)를 완벽하게 갖추기 어렵습니다. 저는 이러한 불완전한 로컬 정보 문제를 해결하기 위해 **특징 표현 인지 지식 증류(Feature Representation-Aware Knowledge Distillation)** 기법을 제안했습니다.

### ✨ 핵심 기여점

모달리티 누락으로 인한 성능 저하를 방지하기 위해, 저는 다음 두 가지 핵심 메커니즘을 개발했습니다:

* **🧩 Disentangled Representation Learning (DRL):**
* 복잡한 영상 특징을 서로 독립적인 잠재 표현(Latent Representation)으로 분리해내는 모듈입니다.
* 이를 통해 특정 모달리티가 없더라도 영상의 구조적 특징을 강건하게 학습할 수 있습니다.


* **🎯 Region-aware Contrastive Learning (RCL):**
* 이미지 전체를 비교하는 기존 방식과 달리, **RCL**은 마스크(Mask)를 활용해 특정 영역(예: 종양 코어 vs 배경)에 집중합니다.
* 동일한 영역 내의 특징 유사도는 높이고 불필요한 배경 잡음과는 거리를 두게 하여, 정교한 분할(Segmentation) 성능을 달성했습니다.



### 🏆 연구 성과

* **학회:** MICCAI 2025 (Medical Image Computing and Computer Assisted Intervention)
* **발표자:** 진세용 (Seyong Jin)
* **주제:** 연합 학습 기반의 뇌 종양 분할 (BraTS)
