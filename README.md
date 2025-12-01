<div align="center">
    <img src="assets/RunIA-logo.png" width="20%" alt="RunIA Logo" />
    <h1 style="font-size: large; font-weight: bold;">Runtime Uncertainty estimation for AI models</h1>
</div>

<div align="center">
    <a href="https://www.python.org/downloads/release/python-390/">
        <img src="https://img.shields.io/badge/Python-3.9+-efefef">
    </a>
    <a href="https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
    <img src="https://img.shields.io/badge/version-2.0.0-blue">
</div>
<br>

---

# Overview

**RunIA** is an open-source Python library for uncertainty estimation and Out-of-Distribution (OoD) detection in AI models. It provides comprehensive tools for evaluating and deploying uncertainty estimation methods across computer vision tasks (image classification, object detection, semantic segmentation) and natural language processing (LLM hallucination detection).

## Key Features

- **Latent Space Uncertainty Estimation**: LaRED (Latent Representations Density) and LaREM (Latent Representations Mahalanobis) methods for OoD detection
- **Monte Carlo Dropout (MCD)**: Epistemic uncertainty estimation through MC sampling
- **Multiple Baseline Methods**: Support for 15+ baseline OoD detection methods (MSP, Energy, Mahalanobis, kNN, ViM, DDU, DICE, ReAct, and more)
- **LLM Uncertainty**: Hallucination detection with methods like semantic entropy, RAUQ, perplexity, and eigen scores
- **Feature Extraction**: Image-level and object-level feature extraction for various architectures
- **Flexible Inference**: Production-ready inference modules for real-time OoD detection
- **Comprehensive Evaluation**: Built-in metrics (AUROC, AUPR, FPR@95) and visualization tools

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
  - [Computer Vision: OoD Detection](#computer-vision-ood-detection)
  - [LLM Uncertainty Estimation](#llm-uncertainty-estimation)
- [Supported Tasks and Architectures](#supported-tasks-and-architectures)
- [API Overview](#api-overview)
- [Hardware Requirements](#hardware-requirements)
- [References](#references)

---

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended for computer vision tasks)

### Using pip

```bash
# Clone the repository
git clone <repository-url>
cd runia

# Create a virtual environment (recommended)
python -m venv runia_env
source runia_env/bin/activate  # On Windows: runia_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install .
```

### Using conda

```bash
# Create a conda environment
conda create -n runia_env python=3.9
conda activate runia_env

# Install dependencies and package
pip install -r requirements.txt
pip install .
```

---

## Quick Start

### Computer Vision OoD Detection

```python
import torch
from runia.evaluation import Hook, get_latent_representation_mcd_samples, get_dl_h_z
from runia.inference import LaRExInference, MCSamplerModule, LaREMPostprocessor
from runia import apply_pca_ds_split

# Setup model with dropout/dropblock layer
model = YourModel()
hooked_layer = Hook(model.dropout_layer)
model.eval()

# Extract MC samples and compute entropy
latent_samples = get_latent_representation_mcd_samples(
    model, dataloader, n_samples=16, hooked_layer
)
_, entropy_samples = get_dl_h_z(latent_samples, mcd_samples_nro=16)

# Setup OoD detector
pca_train, pca_transform = apply_pca_ds_split(entropy_samples, nro_components=256)
detector = LaREMPostprocessor()
detector.setup(pca_train)

# Inference on new images
inference_module = LaRExInference(
    dnn_model=model,
    detector=detector,
    mcd_sampler=MCSamplerModule,
    pca_transform=pca_transform,
    mcd_samples_nro=16,
    layer_type="Conv"
)

prediction, confidence_score = inference_module.get_score(test_image, layer_hook=hooked_layer)
```

### LLM Uncertainty Estimation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from runia.llm_uncertainty import compute_uncertainties

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

gen_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=1.0)

# Define uncertainty methods
uncertainty_requests = [
    {"method_name": "semantic_entropy"},
    {"method_name": "perplexity"},
    {"method_name": "eigen_score"},
    {"method_name": "RAUQ", "token_aggregation": "original", "head_aggregation": "mean_heads"}
]

# Compute uncertainties
generated_text, scores = compute_uncertainties(
    model, tokenizer, "Your prompt here",
    uncertainty_requests, gen_config, num_samples=10
)
```

---

## Supported Tasks and Architectures

### Computer Vision

| Task | Datasets (In-Dist) | Datasets (OoD) | Architectures |
|------|-------------------|----------------|---------------|
| **Image Classification** | CIFAR10 | FMNIST, SVHN, Places365, Textures, iSUN, LSUN | ResNet-18, ResNet-18 + Spectral Norm |
| **Object Detection** | BDD100k, Pascal VOC | COCO, OpenImages | Faster RCNN, YOLOv8, RT-DETR, Deformable DETR, OWLv2 |
| **Semantic Segmentation** | Woodscape, Cityscapes | Woodscape-anomalies, Cityscapes-anomalies | DeepLabv3+, U-Net |

### Natural Language Processing

| Task | Datasets (In-Dist) | Datasets (OoD) | Architectures |
|------|-------------------|----------------|---------------|
| **Hallucination Detection** | SQuADv2 | TriviaQA, Natural Questions, HotpotQA | Llama-3.1, DistilBERT-base |

**Note**: For computer vision tasks, models should include dropout or DropBlock2D layers to enable Monte Carlo Dropout sampling for epistemic uncertainty estimation.

---

## Usage Examples

### Computer Vision: OoD Detection

#### 1. Evaluation Pipeline

Evaluate uncertainty estimation methods on In-Distribution (InD) vs Out-of-Distribution (OoD) datasets:

```python
import torch
from runia.evaluation import Hook, get_latent_representation_mcd_samples, apply_dropout, get_dl_h_z
from runia.evaluation.metrics import log_evaluate_lared_larem

# Setup
N_MCD_SAMPLES = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and hook dropout/dropblock layer
model = YourModel.load_from_checkpoint("model.pt")
hooked_layer = Hook(model.dropout_layer)
model.to(device).eval()
model.apply(apply_dropout)

# Extract MC samples for InD and OoD data
latent_ind_train = get_latent_representation_mcd_samples(
    model, ind_train_loader, N_MCD_SAMPLES, hooked_layer
)
latent_ind_test = get_latent_representation_mcd_samples(
    model, ind_test_loader, N_MCD_SAMPLES, hooked_layer
)
latent_ood_test = get_latent_representation_mcd_samples(
    model, ood_test_loader, N_MCD_SAMPLES, hooked_layer
)

# Compute entropy from MC samples
_, entropy_ind_train = get_dl_h_z(latent_ind_train, mcd_samples_nro=N_MCD_SAMPLES)
_, entropy_ind_test = get_dl_h_z(latent_ind_test, mcd_samples_nro=N_MCD_SAMPLES)
_, entropy_ood_test = get_dl_h_z(latent_ood_test, mcd_samples_nro=N_MCD_SAMPLES)

# Evaluate LaRED and LaREM (returns metrics: AUROC, AUPR, FPR@95)
ood_datasets = {'ood_dataset_name': entropy_ood_test}
metrics_df = log_evaluate_lared_larem(
    ind_train=entropy_ind_train,
    ind_test=entropy_ind_test,
    ood_dict=ood_datasets
)
print(metrics_df)
```

#### 2. Inference Pipeline

Deploy OoD detection in production:

```python
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from runia.evaluation import Hook
from runia.inference import LaRExInference, MCSamplerModule, LaREMPostprocessor
from runia import apply_pca_ds_split

# Load model
model = YourModel.load_from_checkpoint("model.pt")
hooked_layer = Hook(model.dropout_layer)
model.eval()

# Load pre-calculated InD entropies
entropy_train = np.load("ind_train_entropies.npy")
entropy_test = np.load("ind_test_entropies.npy")

# Apply PCA (recommended: 256 components for LaREM)
pca_train, pca_transform = apply_pca_ds_split(entropy_train, nro_components=256)

# Setup LaREM detector
detector = LaREMPostprocessor()
detector.setup(pca_train)

# Calculate threshold (95% confidence)
test_scores = detector.postprocess(pca_transform.transform(entropy_test))
threshold = np.mean(test_scores) - (1.645 * np.std(test_scores))

# Setup inference module
inference = LaRExInference(
    dnn_model=model,
    detector=detector,
    mcd_sampler=MCSamplerModule,
    pca_transform=pca_transform,
    mcd_samples_nro=16,
    layer_type="Conv"
)

# Run inference on new image
image = Image.open("test_image.jpg")
tensor_image = transforms.ToTensor()(image).unsqueeze(0)
prediction, ood_score = inference.get_score(tensor_image, layer_hook=hooked_layer)

print(f"Prediction: {prediction}")
print(f"OoD Score: {ood_score:.4f}")
print(f"Is InD: {ood_score > threshold}")
```

#### 3. Using Baseline Methods

RunIA supports 15+ baseline OoD detection methods:

```python
from runia.baselines import compute_baseline_from_model

# Available methods: 'msp', 'energy', 'mdist', 'knn', 'vim', 'ddu', 'dice', 'react', etc.
baseline_scores = compute_baseline_from_model(
    model=model,
    dataloader=test_loader,
    method_name='energy',  # Energy-based OoD detection
    device='cuda'
)
```

### LLM Uncertainty Estimation

Detect hallucinations and measure uncertainty in LLM outputs:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from runia.llm_uncertainty import compute_uncertainties

# Load model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

gen_config = GenerationConfig(
    max_new_tokens=50,
    do_sample=True,
    top_p=0.9,
    temperature=1.0
)

# Define uncertainty methods
uncertainty_methods = [
    {"method_name": "semantic_entropy"},      # Semantic uncertainty
    {"method_name": "eigen_score"},            # Eigenvalue-based score
    {"method_name": "perplexity"},             # Model perplexity
    {"method_name": "normalized_entropy"},     # Normalized entropy
    {"method_name": "generation_entropy"},     # Generation-level entropy
    {
        "method_name": "RAUQ",                 # Attention-based uncertainty
        "token_aggregation": "original",
        "head_aggregation": "mean_heads",
        "alphas": [0.2, 0.4, 0.6],
        "ablation": True
    }
]

# Compute uncertainties
text, scores = compute_uncertainties(
    model,
    tokenizer,
    prompt="What is the capital of France?",
    uncertainty_requests=uncertainty_methods,
    gen_config=gen_config,
    num_samples=10
)

print(f"Generated: {text}")
print(f"Uncertainty Scores: {scores}")
```

---

## API Overview

### Core Modules

| Module | Description |
|--------|-------------|
| `runia.evaluation` | MC sampling, entropy computation, OoD evaluation metrics |
| `runia.inference` | Production-ready inference with LaRED/LaREM postprocessors |
| `runia.baselines` | 15+ baseline OoD detection methods (MSP, Energy, Mahalanobis, kNN, ViM, DDU, DICE, ReAct, etc.) |
| `runia.feature_extraction` | Image-level and object-level feature extraction |
| `runia.llm_uncertainty` | LLM uncertainty and hallucination detection methods |
| `runia.dimensionality_reduction` | PCA and other dimensionality reduction utilities |

### Key Classes and Functions

**Evaluation:**
- `Hook`: Capture layer outputs during forward pass
- `get_latent_representation_mcd_samples()`: Extract MC dropout samples
- `get_dl_h_z()`: Compute entropy from MC samples
- `log_evaluate_lared_larem()`: Evaluate LaRED/LaREM with metrics

**Inference:**
- `LaRExInference`: Main inference module for OoD detection
- `LaREMPostprocessor`: Mahalanobis distance-based detector (recommended)
- `LaREDPostprocessor`: KDE-based detector
- `MCSamplerModule`: Monte Carlo sampling module

**LLM Uncertainty:**
- `compute_uncertainties()`: Compute multiple uncertainty scores for LLM outputs

---

## Hardware Requirements

- **CPU**: Supported but slow for computer vision tasks
- **GPU**: Required for efficient inference on object detection and segmentation
- **Memory**: Varies by model size (8GB+ GPU memory recommended)

---

## References

### Publications

- **Out-of-Distribution Detection using Deep Neural Network Latent Space**

### Technical Reports

- EC3-FA06 Run-Time Monitoring
- EC3-FA18 Run-Time Monitoring

### Confiance AI Documentation

- [Methodological guidelines](https://irtsystemx.sharepoint.com/:b:/s/IAdeConfiance833/ERYc5y-HkPdAvL0TVAQdp0kBkfsPhJwrXrfZrVsH8CuY8Q?e=1mpavP)
- [Benchmarks](https://irtsystemx.sharepoint.com/:b:/s/IAdeConfiance833/EfaV2zJlJ9VOqMHSr9sk1JIBvXl3CjQGRzHzwAtO_SXiHQ?e=AbUAiM)
- [Use Case application: Valeo - Scene Understanding](https://irtsystemx.sharepoint.com/:b:/s/IAdeConfiance833/EZKRyjRiobZLm58OoerTgTYB9o_PjyuPpVY7PXFb_v0_hg?e=cWNHdI)

---

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

See [LICENSE.txt](LICENSE.txt) for details.

## Authors

- **Fabio Arnez** - fabio.arnez@cea.fr
- **Daniel Montoya** - daniel-alfonso.montoyavasquez@cea.fr

---

## Acknowledgments

This work was developed as part of the Confiance.ai program, focusing on trustworthy AI systems with emphasis on uncertainty estimation and out-of-distribution detection.