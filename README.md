<div align="center">
    <img src="assets/RunIA-logo.png" width="20%" alt="ConfianceAI Logo" />
    <h1 style="font-size: large; font-weight: bold;">Runtime Uncertainty estimation for AI models</h1>
</div><div align="center">
    <a href="https://www.python.org/downloads/release/python-380/">
        <img src="https://img.shields.io/badge/Python-3.9-efefef">
    </a>
    <a href="https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
</div>
<br>

---
# Guideline CEA-LSEA Out-of-Distribution Detection using DNN Latent Representations Uncertainty
---

CEA-LSEA package for Out-of-Distribution (OoD) detection using the uncertainty (entropy) from DNN latent representations.
The package has been used with the following applications, the corresponding DNN architectures and datasets:

- **Simple Classification:**
    - **In-Distribution Dataset:** CIFAR10
    - **Out-of-Distribution Datasets:** FMNIST, SVHN, Places365, Textures, iSUN, LSUN-C, LSUN-R
    - **DNN Architectures:**
        1. ResNet-18
        2. ResNet-18 with Spectral Normalization

- **Object Detection:**
    - **In-Distribution Dataset:** BDD100k
    - **Out-of-Distribution Datasets:** Pascal VOC, Openimages
    - **DNN Architectures:**
      1. Faster RCNN

- **Semantic Segmentation:**
    - **In-Distribution Dataset:** Woodscape  & Cityscapes
    - **Out-of-Distribution Datasets:** Woodscape soiling, Woodscape-anomalies, Cityscapes-anomalies
    - **DNN Architectures:**
      1. Deeplabv3+
      2. U-Net

In all the above cases, the DNNs were slightly modified to capture _epistemic_ uncertainty using the Monte-Carlo Dropout by adding a ``DropBlock2D`` layer.

- [🔍 Specifications](#specifications)
- [🚀 Quick Start](#quickstart)
- [🎮 Usage](#usage)
- [🔀 Description of Inputs and Outputs](#io)
- [💻 Required Hardware](#hardware)
- [📚 References](#references)

<div id='specifications'/>

## 🔍 Specifications
- Version: 1.1.0
- Python Version: python 3.8
- Strong Dependencies: torch, entropy_estimators, numpy, sklearn, dropblock, pandas, mlflow, matplotlib
- Thematic: Computer vision
- Trustworthy: Uncertainty Estimation -  OoD/Anomaly detection
- Hardware : GPU


<div id='id-card'/>

## 🔍 Identity card

 Can be found here: [Identity card](https://git.irt-systemx.fr/confianceai/ec_3/n6_monitoring/component-latent-space/ood-latent-space/-/blob/4-finish-library/docs/identity_card.yml)

<div id='quickstart'/>

## 🚀 Quick Start

To install and use the component , it is recommended to create a Python virtual environment. You can do that with virtualenv, as follows:

### Setting environement

With `virtualenv`
```bash
pip install virtualenv
virtualenv ls_ood_detection_env
source ls_ood_detection_env/bin/activate
```

Or using conda:
```bash
conda create -n ls_ood_detection_env python=3.8
conda activate ls_ood_detection_env
```


### Installation
After creating the environment with python 3.8, install the `requirements.txt` using:
```bash
pip install -r requirements.txt
```

After installing all the requirements, then in the base folder of the repo do `pip install .`


<div id='usage'/>

## 🎮 Usage

Global steps for using the package:

* **To make an evaluation** of how the component performs at classifying one or several Out-of-Distribution (OoD) test datasets with respect to an In-Distribution (InD) test dataset:

    1. Load your Dataloader Modules for InD and OoD datasets. For the InD it is necessary to use the train and a test set; for the OoD only the test set is needed.
    2. Load your DNN Module trained on the InD dataset.
    3. Add Hook to the target layer of the DNN Module for MC samples extraction, making sure dropout or dropblock is activated.
    4. Get Monte-Carlo (MC) samples for InD samples dataloader, and OoD samples dataloader.
        * Note: make sure the dataloader is compatible with any of the functions for samples extraction: Look below
    5. Get Entropy from InD and OoD MC samples
    6. Instantiate density estimator (KDE for LaRED, or Mahalanobis for LaREM)
    7. Evaluate OoD performance results to obtain good configuration parameters: Whether laRED or laREM perform better and how many PCA components to use (if better than not using PCA)

* **To perform inference** using the component, that is: make predictions using the model, obtain the confidence score and classify images as InD or OoD:        
    1. Load your trained DNN Module
    2. Add Hook to the target layer of the DNN Module for MC samples extraction.
    3. Load previously calculated InD entropies from evaluation step, from the train and a test set.
    4. Optionally train a PCA module on the InD train entropies with an appropriate number of components, according to evaluation results.
    5. Instantiate one density estimator (either KDE for LaRED, or Mahalanobis for LaREM). Stability tests favor LaREM so we recommend it as the default option.
    6. Calculate a threshold for classification using the InD test set
    7. Perform inference on new images: get predictions and confidence scores
    8. Visualize predictions and classify new images as InD or OoD according to the score and the threshold

Note that in any case the entropies from the InD train and test set are necessary to be able to perform inference, because the components modules (PCA module and postprocessor module) need to be trained before proceeding to inference, and a threshold needs to be calculated.

Also, stability tests favor LaREM, which tends to work well with about 256 components across different benchmarks. So in case of doubt we recommend testing this configuration.

For a complete illustration of how to use the component, check the notebooks: [Notebook 1: extract MCDz samples](https://git.irt-systemx.fr/confianceai/ec_3/n6_monitoring/component-latent-space/ood-latent-space/-/blob/4-finish-library/examples/1_example_deeplab_segmentation_samples_extraction.ipynb?ref_type=heads),
 [Notebook 2: Evaluate LaREx](https://git.irt-systemx.fr/confianceai/ec_3/n6_monitoring/component-latent-space/ood-latent-space/-/blob/4-finish-library/examples/2_example_deeplab_segmentation_analysis.ipynb?ref_type=heads) 
and [Notebook 3: Inference with LaREx](https://git.irt-systemx.fr/confianceai/ec_3/n6_monitoring/component-latent-space/ood-latent-space/-/blob/4-finish-library/examples/3_performing_inference.ipynb?ref_type=heads)

### Component evaluation example
Here we present first a general overview of how to evaluate the component and obtain detection 
metrics. For a full detailed example of evaluating the component refer to 
[Notebook 1: extract MCDz samples](https://git.irt-systemx.fr/confianceai/ec_3/n6_monitoring/component-latent-space/ood-latent-space/-/blob/4-finish-library/examples/1_example_deeplab_segmentation_samples_extraction.ipynb?ref_type=heads) and  [Notebook 2: Evaluate LaREx](https://git.irt-systemx.fr/confianceai/ec_3/n6_monitoring/component-latent-space/ood-latent-space/-/blob/4-finish-library/examples/2_example_deeplab_segmentation_analysis.ipynb?ref_type=heads).

```python
import torch
from runia.uncertainty_estimation import (
    Hook,
    get_latent_representation_mcd_samples,
    apply_dropout,
    get_dl_h_z
)
from runia.metrics import log_evaluate_lared_larem

N_MCD_SAMPLES = 16
N_PCA_COMPONENTS = 256
z_95 = 1.645
LAYER_TYPE = "Conv"
ind_sample_image_path = "./my_image.jpg"
model_checkpoint_path = "./my_checkpoint.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
trained_model = MyTrainedModel.load_from_checkpoint(checkpoint_path=model_checkpoint_path)
# Hook the dropout or dropblock layer
hooked_layer = Hook(trained_model.my_dropblock_layer)
# Activate dropout or dropblock at inference
trained_model.to(device)
trained_model.eval()
trained_model.apply(apply_dropout)

# Extract MCDz samples
# InD
latent_samples_ind_train = get_latent_representation_mcd_samples(trained_model,
                                                                 my_ind_train_data_loader,
                                                                 N_MCD_SAMPLES,
                                                                 hooked_layer)
latent_samples_ind_test = get_latent_representation_mcd_samples(trained_model,
                                                                my_ind_test_data_loader,
                                                                N_MCD_SAMPLES,
                                                                hooked_layer)
# OoD
latent_samples_ood_test = get_latent_representation_mcd_samples(trained_model,
                                                                my_ood_test_data_loader,
                                                                N_MCD_SAMPLES,
                                                                hooked_layer)
# Get entropy
_, entropy_samples_ind_train = get_dl_h_z(latent_samples_ind_train, mcd_samples_nro=N_MCD_SAMPLES)
_, entropy_samples_ind_test = get_dl_h_z(latent_samples_ind_test, mcd_samples_nro=N_MCD_SAMPLES)
_, entropy_samples_ood_test = get_dl_h_z(latent_samples_ood_test, mcd_samples_nro=N_MCD_SAMPLES)

# Evaluate LaRED and LaREM
# Pass OoD samples as dictionary (you can have several OoD datasets)
ood_datasets_dict = {
    'my_ood_dataset1': entropy_samples_ood_test
}
metrics_pandas_df = log_evaluate_lared_larem(
    ind_train=entropy_samples_ind_train,
    ind_test=entropy_samples_ind_test,
    ood_dict=ood_datasets_dict,
)
```

Pay special attention to the compatibility of the `get_latent_representation_mcd_samples` function 
with the particular dataloader you will use in your project. The main compatibility issue happens
 in the type of line such as `for elements in dataloader: (...)`, where `elements` might contain a 
list, a dictionary, some labels, or other items in a specific order, which makes it difficult to 
have just one extraction function that works with all dataloaders. The `get_latent_representation_mcd_samples`
 implements such line as `for i, (image, label) in enumerate(dataloader): (..)`, therefore it works
with a dataloader where `elements` is a tuple of an `image` and the `label`. For other type of 
contents in the dataloader, you might need to use a custom function.


### Perform inference with the component
As another side note, after several benchmarks, the LaREM component seems to show better stability,
and works well with about 256 PCA components. In case of doubt test this configuration as a default.

Now we present a general overview of the inference process. For a full detailed example refer to
[Notebook 3: Inference with LaREx](https://git.irt-systemx.fr/confianceai/ec_3/n6_monitoring/component-latent-space/ood-latent-space/-/blob/4-finish-library/examples/3_performing_inference.ipynb?ref_type=heads).

```python
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from runia.uncertainty_estimation import Hook, LaRExInference, MCSamplerModule, LaREMPostprocessor
from runia import apply_pca_ds_split

N_MCD_SAMPLES = 16
N_PCA_COMPONENTS = 256
z_95 = 1.645
LAYER_TYPE = "Conv"  # As we get the output of a convolutional layer
image_path = "./my_image.jpg"
model_checkpoint_path = "./my_checkpoint.pt"
train_entropies_path = "./my_train_entropies.npy"
test_entropies_path = "./my_test_entropies.npy"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############### Load modules ############
# Load the trained model
trained_model = MyTrainedModel.load_from_checkpoint(checkpoint_path=model_checkpoint_path)
trained_model.to(device)
trained_model.eval()
# Hook the dropout or dropblock layer
hooked_layer = Hook(trained_model.my_dropblock_layer)

# Load precalculated entropies
entropy_samples_ind_train = np.load(train_entropies_path)
entropy_samples_ind_test = np.load(test_entropies_path)
# Optionally train PCA module and transform InD train samples
pca_ind_train, pca_transformation = apply_pca_ds_split(
    samples=entropy_samples_ind_train,
    nro_components=N_PCA_COMPONENTS
)
# Setup LaREM postprocessor with the PCA transformed features
larem_ds_shift_detector = LaREMPostprocessor()
larem_ds_shift_detector.setup(pca_ind_train)

############# Calculate threshold ##################
# Calculate scores for the InD test set
test_ind_larem = larem_ds_shift_detector.postprocess(pca_transformation.transform(entropy_samples_ind_test))
# Calculate threshold
mean_ind_larem, std_ind_larem = np.mean(test_ind_larem), np.std(test_ind_larem)
# Here we use the 95% confidence z score
threshold_larem = mean_ind_larem - (z_95 * std_ind_larem) 

############ Inference ##############
# Instantiate Inference module and sampler
larem_inference_module = LaRExInference(
    dnn_model=trained_model,
    detector=larem_ds_shift_detector,
    mcd_sampler=MCSamplerModule,
    pca_transform=pca_transformation,
    mcd_samples_nro=N_MCD_SAMPLES,
    layer_type=LAYER_TYPE
)

# Load test images
test_img = Image.open(image_path)

# Apply same transformations as test (usually resize, to tensor and normalize)
img_transforms = transforms.Compose(my_test_transforms)
tensor_img = img_transforms(test_img).unsqueeze(0)  # Optionally unsqueeze to make the image batch 1

# Perform inference and get confidence score
model_prediction, confidence_score = larem_inference_module.get_score(tensor_img, layer_hook=hooked_layer)
# See if image is InD or OoD according to score and threshold
print(f"Image score: {confidence_score}, score above threshold: {confidence_score > threshold_larem}")
```
Both the model predictions and the confidence score are returned by the `get_score` method of the
`LaRExInference` class.

<div id='io'/>

## 🔀 Description of Inputs and Outputs

In general, to perform OoD or anomaly detection with our method you need:
* A well defined In Distribution (InD) dataset
* A model already trained on the InD dataset
* One (or more) dataset(s) defined as Out of Distribution (OoD) or Anomaly to perform evaluation

Then if the model was trained with at least one dropout or dropblock layer, we need to:
* Attach a Hook to the dropblock or dropout layer to catch the outputs of such layer: Use the
  `Hook` class
* Perform Monte Carlo dropout sampling, meaning, inference is performed $N$ times, at each time the
  output of the hooked layer is taken (Use the `get_latent_representation_mcd_samples` function
  (You just need to specify the type of layer in the function parameters):
    * If the layer is convolutional (dropblock layer): we take the mean per channel
    * If the layer is Fully Connected (dropout layer): we take the raw output

* Take the entropy of the previously calculated samples. Use the `get_dl_h_z` function
* Pass the entropy of each image to the LaRED and LaREM estimators to obtain the OoD score. Use the
  `log_evaluate_lared_larem` function.

We will obtain at the end a pandas dataframe with the evaluation metrics and classification
thresholds of LaRED and LaREM. It is possible to plot ROC curves and get FPR@95, AUROC and AUPR metrics for evaluation.

To use the method during inference it is needed to save the InD train and test entropies,
optionally the PCA transformation, and calculate a threshold using a test set. Then using the 
inference module we would obtain the uncertainty estimation at inference time, along with the prediction.
See previous section.

<div id='hardware'/>

## 💻 Required Hardware

GPU is a requirement for our component since we deal with "heavy" tasks in computer vision such as
object detection and image segmentation, which are typically too slow in CPU.

<div id='references'/>

## 📚 References

For more technical and implementation details, we refer the user to the following technical
reports and publications:

Technical Reports:

    - EC3-FA06 Run-Time Monitoring
    - EC3-FA18 Run-Time Monitoring

Publications

    - Out-of-Distribution Detection using Deep Neural Network Latent Space


Confiance AI documents:
* [Methodological guidelines](https://irtsystemx.sharepoint.com/:b:/s/IAdeConfiance833/ERYc5y-HkPdAvL0TVAQdp0kBkfsPhJwrXrfZrVsH8CuY8Q?e=1mpavP)
* [Benchmarks](https://irtsystemx.sharepoint.com/:b:/s/IAdeConfiance833/EfaV2zJlJ9VOqMHSr9sk1JIBvXl3CjQGRzHzwAtO_SXiHQ?e=AbUAiM)
* [Use Case application: Valeo - scene understanding](https://irtsystemx.sharepoint.com/:b:/s/IAdeConfiance833/EZKRyjRiobZLm58OoerTgTYB9o_PjyuPpVY7PXFb_v0_hg?e=cWNHdI)