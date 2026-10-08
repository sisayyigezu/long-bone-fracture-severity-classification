# Experimental Results

This directory contains the experimental results of the research project **Classification of Long Bone Fracture Severity Levels Using Deep Learning**.

## 1. Experimental Overview

The study investigated the classification of long bone X-ray images into three categories: Healthy, Simple Fracture, and Wedge/Complex Fracture.

Four deep learning architectures were evaluated:

- Custom Convolutional Neural Network (CNN)
- ResNet50
- MobileNetV2
- VGG19

The initial experiment evaluated all four models with a maximum training duration of 25 epochs. A separate experiment was conducted using the custom CNN with a maximum of 50 epochs and additional hyperparameter adjustments to investigate its classification performance.

## 2. Model Performance

The models were evaluated using accuracy, precision, recall, F1-score, and confusion matrices.

| Model | Epochs | Test Accuracy |
|---|---:|---:|
| MobileNetV2 | 25 | 99.81% |
| ResNet50 | 25 | 99.63% |
| VGG19 | 25 | 97.40% |
| Custom CNN | 25 | 80.89% |
| Custom CNN | 50 | 90.54% |

The complete numerical results are available in `metrics/model_comparison.csv`.

The custom CNN demonstrated an increase in test accuracy from 80.89% to approximately 90.54% under the extended training configuration. Since other hyperparameters were also adjusted, this difference cannot be attributed solely to the number of epochs.

MobileNetV2 achieved the highest reported test accuracy among the evaluated pretrained architectures.

## 3. Training and Evaluation Visualizations

The `figures/` directory contains:

- Custom CNN architecture diagram
- Custom CNN training accuracy and loss curves
- Custom CNN confusion matrix
- ResNet50 training curves
- MobileNetV2 training curves
- VGG19 training curves
- Model performance comparison chart

The custom CNN visualizations represent the final custom CNN experiment, while the pretrained model visualizations represent the original model comparison experiment.

## 4. Experimental Considerations

The final results combine experiments conducted under different training configurations. Therefore, they should be interpreted as a comparison of recorded experimental outcomes rather than an identical training-budget comparison.

The reported performance reflects the study dataset and does not establish generalization to independent clinical populations.

The original implementation notebooks are available in the repository's `notebooks/` directory.