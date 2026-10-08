# Dataset Information

The research dataset consists of 3,500 long bone X-ray images organized into three classification categories.

## Dataset Distribution

| Class | Images |
|---|---:|
| Healthy | 1,116 |
| Simple Fracture | 1,211 |
| Wedge/Complex Fracture | 1,173 |
| **Total** | **3,500** |

## Dataset Structure

```text
all_original/
├── Healthy/
├── Simple/
└── Wedge_Complex/
```

## Preprocessing

Images were resized to 224 × 224 pixels, converted to tensors, and normalized using ImageNet mean and standard deviation values.

Data augmentation was applied to the training set only and included random resized cropping, horizontal flipping, rotation, and brightness/contrast adjustments.

The dataset was divided into training (70%), validation (15%), and testing (15%) subsets using a fixed random seed of 42.

## Data Availability and Reproducibility

The dataset used in this study consists of long bone X-ray images collected from local hospitals in Ethiopia. Due to data privacy considerations, the full dataset is not publicly distributed through this repository. The repository provides the model implementation, experimental configurations, and evaluation results to support reproducibility.

