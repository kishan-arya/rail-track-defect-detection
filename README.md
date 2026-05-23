# Rail Track Defect Detection using FCCN

Binary classification of rail track images (normal vs defective) using
feature extraction from a pretrained Fully Complex-valued Convolutional
Network (FCCN, ICCV 2023).

## Approach
Extract features using frozen FCCN backbone, classify with SVM, KNN, Random Forest, Logistic Regression

## Results

| Backbone            | Representation | Classifier | Accuracy | Defective Recall |
|---------------------|---------------|------------|----------|-----------------|
| FCCN ResNet18       | Magnitude     | LR         | 83.8%    | 85.5%           |
| FCCN ResNet152      | Real+Imag     | LR         | **95.5%**| **94.5%**       |
| Standard ResNet152  | Magnitude     | RF         | 92.8%    | 89.1%           |

FCCN's complex-valued iHSV colour encoding outperforms standard ResNet152
by +2.7% accuracy and +5.4% defective recall.

### Task 2 — Fine-tuning FCCN ResNet152
| Val Accuracy | Test Accuracy |
|-------------|---------------|
| 97.8%       | 93.7%         |

**Key finding:** Frozen feature extraction (95.5%) outperformed 
fine-tuning (93.7%) on this small dataset (444 training images).

## Dataset
Private dataset — 555 rail track images (444 train, 111 test), balanced binary classes.

## How to Run
1. Open notebook in Google Colab with GPU (T4)
2. Clone FCCN repo: `git clone https://github.com/saurabhya/FCCNs.git`
3. Download pretrained checkpoint from [saurabhya/FCCNs](https://github.com/saurabhya/FCCNs)
4. Update `CKPT_PATH` in notebook to your checkpoint location
5. Run all cells top to bottom

## Reference
Yadav et al., *FCCNs: Fully Complex-valued Convolutional Networks
using Complex-valued Color Model and Loss Function*, ICCV 2023.
