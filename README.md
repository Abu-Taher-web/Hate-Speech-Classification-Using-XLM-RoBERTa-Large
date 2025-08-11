Packages:
transforme
dataset 
numpy
pandas
matplotlib
sklearn
os
collections
evaluate

# Dataset
## Before Filtering

## Dataset Overview
| **Attribute**            | **Description**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dataset Name**         | HateXplain                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **Source**               | [HateXplain – Hugging Face](https://huggingface.co/datasets/literAlbDev/hatexplain)                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **Original Paper**       | Mathew, Binny, et al. *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection*. Proceedings of the AAAI Conference on Artificial Intelligence, 35(17), 14867–14875, 2021. [PDF](https://ojs.aaai.org/index.php/AAAI/article/view/17745)                                                                                                                                                                                                                                                                                      |
| **Language**             | English                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **Dataset Type**         | `datasets.DatasetDict` (Hugging Face format)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **Splits**               | **Train**: 15,383 <br> **Validation**: 1,922 <br> **Test**: 1,924                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| **Features**             | `id` – Unique post identifier <br> `annotators` – Annotator labels and metadata <br> `rationales` – Token-level rationales from annotators <br> `post_tokens` – Tokenized post text                                                                                                                                                                                                                                                                                                                                                         |
| **Data Modality**        | Text (tokenized) + Annotator-provided explanations                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **Cache File Locations** | Train: `/taher/ds/literAlbDev___hatexplain/.../hatexplain-train.arrow` <br> Validation: `/taher/ds/literAlbDev___hatexplain/.../hatexplain-validation.arrow` <br> Test: `/taher/ds/literAlbDev___hatexplain/.../hatexplain-test.arrow`                                                                                                                                                                                                                                                                                                      |
| **License**              | CC BY 4.0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **Intended Use**         | Hate speech detection with explanation; model interpretability research                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **Example Record**       | **ID**: `23107796_gab` <br> **annotators**: `{"label": [0, 2, 2], "annotator_id": [203, 204, 233], "target": [["Hindu", "Islam"], ["Hindu", "Islam"], ["Hindu", "Islam", "Other"]]}` <br> **rationales**: Multiple lists indicating highlighted tokens <br> **post\_tokens**: `["u", "really", "think", "i", "would", "not", "have", "been", "raped", "by", "feral", "hindu", "or", "muslim", "back", "in", "india", "or", "bangladesh", "and", "a", "neo", "nazi", "would", "rape", "me", "as", "well", "just", "to", "see", "me", "cry"]` |


## Feature Summary
| **Feature**      | **Type**          | **Description**                                                                  | **Example Value**                                                                                                                        |
| ---------------- | ----------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **id**           | `string`          | Unique identifier for each post                                                  | `"23107796_gab"`                                                                                                                         |
| **annotators**   | `dict`            | Contains annotator labels, IDs, and target groups for hate speech classification | `{"label": [0, 2, 2], "annotator_id": [203, 204, 233], "target": [["Hindu", "Islam"], ["Hindu", "Islam"], ["Hindu", "Islam", "Other"]]}` |
| **rationales**   | `list[list[int]]` | Token-level binary masks (1 = token marked as rationale) provided by annotators  | `[[0, 0, 0, 0, 0, 0, 0, 0, 1, ...], [...]]`                                                                                              |
| **post\_tokens** | `list[string]`    | Tokenized post text                                                              | `["u", "really", "think", "i", "would", "not", "have", "been", "raped", "by", "feral", "hindu", "or", "muslim", ...]`                    |



## After Filtering
**Filtered HateXplain Dataset Statistics**
| **Split** | **Samples** | **Hatespeech (0)** | **Normal (1)** |
| --------- | ------------: | -------------------: | ---------------: |
| **Train** |        10,999 |                4,748 |            6,251 |
| **Valid** |         1,376 |                  594 |              782 |
| **Test**  |         1,374 |                  781 |              593 |

**🏷 Label Mapping**
| **Label ID** | **Meaning**                                     |
| ------------ | ----------------------------------------------- |
| **0**        | Hatespeech                                      |
| **1**        | Normal                                          |
| **2**        | Offensive *(excluded in this filtered dataset)* |


# Model
## Architecture






## Lyers dimension
### 📏 Notation

- **B** = Batch size  
- **L** = Sequence length (≤ **512**; XLM-R max positions ≈ **514**)  
- **H** = Hidden size = **1024**  
- **A** = Attention heads = **16** (head dimension = 1024 / 16 = **64**)  
- **I** = Intermediate size = **4096**  
- **C** = Number of classes = **2**


| Stage                        | Operation                            | Input → Output shape                                |
| ---------------------------- | ------------------------------------ | --------------------------------------------------- |
| Tokenization                 | text → `input_ids`, `attention_mask` | —                                                   |
| Embeddings                   | lookup + positional                  | `[B, L] → [B, L, 1024]`                             |
| Encoder blk (per layer, ×24) | LayerNorm                            | `[B, L, 1024] → [B, L, 1024]`                       |
|                              | Q/K/VP (linear)                      | `[B, L, 1024] → [B, L, 1024]` (for each of Q, K, V) |
|                              | Reshape for heads                    | `[B, L, 1024] → [B, 16, L, 64]`                     |
|                              | Scaled dot-prod attention            | Q,K,V: `[B,16,L,64] → [B,16,L,64]`                  |
|                              | Concat heads                         | `[B,16,L,64] → [B, L, 1024]`                        |
|                              | Output proj (linear)                 | `[B, L, 1024] → [B, L, 1024]`                       |
|                              | Dropout + Residual                   | `[B, L, 1024] → [B, L, 1024]`                       |
|                              | LayerNorm                            | `[B, L, 1024] → [B, L, 1024]`                       |
|                              | FFN (gelu)                           | `[B, L, 1024] → [B, L, 4096] → [B, L, 1024]`        |
|                              | Dropout + Residual                   | `[B, L, 1024] → [B, L, 1024]`                       |
| Final hidden                 | last layer output                    | `[B, L, 1024]`                                      |
| Pooling                      | take `<s>` token (index 0)           | `[B, L, 1024] → [B, 1024]`                          |
| Classifier                   | Dropout                              | `[B, 1024] → [B, 1024]`                             |
|                              | Linear                               | `[B, 1024] → [B, 2]`                                |
|                              | Softmax (inference)                  | `[B, 2] → [B, 2]`                                   |



### ⚙️ Config Quick Facts – xlm-roberta-large

- **Layers:** 24 encoder blocks  
- **Hidden size:** 1024  
- **Attention heads:** 16 (head dimension = 64)  
- **FFN intermediate size:** 4096  
- **Vocabulary size:** ~250k (SentencePiece)  
- **Max positions:** 514  
- **Parameters:** ~550M (backbone + classification head)



# Results
## Before Augmentation
<img src="Results Before Augmentation/loss_comparison.png" alt="loss_comparison.png" width="600"/>
The training and validation loss curves indicate that the model initially learns effectively, with both losses decreasing from epoch 1 to epoch 2. However, starting from epoch 3, the training loss continues to drop steadily, while the validation loss begins to rise. This divergence suggests overfitting, where the model is becoming increasingly specialized to the training data at the expense of generalization to unseen data. The lowest validation loss is observed around epoch 2, implying that this would have been an optimal early stopping point to maximize performance and avoid overfitting in later epochs.


<img src="Results Before Augmentation/accuracy_comparison.png" alt="loss_comparison.png" width="600"/>
The validation accuracy curve shows that the model maintains a consistently high performance across all epochs, staying around 88–89%. There is only a slight increase from epoch 1 to epoch 3, after which the accuracy stabilizes, with minimal fluctuations. This indicates that while additional training beyond epoch 3 does not significantly degrade validation accuracy, it also does not yield meaningful improvements. Combined with the validation loss trend, this suggests that the model reaches its optimal generalization capacity early in training, and further epochs mainly contribute to overfitting without boosting accuracy.


| **Class / Metric** | **Precision** | **Recall** | **F1-score** | **Support** |
| ------------------ | ------------: | ---------: | -----------: | ----------: |
| **Normal**         |        0.8571 |     0.8600 |       0.8586 |         593 |
| **Hate**           |        0.8935 |     0.8912 |       0.8923 |         781 |
| **Accuracy**       |        0.8777 |     0.8777 |       0.8777 |           — |
| **Macro Avg**      |        0.8753 |     0.8756 |       0.8754 |        1374 |
| **Weighted Avg**   |        0.8778 |     0.8777 |       0.8778 |        1374 |

The model achieves an overall accuracy of 87.77%, showing strong performance in distinguishing between Hate and Normal posts.
For the Normal class, precision (85.71%) and recall (86.00%) are balanced, indicating that the model correctly identifies most normal posts while keeping false positives relatively low.
The Hate class achieves slightly higher precision (89.35%) and recall (89.12%), meaning the model is slightly better at detecting hate speech compared to normal posts.
The F1-scores for both classes (Normal: 85.86%, Hate: 89.23%) confirm balanced performance without significant trade-offs between precision and recall.
Macro average values (~87.54%) show that both classes are treated relatively equally, while weighted averages (~87.78%) reflect the class distribution without a significant performance drop due to imbalance.


<img src="Results Before Augmentation/confusion_matrix.png" alt="loss_comparison.png" width="600"/>
The confusion matrix shows that the model correctly classified 510 Normal posts and 696 Hate posts, demonstrating strong predictive ability for both categories.
Misclassifications are relatively balanced, with 83 Normal posts incorrectly labeled as Hate and 85 Hate posts incorrectly labeled as Normal, indicating no severe bias toward one class.
The high number of correct predictions for Hate posts (true positives) and Normal posts (true negatives) aligns with the strong precision and recall scores in the classification report.
Overall, the confusion matrix confirms that the model performs reliably in detecting hate speech while maintaining accuracy in identifying normal content.

<img src="Results Before Augmentation/roc_curve.png" alt="loss_comparison.png" width="600"/>
The ROC curve demonstrates that the model maintains a high true positive rate while keeping the false positive rate relatively low across varying classification thresholds.
The curve rises steeply towards the top-left corner, indicating strong discriminatory power between the Hate and Normal classes.
An AUC score of 0.94 confirms excellent performance, meaning there is a 94% probability that the model will rank a randomly chosen hate speech example higher than a randomly chosen normal post.
This high AUC suggests that the model is robust to threshold changes and is effective at balancing sensitivity (recall) and specificity.


## After Augmentation


# Reference
[1] A. Conneau, K. Khandelwal, N. Goyal, V. Chaudhary, G. Wenzek, F. Guzmán, E. Grave, M. Ott, L. Zettlemoyer, and V. Stoyanov, “Unsupervised cross-lingual representation learning at scale,” arXiv preprint arXiv:1911.02116, 2019. [Online]. Available: http://arxiv.org/abs/1911.02116

[2] Mathew, Binny, et al. *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection*. Proceedings of the AAAI Conference on Artificial Intelligence, 35(17), 14867–14875, 2021. [PDF](https://ojs.aaai.org/index.php/AAAI/article/view/17745) 






