# **Paleo Oceanography - ResNet 18**

This Repository contains a fine-tuned ResNet18 model, trained to classify Kahanamoku et. al (2023)'s dataset on single-celled organisms, for the course ICS 637 @University of Hawaii.

The following is from a write-up on the project, considered as an exercise in writing an academic description of a Machine Learning project.


# **Paleo Oceanography - Convolutional Neural Network**

## **1. Introduction**
The goal of this project is to train a convolutional neural network capable of classifying single-cell organisms from oceanic ground samples. Such a model is valuable for marine ecology and biodiversity studies, where automated classification can save considerable time and resources compared to manual analysis. Automating this task can support research efforts to understand and monitor marine environments more effectively, contributing to better conservation and resource management.

The model takes RGB images of single-cell organisms captured under a microscope as input, and outputs a predicted class label for each organism, identifying its species or type.

---

## **2. Dataset Introduction**
The dataset consists of labeled images of various single-cell organisms collected from ground samples in the ocean. Each image represents an individual organism, with labels corresponding to the organism’s classification. The dataset was sourced from **Kahanamoku et al. from the University of California at Berkeley**, and contains a total of **10,827 images across 54 classes**, with the class 0 representing non-foram images. Each image has previously been cropped and resized to **224x224 pixels**, ensuring the organism is largely centered.

Given the diversity of single-cell organisms and the subtle differences between classes, the dataset presents a challenging classification problem, making it well-suited for a deep learning approach.

---

## **3. Model Selection**
For this project, the **ResNet-18 architecture** was chosen, which is a popular convolutional neural network architecture known for its effectiveness on image classification tasks. ResNet-18 is an 18-layer deep neural network with **residual connections** that allow for efficient training, even in deeper architectures. These residual connections help mitigate the **vanishing gradient problem**, allowing for a deeper network that captures more complex features and relationships. This proved more efficient than a simpler and shallower Convolutional Neural Network.

The objective function optimized during training was **cross-entropy loss**, which is well-suited for multi-class classification tasks. Cross-entropy loss measures the divergence between the predicted and actual class distributions, helping the model learn to output accurate probabilities for each class.

---

## **4. Feature Engineering and Preprocessing**
### **Input Features**
- RGB images resized to **224x224 pixels**, matching ResNet-18’s input size requirements.
- The images are unit-less, with pixel intensities normalized to a range centered around zero.

### **Data Transformations**
- **Normalization**: Each image was normalized using a mean of `[0.5, 0.5, 0.5]` and a standard deviation of `[0.225, 0.225, 0.225]`. This matches the normalization scheme used in ResNet models pretrained on the ImageNet dataset.

- **Data Augmentation**: 
  - Augmenting the training data for this project resulted in a significant increase in accuracy on both validation and test data. The following augmentations were applied using the `Torchvision.transforms` library:
    - **Color Jitter**
    - **Random Rotations (45°)**
    - **Random Horizontal and Vertical Flips**
    - **Random Resized Crops (scale = 0.8)**
    - **Gaussian Blur and Gaussian Noise**
  - Different variations and amounts of augmentation tuning were done to achieve the best possible model.

---

## **5. Data Splits and Usage**
The total dataset contained **10,827 examples**:
- **Test Set**: 2,174 pre-chosen images.
- **Training Set**: 90% of the remaining 8,653 images.
- **Validation Set**: 10% of the remaining 8,653 images.

The model was trained on the training set, with validation loss used for saving the best model and hyperparameter optimization. The best model was then evaluated on the test set to select the final model.

---

## **6. Hyperparameter Search Space**
The following hyperparameters were tuned to optimize the model’s performance:
- **Learning Rate**: After experimenting with different values, the best proved to be `0.0001`. A scheduler (**StepLR**) was used to reduce the learning rate by a factor of 0.1 every 5 epochs.
- **Layers Unfrozen**: Initially, all layers were kept frozen. The last **10 layers** were progressively unfrozen to allow the architecture to learn better from this specific dataset while retaining the general-purpose features learned during pretraining. Initially, 3 layers were unfrozen, before moving on to 10.
- **Epochs**: Between 20-40 epochs were tested, with a built-in function that saved the model state if validation loss decreased. The final best model was achieved after epoch 19.

Approximately 10 different model configurations were explored, mostly adjusting the learning rate and the number of layers unfrozen. Based on validation loss, the configuration with the best performance was selected for final evaluation.

---

## **7. Hyperparameter Optimization Process**
Hyperparameters were tuned manually, with each configuration evaluated based on validation accuracy and validation loss. After each training run, the model’s performance was recorded, and adjustments were made to either the learning rate or the number of layers unfrozen. The learning rate scheduler was employed to reduce the learning rate every 5 epochs, helping the model converge more effectively by slowing down updates as it approached an optimal solution.

The final model chosen was the model with the **lowest validation loss** (cross-entropy), which was then used to evaluate on the test set. Although some epochs had higher validation accuracy, the model with the lowest validation loss was chosen, as it is more likely to generalize better in the long run.

---

## **8. Model Evaluation**
The final model achieved:
- **Test Accuracy**: **88.45%**
- **Validation Loss**: **0.3100**

### **Training Hyperparameters**
| **Hyperparameter**       | **Value**                  |
|---------------------------|----------------------------|
| Learning Rate             | `0.0001` (with StepLR)    |
| Batch Size                | `32`                      |
| Epochs                    | `19`                      |
| Optimizer                 | Adam                      |
| Loss Function             | Cross-Entropy             |
| Unfrozen Layers           | Last 10/18                |

---

## **9. Datasets**
The test set consists of data similar to the training and validation sets, though the labels were never available to the author during training.

---

## **10. Conclusion**
The ResNet-18 model fine-tuned for this project effectively classifies single-cell organisms from microscopy images, achieving an accuracy of **88.45%** on the test set. The model’s accuracy and robustness suggest that it could be a valuable tool for ecological research, supporting the classification and identification of marine organisms at scale. This classification model is useful for researchers and practitioners in the field, enabling high-throughput analysis of microscopic samples and potentially aiding in conservation efforts through more accurate ecosystem monitoring.

### **Future Improvements**
- Testing deeper architectures (e.g., ResNet-50 or ResNet-101).
- Exploring different hyperparameter combinations.
- Experimenting with other regularization techniques to further improve robustness and accuracy.
