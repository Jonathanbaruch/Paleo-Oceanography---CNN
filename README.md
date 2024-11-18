# Paleo-Oceanography---CNN
Classifying single-cell organisms using a fine-tuned ResNet18. This model achieved an accuracy of 88.45% on the test set, which achieved the first place in a in-class Kaggle Competition for ICS 637 @University of Hawaii.


The following is a write-up on the project, considered as an exercise in writing an academic Machine Learning paper.

Paleo Oceanography - Convolutional Neural Network
1. Introduction
The goal of this project is to train a convolutional neural network capable of classifying single-cell organisms from oceanic ground samples. Such a model is valuable for marine ecology and biodiversity studies, where automated classification can save considerable time and resources compared to manual analysis. Automating this task can support research efforts to understand and monitor marine environments more effectively, contributing to better conservation and resource management.

The model takes RGB images of single-cell organisms captured under a microscope as input, and outputs a predicted class label for each organism, identifying its species or type.



2. Dataset Introduction
The dataset consists of labeled images of various single-cell organisms collected from ground samples in the ocean. Each image represents an individual organism, with labels corresponding to the organism’s classification. The dataset was sourced from kahanamoku et al. from University of California at Berkeley, and contains a total of 10.827 images across 54 classes, with the class 0 representing non-foram images. Each image has previously been cropped and resized to 224x224 pixels so that the organism is largely centered.

Given the diversity of single-cell organisms and the subtle differences between classes, the dataset presents a challenging classification problem, making it well-suited for a deep learning approach.

3. Model Selection
For this project, the ResNet-18 architecture was chosen, which is a popular convolutional neural network architecture known for its effectiveness on image classification tasks. ResNet-18 is a 18-layer deep neural network with residual connections that allow for efficient training, even in deeper architectures. These residual connections help mitigate the vanishing gradient problem, thus allowing for a deeper network, capturing more complex features and relationships. This proved more efficient than a more simple and shallow Convolutional Neural Network.

The objective function optimized during training was cross-entropy loss, which is well-suited for multi-class classification tasks. Cross-entropy loss measures the divergence between the predicted and actual class distributions, helping the model learn to output accurate probabilities for each class.

4. Feature Engineering and Preprocessing
The input features for this model are RGB images, resized to 224x224 pixels, which matches ResNet-18’s input size requirements. The images are unit-less, with pixel intensities normalized to a range centered around zero.

Data Transformations:
Normalization: Each image was normalized using a mean of [0.5, 0.5, 0.5] and standard deviation of [0.225, 0.225, 0.225]. This matches the normalization scheme used in pretraining of ResNet models on the ImageNet dataset.

Data Augmentation:
Augmenting the training data for this project resulted in a rather large increase in the accuracy of the final model on both validation and test data. To improve the model’s ability to generalize and handle variability in the images, the following augmentations were applied, using the ‘Torchvision transforms’ library: Color Jitter, Random Rotations (45 degrees), Random Horizontal and Vertical Flips, Random Resized Crop (scale 0.8), and finally Gaussian Blur and Gaussian Noise. Different variations and amounts of augmentation tuning were done to get the best possible model, landing on the values seen in the project code.

5. Data Splits and Usage
The total dataset contained 10.827 examples. 2174 of these were pre-chosen as a test set. Of the remaining 8563 pictures, 90% were chosen for training (to maximize the amount of data for training), and 10% for validation. The model was then trained on the training set, with validation loss used for saving the best model and hyperparameter optimization. The best model was then evaluated on the test set, to choose a final model.


6. Hyperparameter Search Space
The following hyperparameters were tuned to optimize the model’s performance:

Learning Rate: After experimenting with different ones, the best proved to be 0.0001. A scheduler (StepLR) was used to reduce the learning rate by a factor of 0.1 every 5 epochs, allowing finer adjustments during later training.

Layers Unfrozen: Initially, all layers were kept frozen. However, since ResNet-18 was pretrained on the ImageNet dataset, the last 10 layers were progressively unfrozen, to allow the architecture to learn better from this specific dataset, while retaining the general-purpose features learned during pretraining. Initially, 3 layers were unfrozen, before moving on to 10.

Epochs: Between 20-40 epochs were done, with a built-in function that saved the model state, if validation loss decreased. The final best model was achieved after epoch 19. This also serves as the only explicit form of regularization in this project, as there is no dropout, or L1/L2-Regularization, to stop it from overfitting.

Approximately 10 different model configurations were explored, mostly adjusting the learning rate and the number of layers unfrozen. Based on the validation loss, the configuration with the best performance was selected for final evaluation.

7. Hyperparameter Optimization Process
Hyperparameters were tuned manually, with each configuration evaluated based on validation accuracy and validation loss. After each training run, the model’s performance was recorded, and adjustments were made to either the learning rate or the number of layers unfrozen. The learning rate scheduler was employed to reduce the learning rate every 5 epochs, helping the model converge more effectively by slowing down updates as it approached an optimal solution.

The final model chosen was the model with the lowest validation loss (Cross-entropy), which was then used to evaluate on the test set. Another option could have been to choose the model with the highest accuracy instead, as some epochs had a higher validation loss, but also higher accuracy, than the final model with the lowest validation loss. As the test set is bigger than the validation set, the model with the lowest loss was selected instead of the one with highest accuracy, as this model in the long run should outperform by being a better overall classifier.

8. Model Evaluation
The final model achieved an accuracy of 88.45% on the Test Set. This model had a Validation Loss of 0.3100. The model was trained with the following hyperparameters:

Learning rate: 0.001 (with scheduler of step size 5, gamma of 0.1)
Batch size: 32
Epochs: 19
Optimizer: Adam
Loss function: Cross-Entropy
Unfrozen layers: Last 10/18

It should be noted that the test- and validation data underwent the same transformation as the training data for the final and most accurate model, meaning that they underwent data augmentation before using the model to infer the class. Leaving out the data augmentation techniques when predicting test data thus resulted in a lower accuracy.


9. Datasets
The test-set consists of data similar to the training and validation set, even though the labels were never in the possession of the author.


10. Conclusion
The ResNet-18 model fine-tuned for this project effectively classifies single-cell organisms from microscopy images, achieving an accuracy of 88.45% on the test set. The model’s accuracy and robustness suggest that it could be a valuable tool for ecological research, supporting the classification and identification of marine organisms at scale. This classification model is useful for researchers and practitioners in the field, enabling high-throughput analysis of microscopic samples and potentially aiding in conservation efforts through more accurate ecosystem monitoring.

Future improvements could involve testing deeper architectures (e.g., ResNet-50 or ResNet-101), different hyperparameter choices, and experimenting with other regularization techniques to further improve the model’s robustness and accuracy.
