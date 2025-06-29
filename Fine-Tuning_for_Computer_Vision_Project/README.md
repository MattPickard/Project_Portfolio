# Adapting Neural Networks: Fine-Tuning for Computer Vision

<p align="center">
<img src="https://awaywithideas.com/assets/images/2020/10/mnist_extended_4_0.png" style="width: 40%;">
</p>

## Table of Contents
<table>
<tr>
<td>
<a href="#introduction">Introduction</a><br>
<a href="#cnn-training">Base Model CNN Training</a><br>
<a href="#experience-replay-fine-tuning">Experience Replay Fine-tuning</a><br>
<a href="#sequential-fine-tuning">Sequential Fine-tuning</a><br>
<a href="#lora-fine-tuning">LoRA Fine-tuning</a><br>
<a href="#conclusion">Conclusion</a>
</td>
</tr>
</table>

## Introduction
<a name="introduction"></a>
In this project, I explored various fine-tuning techniques for adapting a neural network to new data. Using the MNIST dataset of handwritten digits, I first trained a convolutional neural network (CNN) to recognize digits 1-9, then fine-tuned it to recognize the digit 0 using three different approaches: experience replay, sequential fine-tuning, and Low-Rank Adaptation (LoRA).

The goal was to simulate a common challenge in machine learning: how to adapt a model to new classes or data distributions without losing performance on previously learned tasks. I was especially curious about the effects of fine-tuning when using exclusively new task data, which I simulated in the sequential fine-tuning and LoRA experiments.

These experiments simulated real-world deep learning applications where:
- The full original training data may no longer be available
- Retraining from scratch and naive fine-tuning are too computationally expensive
- New classes or data distributions emerge over time

The MNIST dataset used in this project is a widely used dataset consisting of 70,000 grayscale images of handwritten digits (0-9), each 28x28 pixels. By treating digit 0 as a "new class" that the model learns separately after being trained on digits 1-9, I was able to evaluate different fine-tuning strategies and their performance on both the new and previously learned classes.

<p align="center">
<img src="https://github.com/MattPickard/Project_Portfolio/blob/main/Images/fine-tuning_comparison.png?raw=true" style="width: 50%;">
</p>

## Preprocessing

The data was reshaped to be (28, 28, 1) for input into the neural network. This represents the 28x28 pixel images of 1 channel for grayscale images (RGB has 3 channels). The values were then normalized to be between 0 and 1 by dividing by the maximum pixel value of 255:
```python
# Reshape data to be 28x28x1 and normalize pixel values
train_images = np.array(train_images).reshape((-1, 28, 28, 1)) / 255.0
train_labels = np.array(train_labels)
test_images = np.array(test_images).reshape((-1, 28, 28, 1)) / 255.0
test_labels = np.array(test_labels)
```

These transformations must be made for any data used as input into the model or subsequent fine-tuned models.

## Base Model CNN Training
<a name="cnn-training"></a>
**Code:** [**Base Model CNN Training**](https://github.com/MattPickard/Project_Portfolio/blob/main/Fine-Tuning_for_Computer_Vision_Project/cnn_training.ipynb)

For the base model, a CNN was trained exclusively on digits 1-9 from the MNIST dataset, excluding digit 0. This model was then treated as the "pre-trained" model for all subsequent fine-tuning experiments.

The architecture of the CNN model consisted of:
- Three convolutional layers with 32, 64, and 64 filters, respectively
- Two max pooling layers for dimensionality reduction
- Dropout layers (10% dropout rate) for regularization
- A flatten layer
- Two fully connected layers with 128 neurons each
- A final output layer with 10 neurons in anticipation of future fine-tuning on digit 0, although this was not necessary

The model was trained using the Adam optimizer with a learning rate of 0.0003 and sparse categorical cross-entropy loss. Early stopping was implemented to prevent overfitting, monitoring validation loss with a patience of 5 epochs.

After training, the model achieved an accuracy of 99.25% on the test set containing only digits 1-9, establishing a strong baseline for the fine-tuning experiments.

## Experience Replay Fine-tuning
<a name="experience-replay-fine-tuning"></a>
**Code:** [**Replay Fine-tuning**](https://github.com/MattPickard/Project_Portfolio/blob/main/Fine-Tuning_for_Computer_Vision_Project/replay_fine-tuning.ipynb)  

Experience replay is a technique in which a model is fine-tuned using both new data and a subset of the original training data. This approach helps prevent catastrophic forgetting, where a model loses performance on previously learned tasks when adapting to new ones.

For this experiment, I simulated experience replay by fine-tuning the base model on the full MNIST dataset, including both the previously trained digits 1-9 samples and the "new" digit 0. This represents an ideal scenario where historical training data remains available. To account for the potential computational expense of fine-tuning in real-world applications, I froze training on all but the last 2 dense layers and the output layer, reducing the computational cost. By freezing training on early layers, it is assumed that the early convolutional layers have effectively learned features that can be used to classify the new digit 0. However, this assumption may not hold true in all situations, particularly if the new task is significantly different from the previous tasks the model has learned.

### **Results:**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Overall test accuracy:&nbsp;&nbsp;  **99.31%**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Accuracy for digit 0:&nbsp;&nbsp;&nbsp;&nbsp;  **99.69%**  

Experience replay proved to be highly effective at mitigating catastrophic forgetting. It preserved model accuracy of the original 1-9 digits while achieving near-perfect accuracy on the new digit 0. This approach is ideal when the original or previous training data is still available. The next two approaches will simulate scenarios where the original training data is no longer available.

## Sequential Fine-tuning
<a name="sequential-fine-tuning"></a>
**Code:** [**Sequential Fine-tuning**](https://github.com/MattPickard/Project_Portfolio/blob/main/Fine-Tuning_for_Computer_Vision_Project/sequential_fine-tuning.ipynb)  

Sequential fine-tuning represents a more challenging scenario where only new task data (digit 0) is available for training. This may be used in situations where the original training data is no longer accessible. Sequential fine-tuning is highly susceptible to catastrophic forgetting, so it presents a delicate balance between maximizing performance on the new task and preserving performance on the old tasks. 

Similar to the experience replay experiment, I froze all but the last 2 dense layers and the output layer. Then hyperparameter optimization was performed using an Optuna study to find the optimal learning rate and number of epochs. It's worth noting that this introduces slight data leakage, as the number of epochs and learning rate were optimized while maximizing the test set accuracy. In a real-world scenario, a separate validation set should be used, and early stopping can be implemented using the validation set.

### **Results:**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Overall test accuracy:&nbsp;&nbsp;  **98.22%**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Accuracy for digit 0:&nbsp;&nbsp;&nbsp;&nbsp;  **97.86%**  

The decrease in overall accuracy from 99.31% in the experience replay experiment to 98.22% in the sequential fine-tuning experiment suggests that the model experienced catastrophic forgetting as a result of only training on the new digit 0. While this approach doesn't achieve the same level of performance as experience replay, this experiment reveals that sequential fine-tuning can be a valuable technique when original or comprehensive training data is unavailable and quick adaptation to new classes is needed.

## LoRA Fine-tuning
<a name="lora-fine-tuning"></a>
**Code:** [**LoRA Fine-tuning**](https://github.com/MattPickard/Project_Portfolio/blob/main/Fine-Tuning_for_Computer_Vision_Project/lora_fine-tuning.ipynb)  

Low-Rank Adaptation (LoRA) is a fine-tuning technique that introduces small, trainable low-rank matrices into the output of the original model's layers. This approach significantly reduces the number of trainable parameters compared to other fine-tuning methods while still allowing the model to adapt to new data. For example, by using LoRA to fine-tune the last two dense layers of this model, the number of trainable parameters compared to the other two experiments was reduced from 90,240 to 2,052, around a ~45x reduction in trainable parameters. 

For this implementation, I created a custom TensorFlow layer that applies a LoRA update to the weights of a frozen layer. Specifically, it adds the product of two low-rank matrices (A and B) to the frozen layer's weight parameters, and disables the original bias parameters.

The matrices Lora_A and Lora_B have shapes (d×r) and (r×k) where:
- d is the input dimension,
- k is the output dimension, and
- r is the LoRA rank (set to 4)

The forward pass computes the output as:

    y=Wx+αABx

Where:
- W represents the frozen base weights,
- x is the input,
- AB is the low-rank update, and
- α is a strength scaling factor, explained in the section "Adjustable LoRA Strength" below

Similar to the sequential fine-tuning experiment, I restricted LoRA fine-tuning to training only on data from the digit 0. The goal was to see the effects of catastrophic forgetting in a LoRA-based model, where the underlying pre-trained parameters remain unchanged.

As in the previous experiment, I performed hyperparameter tuning using an Optuna study to optimize the number of epochs and learning rate, maximizing test set accuracy. In a real-world scenario, a separate validation set should be used for tuning, and early stopping can be applied based on validation performance.

### Adjustable LoRA Strength

A unique characteristic of LoRA models is that a strength adjuster can be implemented to allow for post-training tuning, meaning it can be adjusted while making predictions. In a real-world scenario, this provides the user the ability to tune the impact LoRA has over the predictions, which can be useful in situations where either false positives or false negatives for the new task are more costly than the other. I implemented the ability to adjust the strength (the alpha value in the LoRA Dense layers) and plotted the effect of changing the LoRA strength on accuracy:

<p align="center">
<img src="https://github.com/MattPickard/Project_Portfolio/blob/main/Images/LoRa_Strength.png?raw=true" style="width: 50%;">
</p>

### **Results:**   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Overall test accuracy:&nbsp;&nbsp; **97.74%**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Accuracy for digit 0:&nbsp;&nbsp;&nbsp;&nbsp;  **96.73%**  

Considering this LoRA implementation was trained using only 0 digit data and utilized significantly fewer trainable parameters, it's not surprising that the performance is lower than the other two experiments. It shows that LoRA fine-tuning can be a valuable option when training computational resources are limited. It also allows for efficient storage if multiple specialized versions of a model are needed for different tasks by simply swapping the small LoRA weights, as opposed to storing a full separate model for each task. Finally, the ability to adjust the LoRA strength factor provides the user a unique ability to balance performance on the new and existing classes.

## Conclusion
<a name="conclusion"></a>

<p align="center">
<img src="https://github.com/MattPickard/Project_Portfolio/blob/main/Images/fine-tuning_comparison.png?raw=true" style="width: 50%;">
</p>

This project demonstrated three different approaches to fine-tuning a pre-trained neural network for a new class, each with its own strengths and trade-offs:
| Method | Overall Accuracy | Digit 0 Accuracy | Trainable Parameters | Used Historical Data |
|--------|------------------|------------------|----------------------|------------------------|
| Experience Replay | 99.31% | 99.69% | 90,240 | Yes |
| Sequential Fine-tuning | 98.22% | 97.86% | 90,240 | No |
| LoRA Fine-Tuning | 97.74% | 96.73% | 2,052 | No |

**Key Takeaways:**

- **Experience Replay** provides the best performance by mitigating catastrophic forgetting but requires access to original training data.
- **Sequential Fine-tuning**, which can be used when old data is unavailable, leads to a noticeable drop in overall accuracy, demonstrating catastrophic forgetting.
- **LoRA Fine-tuning** offers significant parameter efficiency, drastically reducing the number of trainable weights, and demonstrating the ability to be fine-tuned using only the new class data. Its primary advantage lies in reduced computational cost and storage for a model with many fine-tuned variants. Additionally, the ability to adjust the LoRA strength factor provides the user with a unique ability to balance performance on the new and existing classes.

These techniques have innovative broad-reaching applications beyond digit recognition, including:
- Extending natural language models to new domains or specific tasks
- Updating recommendation systems to accommodate new product categories
- Enhancing computer vision and medical imaging systems to recognize new objects
- Adapting neural networks to new data distributions, classes, and tasks

**Use of Fine-Tuning:**
Fine-tuning is a useful technique when training large models from scratch is both computationally expensive and time-consuming. For many real-world applications, it is more practical to fine-tune a pre-trained model on a specific task rather than training a new model from scratch. For many domains, there are thriving machine learning communities that open-source or open-weight pre-trained models, further increasing the importance of fine-tuning techniques.

In conclusion, the findings highlight the critical role of data quality and diversity in the fine-tuning process. It is essential to recognize that the most effective approach is context-dependent, requiring a balance between achieving high accuracy and addressing constraints such as data availability and computational efficiency.
