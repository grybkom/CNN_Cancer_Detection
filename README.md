# CNN for Cancer Detection
Binary image classification of metastatic cancer in tissue samples using a convolutional neural network (CNN).
---

## Background

The aim of this work is to build a CNN that performs binary image classification of metastatic cancer in tissue samples. According to the National Cancer Institute (2024) in the United States alone it is estimated that over 2 million new cases of cancer will be diagnosed and over 600,000 people will die from cancer. The World Health Organization (2023) emphasizes the importance of early cancer diagnosis in fighting the disease. Early detection increases the success rate of treatments, deceases the severity of side effects from treatment, and lowers cost of care (World Health Organization, 2023). Given the prevalence of cancer and the importance of early detection in its treatment, a model that can detect cancer in tissue images could help improve the outcome of those suffering from the disease. 

---

## Data & Methodology

- The data used for this work is available at Kaggle, Histopathologic Cancer Detection:
Will Cukierski. (2018). Histopathologic Cancer Detection. Kaggle. https://kaggle.com/competitions/histopathologic-cancer-detection
- This data set contains images of metastatic cancer. The National Cancer Institute (2024) defines metastasis as the spread of cancer cells from the place where they first formed to other parts of the body. 
- This is a binary classification task with a positive label indicating the center 32x32px region of an image contains at least one pixel of tumor tissue. 
- There are just over 220,000 training images of size 96x96x3, with approximately 60% of the images being of cancer free tissue and approximately 40% of the images containing cancerous tissue. 

![training_data_distribution](https://github.com/user-attachments/assets/aa273beb-0c52-4cb7-8cb4-5aa75175d5cd)

### Languge: Python
  - [TensorFlow](https://www.tensorflow.org/)
 
### Hardware: 
  - Dual NVIDIA T4 GPUs T4 x2 accelerator (ran on Kaggle's platform)

### CNN Architecture

- **Convolutional Layers:** Four convolutional blocks with increasing depth: 64 → 128 → 256 → 512 filters.

- **Spatial Attention:**
  - Two spatial attention blocks early in the network to help the model focus on informative regions in the image.
  - Two spatial attention blocks applied again after deep feature extraction to refine focus.
  - Uses varying kernel sizes of 9 and 3 to capture both global and local spatial dependencies.
  
- **Regularization and Normalization:**
  - Dropout after each block (progressively increasing)
  - BatchNormalization throughout the network
  - L2 kernel regularization to reduce overfitting
  
- **Pooling:**
  - MaxPooling2D after each convolutional block to reduce spatial dimensions
  - GlobalAveragePooling2D before fully connected layer
  
- **Fully Connected Layer:**
  - Dense layer with 256 units, followed by Dropout and BatchNormalization
  - Final output layer: 1 unit with sigmoid activation for binary classification

### Training Strategy

- **Optimizer:** Adam with a learning rate initialized at 0.0002

- **Loss Function:** Binary Cross-Entropy 

- **Learning Rate Scheduler:** A scheduler was used ([ReduceLROnPlateau]( https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)) to reduce the learning rate during training to help with model stability.

## Results
![cnn_training_results](https://github.com/user-attachments/assets/5f59d829-49f4-4421-93d9-683e590b05c9)
![cnn_training_confusion](https://github.com/user-attachments/assets/74ad841f-39ca-4e12-89ff-432c9bbd5e6b)


## To Do

- Trian the model with a larger batch size, increase from 32 to 64. The model's performance imporoved when the batch size was increased from 16 to 32. 

- Assess the model’s performance with different kernel sizes in the spatial attention layers.

- The current notebook includes a gaussian smoothing block, which will slightly blur the images while the edges are maintained. The current versions do not include this step, it is commented out in the data generator. The network has had trouble learning when smoothing has been applied to the images. However, this has not been attempted with the most recent versions of the network. 

## References

Cancer statistics. (2024, May 9). National Cancer Institute. https://www.cancer.gov/about-cancer/understanding/statistics

Promoting cancer early diagnosis. (2023, October 27). World Health Organization. https://www.who.int/activities/promoting-cancer-early-diagnosis

Verma, M. (2024, May 17). Binary classification using Convolution Neural Network (CNN) model. Medium. https://medium.com/@mayankverma05032001/binary-classification-using-convolution-neural-network-cnn-model-6e35cdf5bdbb


## Author

Michael Grybko - GitHub username: grybkom
