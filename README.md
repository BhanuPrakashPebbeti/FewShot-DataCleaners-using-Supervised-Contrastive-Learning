# Supervised Contrastive Approach for Cleaning Dataset
One of the major hurdles in training a model is dealing with outliers that pulls the model away from the ideal optimums. To mitigate effects of such noises and outliers in the data used, we explore an idea of automating the dataset cleaning process, which can be a pre-processing step to yield better results. Inspired from recent development in DL techniques to form better feature spaces that are able to capture essential data features, we propose a supervised contrastive learning mechanism to clean the datasets.

# Model Architecture
Supervised contrastive learning was used to learn a good representations. It is a supervised approach for self supervised contrastive method SIMCLR.\
\
<img src="https://github.com/BhanuPrakashPebbeti/Supervised-Contrastive-Approach-for-Cleaning-Dataset/blob/main/Model%20Architecture/model%20arch.png" width="600" height="700">
# Augmentations
Augmentations is very key in Contrastive learning approach. Below are some augmentations used in our work.\
\
![](https://github.com/BhanuPrakashPebbeti/Supervised-Contrastive-Approach-for-Cleaning-Dataset/blob/main/Augmentations/Aug.png)
# Results
Here are some outliers detected by supervised contrastive approach.\
\
![](https://github.com/BhanuPrakashPebbeti/Supervised-Contrastive-Approach-for-Cleaning-Dataset/blob/main/Results/Supervised%20Contrastive%20Model%20%20Predicted%20Noise.png)
# Latent Space Visualization
We used Imagenet and some medical images as negative samples in our supervised contrastive based learning. Here is the visualization of latent space.\
\
![](https://github.com/BhanuPrakashPebbeti/Supervised-Contrastive-Approach-for-Cleaning-Dataset/blob/main/Latent%20Space%20Visualizations/Supervised%20Contrative%20Learned%20Latent%20space.png)
