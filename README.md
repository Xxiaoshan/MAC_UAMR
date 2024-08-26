# MAC_UAMR
We have released the "Multi-Representation Domain Attentive Contrastive Learning Based Unsupervised Automatic Modulation Recognition (MAC)" and our codebase to the community. In this work, we propose an unsupervised automatic modulation recognition (UAMR) method based on multi-domain contrastive learning. Existing deep learning-based AMR methods suffer from significant performance degradation when there is a lack of labeled samples. We evaluated the proposed method on three public datasets and across different sample sizes. By leveraging contrastive learning and multi-domain representation, MAC achieves representation learning of modulation signals using unlabeled signals. After unsupervised training, the encoder can achieve surprisingly effective semi-supervised modulation recognition when fine-tuned with a small amount of labeled samples.

# Motivation

The existing data augmentation schemes for AMR primarily draw from those developed in computer vision or audio sequence fields. such as *flipping*, *rotating*, *adding Gaussian noise*. However, these methods do not consider data augmentation from the perspective of signal modulation characteristics, just like processing an image.

<img src="https://github.com/user-attachments/assets/d0c2cadd-dbcd-4f9f-8d28-e963b5f888d1" width="70%">

MAC inter-domain contrastive learning engages different representational domains that observe signal modulation characteristics in a high-dimensional space. This is analogous to viewing an object from multiple perspectives—frontal, lateral, and overhead—which provides a more comprehensive set of features (information) and makes the modulated signals recognizable.

<img src="https://github.com/user-attachments/assets/0ad78381-d45c-476c-a0d7-8f6a8a1b9442" width="70%">

# Benchmark Datasets

|Dataset|Modulation type|Signal sample size|Signal length|SNR|
|:---|:---|:---|:---|:---|
|RML2016.10A|11classes|220000|2\*128|-20:2:20|
|列1的内容2|列2的内容2|

# requirements

main.py is the main function of training for the entire unsupervised inter-domain contrastive learning.

LinearProbing.py is the main function of the linear evaluation stage.

sp_AMR.py is training function for various supervised comparison methods.

read2016_v1.py is the dataset reading code.
