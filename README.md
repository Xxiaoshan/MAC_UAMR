# MAC_UAMR
We have released the "Multi-Representation Domain Attentive Contrastive Learning Based Unsupervised Automatic Modulation Recognition (MAC)" and our codebase to the community. In this work, we propose an unsupervised automatic modulation recognition (UAMR) method based on multi-domain contrastive learning. Existing deep learning-based AMR methods suffer from significant performance degradation when there is a lack of labeled samples. We evaluated the proposed method on three public datasets and across different sample sizes. By leveraging contrastive learning and multi-domain representation, MAC achieves representation learning of modulation signals using unlabeled signals. After unsupervised training, the encoder can achieve surprisingly effective semi-supervised modulation recognition when fine-tuned with a small amount of labeled samples.

# Motivation

The existing data augmentation schemes for AMR primarily draw from those developed in computer vision or audio sequence fields. such as *flipping*, *rotating*, *adding Gaussian noise*. However, these methods do not consider data augmentation from the perspective of signal modulation characteristics, just like processing an image.
<div align=center>
<img src="https://github.com/user-attachments/assets/d0c2cadd-dbcd-4f9f-8d28-e963b5f888d1" width="70%">
</div>
MAC inter-domain contrastive learning engages different representational domains that observe signal modulation characteristics in a high-dimensional space. This is analogous to viewing an object from multiple perspectives—frontal, lateral, and overhead—which provides a more comprehensive set of features (information) and makes the modulated signals recognizable.
<div align=center>
<img src="https://github.com/user-attachments/assets/0ad78381-d45c-476c-a0d7-8f6a8a1b9442" width="70%">
</div>

# requirements
**Window+Pytorch**
```
pip install -r requirements.txt
```


# Benchmark Datasets

|Dataset|Modulation type|Signal sample size|Signal length|SNR|
|:---|:---|:---|:---|:---|
|RML2016.10A|11classes|220000|2\*128|-20:2:20|
|RML2016.10B|10classes|1200000|2\*128|-20:2:20|
|RML2018.01A|24classes|2555904|2\*1024|-20:2:30|

***RML2016.10A*: 8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAW4, QAM16, QAM64, QPSK, and WBFM.**

***RML2016.10B*: 8PSK, AM-SSB, BPSK, CPFSK, GFSK, PAW4, QAM16, QAM64, QPSK, and WBFM.**

***RML2018.01A*: 32PSK, 16APSK, 32QAM, FM, GMSK, 32APSK, OQPSK, 8ASK, BPSK, 8PSK, AM-SSB-SC, 4ASK, 16PSK, 64APSK, 128QAM, 128APSK, AM-DSB-SC, AM-SSB-WC, 64QAM, QPSK, 256QAM, AM-DSB-WC, OOK, 16QAM.**



main.py is the main function of training for the entire unsupervised inter-domain contrastive learning.

LinearProbing.py is the main function of the linear evaluation stage.

sp_AMR.py is training function for various supervised comparison methods.

read2016_v1.py is the dataset reading code.
