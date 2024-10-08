# MAC_UAMR
We have released the "Multi-Representation Domain Attentive Contrastive Learning Based Unsupervised Automatic Modulation Recognition (MAC)" and our codebase to the community. In this work, we propose an unsupervised automatic modulation recognition (UAMR) method based on multi-domain contrastive learning. Existing deep learning-based AMR methods suffer from significant performance degradation when there is a lack of labeled samples. We evaluated the proposed method on three public datasets and across different sample sizes. By leveraging contrastive learning and multi-domain representation, MAC achieves representation learning of modulation signals using unlabeled signals. After unsupervised training, the encoder can achieve surprisingly effective semi-supervised modulation recognition when fine-tuned with a small amount of labeled samples.

## Article PDF
[Multi-Representation Domain Attentive Contrastive Learning Based Unsupervised Automatic Modulation Recognition](https://www.researchsquare.com/article/rs-3696311/v1)

Version 1

posted 21 Jan, 2024

## Motivation

The existing data augmentation schemes for AMR primarily draw from those developed in computer vision or audio sequence fields. such as *flipping*, *rotating*, *adding Gaussian noise*. However, these methods do not consider data augmentation from the perspective of signal modulation characteristics, just like processing an image.
<div align=center>
<img src="https://github.com/user-attachments/assets/d0c2cadd-dbcd-4f9f-8d28-e963b5f888d1" width="70%">
</div>
MAC inter-domain contrastive learning engages different representational domains that observe signal modulation characteristics in a high-dimensional space. This is analogous to viewing an object from multiple perspectives—frontal, lateral, and overhead—which provides a more comprehensive set of features (information) and makes the modulated signals recognizable.
<div align=center>
<img src="https://github.com/user-attachments/assets/10bdbf56-def1-4c84-88d1-bb8e2a57d95c" width="70%">
</div>

# requirements
**Window+Pytorch**
```
pip install -r requirements.txt
```

# Benchmark Datasets

|Dataset|Modulation type|Signal sample size|Signal length|SNR|
|:---|:---|:---|:---|:---|
|[RML2016.10A](https://www.deepsig.ai/datasets/)|11classes|220000|2\*128|-20:2:20|
|[RML2016.10B](https://www.deepsig.ai/datasets/)|10classes|1200000|2\*128|-20:2:20|
|[RML2018.01A](https://www.deepsig.ai/datasets/)|24classes|2555904|2\*1024|-20:2:30|

***RML2016.10A*: 8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAW4, QAM16, QAM64, QPSK, and WBFM.**

***RML2016.10B*: 8PSK, AM-SSB, BPSK, CPFSK, GFSK, PAW4, QAM16, QAM64, QPSK, and WBFM.**

***RML2018.01A*: 32PSK, 16APSK, 32QAM, FM, GMSK, 32APSK, OQPSK, 8ASK, BPSK, 8PSK, AM-SSB-SC, 4ASK, 16PSK, 64APSK, 128QAM, 128APSK, AM-DSB-SC, AM-SSB-WC, 64QAM, QPSK, 256QAM, AM-DSB-WC, OOK, 16QAM.**

# About

**Pretraing_MAC.py**: Implement unsupervised pre-training phase for MAC

**Fine_tuning_Times.py**: Implement the linear evaluation and Fine-tuning phase of MAC (semi-supervised modulation recognition)

# Citation
If this repository is helpful to you, please cite our work.
```
@inproceedings{li_unsupervised_2023,
	title = {Unsupervised {Modulation} {Recognition} {Method} {Based} on {Multi}-{Domain} {Representation} {Contrastive} {Learning}},
	booktitle = {2023 {IEEE} {International} {Conference} on {Signal} {Processing}, {Communications} and {Computing} ({ICSPCC})},
	author = {Li, Yu and Shi, Xiaoran and Yang, Xinyao and Zhou, Feng},
	month = nov,
	year = {2023},
	pages = {1--6},
}
```
```
@article{multi-representation_2024,
	title = {Multi-{Representation} {Domain} {Attentive} {Contrastive} {Learning} {Based} {Unsupervised} {Automatic} {Modulation} {Recognition}},
	url = {https://www.researchsquare.com},
	author = {Li, Yu and Shi, Xiaoran and Yang, Xinyao and Zhou, Feng},
	language = {en},
	urldate = {2024-03-08},
	month = jan,
	year = {2024},
}
```
