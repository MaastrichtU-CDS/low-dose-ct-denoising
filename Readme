# Are-All-Shortcuts-in-Encoder-Decoder-Networks-Necessary-for-Computed-Tomography-Denoising-
Code and Supplementary Materials for manuscript 'Are All Shortcuts in Encoder-Decoder Networks Necessary for Computed Tomography Denoising?'

The aim of this project is finding out the necessarity of all shortcuts in encoder-decoder network for medcial image denoising task. In deep learning era, a series of publications have shown that generative model based methods has become the state-of-art method. As an important generative model, encoder-decoder network has been used in medical image denoising. Shortcuts are improtant parts for encoder-decoder network when it was used for segmentation and style transfer, however, are they still important for denoising task? More details of about this paper you can find in our papers. Here are the codes of paper'Are All Shortcuts in Encoder-Decoder Networks Necessary for CT Denoising?'

First of all, datasets which were used in our papaer can be found in Low Dose CT Grand Challenge (https://www.aapm.org/GrandChallenge/LowDoseCT/, called as LDGC in the following section)  and NSCLC-Radiomics(https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics, called as LUNG 1 in the following section).

As we mentionded in the paper, we simulated noisy images based on LDGC and LUNG 1, we used Addnoiseinradiomic.m to finish this task, a matlab script.

We used tfrecord as the input file of our encoder-decoder network, therefore, we need make tfrecord based on paired noised and standard images at first. This part of job was finsihed by using Tfwrite.py, a python script.

The generated tfrecords were used to train the encoder-decoder network and then the trained network used to denoise images. This part of job was finished by Second-Encoder-Decoder-networkRealdata.py, a python script.

We used pretrained VGG-16 network to calculate content loss, the pretrained network can be found in https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM
