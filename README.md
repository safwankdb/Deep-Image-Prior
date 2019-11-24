# Deep Image Prior

PyTorch implementation of the CVPR 2018 paper [Deep Image Prior](https://dmitryulyanov.github.io/deep_image_prior) by Dmitry Ulyanov et. al. This codebase is a part of final project for **CS 663: Fundamentals of Digital Image Processing** at IITB. The team members are:

- [Mohd Safwan](https://safwankdb.github.io)
- [Manas Vashistha](https://home.iitb.ac.in/~manasv)
- Sumrit Gupta

We use the method described in the paper for image inpainting. We randomly delete some pixels of a 512*512 RGB image and then try to fill them using the said technique.

## Results

### 50% pixels removed
<div align='center'>
    <img src='imgs/lena_start.jpg' height="256">
    <img src='imgs/lena_final.jpg' height="256">
    <img src='imgs/lena_progress.gif' height="256">
    <img src='imgs/start.jpg' height="256">
    <img src='imgs/final.jpg' height="256">
    <img src='imgs/barbara_progress.gif' height="256">
</div>

The images used are the standard **Lena** and **Barbara** test images. The input and resultant images are as shown.The progress in the initial iterations is shown on the rightmost column.

### 80% pixels removed
<div align='center'>
    <img src='imgs/lena20_start.jpg' height="256">
    <img src='imgs/lena20_final.jpg' height="256">
    <img src='imgs/lena20_progress.gif' height="256">
</div>

## Training
This are the loss curves we obtained during the training on 50% and 80% images respectively. We can observe that the reconstruction loss decreases to negligible value within very few iterations.
<div align='center'>
    <img src='imgs/loss.jpg' height="256">
    <img src='imgs/loss2.jpg' height="256">
</div>

## References
1. **Dmitry Ulyanov et. al** *Deep Image Prior* [[arxiv](https://arxiv.org/abs/1711.10925)]
2. Test Images [Public-Domain Test Images for Homeworks and Projects](https://homepages.cae.wisc.edu/~ece533/images/)
