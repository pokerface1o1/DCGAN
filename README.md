# DCGAN

GANs were first introduced in 2014 by a group of researchers at the University of Montreal lead by Ian Goodfellow. 
The main idea behind a GAN is to have two competing neural network models. One takes noise as input and generates samples (and so is called the generator). The other model (called the discriminator) receives samples from both the generator and the training data, and has to be able to distinguish between the two sources. These two networks play a continuous game, where the generator is learning to produce more and more realistic samples, and the discriminator is learning to get better and better at distinguishing generated data from real data. These two networks are trained simultaneously, and the hope is that the competition will drive the generated samples to be indistinguishable from real data.

![gan](https://user-images.githubusercontent.com/28016169/27252801-a77cd8d8-5384-11e7-86ab-7604da5d5c21.png)

# Result
![fake_images-200](https://user-images.githubusercontent.com/28016169/27252817-1d2c8010-5385-11e7-91eb-95c277648f28.png)
![fake_images-199](https://user-images.githubusercontent.com/28016169/27252945-a5fc9824-5387-11e7-9241-e3144cb4bf89.png)
![fake_images-198](https://user-images.githubusercontent.com/28016169/27252950-bc68ba0c-5387-11e7-9cd0-1a6d848ce738.png)


# Note
1. Use more convolutional layers in generator for better performance.
2. Binary noise(sequence of -1, 1) works amazingly well.

# Credit
Most of the code is borrowed from [yunjey](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/deep_convolutional_gan)
