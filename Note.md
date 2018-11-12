# StarGAN

## 01

batch size: 16
Generator - ResBlock: IN-Swish-Conv-IN-Swish-Conv-SEUnit
Discriminator2: VGG-like

## 02

Discriminator: DenseNet-like

## 03

WGAN-GP: fixed the shape of random generator in random interpolate

## 04

(unchanged)
WGAN-GP: used DRAGAN interpolation

## 05

(unchanged)
Generator - EBlock, DBlock: added InstanceNorm

## 06

Generator - ResBlock: Swish-IN-Conv-Swish-IN-Conv-SEUnit

