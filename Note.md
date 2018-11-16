# StarGAN

## 01

batch size: 16
Generator - ResBlock: IN-Swish-Conv-IN-Swish-Conv-SEUnit
Discriminator2: VGG-like

## 02

Discriminator: DenseNet-like

## 03

WGAN-GP: fixed the shape of random generator in random interpolate
restore to 256000 steps

## 04

(unchanged)
WGAN-GP: used DRAGAN interpolation

## 05

(unchanged)
Generator - EBlock, DBlock: added InstanceNorm

## 06

(unchanged)
Generator - ResBlock: Swish-IN-Conv-Swish-IN-Conv-SEUnit

## 07

steps: 128000 => 512000

## 08

LR policy: SGDR + exponential
steps: 511000
