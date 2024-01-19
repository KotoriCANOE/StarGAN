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

(unchanged)
LR policy: SGDR + exponential
steps: 511000

## 09

steps: 256000

## 10

(unchanged)
adv loss: WGAN-GP => DRAGAN
- mode collapse

## 11

steps: 256000
batch size: 8
adv loss: WGAN-GP => WGAN-div

## 12

steps: 256000
batch size: 12
adv loss: WGAN-div

## 13

steps: 256000
batch size: 12
adv loss: WGAN-GP

## 14

steps: 128000
batch size: 12
adv loss: WGAN-div
LR: 0=2e-5, 64000=2e-5, 128000=0
