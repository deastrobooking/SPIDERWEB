//! Computer vision utilities and models

use crate::tensor::Tensor;
use crate::nn::{Module, Sequential, Conv2d, ReLU, MaxPool2d, Linear, BatchNorm2d, AdaptiveAvgPool2d};
use anyhow::Result;

/// ResNet block implementation
#[derive(Debug, Clone)]
pub struct ResNetBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    relu: ReLU,
    downsample: Option<Sequential>,
    stride: usize,
    training: bool,
}

impl ResNetBlock {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        let downsample = if stride != 1 || in_channels != out_channels {
            Some(Sequential::new()
                .add(Conv2d::new(in_channels, out_channels, 1, stride, 0))
                .add(BatchNorm2d::new(out_channels)))
        } else {
            None
        };

        Self {
            conv1: Conv2d::new(in_channels, out_channels, 3, stride, 1),
            bn1: BatchNorm2d::new(out_channels),
            conv2: Conv2d::new(out_channels, out_channels, 3, 1, 1),
            bn2: BatchNorm2d::new(out_channels),
            relu: ReLU::new(),
            downsample,
            stride,
            training: true,
        }
    }
}

impl Module for ResNetBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let identity = if let Some(ref downsample) = self.downsample {
            downsample.forward(input)?
        } else {
            input.clone_tensor()
        };

        let out = self.conv1.forward(input)?;
        let out = self.bn1.forward(&out)?;
        let out = self.relu.forward(&out)?;

        let out = self.conv2.forward(&out)?;
        let out = self.bn2.forward(&out)?;

        let out = &out + &identity;
        self.relu.forward(&out)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        if let Some(ref downsample) = self.downsample {
            params.extend(downsample.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters_mut());
        params.extend(self.bn1.parameters_mut());
        params.extend(self.conv2.parameters_mut());
        params.extend(self.bn2.parameters_mut());
        if let Some(ref mut downsample) = self.downsample {
            params.extend(downsample.parameters_mut());
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
        self.conv1.train();
        self.bn1.train();
        self.conv2.train();
        self.bn2.train();
        if let Some(ref mut downsample) = self.downsample {
            downsample.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        self.conv1.eval();
        self.bn1.eval();
        self.conv2.eval();
        self.bn2.eval();
        if let Some(ref mut downsample) = self.downsample {
            downsample.eval();
        }
    }

    fn training(&self) -> bool { self.training }
    fn name(&self) -> &str { "ResNetBlock" }
    
    fn clone_module(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// Create ResNet-18 model
pub fn resnet18(num_classes: usize) -> Sequential {
    Sequential::new()
        .add(Conv2d::new(3, 64, 7, 2, 3))
        .add(BatchNorm2d::new(64))
        .add(ReLU::new())
        .add(MaxPool2d::new(3, 2))
        .add(ResNetBlock::new(64, 64, 1))
        .add(ResNetBlock::new(64, 64, 1))
        .add(ResNetBlock::new(64, 128, 2))
        .add(ResNetBlock::new(128, 128, 1))
        .add(ResNetBlock::new(128, 256, 2))
        .add(ResNetBlock::new(256, 256, 1))
        .add(ResNetBlock::new(256, 512, 2))
        .add(ResNetBlock::new(512, 512, 1))
        .add(AdaptiveAvgPool2d::new(1))
        .add(Linear::new(512, num_classes))
}

/// Create VGG-like model
pub fn vgg11(num_classes: usize) -> Sequential {
    Sequential::new()
        .add(Conv2d::new(3, 64, 3, 1, 1))
        .add(ReLU::new())
        .add(MaxPool2d::new(2, 2))
        .add(Conv2d::new(64, 128, 3, 1, 1))
        .add(ReLU::new())
        .add(MaxPool2d::new(2, 2))
        .add(Conv2d::new(128, 256, 3, 1, 1))
        .add(ReLU::new())
        .add(Conv2d::new(256, 256, 3, 1, 1))
        .add(ReLU::new())
        .add(MaxPool2d::new(2, 2))
        .add(Conv2d::new(256, 512, 3, 1, 1))
        .add(ReLU::new())
        .add(Conv2d::new(512, 512, 3, 1, 1))
        .add(ReLU::new())
        .add(MaxPool2d::new(2, 2))
        .add(Conv2d::new(512, 512, 3, 1, 1))
        .add(ReLU::new())
        .add(Conv2d::new(512, 512, 3, 1, 1))
        .add(ReLU::new())
        .add(MaxPool2d::new(2, 2))
        .add(AdaptiveAvgPool2d::new(7))
        .add(Linear::new(512 * 7 * 7, 4096))
        .add(ReLU::new())
        .add(Linear::new(4096, 4096))
        .add(ReLU::new())
        .add(Linear::new(4096, num_classes))
}

/// Simple CNN for CIFAR-10/MNIST
pub fn simple_cnn(input_channels: usize, num_classes: usize) -> Sequential {
    Sequential::new()
        .add(Conv2d::new(input_channels, 32, 3, 1, 1))
        .add(ReLU::new())
        .add(MaxPool2d::new(2, 2))
        .add(Conv2d::new(32, 64, 3, 1, 1))
        .add(ReLU::new())
        .add(MaxPool2d::new(2, 2))
        .add(Conv2d::new(64, 128, 3, 1, 1))
        .add(ReLU::new())
        .add(AdaptiveAvgPool2d::new(1))
        .add(Linear::new(128, num_classes))
}