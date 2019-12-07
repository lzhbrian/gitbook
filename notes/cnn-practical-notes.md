---
description: some notes on my training with CNN
---

# CNN practical notes

## Train CIFAR10

* On 2019.12.6, I tried to use `torchvision.models.resnet18` to train CIFAR10
* Some findings in time sequential orders
  * without weight decay, model can only reach 70% something.
  * pretrained weights from ImageNet does not actually helps.
  * using `torchvision.models.resnet18` can only achieve 85% something accuracy. This is because `torchvision.models` is perfectly tuned for ImageNet, and when training on other datasets, the results usually won't went well.
    * See also
      * [How to train resnet18 to the best accuracy? \#1166](https://github.com/pytorch/vision/issues/1166)
      * [Adding "narrow" ResNet models for CIFAR-10 \#1570](https://github.com/pytorch/vision/issues/1570)
      * [https://github.com/akamaster/pytorch\_resnet\_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10) \(I use this implementation and achieve comparable claimed test set accuracy 8.27%\)
      * [https://github.com/kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
* Takeaway
  * weight decay is important, yet `torch.optim` disable it by default. set it to 1e-4 or 5e-4.
  * It's not preferrable to directly use `torchvision.models` or other pretrained model architectures on datasets other than ImageNet. That's what so called 'Hyperparameter tuning is important'.

## Pretrained models

* [https://github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
* [https://github.com/Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)

