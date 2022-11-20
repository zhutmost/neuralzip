import torchvision.models.resnet as resnet_tv


from typing import Any

def resnet18(pretrained: bool=True, **kwargs: Any) -> resnet_tv.ResNet:
    """Wrapper of ResNet from TorchVision to keep compatibility.

        Args:
            pretrained (bool): Whether the pretrained weights are loaded.
            **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet`` base class.
    """
    w = resnet_tv.ResNet18_Weights.DEFAULT if pretrained else None
    m = resnet_tv.resnet18(weights=w, **kwargs)
    return m


def resnet34(pretrained: bool=True, **kwargs: Any) -> resnet_tv.ResNet:
    """Wrapper of ResNet from TorchVision to keep compatibility.

        Args:
            pretrained (bool): Whether the pretrained weights are loaded.
            **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet`` base class.
    """
    w = resnet_tv.ResNet34_Weights.DEFAULT if pretrained else None
    m = resnet_tv.resnet34(weights=w, **kwargs)
    return m

def resnet50(pretrained: bool=True, **kwargs: Any) -> resnet_tv.ResNet:
    """Wrapper of ResNet from TorchVision to keep compatibility.

        Args:
            pretrained (bool): Whether the pretrained weights are loaded.
            **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet`` base class.
    """
    w = resnet_tv.ResNet50_Weights.DEFAULT if pretrained else None
    m = resnet_tv.resnet50(weights=w, **kwargs)
    return m

def resnet101(pretrained: bool=True, **kwargs: Any) -> resnet_tv.ResNet:
    """Wrapper of ResNet from TorchVision to keep compatibility.

        Args:
            pretrained (bool): Whether the pretrained weights are loaded.
            **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet`` base class.
    """
    w = resnet_tv.ResNet101_Weights.DEFAULT if pretrained else None
    m = resnet_tv.resnet101(weights=w, **kwargs)
    return m

def resnet152(pretrained: bool=True, **kwargs: Any) -> resnet_tv.ResNet:
    """Wrapper of ResNet from TorchVision to keep compatibility.

        Args:
            pretrained (bool): Whether the pretrained weights are loaded.
            **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet`` base class.
    """
    w = resnet_tv.ResNet152_Weights.DEFAULT if pretrained else None
    m = resnet_tv.resnet152(weights=w, **kwargs)
    return m

def resnext50_32x4d(pretrained: bool=True, **kwargs: Any) -> resnet_tv.ResNet:
    """Wrapper of ResNet from TorchVision to keep compatibility.

        Args:
            pretrained (bool): Whether the pretrained weights are loaded.
            **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet`` base class.
    """
    w = resnet_tv.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
    m = resnet_tv.resnet18(weights=w, **kwargs)
    return m

def resnext101_32x8d(pretrained: bool=True, **kwargs: Any) -> resnet_tv.ResNet:
    """Wrapper of ResNet from TorchVision to keep compatibility.

        Args:
            pretrained (bool): Whether the pretrained weights are loaded.
            **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet`` base class.
    """
    w = resnet_tv.ResNeXt101_32X8D_Weights.DEFAULT if pretrained else None
    m = resnet_tv.resnext101_32x8d(weights=w, **kwargs)
    return m

def resnext101_64x4d(pretrained: bool=True, **kwargs: Any) -> resnet_tv.ResNet:
    """Wrapper of ResNet from TorchVision to keep compatibility.

        Args:
            pretrained (bool): Whether the pretrained weights are loaded.
            **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet`` base class.
    """
    w = resnet_tv.ResNeXt101_64X4D_Weights.DEFAULT if pretrained else None
    m = resnet_tv.resnext101_64x4d(weights=w, **kwargs)
    return m

def wide_resnet50_2(pretrained: bool=True, **kwargs: Any) -> resnet_tv.ResNet:
    """Wrapper of ResNet from TorchVision to keep compatibility.

        Args:
            pretrained (bool): Whether the pretrained weights are loaded.
            **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet`` base class.
    """
    w = resnet_tv.Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
    m = resnet_tv.wide_resnet50_2(weights=w, **kwargs)
    return m

def wide_resnet101_2(pretrained: bool=True, **kwargs: Any) -> resnet_tv.ResNet:
    """Wrapper of ResNet from TorchVision to keep compatibility.

        Args:
            pretrained (bool): Whether the pretrained weights are loaded.
            **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet`` base class.
    """
    w = resnet_tv.Wide_ResNet101_2_Weights.DEFAULT if pretrained else None
    m = resnet_tv.wide_resnet101_2(weights=w, **kwargs)
    return m
