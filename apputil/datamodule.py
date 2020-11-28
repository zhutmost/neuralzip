from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule


def get_datamodule(cfg):
    dataset_cfg = cfg.dataset
    if cfg.dataset.name == 'cifar10':
        dm = CIFAR10DataModule(cfg.dataset.data_dir,
                               num_workers=dataset_cfg.workers,
                               batch_size=dataset_cfg.batch_size,
                               seed=cfg.seed)
    elif cfg.dataset.name == 'imagenet':
        dm = ImagenetDataModule(cfg.dataset.data_dir,
                                num_workers=dataset_cfg.workers,
                                batch_size=dataset_cfg.batch_size)
    else:
        raise ValueError(f'get_datamodule does not support dataset {dataset_cfg.name}')
    return dm
