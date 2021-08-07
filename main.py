import logging
import logging.config
import os
from pathlib import Path

import pytorch_lightning as pl
import torch as t
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

import apputil
import neuralzip as nz
from apputil import load_obj
from lightning import LitModuleWrapper


def run(cfg: DictConfig):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # The logs & checkpoints are dumped in: ${cfg.output_dir}/${cfg.experiment_name}/vN, where vN
    # is v0, v1, .... The version number increases automatically.
    script_dir = Path.cwd()
    experiment_dir = script_dir / cfg.output_dir / cfg.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    existing_ver = list()
    for d in experiment_dir.iterdir():
        if d.name.startswith('v') and d.name[1:].isdecimal() and d.is_dir():
            existing_ver.append(int(d.name[1:]))
    if local_rank == 0:
        current_ver = max(existing_ver) + 1 if existing_ver else 0
        output_dir = experiment_dir / f'v{current_ver}'
        output_dir.mkdir()
    else:
        # Use the same directory for output with the main process.
        current_ver = max(existing_ver)
        output_dir = experiment_dir / f'v{current_ver}'

    pl_logger = logging.getLogger('lightning')
    logging.config.fileConfig(script_dir / 'logging.conf', disable_existing_loggers=False,
                              defaults={'log_filename': output_dir / f'run_rank{local_rank}.log'})

    pl_logger.info(f'Output logs & checkpoints in: {output_dir}')
    # Dump experiment configurations for reproducibility
    if local_rank == 0:
        with open(output_dir / "cfg.yaml", "w") as yaml_file:
            yaml_file.write(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)

    pl_logger.info('Tensorboard logger initialized in: ./tb_runs')
    tb_logger = TensorBoardLogger(output_dir / 'tb_runs', name=cfg.experiment_name, log_graph=True)

    # Create model
    net = load_obj(cfg.model.class_name, 'torchvision.models')(**cfg.model.params)
    pl_logger.info(f'Create model "{type(net)}". You can view its graph using TensorBoard.')

    # Inject quantizers into the model
    net = nz.quantizer_inject(net, cfg.quan)
    quan_cnt, quan_dict = nz.quantizer_stat(net)
    msg = f'Inject {quan_cnt} quantizers into the model:'
    for k, n in quan_dict.items():
        msg += f'\n                {str(k)} = {n}'
    pl_logger.info(msg)

    # Prepare the dataset
    dm = apputil.get_datamodule(cfg)
    pl_logger.info(f'Prepare the "{cfg.dataset.name}" dataset from: {cfg.dataset.data_dir}')
    msg = f'The dataset samples are split into three sets:' \
          f'\n              Training Set = {len(dm.train_dataloader().sampler)} ({len(dm.train_dataloader())})' \
          f'\n            Validation Set = {len(dm.val_dataloader().sampler)} ({len(dm.val_dataloader())})' \
          f'\n                  Test Set = {len(dm.test_dataloader().sampler)} ({len(dm.test_dataloader())})'
    pl_logger.info(msg)

    progressbar_cb = apputil.ProgressBar(pl_logger)
    # gpu_stats_cb = pl.callbacks.GPUStatsMonitor()
    lr_monitor_cb = pl.callbacks.LearningRateMonitor()
    checkpoint_cb = pl.callbacks.ModelCheckpoint(dirpath=output_dir / 'checkpoints',
                                                 filename='{epoch}-{val_loss_epoch:.2f}-{val_acc_epoch:.2f}',
                                                 monitor='val_acc_epoch',
                                                 save_top_k=3,
                                                 save_last=True)
    pl_logger.info('Checkpoints of the best 3 models as well as the last one will be saved to: ./checkpoints')

    # Wrap model with LightningModule
    lit = LitModuleWrapper(net, cfg)
    # A fake input array for TensorBoard to generate graph
    lit.example_input_array = t.rand(dm.size()).unsqueeze(dim=0)

    # Initialize the Trainer
    trainer = pl.Trainer(logger=[tb_logger],
                         callbacks=[checkpoint_cb, lr_monitor_cb, progressbar_cb],
                         resume_from_checkpoint=cfg.checkpoint.path,
                         **cfg.trainer)
    if cfg.checkpoint.path:
        assert Path(cfg.checkpoint.path).is_file(), f'Checkpoint path is not a file: {cfg.checkpoint.path}'
        pl_logger.info(f'Resume training checkpoint from: {cfg.checkpoint.path}')
    pl_logger.info(f'The model is distributed to {trainer.num_gpus} GPUs with {cfg.trainer.accelerator} backend.')

    if cfg.eval:
        pl_logger.info('Training process skipped. Evaluate the resumed model.')
        assert cfg.checkpoint.path is not None, 'Try to evaluate the model resumed from the checkpoint, but got None'
        trainer.test(lit, datamodule=dm, verbose=False)
    else:  # train + eval
        pl_logger.info('Training process begins.')
        trainer.fit(lit, datamodule=dm)

        pl_logger.info('Evaluate the best trained model.')
        trainer.test(datamodule=dm, ckpt_path='best', verbose=False)

    pl_logger.info('Program completed successfully. Exiting...')
    pl_logger.info('If you have any questions or suggestions, please visit: github.com/zhutmost/neuralzip')


if __name__ == '__main__':
    # When in the `ddp` parallelism mode, this script will be executed many times, and Hydra will
    # generate multiple directories for logging. To avoid confusion, these directories are renamed
    # as `${experiment_name}-${current_time}-${current_local_rank}`.
    # Note that most outputs are only dumped in the directory with LOCAL_RANK = 0.
    conf = apputil.get_config(base_conf_filepath=Path.cwd() / 'conf' / 'template.yaml')

    run(cfg=conf)
