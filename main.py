import logging
import logging.config
import os
from pathlib import Path

import pytorch_lightning as pl
import yaml
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

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
    # Only the process with LOCAL_RANK = 0 will print logs on the console.
    # And all the processes will print logs in their own log files.
    if local_rank != 0:
        root_logger = logging.getLogger()
        root_logger.removeHandler(root_logger.handlers[0])

    pl_logger.info(f'Output logs & checkpoints in: {output_dir}')
    # Dump experiment configurations for reproducibility
    if local_rank == 0:
        with open(output_dir / 'cfg.yaml', 'w') as yaml_file:
            yaml_file.write(OmegaConf.to_yaml(cfg))
    pl_logger.info('The final experiment setup is dumped as: ./cfg.yaml')

    pl.seed_everything(cfg.seed, workers=True)

    # Create model
    net = load_obj(cfg.model.class_name, 'torchvision.models')(**cfg.model.params)
    pl_logger.info(f'Create model "{type(net)}". You can view its graph using TensorBoard.')

    # Inject quantizers into the model
    net = nz.quantizer_inject(net, cfg.quan)
    quan_cnt, quan_dict = nz.quantizer_stat(net)
    msg = f'Inject {quan_cnt} quantizers into the model:'
    for k, v in quan_dict.items():
        msg += f'\n                {k} = {len(v)}'
    yaml.safe_dump(quan_dict, open(output_dir / 'quan_stat.yaml', 'w'))
    pl_logger.info(msg)
    pl_logger.info('A complete list of injected quantizers is dumped as: ./quan_stat.yaml')

    # Prepare the dataset
    dm = apputil.get_datamodule(cfg)
    pl_logger.info(f'Prepare the "{cfg.dataset.name}" dataset from: {cfg.dataset.data_dir}')
    msg = f'The dataset samples are split into three sets:' \
          f'\n         Train = {len(dm.train_dataloader())} batches (batch size = {dm.train_dataloader().batch_size})' \
          f'\n           Val = {len(dm.val_dataloader())} batches (batch size = {dm.val_dataloader().batch_size})' \
          f'\n          Test = {len(dm.test_dataloader())} batches (batch size = {dm.test_dataloader().batch_size})'
    pl_logger.info(msg)

    progressbar_cb = apputil.ProgressBar(pl_logger)
    # gpu_stats_cb = pl.callbacks.GPUStatsMonitor()

    if cfg.checkpoint.path:
        assert Path(cfg.checkpoint.path).is_file(), f'Checkpoint path is not a file: {cfg.checkpoint.path}'
        pl_logger.info(f'Resume training checkpoint from: {cfg.checkpoint.path}')

    if cfg.eval:
        pl_logger.info('Training process skipped. Evaluate the resumed model.')
        assert cfg.checkpoint.path is not None, 'Try to evaluate the model resumed from the checkpoint, but got None'

        # Initialize the Trainer
        trainer = pl.Trainer(callbacks=[progressbar_cb], **cfg.trainer)
        pl_logger.info(f'The model is distributed to {trainer.num_gpus} GPUs with {cfg.trainer.accelerator} backend.')

        pretrained_lit = LitModuleWrapper.load_from_checkpoint(checkpoint_path=cfg.checkpoint.path, model=net, cfg=cfg)
        trainer.test(pretrained_lit, datamodule=dm, verbose=False)
    else:  # train + eval
        tb_logger = TensorBoardLogger(output_dir / 'tb_runs', name=cfg.experiment_name, log_graph=True)
        pl_logger.info('Tensorboard logger initialized in: ./tb_runs')

        lr_monitor_cb = pl.callbacks.LearningRateMonitor()
        checkpoint_cb = pl.callbacks.ModelCheckpoint(dirpath=output_dir / 'checkpoints',
                                                     filename='{epoch}-{val_loss_epoch:.4f}-{val_acc_epoch:.4f}',
                                                     monitor='val_loss_epoch',
                                                     mode='min',
                                                     save_top_k=3,
                                                     save_last=True)
        pl_logger.info('Checkpoints of the best 3 models as well as the last one will be saved to: ./checkpoints')

        # Wrap model with LightningModule
        lit = LitModuleWrapper(net, cfg)
        # Generate a fake input array TensorBoard to generate graph
        lit.example_input_array = dm.val_dataloader().dataset[0][0].unsqueeze(dim=0)

        # Initialize the Trainer
        trainer = pl.Trainer(logger=tb_logger,
                             callbacks=[checkpoint_cb, lr_monitor_cb, progressbar_cb],
                             resume_from_checkpoint=cfg.checkpoint.path,
                             plugins=DDPPlugin(find_unused_parameters=False),
                             **cfg.trainer)
        pl_logger.info(f'The model is distributed to {trainer.num_gpus} GPUs with {cfg.trainer.strategy} backend.')

        pl_logger.info('Training process begins.')
        trainer.fit(model=lit, datamodule=dm)

        pl_logger.info('Evaluate the best trained model.')
        trainer.test(datamodule=dm, verbose=False)

    pl_logger.info('Program completed successfully. Exiting...')
    pl_logger.info('If you have any questions or suggestions, please visit: github.com/zhutmost/neuralzip')


if __name__ == '__main__':
    conf = apputil.get_config(base_conf_filepath=Path.cwd() / 'conf' / 'template.yaml')
    run(cfg=conf)
