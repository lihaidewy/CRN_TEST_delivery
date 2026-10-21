# Copyright (c) Megvii Inc. All rights reserved.
import os
import torch  # 🚨 新增：用于加载权重
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_summary import ModelSummary
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from callbacks.ema import EMACallback
from callbacks.rolling_epoch_logger import RollingEpochLogger
from utils.torch_dist import all_gather_object, synchronize

from .base_exp import BEVDepthLightningModel


def run_cli(model_class=BEVDepthLightningModel,
            exp_name='base_exp',
            use_ema=False,
            ckpt_path=None):
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('-e',
                               '--evaluate',
                               dest='evaluate',
                               action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('-p',
                               '--predict',
                               dest='predict',
                               action='store_true',
                               help='predict model on testing set')
    parent_parser.add_argument('-b', '--batch_size_per_device', type=int)
    parent_parser.add_argument('--seed',
                               type=int,
                               default=0,
                               help='seed for initializing training.')
    parent_parser.add_argument('--ckpt_path', type=str)
    parent_parser.add_argument(
        '--disable_early_stopping',
        action='store_true',
        help='Run the configured max_epochs without the EarlyStopping callback.',
    )
    parent_parser.add_argument(
        '--checkpoint_on_train_epoch_end',
        action='store_true',
        help=(
            'Save checkpoints at train-epoch end. Required when validation '
            'is intentionally disabled (for example, to avoid reading a held-out split).'
        ),
    )
    parent_parser.add_argument(
        '--rolling_terminal_log',
        action='store_true',
        help=(
            'Keep the live tqdm progress bar and append one persistent summary '
            'line for each train/validation epoch.'
        ),
    )
    parent_parser.add_argument(
        '--checkpoint_every_n_epochs',
        type=int,
        default=8,
        help='Save one intermediate checkpoint every N epochs (default: 8).',
    )
    parser = BEVDepthLightningModel.add_model_specific_args(parent_parser)
    parser.set_defaults(profiler='simple',
                        deterministic=False,
                        max_epochs=50,
                        # strategy='ddp',
                        # strategy='single_device',
                        # strategy='ddp_find_unused_parameters_false',
                        num_sanity_val_steps=0,
                        check_val_every_n_epoch=2,  # 每2个epoch验证一次，减少验证开销
                        gradient_clip_val=5,
                        limit_val_batches=0.25,
                        log_every_n_steps=50,
                        enable_checkpointing=True,
                        precision=16,
                        default_root_dir=os.path.join('./outputs/', exp_name))
    args = parser.parse_args()
    if args.seed is not None:
        pl.seed_everything(args.seed)

    model = model_class(**vars(args))
    # 每隔 8 个 epoch 保存中间权重，方便观察不同阶段的收敛情况
    # save_last=True 确保训练结束时始终保留最后一个 epoch 的权重
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=args.checkpoint_every_n_epochs,
        save_top_k=-1,
        save_last=True,
        # Preserve prior Lightning default unless this explicit opt-in is
        # passed.  With limit_val_batches=0 the default callback may never
        # receive a validation-end hook and consequently save no checkpoint.
        save_on_train_epoch_end=True if args.checkpoint_on_train_epoch_end else None,
    )

    # EarlyStopping: monitor validation detection loss.
    early_stop_callback = EarlyStopping(
        monitor='val/detection',
        min_delta=0.001,
        patience=10,
        verbose=True,
        mode='min',
        check_on_train_epoch_end=False,
    )

    # 训练进度通知回调（可选，通过环境变量 SERVERCHAN_KEY 配置微信推送）
    notify_callback = None
    _notify_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts', 'notify_callback.py')
    if os.path.exists(_notify_path) and os.environ.get('SERVERCHAN_KEY', ''):
        import importlib.util
        _spec = importlib.util.spec_from_file_location('notify_callback', _notify_path)
        _notify_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_notify_mod)
        notify_callback = _notify_mod.NotifyCallback()

    if use_ema:
        train_dataloader = model.train_dataloader()
        ema_callback = EMACallback(
            len(train_dataloader.dataset) * args.max_epochs)
        cb_list = [ema_callback, checkpoint_callback, ModelSummary(max_depth=3)]
        if not args.disable_early_stopping:
            cb_list.append(early_stop_callback)
        if notify_callback:
            cb_list.append(notify_callback)
        if args.rolling_terminal_log:
            cb_list.append(RollingEpochLogger())
        trainer = pl.Trainer.from_argparse_args(args, callbacks=cb_list,
        accelerator="gpu",
        devices=[0]
        )
    else:
        cb_list = [checkpoint_callback, ModelSummary(max_depth=3)]
        if not args.disable_early_stopping:
            cb_list.append(early_stop_callback)
        if notify_callback:
            cb_list.append(notify_callback)
        if args.rolling_terminal_log:
            cb_list.append(RollingEpochLogger())
        trainer = pl.Trainer.from_argparse_args(args, callbacks=cb_list,
        accelerator="gpu",
        devices=[0])
        
    if args.evaluate:
        trainer.test(model, ckpt_path=args.ckpt_path)
    elif args.predict:
        predict_step_outputs = trainer.predict(model, ckpt_path=args.ckpt_path)
        all_pred_results = list()
        all_img_metas = list()
        for predict_step_output in predict_step_outputs:
            for i in range(len(predict_step_output)):
                all_pred_results.append(predict_step_output[i][:3])
                all_img_metas.append(predict_step_output[i][3])
        synchronize()
        len_dataset = len(model.test_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:len_dataset]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:len_dataset]
        model.evaluator._format_bbox(all_pred_results, all_img_metas,
                                     os.path.dirname(args.ckpt_path))
    # else:
    #     # =====================================================================
    #     #  增量训练 / 微调 
    #     # =====================================================================
    #     # 之前训练好的最优权重路径
    #     pretrained_ckpt = "outputs/det/CRN_r50_256x704_128x128_4key/lightning_logs/version_81/checkpoints/epoch=23-step=4320.ckpt"
        
    #     if os.path.exists(pretrained_ckpt):
    #         print(f" [增量训练] 正在提取记忆 (权重): {pretrained_ckpt}")
    #         checkpoint = torch.load(pretrained_ckpt, map_location="cpu")
            
    #         # 严格把权重灌入当前模型中
    #         model.load_state_dict(checkpoint['state_dict'], strict=True)
    #         print(" 权重灌注成功！即将以全新的优化器状态，从 Epoch 0 开始增量微调...")
    #     else:
    #         print(f"⚠️ 警告：找不到权重文件 {pretrained_ckpt}，将从零开始训练！")

    #     # 注意：这里千万不要加 ckpt_path 参数，让 Lightning 以为这是一个全新的训练任务
    #     trainer.fit(model)
    #     # =====================================================================
    else:
        print(
            '未通过 --ckpt_path 恢复完整训练状态（epoch、优化器和调度器从头计数）。'
            '若实验配置在 __init__ 中加载了初始化权重，当前训练仍属于基于该权重的微调。'
        )
        trainer.fit(model, ckpt_path=args.ckpt_path)
