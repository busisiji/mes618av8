# YOLOv8 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv8 model on a custom dataset.
Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov8ns.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov8ns.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
"""

import argparse
import glob
import logging
import os
import random
import shutil
import sys
import time

from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

import yaml
from torch.optim import lr_scheduler


from ultralytics import YOLO

from xml2txt import convert_annotation

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks

from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, check_amp, check_dataset, check_file, check_git_info,
                           check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds,methods, one_cycle, print_args, print_mutation,yaml_save)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume

from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (ModelEMA, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first, run_start_log)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
logging.basicConfig(filename='example.log', level=logging.DEBUG,format='\r\n%(asctime)s %(levelname)s：%(message)s')
# 打开日志文件并将其截断为零字节
with open('example.log', 'w'):
    pass
# GIT_INFO = check_git_info()


def train(hyp, opt, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    task, device, save_dir, epochs, batch_size, weights, \
        lr0, lrf, momentum, weight_decay, \
        warmup_epochs, warmup_momentum, warmup_bias_lr, \
        box, cls, cls_pw, obj, obj_pw, iou_t, \
        anchor_t, fl_gamma, label_smoothing, \
        nbs, overlap_mask, mask_ratio, dropout, \
        single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        opt.task, opt.device, Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, \
            opt.lr0, opt.lrf, opt.momentum, opt.weight_decay, \
            opt.warmup_epochs, opt.warmup_momentum, opt.warmup_bias_lr, \
            opt.box, opt.cls, opt.cls_pw, opt.obj, opt.obj_pw, opt.iou_t, \
            opt.anchor_t, opt.fl_gamma, opt.label_smoothing, \
            opt.nbs, opt.overlap_mask, opt.mask_ratio, opt.dropout, \
            opt.single_cls, opt.evolve, opt.data, opt.cfg, \
            opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

    try:
        callbacks.run('on_pretrain_routine_start')

        # Directories
        w = save_dir / 'weights'  # weights dir
        # if not resume:
        print(w)
        (w.parent if evolve else w).mkdir(parents=True, exist_ok=opt.exist_ok)  # make dir
        last, best = w / 'last.pt', w / 'best.pt'

        # Hyperparameters
        if isinstance(hyp, str):
            with open(hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)  # load hyps dict
        LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
        opt.hyp = hyp.copy()  # for saving hyps to checkpoints

        # Save run settings
        if not evolve:
            yaml_save(save_dir / 'hyp.yaml', hyp)
            yaml_save(save_dir / 'opt.yaml', vars(opt))

        # Loggers
        data_dict = None
        if RANK in {-1, 0}:
            loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance

            # Register actions
            for k in methods(loggers):
                callbacks.register_action(k, callback=getattr(loggers, k))

            # Process custom dataset artifact link
            data_dict = loggers.remote_dataset
            if resume:  # If resuming runs from remote artifact
                weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

        # Config
        plots = not evolve and not opt.noplots  # create plots
        # cuda = device.type != 'cpu'
        init_seeds(opt.seed + 1 + RANK, deterministic=True)
        with torch_distributed_zero_first(LOCAL_RANK):
            data_dict = data_dict or check_dataset(data)  # check if None
        train_path, val_path = data_dict['train'], data_dict['val']
        nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
        names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
        is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset
        fine_moedl_file = increment_path(Path(str(ROOT / 'runs/detect')) / 'train', exist_ok=opt.exist_ok) / 'weights'

        # Model
        check_suffix(weights, '.pt')  # check weights
        pretrained = weights.endswith('.pt')

        if pretrained:
            with torch_distributed_zero_first(LOCAL_RANK):
                weights = attempt_download(weights)  # download if not found locally
            # ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
            #
            # load model
            model = YOLO(weights)
            print(device)
            model.train(data=data, epochs=epochs, lr0=lr0, lrf=lrf, batch=batch_size, momentum=momentum, weight_decay=weight_decay,
                        warmup_epochs=warmup_epochs, warmup_momentum=warmup_momentum, warmup_bias_lr=warmup_bias_lr,
                        box=box, cls=cls, device=device)  # DDP mode

            exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
            # csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            # LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
        else:
            # DDP mode
            model = YOLO(weights)
            model.train(data=data, epochs=epochs, lr0=lr0, lrf=lrf, batch=batch_size, momentum=momentum,
                       weight_decay=weight_decay,
                       warmup_epochs=warmup_epochs, warmup_momentum=warmup_momentum, warmup_bias_lr=warmup_bias_lr,
                       box=box, cls=cls, device=device)  # DDP mode
    # amp = check_amp(model)  # check AMP
    except Exception as e:
        # 记录异常信息
        logging.exception("【模型训练报错】")

    try:
        '''模型文件移到最外层'''
        shutil.copy(fine_moedl_file / 'best.pt', 'best.pt')
        print(save_dir / 'opt.yaml', increment_path(Path(str(ROOT / 'runs/detect')) / 'train', exist_ok=False) / 'opt.yaml')

    # shutil.copy(save_dir / 'opt.yaml', increment_path(Path(str(ROOT / 'runs/detect')) / 'train', exist_ok=False) / 'opt.yaml')
    except Exception as e:
        # 记录异常信息
        logging.exception("【模型文件未保存】")




def parse_opt(known=False):
    '''
- `-task`：选择训练任务，可以是`detect`、`classify`或`seg`。
- `-device`：选择 GPU 设备，可以是单个 GPU 设备的编号，例如`0`，或多个 GPU 设备的编号，例如`0,1,2,3`，或者选择 CPU 设备，即`cpu`。
- `-weights`：初始化权重路径。
- `-cfg`：模型配置文件路径。
- `-data`：数据集配置文件路径。
- `-hyp`：超参数文件路径。
- `-epochs`：训练的总轮数。
- `-batch-size`：每个 GPU 设备上的 batch size 的大小，可以是`1`以自动计算 batch size。
- `-imgsz`：训练和验证图像的大小（像素）。
- `-optimizer`：选择优化器，可以是`SGD`、`Adam`或`AdamW`。
- `-single-cls`：将多类别数据视为单类别进行训练。
- `-image-weights`：在训练时使用加权图像选择。
- `-rect`：使用矩形域进行训练。
- `-cos-lr`：使用余弦退火学习率调度器。
- `-lr0`：初始学习率。
- `-lrf`：OneCycleLR 的最终学习率。
- `-momentum`：SGD 的动量或 Adam 的 beta1。
- `-weight-decay`：优化器的权重衰减。
- `-warmup-epochs`：学习率热身的轮数。
- `-warmup-momentum`：热身时的动量。
- `-warmup-bias_lr`：热身时的偏置学习率。
- `-box`：box 损失的权重。
- `-cls`：cls 损失的权重。
- `-cls-pw`：cls BCELoss 的正权重。
- `-obj`：obj 损失的权重（与像素缩放比例相关）。
- `-obj-pw`：obj BCELoss 的正权重。
- `-iou-t`: IOU 训练阈值。
- `-anchor-t`: 锚框多倍阈值。
- `-fl-gamma`: 聚焦损失伽马。
- `-label-smoothing`: 标签平滑。
- `-nbs`: 标准批次大小。
- `-overlap-mask`: 使用训练期间的掩模重叠。
- `-mask_ratio`: 设置掩模下采样。
- `-dropout`: 训练时使用 dropout。
- `-resume`: 恢复最近的训练。
- `-nosave`: 只保存最终的检查点。
- `-noval`: 只验证最终的轮数。
- `-noautoanchor`: 禁用 AutoAnchor。
- `-noplots`: 不保存绘图文件。
- `-evolve`: 演化超参数 x 代。
- `-bucket`: gsutil 存储桶。
- `-cache`: 图像缓存
- `-multi-scale`: 改变图像大小。
- `-sync-bn`: 使用 SyncBatchNorm，仅在 DDP 模式下可用。
- `-workers`: 每个 DDP 模式下的最大数据加载器工作进程数。
- `-project`: 保存到项目/名称。
- `-name`: 保存到项目/名称。
- `-exist-ok`: 现有项目/名称是否可用。
- `-quad`: 使用四重数据加载器。
- `-patience`: EarlyStopping 的等待轮数。
- `-freeze`: 冻结层。
- `-save-period`: 每 x 轮保存检查点。
- `-seed`: 全局训练种子。
- `-local_rank`: 自动 DDP 多 GPU 的参数。
- `-entity`: 实体名称。
- `-upload_dataset`: 上传数据集。
- `-bbox_interval`: 设置边界框图像记录间隔。
- `-artifact_alias`: 使用数据集版本名称。
    '''
    try:
        parser = argparse.ArgumentParser()

        parser.add_argument('--task', default='detect', help='select train task, i.e.  detect or classify, seg')
        parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--weights', type=str, default=ROOT / 'yolov8n.pt', help='initial weights path')
        parser.add_argument('--cfg', type=str, default=ROOT / 'models/yolov8/yolov8n.yaml', help='model.yaml path')
        parser.add_argument('--data', type=str, default=ROOT / 'data/618A.yaml', help='dataset.yaml path')
        parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
        parser.add_argument('--epochs', type=int, default=1, help='total training epochs')
        parser.add_argument('--batch-size','--batch',  type=int, default=4,
                            help='total batch size for all GPUs, -1 for autobatch')
        parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
        parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
        parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class.')
        parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training.')
        parser.add_argument('--rect', action='store_true', help='rectangular training.')
        parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler.')

        parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate.')
        parser.add_argument('--lrf', type=float, default=0.01, help='Final OneCycleLR learning rate.')
        parser.add_argument('--momentum', type=float, default=0.937, help='Use as momentum for SGD and beta1 for Adam.')
        parser.add_argument('--weight-decay', type=float, default=0.0005, help='Optimizer weight decay.')
        parser.add_argument('--warmup-epochs','--warmup_epochs', type=float, default=3.0, help='Warmup epochs. Fractions are ok.')
        parser.add_argument('--warmup-momentum', type=float, default=0.8, help='Warmup initial momentum.')
        parser.add_argument('--warmup-bias_lr', type=float, default=0.1, help='Warmup initial bias lr.')
        parser.add_argument('--box', type=float, default=0.05, help='Box loss gain.')
        parser.add_argument('--cls', type=float, default=0.5, help='cls loss gain.')
        parser.add_argument('--cls-pw', type=float, default=1.0, help='cls BCELoss positive_weight.')
        parser.add_argument('--obj', type=float, default=1.0, help='bj loss gain (scale with pixels).')
        parser.add_argument('--obj-pw', type=float, default=1.0, help='obj BCELoss positive_weight.')
        parser.add_argument('--iou-t', type=float, default=0.20, help='IOU training threshold.')
        parser.add_argument('--anchor-t', type=float, default=4.0, help='anchor-multiple threshold.')
        parser.add_argument('--fl-gamma', type=float, default=0.0, help='focal loss gamma.')
        parser.add_argument('--label-smoothing', type=float, default=0.0, help='label smoothing.')
        parser.add_argument('--nbs', type=int, default=64, help='nominal batch size.')

        parser.add_argument('--overlap-mask', default=True, action='store_true',
                            help='Segmentation: Use mask overlapping during training')
        parser.add_argument('--mask_ratio', type=float, default=4.0, help='Segmentation: Set mask downsampling.')
        parser.add_argument('--dropout', default=False, action='store_true',
                            help='Classification: Use dropout while training')

        parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
        parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
        parser.add_argument('--noval', action='store_true', help='only validate final epoch')
        parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
        parser.add_argument('--noplots', action='store_true', help='save no plot files')
        parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
        parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
        parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')

        parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
        parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
        parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
        parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
        parser.add_argument('--name', default='exp', help='save to project/name')
        # parser.add_argument('--project', default=ROOT / 'runs/detect', help='save to project/name')
        # parser.add_argument('--name', default='train', help='save to project/name')
        parser.add_argument('--exist-ok', default=False,action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--quad', action='store_true', help='quad dataloader')
        parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
        parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
        parser.add_argument('--save-period','--save_period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
        parser.add_argument('--seed', type=int, default=0, help='Global training seed')
        parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

        # Logger arguments
        parser.add_argument('--entity', default=None, help='Entity')
        parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
        parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
        parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

        return parser.parse_known_args()[0] if known else parser.parse_args()
    except Exception as e:
        # 记录异常信息
        logging.exception("【输入参数报错】")


def main(opt, callbacks=Callbacks()):
    '''这是一个函数main，可以用来训练模型。它有许多参数，例如resume，evolve，data，cfg，hyp，weights，project等。resume参数可以用来从之前的训练中恢复模型，evolve参数可以用来进行超参数优化，data，cfg，hyp，weights和project等参数可以用来指定数据集、模型配置文件、超参数文件、权重文件和项目名称。此外，函数还包括一些检查和错误处理的代码，以确保训练过程的正确性。如果LOCAL_RANK不等于-1，函数将以DDP模式运行，这意味着多个GPU将同时使用以提高训练速度。如果不进行超参数优化，函数将执行训练过程。如果进行超参数优化，函数将使用遗传算法来优化超参数，并输出优化结果。最后，函数将保存优化结果并绘制优化曲线。'''
    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))
        # check_git_status()
        run_start_log()
        check_requirements()

    # Resume (from specified or most recent last.pt)
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location='cpu')['opt']
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = '', str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    # device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        # device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, callbacks)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {
            'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)pi list
            'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
            'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
            'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
            'box': (1, 0.02, 0.2),  # box loss gain
            'cls': (1, 0.2, 4.0),  # cls loss gain
            'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
            'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
            'iou_t': (0, 0.1, 0.7),  # IoU training threshold
            'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
            'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
            'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
            'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
            'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
            'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
            'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
            'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
            'mixup': (1, 0.0, 1.0),  # image mixup (probability)
            'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3

        if opt.noautoanchor:
            del hyp['anchors'], meta['anchors']
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {evolve_csv}')  # download evolve.csv if exists

        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, callbacks)
            callbacks = Callbacks()
            # Write mutation results
            keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss',
                    'val/obj_loss', 'val/cls_loss')
            print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished {opt.evolve} generations\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Usage example: $ python train.py --hyp {evolve_yaml}')

def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt

def FileClean(path):
    '''清空文件夹'''
    # 确保路径存在
    if os.path.exists(path):
        # 循环遍历文件夹中的内容并删除
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print(f"{path} does not exist.")
def FileSetLimits(path=ROOT / 'runs/detect',num=100):
    '''文件超过上限数量则清空'''
    try:
        if len([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]) > num:
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                try:
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
    except Exception as e:
        # 记录异常信息
        logging.exception("【清理文件报错】")

def set_datasets(file):
    try:
        """修改datasets根地址"""
        username = os.getlogin()
        appdata_path = os.path.join(os.environ['USERPROFILE'], 'AppData', 'Roaming', 'Ultralytics','settings.yaml')
        print(f'AppData path for {username}: {appdata_path}')
        # Load the settings.yaml file
        with open(appdata_path, 'r',encoding='utf-8') as f:
            settings = yaml.safe_load(f)
        with open(file, 'r',encoding='utf-8') as f:
            yaml_path = yaml.safe_load(f)['path']
        # Modify the datasets_dir parameter
        num_dots = yaml_path.count('..')
        code_path = FILE.parents[0]
        for i in range(num_dots):
            code_path = code_path.parents[0]
        datasets_path = code_path / yaml_path.split('../')[-1]
        datasets_path = os.path.normpath(datasets_path)
        settings['datasets_dir'] = datasets_path
        # Save the modified settings.yaml file
        with open(appdata_path, 'w',encoding='utf-8') as f:
            yaml.dump(settings, f)
    except Exception as e:
        # 记录异常信息
        logging.exception("【修改datasets根地址报错】")
def set_xml2txt(file = 'data/618A.yaml',xml_file = r'../data/Annotations',txt_file = r'../data/labels'):
    try:
        '''xml转txt'''
        FileClean(txt_file)
        with open(file, 'r', encoding='utf-8') as f:
            data_yaml = yaml.load(f, Loader=yaml.FullLoader)
        class_names = []  # 标签类别
        for value in data_yaml['names'].values():
            class_names.append(value)
        name = glob.glob(os.path.join(xml_file, "*.xml"))
        for i in name:
            name_id = os.path.basename(i)[:-4]
            convert_annotation(name_id, class_names,xml_file,txt_file)
    except Exception as e:
        # 记录异常信息
        # logging.error('xml转txt文件报错：\n')
        logging.exception("【xml转txt文件报错】")

if __name__ == "__main__":
    # params = {
    #     "gpuOpenOrClose": 0 ,  # GPU开关
    #     "dataReadyOrNot": True,
    #     "initH5File": 'best.pt',
    #     "trainval_percent": 0.9,
    #     "last2LayersFrozenOrNot": True ,
    #     "last2LayersLearningRate": 0.001,
    #     "last2LayersEpochs": 3,
    #     "last2LayersBatchSize": 1,
    #     "allLayersFrozenOrNot": True,
    #     "allLayersLearningRate": 0.001,
    #     "allLayersEpochs": 30,
    #     "allLayersBatchSize": 2,
    #     "LearningDownRate": 0.1,
    #     "DownAboutEpochs": 3,
    #     "userId": "VisionMaker"
    # }
    # params = {
    #     "gpuOpenOrClose": 0 if (str(sys.argv[1]) == "True") else -1,  # GPU开关
    #     "dataReadyOrNot": True if (str(sys.argv[2]) == "True") else False,
    #     "initH5File": str(sys.argv[3]),
    #     "trainval_percent": float(sys.argv[4]),
    #     "last2LayersFrozenOrNot": True if (str(sys.argv[5]) == "True") else False,
    #     "last2LayersLearningRate": float(sys.argv[6]),
    #     "last2LayersEpochs": int(sys.argv[7]),
    #     "last2LayersBatchSize": int(sys.argv[8]),
    #     "allLayersFrozenOrNot": True if (str(sys.argv[9]) == "True") else False,
    #     "allLayersLearningRate": float(sys.argv[10]),
    #     "allLayersEpochs": int(sys.argv[11]),
    #     "allLayersBatchSize": int(sys.argv[12]),
    #     "LearningDownRate": float(sys.argv[13]),
    #     "DownAboutEpochs": int(sys.argv[14]),
    #     "userId": str(sys.argv[15])
    # }
    opt = parse_opt()
    opt.device = '0' if opt.device else 'cpu'
    FileSetLimits()
    FileSetLimits(ROOT / 'runs/train',99)
    set_datasets(opt.data)
    set_xml2txt(opt.data)
    main(opt)

