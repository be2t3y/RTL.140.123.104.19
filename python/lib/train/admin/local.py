import os

from lib.data_root import REPO_PYTHON, resolve_data_root

_PRJ = REPO_PYTHON


class EnvironmentSettings:
    def __init__(self):
        data = resolve_data_root()
        self.workspace_dir = _PRJ
        self.tensorboard_dir = os.path.join(_PRJ, 'tensorboard')
        self.pretrained_networks = os.path.join(_PRJ, 'pretrained_networks')
        self.lasot_dir = os.path.join(data, 'lasot')
        self.got10k_dir = os.path.join(data, 'got10k/train')
        self.got10k_val_dir = os.path.join(data, 'got10k/val')
        self.lasot_lmdb_dir = os.path.join(data, 'lasot_lmdb')
        self.got10k_lmdb_dir = os.path.join(data, 'got10k_lmdb')
        self.trackingnet_dir = os.path.join(data, 'trackingnet')
        self.trackingnet_lmdb_dir = os.path.join(data, 'trackingnet_lmdb')
        self.coco_dir = os.path.join(data, 'coco')
        self.coco_lmdb_dir = os.path.join(data, 'coco_lmdb')
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = os.path.join(data, 'vid')
        self.imagenet_lmdb_dir = os.path.join(data, 'vid_lmdb')
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.uav123_dir = os.path.join(data, 'uav123/UAV123')
