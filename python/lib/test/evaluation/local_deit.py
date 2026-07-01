import os

from lib.data_root import REPO_PYTHON, resolve_data_root
from lib.test.evaluation.environment import EnvSettings

from lib.data_root import REPO_PYTHON, resolve_data_root

_PRJ = REPO_PYTHON
_OUT = os.path.join(_PRJ, 'output')


def local_env_settings():
    data = resolve_data_root()
    settings = EnvSettings()

    settings.davis_dir = ''
    settings.dtb70_path = os.path.join(data, 'dtb70/DTB70')
    settings.got10k_lmdb_path = os.path.join(data, 'got10k_lmdb')
    settings.got10k_path = os.path.join(data, 'got10k')
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = os.path.join(data, 'itb')
    settings.lasot_extension_subset_path_path = os.path.join(data, 'lasot_extension_subset')
    settings.lasot_lmdb_path = os.path.join(data, 'lasot_lmdb')
    settings.lasot_path = os.path.join(data, 'lasot')
    settings.network_path = os.path.join(_OUT, 'test/networks')
    settings.nfs_path = os.path.join(data, 'nfs')
    settings.otb_path = os.path.join(data, 'otb')
    settings.prj_dir = _PRJ
    settings.result_plot_path = os.path.join(_OUT, 'test/result_plots')
    settings.results_path = os.path.join(_OUT, 'test/tracking_results')
    settings.save_dir = _OUT
    settings.segmentation_path = os.path.join(_OUT, 'test/segmentation_results')
    settings.tc128_path = os.path.join(data, 'TC128')
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = os.path.join(data, 'tnl2k')
    settings.tpl_path = ''
    settings.trackingnet_path = os.path.join(data, 'trackingnet')
    settings.uav123_10fps_path = os.path.join(data, 'uav123_10fps/UAV123_10fps')
    settings.uav123_path = os.path.join(data, 'uav123/UAV123')
    settings.uav_path = os.path.join(data, 'uav')
    settings.uavdt_path = os.path.join(data, 'uavdt/home/data/uavdt')
    settings.uavtrack_path = os.path.join(data, 'uavtrack112/home/data/V4RFlight112')
    settings.visdrone_path = os.path.join(data, 'visdrone/VisDrone2018-SOT-test-dev')
    settings.vot18_path = os.path.join(data, 'vot2018')
    settings.vot22_path = os.path.join(data, 'vot2022')
    settings.vot_path = os.path.join(data, 'VOT2019')
    settings.youtubevos_dir = ''

    return settings
