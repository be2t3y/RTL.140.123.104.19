import os

from lib.test.evaluation.environment import EnvSettings

_PRJ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_DATA = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data'
_OUT = os.path.join(_PRJ, 'output_vit')


def local_env_settings():
    settings = EnvSettings()

    settings.davis_dir = ''
    settings.dtb70_path = os.path.join(_DATA, 'dtb70/DTB70')
    settings.got10k_lmdb_path = os.path.join(_DATA, 'got10k_lmdb')
    settings.got10k_path = os.path.join(_DATA, 'got10k')
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = os.path.join(_DATA, 'itb')
    settings.lasot_extension_subset_path_path = os.path.join(_DATA, 'lasot_extension_subset')
    settings.lasot_lmdb_path = os.path.join(_DATA, 'lasot_lmdb')
    settings.lasot_path = os.path.join(_DATA, 'lasot')
    settings.network_path = os.path.join(_OUT, 'test/networks')
    settings.nfs_path = os.path.join(_DATA, 'nfs')
    settings.otb_path = os.path.join(_DATA, 'otb')
    settings.prj_dir = _PRJ
    settings.result_plot_path = os.path.join(_OUT, 'test/result_plots')
    settings.results_path = os.path.join(_OUT, 'test/tracking_results')
    settings.save_dir = _OUT
    settings.segmentation_path = os.path.join(_OUT, 'test/segmentation_results')
    settings.tc128_path = os.path.join(_DATA, 'TC128')
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = os.path.join(_DATA, 'tnl2k')
    settings.tpl_path = ''
    settings.trackingnet_path = os.path.join(_DATA, 'trackingnet')
    settings.uav123_10fps_path = os.path.join(_DATA, 'uav123_10fps/UAV123_10fps')
    settings.uav123_path = os.path.join(_DATA, 'uav123/UAV123')
    settings.uav_path = os.path.join(_DATA, 'uav')
    settings.uavdt_path = os.path.join(_DATA, 'uavdt/home/data/uavdt')
    settings.uavtrack_path = os.path.join(_DATA, 'uavtrack112/home/data/V4RFlight112')
    settings.visdrone_path = os.path.join(_DATA, 'visdrone/VisDrone2018-SOT-test-dev')
    settings.vot18_path = os.path.join(_DATA, 'vot2018')
    settings.vot22_path = os.path.join(_DATA, 'vot2022')
    settings.vot_path = os.path.join(_DATA, 'VOT2019')
    settings.youtubevos_dir = ''

    return settings
