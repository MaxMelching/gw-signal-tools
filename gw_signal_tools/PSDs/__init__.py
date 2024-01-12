from os.path import dirname as _path_dirname

PSD_PATH = _path_dirname(__file__)

from os.path import join as _path_join
from ..inner_product import psd_from_file_to_FreqSeries as _psd_reader

# TODO: put PSDs sampled at 0.625Hz in there? Or even 0.0125 like O3 ones?
# Otherwise we have resampling each time function is called?
# Maybe do that in file, could also be good way to handle it


psd_gw150914 = _psd_reader(_path_join(PSD_PATH, 'GW150914_psd.txt'), name='PSD around GW150914')
psd_o3_h1 = _psd_reader(_path_join(PSD_PATH, 'O3_H1_asd.txt'), is_asd=True, name='Typical PSD of Hanford detector in O3')
psd_o3_l1 = _psd_reader(_path_join(PSD_PATH, 'O3_L1_asd.txt'), is_asd=True, name='Typical PSD of Livingston detector in O3')
psd_o3_v1 = _psd_reader(_path_join(PSD_PATH, 'O3_V1_asd.txt'), is_asd=True, name='Typical PSD of VIRGO detector in O3')
psd_sim = _psd_reader(_path_join(PSD_PATH, 'sim_psd.txt'), name='PSD values as simulated by `SimNoisePSDaLIGOZeroDetHighPower`')
psd_no_noise = _psd_reader(_path_join(PSD_PATH, 'no_psd.txt'), name='PSD values that indicate no noise being present')

# TODO: decide if names shall be capital, indicating they are constants
# PSD_GW_150914 = _psd_reader(_path_join(PSD_PATH, 'GW150914_psd.txt'), name='PSD around GW150914')
# PSD_O3_H1 = _psd_reader(_path_join(PSD_PATH, 'O3_H1_asd.txt'), is_asd=True, name='Typical PSD of Hanford detector in O3')
# PSD_O3_L1 = _psd_reader(_path_join(PSD_PATH, 'O3_L1_asd.txt'), is_asd=True, name='Typical PSD of Livingston detector in O3')
# PSD_O3_V1 = _psd_reader(_path_join(PSD_PATH, 'O3_V1_asd.txt'), is_asd=True, name='Typical PSD of VIRGO detector in O3')
# PSD_SIM = _psd_reader(_path_join(PSD_PATH, 'sim_psd.txt'), name='PSD values as simulated by `SimNoisePSDaLIGOZeroDetHighPower`')
# PSD_NO_NOISE = _psd_reader(_path_join(PSD_PATH, 'no_psd.txt'), name='PSD values that indicate no noise being present')