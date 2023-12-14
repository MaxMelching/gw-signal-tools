import os
import sys

psd_path = psd_path = os.path.dirname(__file__)


sys.path.append(os.path.join(psd_path, '../'))

from inner_product import psd_from_file_to_FreqSeries


psd_gw150914 = psd_from_file_to_FreqSeries(os.path.join(psd_path, 'GW150914_psd.txt'))
psd_o3_h1 = psd_from_file_to_FreqSeries(os.path.join(psd_path, 'O3_H1_asd.txt'), is_asd=True)
psd_o3_l1 = psd_from_file_to_FreqSeries(os.path.join(psd_path, 'O3_L1_asd.txt'), is_asd=True)
psd_o3_v1 = psd_from_file_to_FreqSeries(os.path.join(psd_path, 'O3_V1_asd.txt'), is_asd=True)
psd_sim = psd_from_file_to_FreqSeries(os.path.join(psd_path, 'sim_psd.txt'))