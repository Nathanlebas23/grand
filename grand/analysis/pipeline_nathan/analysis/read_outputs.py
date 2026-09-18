import sys
sys.path.append("/home/lpnhe/grand")

from grand.dataio import TShower

shower = TShower("/home/lpnhe/grand/grand/analysis/pipeline_nathan/outputs/GP80_20260915_095047_RUN10444_CD_20dB-GP65-58DUs-512trace-FY2Float-Normal-newcsdaq-3steps-timedriven-CD-100000-14/shower_20260918_143047_0-0_L1_0000.root")
print("TShower:", shower.get_list_of_events())


