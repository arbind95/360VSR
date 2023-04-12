#!/bin/bash
 
# Options SBATCH :
#SBATCH --job-name=NoFeat      # Job Name
#SBATCH --gpus=1        
#SBATCH --partition=gpu          # Name of the Slurm partition used
#SBATCH --nodelist=boromir   

module purge
# module load pytorch
# module load opencv
module load python
# pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html --user
 
python3 test.py --select 0 --vid '360degreevideocrosscountryskiing-Scene-003_frames' > UHD_test_BI_epoch145_360degreevideocrosscountryskiing-Scene-003_frames.log
python3 test.py --select 0 --vid '360° SunsetTimelapseftevolv360VideoVirtualRealityRelaxationExperience-Scene-006_frames' > UHD_test_BI_epoch145_360° SunsetTimelapseftevolv360VideoVirtualRealityRelaxationExperience-Scene-006_frames.log
python3 test.py --select 0 --vid '2016BelmontChristmasParade360videoRideAlong!Part1-Scene-002_frames' > UHD_test_BI_epoch145_2016BelmontChristmasParade360videoRideAlong!Part1-Scene-002_frames.log
python3 test.py --select 0 --vid 'Birds-Scene-001_frames' > UHD_test_BI_epoch145_Birds-Scene-001_frames.log
python3 test.py --select 0 --vid 'Welder-Scene-001_frames' > UHD_test_BI_epoch145_Welder-Scene-001_frames.log
python3 test.py --select 0 --vid 'G4DrewBrees-Scene-001_frames' > UHD_test_BI_epoch145_G4DrewBrees-Scene-001_frames.log
python3 test.py --select 0 --vid 'G5Neighborhood-Scene-001_frames' > UHD_test_BI_epoch145_G5Neighborhood-Scene-001_frames.log
# python3 test.py --select 0 --vid 'G4Alcatraz-Scene-001_frames' > UHD_test_BI_epoch145_G4Alcatraz-Scene-001_frames.log