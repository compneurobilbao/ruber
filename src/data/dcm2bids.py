# -*- coding: utf-8 -*-

from src.env import RAW_DATA
import shutil
import os
import os.path as op
import json
import subprocess


if __name__ == "__main__":

    subjects = os.listdir(op.join(RAW_DATA, 'DICOM'))
    
    try:
        dcm2bids = json.load(open(op.join(RAW_DATA, 'dcm2bidsdict')))
    except:
        dcm2bids = {}
    
    for idx, subject in enumerate(subjects, start=1):
        idx = '{0:03}'.format(idx)
        
        if not idx in dcm2bids.keys():
            dcm2bids[idx] = subject
            os.rename(op.join(RAW_DATA, 'DICOM', subject),
                      op.join(RAW_DATA, 'DICOM', idx))
            for i in range(6):
                try:
                    command = 'heudiconv -d \'' + RAW_DATA + '/DICOM' + \
                              '/{subject}/DICOM/*/*/*0' + str(i) + '/*\'' + ' -s ' + \
                              idx + ' -f ' + RAW_DATA + \
                              '/convertall.py -c dcm2niix -b -o ' + \
                              RAW_DATA + '/bids'
                    subprocess.check_output(command, shell=True)
                    shutil.rmtree(op.join(RAW_DATA,'bids', 'sub-'+idx, 'info'))
                except:
                    pass            
        else:
            print('Subject ', subject, ' already processed')

    
    
    json.dump(dcm2bids, open(op.join(RAW_DATA, 'dcm2bidsdict'),'w'))


    

heudiconv -d '/home/asier/git/ruber/data/raw/DICOM/{subject}/DICOM/*/*/*03/*' \
              -s 001 \
              -f /home/asier/git/ruber/data/raw/convertall.py \
              -c dcm2niix -b -o /home/asier/git/ruber/data/raw/bids
              
              
            