# -*- coding: utf-8 -*-

from src.env import RAW_DATA, HEUDICONV_BIN
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

        if (idx not in dcm2bids.keys() or
           not op.exists(op.join(RAW_DATA, 'bids', 'sub-'+idx))):

            dcm2bids[idx] = subject
            os.rename(op.join(RAW_DATA, 'DICOM', subject),
                      op.join(RAW_DATA, 'DICOM', idx))
            for i in range(6):
                try:
                    data_dir = RAW_DATA + '/DICOM' + \
                               '/{subject}/DICOM/*/*/*0' + str(i) + '/*'

                    command = [
                       HEUDICONV_BIN,
                       "-d",
                       data_dir,
                       "-s",
                       idx,
                       "-f",
                       op.join(RAW_DATA, 'convertall.py'),
                       "-c",
                       "dcm2niix",
                       "-b",
                       "-o",
                       op.join(RAW_DATA, 'bids')
                    ]

                    output, error = subprocess.Popen(
                                        command, universal_newlines=True,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE).communicate()

                    shutil.rmtree(op.join(RAW_DATA, 'bids', 'sub-'+idx, 'info'))
                except:
                    if op.exists(op.join(RAW_DATA, 'bids', 'sub-'+idx, 'info')):
                        shutil.rmtree(op.join(RAW_DATA, 'bids', 'sub-'+idx, 'info'))
        else:
            print('Subject ', subject, ' already processed')

    json.dump(dcm2bids, open(op.join(RAW_DATA, 'dcm2bidsdict'), 'w'))

