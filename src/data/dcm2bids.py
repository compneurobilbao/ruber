#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import os.path as op
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import RAW_DATA, HEUDICONV_BIN, HEUDICONV_FOLDER
import shutil
import subprocess
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", nargs='+',
                        help="subject id for heudiconv convertion",
                        required=True)
    parser.add_argument("-ses", "--session", nargs='+',
                        help="session id for heudiconv convertion",
                        required=True)

    args = parser.parse_args()
    sub_ses_comb = [[subject, session] for subject in args.subject
                    for session in args.session]

    for sub, ses in sub_ses_comb:

        if not op.exists(op.join(RAW_DATA, 'bids', 'sub-' + sub,
                                 'ses-' + ses)):
            print('Calculating: Subject ', sub, ' and session', ses)

            for i in range(6):
                try:
                    data_dir = RAW_DATA + '/DICOM' + \
                               '/' + sub + '/' + ses \
                               + '/' \
                               + 'DICOM/*/*/*0' + \
                               str(i) + '/*'

                    command = [
                       HEUDICONV_BIN,
                       "-d",
                       data_dir,
                       "-s",
                       sub,
                       "-ss",
                       ses,
                       "-f",
                       op.join(HEUDICONV_FOLDER, 'convertall.py'),
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

                    shutil.rmtree(op.join(RAW_DATA, 'bids', 'sub-' + sub,
                                 'ses-' + ses, 'info'))
                except:
                    print(error)
                    if op.exists(op.join(RAW_DATA, 'bids', 'sub-' + sub,
                                 'ses-' + ses, 'info')):
                        shutil.rmtree(op.join(RAW_DATA, 'bids', 'sub-' + sub,
                                 'ses-' + ses, 'info'))
        else:
            print('Subject ', sub, ' and session', ses, ' already processed')

