# -*- coding: utf-8 -*-
"""
ONLY WORKS WITH python 2.7 DUE TO DCMSTACK. "source activate ruber"
"""
import sys
import shutil
import subprocess
import argparse
import os.path as op
sys.path.append(op.dirname(op.dirname(op.abspath(__file__))))
from env import RAW_DATA, HEUDICONV_BIN, HEUDICONV_FOLDER, SESSION_TYPES


def electrodes_session_processing(sub, ses):

    try:
        data_dir = RAW_DATA + '/DICOM' + \
                   '/' + sub + '/' + ses \
                   + '/*/*/*'

        command = [
           HEUDICONV_BIN,
           "-d",
           data_dir,
           "-s",
           sub,
           "-ss",
           ses,
           "-f",
           op.join(HEUDICONV_FOLDER, 'convertall_electrodes.py'),
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
        if op.exists(op.join(RAW_DATA, 'bids', 'sub-' + sub,
                     'ses-' + ses, 'info')):
            shutil.rmtree(op.join(RAW_DATA, 'bids', 'sub-' + sub,
                          'ses-' + ses, 'info'))

    return


def normal_sesssion_processing(sub, ses):

    for i in range(6):
        try:
            data_dir = RAW_DATA + '/DICOM' + \
                       '/' + sub + '/' + ses \
                       + '/' \
                       + 'DICOM/*/*/*0' + \
                       str(i) + '/*'

            command = [
               HEUDICONV_BIN,
               '-d',
               data_dir,
               '-s',
               sub,
               '-ss',
               ses,
               '-f',
               op.join(HEUDICONV_FOLDER, 'convertall.py'),
               '-c',
               'dcm2niix',
               '-b',
               '-o',
               op.join(RAW_DATA, 'bids')
            ]

            output, error = subprocess.Popen(
                                command, universal_newlines=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE).communicate()

            shutil.rmtree(op.join(RAW_DATA, 'bids', 'sub-' + sub,
                          'ses-' + ses, 'info'))
        except:
            if op.exists(op.join(RAW_DATA, 'bids', 'sub-' + sub,
                         'ses-' + ses, 'info')):
                shutil.rmtree(op.join(RAW_DATA, 'bids', 'sub-' + sub,
                              'ses-' + ses, 'info'))
    return

if __name__ == "__main__":
    """Process list of dicoms and creates bids dirs

    Parameters
    ----------
    subject : list of str
      List of subjects to consider
    session : list of str
      List of session ids, corresponds to multiple sessions

    Returns
    -------
    Converts DICOMs to BIDS protocol
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subject', nargs='+',
                        help='subject id for heudiconv convertion',
                        required=True)
    parser.add_argument('-ses', '--session', nargs='+',
                        help='session id for heudiconv convertion',
                        required=True)

    args = parser.parse_args()
    sub_ses_comb = [[subject, session] for subject in args.subject
                    for session in args.session]

    for sub, ses in sub_ses_comb:

        if not op.exists(op.join(RAW_DATA, 'bids', 'sub-' + sub,
                                 'ses-' + ses)):
            print('Calculating: Subject ', sub, ' and session', ses)
            
            if ses == 'electrodes':
                electrodes_session_processing(sub, ses)
            elif ses in SESSION_TYPES:
                normal_sesssion_processing(sub, ses)
            else:
                print('Session type not defined')
                    
        else:
            print('Subject ', sub, ' and session', ses, ' already processed')
