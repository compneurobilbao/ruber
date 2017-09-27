# First of all, copy the DICOM dir to /ruber/data/raw/DICOM/XXX/{presurg/postsurg/electrodes/}/

heudiconv -d "/home/asier/git/ruber/data/raw/DICOM/{subject}/DICOM/*/*/*02/*" \
-s 001 -f /home/asier/git/ruber/data/raw/convertall.py \
-c dcm2niix -b -o /home/asier/git/ruber/data/raw/bids


#subject 2

heudiconv -d "/home/asier/git/ruber/data/raw/DICOM/{subject}/DICOM/*/*/*02/*" \
-s 001 -f /home/asier/git/ruber/data/raw/convertall.py \
-c dcm2niix -b -o /home/asier/git/ruber/data/raw/bids