# [``ISpyMSI``](https://bitbucket.astrazeneca.com/projects/IMGGRP/repos/ispymsi/browse): Mass spectrometry tissue segmentation project.



## Installation
To install the repository, first clone it with
```bash
git clone git@github.com:AZU-BioPharmaceuticals-RD/ISpyMSI.git
cd path/to/repo/
pip install .
```
Note: it would be a good idea to do this inside a virtual Python env.


## Using this method

### Clone the repo
```bash
git clone ...
```

### Create the virtual environment

### Recreating the paper analysis (with your own data)

### Structuring your project
Create a directory, which will serve as the parent directory for your entire project. For the sake of this example, we will call it ``proj-parent``.

The structure should look like this:
```bash
├── H&E
├── MSI
└── metadata.csv
```

- The H&E folder should contain the histological whole-slide images.
- The MSI folder should contain your ``.imzML`` and ``.ibd`` files.
- The file ``metadata.csv`` should be a csv file with the fields: "he_img", "msi", "tissue_type", "organism", "msi_microns_pp", "dataset_id", "ion_mode", "split" and "Notes".
    - "dataset_id" can be anything, but integers make sense.
    - "split" should be either "train" or "test".
    - "msi_microns_pp" should be a float.


#### Labelling the slides: QuPath
In this work we used [``QuPath v0.4.3``](https://github.com/qupath/qupath/releases/tag/v0.4.3).

- Install QuPath, create a project, and add your H&Es. Your project dir might now look like:
```bash
├── H&E
├── MSI
├── metadata.csv
└── qupath-project
```
- Annotate the tissue, setting the class name to ``"tissue"``.
    - Note, if you have small fragments of tissue surrounding bigger fragments, group them as single annotation objects.
- Save the annotations as ``.geojson`` files by using the option available when you click "file".
    - Note, you _must_ uncheck all of the boxes before saving.
- Save the to a folder called "ROI". Now your project should look like
```bash
├── H&E
├── MSI
├── ROI
├── metadata.csv
└── qupath-project
```

#### Prepare your data

#### Extract information from the MSI files
Run
```bash
./scripts/extract_mass_spec_info.py /path/to/proj-parent
```
where the path is to the folder containing your ``.imzML`` and ``.ibd`` files. This will extract ion images from the MSI, and after it runs, your project directory should now look like
```bash
├── H&E
├── MSI
├── ROI
├── ion-imgs
├── metadata.csv
└── qupath-project
```


#### Record landmarks
Run
```bash
./scripts/record_landmarks.py /path/to/proj-parent
```
where the directory is the parent directory where all of data for this project are saved. Recording the landmarks should be very self explanatory, and boring. The landmarks will be saved as a csv file in the current working directory.

#### Project masks

```bash
./scripts/project_masks.py /project/base/proj-parent/
```
Again, the path is to the parent directory for the project. Remember to use the ``--help`` argument to look at all of the command-line options.

After completing this step, your folder should look like
```bash
├── H&E
├── MSI
├── ROI
├── images-and-masks
├── ion-imgs
├── metadata.csv
└── qupath-project
```


#### Extract patches
To extract the patches, run
```bash
./scripts/extract_patches.py /path/to/proj-parent/images-and-masks/
```
Again, use ``--help`` for more info.


#### Training the model
To train the model, run
```bash
./scripts/train_segmentation_model.py /patch/parent/dir/
```
It is strongly recommended to run this with ``--help`` first.


#### Testing the model
To test the model, on unseen data, first run
```bash
./scripts/test_segmentation_model.py --help
```
and then proceed as you see fit.