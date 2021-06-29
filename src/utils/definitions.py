import os

# PATHS
HOME = '/home/lucasf'
REPO_PATH = os.path.join(HOME, 'workspace', 'FetalBrainRegistration')
NIFTYREG_PATH = os.path.join(HOME, 'workspace', 'niftyreg_stable', 'build', 'reg-apps')
ATLAS_FOLDER = os.path.join('data', 'fetal_brain_srr_parcellation_Nov20_atlas_partialseg')
DATA_DIR = os.path.join(HOME, 'data', 'Fetal_SRR_and_Seg', 'SRR_and_Seg_Nada_cases_group')
UCLH_MMC = os.path.join(DATA_DIR, 'UCLH_MMC')
UCLH_MMC_2 = os.path.join(DATA_DIR, 'UCLH_MMC_2')
LEUVEN_MMC = os.path.join(DATA_DIR, 'Leuven_MMC')

NUM_CLASSES = 9

# REG PARAMS
IMG_RES = 0.8  # in mm (isotropic)

# ATLAS PARAMS
ATLAS_SAVE_FOLDER = os.path.join(HOME, 'data', 'SB_atlas')
MEAN_INTENSITY = 2000.
STD_INTENSITY = 500.
MIN_CASES_PER_GA = 6
GA_STD = 3
MAX_GA_DIFF = 3 * GA_STD  # in days
GA_LIST = [i for i in range(21, 34)]
N_ITER = 3
USE_RIGHT_LEFT_FLIP = True

# LMKS
LMKS_PATH = os.path.join(
    REPO_PATH,
    'data',
    'RecordOfManualLandmarkSelection.xlsx',
)
LMKS_NAMES = [  # in order of the columns
    "Anterior Horn of the Right Lateral Ventricle",  # 1
    "Anterior Horn of the Left Lateral Ventricle",   # 2
    "Posterior Tectum edge",                         # 3
    "Left Cerebellar-Brainstem Junction",            # 4
    "Right Cerebellar-Brainstem Junction",           # 5
    "Left Deep Grey Border at Anterior CSP line",    # 6
    "Right Deep Grey Border at Anterior CSP line",   # 7
    "Left Deep Grey Border at Posterior CSP Line",   # 8
    "Right Deep Grey Border at Posterior CSP line",  # 9
    "Right Deep Grey Border at FOM",                 # 10
    "Left Deep Grey Border at FOM",                  # 11
]
LMKS_LABELS = {
    s: i + 1
    for i,s in enumerate(LMKS_NAMES)
}
UNRELIABLE_LMKS = [
    "Left Deep Grey Border at Anterior CSP line",    # 6
    "Right Deep Grey Border at Anterior CSP line",   # 7
    "Left Deep Grey Border at Posterior CSP Line",   # 8
    "Right Deep Grey Border at Posterior CSP line",  # 9
]
LIZZY_LMKS_LABELS = {
    "Anterior Horn of the Right Lateral Ventricle": 2,
    "Anterior Horn of the Left Lateral Ventricle": 3,
    "Posterior Tectum edge": 6,
    "Left Cerebellar-Brainstem Junction": 10,
    "Right Cerebellar-Brainstem Junction": 11,
    "Left Deep Grey Border at Anterior CSP line": 4,
    "Right Deep Grey Border at Anterior CSP line": 7,
    "Left Deep Grey Border at Posterior CSP Line": 8,
    "Right Deep Grey Border at Posterior CSP line": 9,
    "Right Deep Grey Border at FOM": 14,
    "Left Deep Grey Border at FOM": 13,
}

# ROTATIONS
ROTATIONS = {
    'tenplate057': os.path.join(REPO_PATH, 'data', 'RotationCorrection057tenplate.txt'),
    'tenplate332': os.path.join(REPO_PATH, 'data', 'RotationCorrection332tenplate.txt'),
    'tenplate360': os.path.join(REPO_PATH, 'data', 'tenplate360.txt'),
}
ROTATIONS_INV = {
    'tenplate057': os.path.join(REPO_PATH, 'data', 'RotationCorrection057tenplate_inv.txt'),
    'tenplate332': os.path.join(REPO_PATH, 'data', 'RotationCorrection332tenplate_inv.txt'),
    'tenplate360': os.path.join(REPO_PATH, 'data', 'tenplate360_inv.txt'),
}
