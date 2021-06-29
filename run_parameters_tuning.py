import csv
import pandas as pd
import time
import numpy as np
import nibabel as nib
from src.data.landmarks import get_lmks_data
from src.evaluation.metrics import mean_dice_score
from src.registration.tools import register, warp_img
from src.data.landmarks import Landmarks
from src.utils.definitions import *

RES_EXCEL_PATH = os.path.join(REPO_PATH, 'data', 'parameters_tuning.csv')
COLUMNS = ['use landmarks', 'landmarks weight', 'le', 'be', 'loss', 'loss param', 'grid spacing', 'mean dice']
MAX_GA_DIFF = 3  # in days
MAXIT = 300
GRID_SPACING = 3
BE_LIST = [0.001, 0.01, 0.03, 0.1, 0.3]
LE_LIST = [0.01, 0.03, 0.1]
LMKS_W_LIST = [0.0003, 0.001, 0.003]
LNCC_SCALE_LIST = [1, 2, 4, 6, 8]


def load_csv_results():
    if not os.path.exists(RES_EXCEL_PATH):
        print('Parameters tuning csv %s not found.' % RES_EXCEL_PATH)
        return None
    data = pd.read_csv(RES_EXCEL_PATH)
    return data


def are_params_already_evaluated(params):
    data = load_csv_results()
    if data is None:
        return False
    for k in COLUMNS[:(-1)]:
        data = data[data[k] == params[k]]
        if len(data) == 0:
            return False
    return True


def update_csv(params, dice_val):
    row = [params[k] for k in COLUMNS[:(-1)]] + [dice_val]
    if not os.path.exists(RES_EXCEL_PATH):
        print('Create %s' % RES_EXCEL_PATH)
        with open(RES_EXCEL_PATH, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            writer.writerow(row)
    else:
        with open(RES_EXCEL_PATH, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)


def eval(pairs, params_dict):
    print('\nStart evaluation ')
    print(params_dict)
    dsc = []
    save_folder_eval = 'tmp_param_selection'
    if not os.path.exists(save_folder_eval):
        os.mkdir(save_folder_eval)
    for imgs in pairs:
        save_folder = os.path.join(save_folder_eval, '%s_reg_to_%s' % (imgs[0], imgs[1]))
        if not os.path.exists(save_folder):
                os.mkdir(save_folder)
        img1_path = os.path.join(UCLH_MMC, imgs[0], 'srr_template.nii.gz')
        os.system('cp %s %s' % (img1_path, os.path.join(save_folder, '%s.nii.gz' % imgs[0])))
        mask1_path = os.path.join(UCLH_MMC, imgs[0], 'srr_template_mask.nii.gz')
        img2_path = os.path.join(UCLH_MMC, imgs[1], 'srr_template.nii.gz')
        os.system('cp %s %s' % (img2_path, os.path.join(save_folder, '%s.nii.gz' % imgs[1])))
        mask2_path = os.path.join(UCLH_MMC, imgs[1], 'srr_template_mask.nii.gz')
        if params_dict['use landmarks']:
            lmks1 = Landmarks.fromexcel(study_name=imgs[0], srr_path=img1_path)
            lmks2 = Landmarks.fromexcel(study_name=imgs[1], srr_path=img2_path)
        else:
            lmks1 = None
            lmks2 = None
        # Run the registration
        register(
            img1_path, img2_path,
            mask1_path=mask1_path, mask2_path=mask2_path,
            lmks1=lmks1, lmks2=lmks2,
            save_folder=save_folder,
            verbose=False,
            grid_spacing=params_dict['grid spacing'],
            be=params_dict['be'],
            le=params_dict['le'],
            lmks_weight=params_dict['landmarks weight'],
            loss=params_dict['loss'],
            loss_param=params_dict['loss param'],
            maxit=MAXIT,
        )
        affine_out = os.path.join(save_folder, 'outputAffine.txt')
        cpp_out = os.path.join(save_folder, 'cpp.nii.gz')

        # Load and warp the segmentation
        seg1 = os.path.join(UCLH_MMC, imgs[0], 'parcellation.nii.gz')
        seg1_warped_aff = os.path.join(save_folder, 'seg_%s_warped_aff.nii.gz' % imgs[0])
        seg1_warped = os.path.join(save_folder, 'seg_%s_warped.nii.gz' % imgs[0])
        seg2 = os.path.join(UCLH_MMC, imgs[1], 'parcellation.nii.gz')
        warp_img(
            seg1,
            trans=affine_out,
            ref_img=img2_path,
            save_path=seg1_warped_aff,
            is_seg=True,
        )
        warp_img(
            seg1_warped_aff,
            trans=cpp_out,
            ref_img=img2_path,
            save_path=seg1_warped,
            is_seg=True,
        )
        seg1_np = nib.load(seg1_warped).get_fdata().astype(np.uint8)
        seg2_np = nib.load(seg2).get_fdata().astype(np.uint8)
        dice_val = mean_dice_score(seg1_np, seg2_np, labels_list=[1, 2, 3])
        dsc.append(dice_val)

    mean_dice_val = np.mean(dsc)
    return mean_dice_val


def get_pairs_to_register():
    #TODO replace pairs of study names by pairs of studies
    def get_ga_days(study_name):
        data_s = data[data['File Name/Study'] == study_name]
        ga = data_s['Gestational Age'].values[0].split(', ')
        ga_days = 7 * int(ga[0]) + int(ga[1])
        return ga_days
    def is_in_UCLH_MMC(study_name):
        name = study_name.replace('-', '')
        for f_name in os.listdir(UCLH_MMC):
            if name == f_name:
                return True
        return False
    pairs = []
    data = get_lmks_data()
    study_names = [
        n for n in data['File Name/Study'].to_list()
        if n == n and 'Study' in n
    ]
    UCLH_MMC_study_names = [
        n for n in study_names
        if is_in_UCLH_MMC(n)
    ]
    for i, s in enumerate(UCLH_MMC_study_names):
        ga = get_ga_days(s)  # in days
        for s2 in UCLH_MMC_study_names[i:]:
            if s2 == s:
                continue
            ga2 = get_ga_days(s2)
            # Condition of inclusion based on GA
            if np.abs(ga - ga2) > MAX_GA_DIFF:
                continue
            pairs.append([s, s2])
    return pairs


def get_params_list(be_list, le_list, lmks_w_list, lncc_w_list):
    params_list = []
    for be in be_list:
        for le in le_list:
            for loss in ['nmi', 'lncc']:
                for use_lmks in [True, False]:
                    if use_lmks:
                        for lmks_w in lmks_w_list:
                            if loss == 'lncc':
                                for lncc_w in lncc_w_list:
                                    params = {}
                                    params['grid spacing'] = GRID_SPACING
                                    params['use landmarks'] = use_lmks
                                    params['be'] = be
                                    params['le'] = le
                                    params['landmarks weight'] = lmks_w
                                    params['loss'] = loss
                                    params['loss param'] = lncc_w
                                    params_list.append(params)
                            else:  # NMI
                                params = {}
                                params['grid spacing'] = GRID_SPACING
                                params['use landmarks'] = use_lmks
                                params['be'] = be
                                params['le'] = le
                                params['landmarks weight'] = lmks_w
                                params['loss'] = loss
                                params['loss param'] = 0
                                params_list.append(params)
                    else:  # No Landmarks
                        if loss == 'lncc':
                                for lncc_w in lncc_w_list:
                                    params = {}
                                    params['grid spacing'] = GRID_SPACING
                                    params['use landmarks'] = use_lmks
                                    params['be'] = be
                                    params['le'] = le
                                    params['landmarks weight'] = 0.
                                    params['loss'] = loss
                                    params['loss param'] = lncc_w
                                    params_list.append(params)
                        else:  # NMI
                            params = {}
                            params['grid spacing'] = GRID_SPACING
                            params['use landmarks'] = use_lmks
                            params['be'] = be
                            params['le'] = le
                            params['landmarks weight'] = 0.
                            params['loss'] = loss
                            params['loss param'] = 0
                            params_list.append(params)
    return params_list


def main():
    pairs = get_pairs_to_register()
    print('Found %d pairs' % len(pairs))
    print(pairs)
    params_list = get_params_list(
        be_list=BE_LIST,
        le_list=LE_LIST,
        lmks_w_list=LMKS_W_LIST,
        lncc_w_list=LNCC_SCALE_LIST,
    )
    print('Found %d parameters values config to test' % len(params_list))
    for params in params_list:
        if are_params_already_evaluated(params):
            print('\nParameter values already evaluated. Skip')
            print(params)
            continue
        t0 = time.time()
        dice = eval(pairs, params)
        update_csv(params, dice)
        t1 = time.time()
        delta_t = int(t1 - t0)
        print('\nParameters selection performed in %dmin%dsec' % (delta_t // 60, delta_t % 60))
        print('Mean DSC=%.2f' % (100 * dice))


if __name__ == '__main__':
    main()
