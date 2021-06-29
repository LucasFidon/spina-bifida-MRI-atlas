import os
import time
from typing import Sequence
import numpy as np
from argparse import ArgumentParser
from src.data.study import Study
from src.data.landmarks import get_lmks_data
from src.utils.definitions import MAX_GA_DIFF, MIN_CASES_PER_GA, N_ITER, ATLAS_SAVE_FOLDER, USE_RIGHT_LEFT_FLIP, GA_STD, LMKS_NAMES
from src.utils.maths import gaussian_kernel
from src.registration.procrustes import WeightedGeneralizedProcrustes

parser = ArgumentParser("Create the atlas template volume for the specified GA")
parser.add_argument('--ga', type=int, required=True)  #todo allow for list of ga
parser.add_argument('--no_landmarks', action='store_true')
parser.add_argument('--debug', action='store_true', help='Do not delete the intermediate results')


def register_all_to_template(study_list: Sequence[Study], template: Study,
                             save_folder: str, use_landmarks: bool = True, lp=3):
    warped_studies = []
    if not use_landmarks:
        print('Warning! Landmarks are not used')

    for study in study_list:
        # Register the study to the template
        save_case_path = os.path.join(save_folder, study.name)
        out_name = '%s_to_%s' % (study.name, template.name)
        if use_landmarks:
            warped_study = Study.register(
                study, template, out_folder=save_case_path, out_name=out_name, lp=lp)
        else:
            warped_study = Study.register(
                study, template, out_folder=save_case_path, out_name=out_name, lmks_weight=0, lp=lp)
        warped_studies.append(warped_study)

    return warped_studies


def initialize_template_linear(study_list: Sequence[Study], save_folder: str,
                               template_name: str, target_GA: int):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Find the case with the brain volume the closest to the mean brain volume
    brain_volumes = []
    mean_brain_volume = 0
    w_sum = 0.
    for s in study_list:
        brain_vol = np.sum(s.mask > 0)
        w = gaussian_kernel(s.ga, target_ga=target_GA, std=GA_STD)
        brain_volumes.append(brain_vol)
        mean_brain_volume += w * brain_vol
        w_sum += w
    mean_brain_volume /= w_sum
    ref_idx = np.argmin(np.abs(np.array(brain_volumes) - mean_brain_volume))

    # Register all the studies linearly
    warped_studies = []
    for i, s in enumerate(study_list):
        out_folder = os.path.join(save_folder, s.name)
        warped_s = Study.register_linear(
            s, study_list[ref_idx], out_name=s.name, out_folder=out_folder)
        warped_studies.append(warped_s)

    # Compute the average
    out_folder = os.path.join(save_folder, template_name)
    init_template = Study.average(warped_studies, name=template_name, save_folder=out_folder, target_GA=target_GA)
    return init_template


def initialize_template_procrustes(study_list: Sequence[Study], save_folder: str,
                                   template_name: str, target_GA: int):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Create the landmarks matrix for the Procrustes method
    lmks = np.stack([s.lmks.numpy() for s in study_list], axis=0)  # N,K,D
    num_lmks = lmks.shape[1]

    # Create the weights matrix for the Procrustes method
    weight_list = []
    for s in study_list:
        w = np.array([gaussian_kernel(s.ga, target_ga=target_GA, std=GA_STD)] * num_lmks)
        weight_list.append(w)
    weights = np.stack(weight_list, axis=0)

    # Compute the affine transformation for all the studies
    # using the weighted generalized procrustes
    procrustes = WeightedGeneralizedProcrustes(landmarks=lmks, weights=weights, mode='scaling')
    procrustes.solve(max_iter=10)

    # Apply the affine transformations
    warped_studies = []
    for i, s in enumerate(study_list):
        out_folder = os.path.join(save_folder, s.name)
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        # Save the affine in the NyftyReg format
        m_linear = procrustes.linear[i,:,:]
        translation = procrustes.translation[i,:]
        aff_txt = os.path.join(out_folder, 'affine.txt')
        with open(aff_txt, 'w') as f:
            for l in range(3):
                line = '%f %f %f %f\n' % (m_linear[l,0], m_linear[l,1], m_linear[l,2], translation[l])
                f.write(line)
            f.write('0 0 0 1')

        # Apply the affine transformation
        warp_s = Study.warp(study=s, trans=aff_txt, out_folder=out_folder)
        warped_studies.append(warp_s)

    # Compute the average
    out_folder = os.path.join(save_folder, template_name)
    init_template = Study.average(warped_studies, name=template_name, save_folder=out_folder, target_GA=target_GA)
    return init_template


def create_template(study_list: Sequence[Study], save_folder: str, template_name: str, target_GA: int):
    template_study = Study.average(
        study_list=study_list,
        name=template_name,
        save_folder=save_folder,
        target_GA=target_GA,  # must be in days
    )
    return template_study


def group_studies(save_folder_per_ga):
    def get_ga_days(study_name):
        data_s = data[data['File Name/Study'] == study_name]
        ga = data_s['Gestational Age'].values[0].split(', ')
        ga_days = 7 * int(ga[0]) + int(ga[1])
        return ga_days

    def get_op_status(study_name):
        data_s = data[data['File Name/Study'] == study_name]
        op_code = int(data_s['Unnamed: 1'].values[0])
        if op_code == 0:
            return 'notoperated'
        elif op_code == 1:
            return 'operated'
        else:
            raise ValueError('Unknown operation code', op_code)

    ga_list = list(save_folder_per_ga.keys())
    studies_dict = {
        i: {'notoperated': [], 'operated': []} for i in ga_list
    }

    data = get_lmks_data()
    study_names = [
        n for n in data['File Name/Study'].to_list()
        if n == n and 'Study' in n
    ]

    print('\nLoad all the studies...')
    t0 = time.time()
    for s in study_names:
        ga_s = get_ga_days(s)  # in days
        for ga in ga_list:
            if np.abs(7*ga - ga_s) <= MAX_GA_DIFF:
                op_status = get_op_status(s)
                inputs_f = os.path.join(save_folder_per_ga[ga][op_status], 'inputs')
                if not os.path.exists(inputs_f):
                    os.mkdir(inputs_f)
                study_folder = os.path.join(inputs_f, s)
                study = Study.from_study_name(
                    study_name=s, df=data, save_folder=study_folder)
                studies_dict[ga][op_status].append(study)

                # Right/left flip data augmentation
                if USE_RIGHT_LEFT_FLIP:
                    flip_study_name = '%s_flip' % s
                    flip_study_folder = os.path.join(inputs_f, flip_study_name)
                    flip_study = Study.flip(
                        study=study, name=flip_study_name, save_folder=flip_study_folder)
                    studies_dict[ga][op_status].append(flip_study)

    for ga in ga_list:
        for op_status in ['notoperated', 'operated']:
            print('Found %d cases for %s ga=%d weeks' %
                  (len(studies_dict[ga][op_status]), op_status, ga))

    delta_t = time.time() - t0
    print('It took %.0fsec to load all the studies' % delta_t)

    return studies_dict


def check_inclusion_group(study_list, target_GA):
    to_include = True

    # Check minimum number of studies.
    if len(study_list) < MIN_CASES_PER_GA:
        to_include = False

    # Check that there are GA on both sides of target_GA (in days).
    # We don't want to extrapolate.
    lower = False
    higher = False
    for s in study_list:
        if s.ga <= target_GA:
            lower = True
        if s.ga >= target_GA:
            higher = True
    if not lower or not higher:
        to_include = False

    return to_include



def main(args):
    ga_list = [args.ga]
    if not os.path.exists(ATLAS_SAVE_FOLDER):
        os.mkdir(ATLAS_SAVE_FOLDER)

    # Set the folders where to save results
    save_folders = {
        ga: {
            'notoperated': os.path.join(ATLAS_SAVE_FOLDER, 'fetal_SB_atlas_GA%d_notoperated' % ga),
            'operated': os.path.join(ATLAS_SAVE_FOLDER, 'fetal_SB_atlas_GA%d_operated' % ga),
        }
        for ga in ga_list
    }
    for ga in ga_list:
        for op_status in ['notoperated', 'operated']:
            if os.path.exists(save_folders[ga][op_status]):
                os.system('rm -r %s' % save_folders[ga][op_status])
            os.mkdir(save_folders[ga][op_status])

    # Get the studies per GA
    studies_per_ga = group_studies(save_folders)

    # Create atlases for each GA in args.ga (if enough studies)
    for ga in ga_list:
        for op_status in ['notoperated', 'operated']:

            # Check inclusion criteria
            to_include = check_inclusion_group(studies_per_ga[ga][op_status], 7 * ga)
            if not to_include:
                print('\nSkip ga=%d weeks for %s' % (ga, op_status))
                print('Inclusion criteria not matched')
                os.system('rm -r %s' % save_folders[ga][op_status])
                continue

            print('\nCreate the atlas for %s ga=%d weeks (%d cases)' %
                  (op_status, ga, len(studies_per_ga[ga][op_status])))
            studies = studies_per_ga[ga][op_status]
            warped_studies = studies

            # Iterate for the creation of the template
            for iter in range(N_ITER):
                t0 = time.time()
                print('\n*** Iter %d' % iter)
                iter_folder = os.path.join(save_folders[ga][op_status], 'iter%d' % iter)
                if not os.path.exists(iter_folder):
                    os.mkdir(iter_folder)

                # Update template
                if iter == 0:
                    print('Initialize the template')
                    template_n = 'template_iter%d' % iter
                    save_folder = os.path.join(save_folders[ga][op_status], 'initialization')
                    if args.no_landmarks:  # Initialization with a reference template
                        template = initialize_template_linear(
                            study_list=warped_studies,
                            save_folder=save_folder,
                            template_name=template_n,
                            target_GA=7*ga,
                        )
                    else:  # Initialization using procrustes method
                        template = initialize_template_procrustes(
                            study_list=warped_studies,
                            save_folder=save_folder,
                            template_name=template_n,
                            target_GA=7*ga,
                        )
                else:
                    print('Update the template...')
                    template_n = 'template_iter%d' % iter
                    template_folder = os.path.join(iter_folder, template_n)
                    template = create_template(
                        study_list=warped_studies,
                        save_folder=template_folder,
                        template_name=template_n,
                        target_GA=7*ga,
                    )

                # Update warped_studies
                print('Register all the studies to the new template...')
                warped_studies = register_all_to_template(
                    study_list=studies,
                    template=template,
                    save_folder=iter_folder,
                    use_landmarks=not(args.no_landmarks),
                    lp=iter + 1,
                )
                delta_t = int(time.time() - t0)
                min = delta_t // 60
                sec = delta_t % 60
                print('Iter %d took %dmin%dsec' % (iter, min, sec))

            print('Compute the final template')
            template_n = 'SB_template_GA%d' % ga
            _ = create_template(
                study_list=warped_studies,
                save_folder=save_folders[ga][op_status],
                template_name=template_n,
                target_GA=7*ga,
            )

            if not args.debug:
                # Delete the intermediate output
                for f in ['inputs', 'initialization'] + ['iter%d' % i for i in range(N_ITER)]:
                    folder = os.path.join(save_folders[ga][op_status], f)
                    if os.path.exists(folder):
                        os.system('rm -r %s' % folder)


if __name__ == '__main__':
    t0 = time.time()
    args = parser.parse_args()
    if args.no_landmarks:
        ATLAS_SAVE_FOLDER = '%s_noLandmarks' % ATLAS_SAVE_FOLDER
    main(args)
    delta_t = time.time() - t0
    delta_min = int(delta_t / 60)
    hour = delta_min // 60
    min = delta_min % 60
    print('\nTotal time: %dh %dmin' % (hour, min))
