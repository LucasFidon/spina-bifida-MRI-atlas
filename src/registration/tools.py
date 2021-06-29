import yaml
import inspect
import numpy as np
import nibabel as nib
from src.utils.definitions import *
from src.data import landmarks

SUPPORTED_INTENSITY_SIM = [
    'nmi',
    'lncc',
]


def register_affine(img1_path, img2_path, save_folder,
                    mask1_path=None, mask2_path=None,
                    lmks1=None, seg1_path=None,
                    rig_only=False, verbose=False):
    affine_path = os.path.join(save_folder, 'outputAffine.txt')
    flo = os.path.join(save_folder, 'output_affine.nii.gz')
    affine_reg_cmd = '%s/reg_aladin -ref %s -flo %s -res %s -aff %s -pad 0' % \
        (NIFTYREG_PATH, img2_path, img1_path, flo, affine_path)
    if mask1_path is not None:
        affine_reg_cmd += ' -fmask %s' % mask1_path
    if mask2_path is not None:
        affine_reg_cmd += ' -rmask %s' % mask2_path
    if rig_only:
        affine_reg_cmd += ' -rigOnly'
    if not verbose:
        affine_reg_cmd += ' -voff'
    os.system(affine_reg_cmd)

    # Warp the mask
    if mask1_path is not None:
        flo_mask = os.path.join(save_folder, 'output_affine_mask.nii.gz')
        warp_img(mask1_path, affine_path, img2_path, flo_mask, is_seg=True)
    else:
        flo_mask = None

    # Warp the landmarks
    if lmks1 is not None:
        flo_lmks = landmarks.transform(lmks1, trans=affine_path, ref_img=img2_path)
        aff_flo_lmks_path = os.path.join(save_folder, 'output_affine_lmks.nii.gz')
        flo_lmks.save_as_seg(save_path=aff_flo_lmks_path)
    else:
        flo_lmks = None

    # Warp the segmentation
    if seg1_path is not None:
        aff_flo_seg_path = os.path.join(save_folder, 'output_affine_parcellation_softmax.nii.gz')
        warp_img(seg1_path, affine_path, img2_path, aff_flo_seg_path, is_proba_seg=True)
    else:
        aff_flo_seg_path = None

    return flo, flo_mask, flo_lmks, aff_flo_seg_path, affine_path

# for grid=2: be=0.03, le=0.1, lncc, loss_param=6, lmks_weight=0.001
def register(img1_path, img2_path, save_folder,
             mask1_path=None, mask2_path=None,
             lmks1=None, lmks2=None,
             seg1_path=None, seg2_path=None,
             lmks_weight=0.001,
             use_affine=True, grid_spacing=3, be=0.1, le=0.3, lp=3,
             loss='lncc', loss_param=6,
             maxit=300,
             save_output_landmarks=True,
             verbose=False):
    assert loss in SUPPORTED_INTENSITY_SIM, "Received loss name %s but expected one of %s" % \
        (loss, str(SUPPORTED_INTENSITY_SIM))

    if verbose:
        print('Use NiftyReg installed at %s\n' % NIFTYREG_PATH)

    # Create the save folder and save the registration parameters
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    dict_param = {
        i: values[i]
        for i in args if 'lmks' not in i
    }
    yaml_path = os.path.join(save_folder, 'parameters.yml')
    with open(yaml_path, 'w') as f:
        yaml.dump(dict_param, f)

    # Affine registration
    if use_affine:
        if verbose:
            print('Start with an affine registration...')
        flo, flo_mask, flo_lmks, flo_seg, affine_path = register_affine(
            img1_path=img1_path,
            img2_path=img2_path,
            save_folder=save_folder,
            mask1_path=mask1_path,
            mask2_path=mask2_path,
            lmks1=lmks1,
            seg1_path=seg1_path,
            verbose=verbose,
        )
    else:  # No affine transformation
        flo = img1_path
        flo_mask = mask1_path
        flo_lmks = lmks1
        flo_seg = seg1_path
        affine_path = None

    # Non-linear registration
    res_path = os.path.join(save_folder, 'output.nii.gz')
    cpp_path = os.path.join(save_folder, 'cpp.nii.gz')
    if loss == 'lncc':
        reg_loss_options = '-lncc 0 %d ' % loss_param
    else:  # NMI
        reg_loss_options = ''
    reg_options = '%s-be %f -le %f -sx %s -ln 3 -lp %d -pad 0 -maxit %d' % \
        (reg_loss_options, be, le, grid_spacing, lp, maxit)
    reg_cmd = '%s/reg_f3d -ref %s -flo %s -res %s -cpp %s %s' % \
        (NIFTYREG_PATH, img2_path, flo, res_path, cpp_path, reg_options)
    if flo_mask is not None:
        reg_cmd += ' -fmask %s' % flo_mask
    if mask2_path is not None:
        reg_cmd += ' -rmask %s' % mask2_path
    if flo_lmks is not None and lmks2 is not None:
        if verbose:
            print('Use the landmarks (floating):')
            print(flo_lmks)
            print('Use the landmarks (reference):')
            print(lmks2)
        lmks_txt = os.path.join(save_folder, 'lmks.txt')
        landmarks.write_lmks_file(lmks_ref=lmks2, lmks_flo=flo_lmks, save_path=lmks_txt)
        reg_cmd += ' -land %f %s' % (lmks_weight, lmks_txt)
    if not verbose:
        reg_cmd += ' -voff'
    os.system(reg_cmd)

    # Warp and save the mask (if present)
    if flo_mask is not None:
        save_flo_mask = os.path.join(save_folder, 'output_mask.nii.gz')
        warp_img(flo_mask, trans=cpp_path, ref_img=img2_path, save_path=save_flo_mask, is_seg=True)
        if verbose:
            print('The output mask was saved at', save_flo_mask)

    # Warp and save the parcellation (if present)
    if flo_seg is not None:
        save_flo_seg = os.path.join(save_folder, 'output_parcellation_softmax.nii.gz')
        warp_img(flo_seg, trans=cpp_path, ref_img=img2_path, save_path=save_flo_seg, is_proba_seg=True)
        if verbose:
            print('The output parcellation was saved at', save_flo_seg)

    # Warp and save the landmarks (if present)
    if flo_lmks is not None and save_output_landmarks:
        out_lmks = landmarks.transform(flo_lmks, trans=cpp_path, ref_img=img2_path)
        lmks_save_path = os.path.join(save_folder, 'output_lmks.nii.gz')
        out_lmks.save_as_seg(lmks_save_path)
        if verbose:
            print('The output landmarks were saved at', lmks_save_path)


def warp_img(img, trans, ref_img, save_path, is_seg=False, is_proba_seg=False):
    #todo: if is_seg convert to one-hot encoding and use linear interpolation
    cmd = '%s/reg_resample -ref %s -flo %s -trans %s -res %s -pad 0 -voff' % \
        (NIFTYREG_PATH, ref_img, img, trans, save_path)
    if is_seg:
        cmd += ' -inter 0'
    else:
        cmd += ' -inter 1'
    os.system(cmd)
    if is_proba_seg:
        # Set the padded values to background
        proba_nii = nib.load(save_path)
        out_proba = proba_nii.get_fdata().astype(np.float32)
        out_proba[out_proba.max(axis=-1) == 0, 0] = 1
        out_proba_nii = nib.Nifti1Image(out_proba, proba_nii.affine, proba_nii.header)
        nib.save(out_proba_nii, save_path)
