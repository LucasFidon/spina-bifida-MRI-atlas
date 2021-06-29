import time
import pandas as pd
import numpy as np
import nibabel as nib
import random
from src.utils.definitions import *
from src.utils.maths import gaussian_kernel

NROWS = 100  # number of rows that will be loaded


def get_lmks_data(sheet_name=0):
    print('\nRead the excel file...')
    assert sheet_name in [0,1], 'There are only two sheets'
    t0 = time.time()
    data = pd.read_excel(
        LMKS_PATH,
        engine='openpyxl',
        sheet_name=sheet_name,
        nrows=NROWS,
    )
    delta_t = time.time() - t0
    print('It took %.1fsec to read the excel file' % delta_t)
    return data

def write_lmks_file(lmks_ref, lmks_flo, save_path):
    """
    Save the landmarks in the format expected by reg_f3d
    i.e. a text file with line format:
    <refX> <refY> <refZ> <floX> <floY> <floZ>
    """
    assert isinstance(lmks_ref, Landmarks)
    assert isinstance(lmks_flo, Landmarks)
    # print('The following (unreliable) landmarks are not used during the registration')
    # print(UNRELIABLE_LMKS)
    with open(save_path, 'w') as f:
        for roi in LMKS_NAMES:
            if roi in UNRELIABLE_LMKS:
                continue
            elif not(lmks_ref.coord[roi] is None or lmks_flo.coord[roi] is None):
                # The landmarks positions are written by pair
                # corresponding to the same landmark in img1 and img2
                line = '%f %f %f %f %f %f\n' % \
                    (lmks_ref.coord[roi][0],
                    lmks_ref.coord[roi][1],
                    lmks_ref.coord[roi][2],
                    lmks_flo.coord[roi][0],
                    lmks_flo.coord[roi][1],
                    lmks_flo.coord[roi][2],
                    )
                f.write(line)

def transform(lmks, trans, ref_img):
    tmp = './tmp_lmks_for_transform_%s' % (os.path.basename(os.path.dirname(trans)))
    while os.path.exists(tmp):
        tmp = '%s%d' % (tmp, random.randint(1, 100))
    if not os.path.exists(tmp):
        os.mkdir(tmp)
    # Save the lmks as a seg
    seg = os.path.join(tmp, 'lmks_seg.nii.gz')
    lmks.save_as_seg(seg, dilation=True)
    # Warp the segmentation
    warped_seg = os.path.join(tmp, 'warped_lmks_seg.nii.gz')
    cmd = '%s/reg_resample -ref %s -flo %s -trans %s -res %s -inter 0 -pad 0 -voff' % \
        (NIFTYREG_PATH, ref_img, seg, trans, warped_seg)
    os.system(cmd)
    # Create a new Landmarks instance from the warped seg
    warped_lmks = Landmarks.fromseg(warped_seg, ga=lmks.ga)
    os.system('rm -r %s' % tmp)
    return warped_lmks


class Landmarks:
    def __init__(self, coord, affine, header, ga, rotation=None):
        # Check the coord dict contain coordinates
        # for all the anatomical landmarks defined by Lizzy
        # and only for them.
        for roi in LMKS_NAMES:
            assert roi in list(coord.keys())
        for roi in list(coord.keys()):
            assert roi in LMKS_NAMES
        self.affine = affine
        self.header = header
        self.coord = coord
        self.rotation = rotation
        self.ga=ga

    @classmethod
    def fromseg(cls, seg_path, ga):
        seg_nii = nib.load(seg_path)
        affine = seg_nii.affine
        header = seg_nii.header
        seg = seg_nii.get_fdata().astype(np.uint8)
        ori = affine[:3, 3].astype(np.float64)
        mat = affine[:3, :3].astype(np.float64)
        coord = {}
        for roi in LMKS_NAMES:
            x, y, z = np.where(seg == LMKS_LABELS[roi])
            if x.size == 0:  # landmark not present
                coord[roi] = None
            else:
                # Take the closest integer to mean coordinates
                # in case the segmentation is not a single point
                coord_vox = np.array([
                    np.mean(x), np.mean(y), np.mean(z)
                ]).astype(np.float64)
                coord_mm = mat.dot(coord_vox) + ori
                coord[roi] = coord_mm
        return cls(coord, affine, header, ga=ga)

    @classmethod
    def fromexcel(cls, study_name, srr_path, df=None):
        def get_ga_days():
            data_s = df[df['File Name/Study'] == study_name]
            ga = data_s['Gestational Age'].values[0].split(', ')
            ga_days = 7 * int(ga[0]) + int(ga[1])
            return ga_days

        if df is None:
            # Read data in the excel file
            df = get_lmks_data(sheet_name=0)
        raw_data = df[df['File Name/Study'] == study_name]

        # Nifti image data
        srr_nii = nib.load(srr_path)
        affine = srr_nii.affine
        header = srr_nii.header

        # Get the rotation used during the annotation
        rotation_key = raw_data['Unnamed: 6'].values[0]
        if rotation_key not in list(ROTATIONS.keys()):
            rotation_path = None
        else:
            rotation_path = ROTATIONS[rotation_key]

        # Load the landmarks (in mm)
        coord_dict = {}
        for roi in LMKS_NAMES:
            coord_str = raw_data[roi].values  # numpy array
            # Parse the coordinates
            if coord_str[0] != coord_str[0]:  # nan value
                coord_dict[roi] = None
            elif roi in UNRELIABLE_LMKS:  # exclude unreliable landmarks
                coord_dict[roi] = None
            else:
                # We need to substract 1 because in ITK-Snap
                # voxel indices start from 1 not 0...
                coord = np.array([int(s) - 1 for s in coord_str[0].split(',')]).astype(np.float64)
                ori = affine[:3, 3].astype(np.float64)
                mat = affine[:3, :3].astype(np.float64)
                coord_mm = mat.dot(coord) + ori
                coord_dict[roi] = coord_mm

        # Get the GA
        ga = get_ga_days()

        lmks = cls(coord_dict, affine, header, ga, rotation_path)

        return lmks

    def numpy(self):
        coord_list = []
        for roi in LMKS_NAMES:
            if self.coord[roi] is None:
                coord_list.append([np.nan] * 3)
            else:
                coord_list.append([
                    self.coord[roi][0], self.coord[roi][1], self.coord[roi][2]
                ])
        coord = np.array(coord_list)  # K,3
        return coord


    def save_as_txt(self, save_path):
        with open(save_path, 'w') as f:
            for roi in LMKS_NAMES:
                if not self.coord[roi] is None:
                    line = '%f %f %f\n' % \
                        (self.coord[roi][0], self.coord[roi][1], self.coord[roi][2])
                    f.write(line)

    def save_as_seg(self, save_path, dilation=False):
        # Save the landmarks as segmentation
        ori = self.affine[:3, 3].astype(np.float64)
        mat = self.affine[:3, :3].astype(np.float64)
        mat_inv = np.linalg.inv(mat)
        seg = np.zeros(self.header.get_data_shape())
        for roi in LMKS_NAMES:
            if self.coord[roi] is None:
                continue
            # Convert the landmarks coordinates into voxels indices
            coord_vox = mat_inv.dot(self.coord[roi].astype(np.float64) - ori)
            coord_vox = np.rint(coord_vox).astype(np.uint8)
            seg[coord_vox[0], coord_vox[1], coord_vox[2]] = LMKS_LABELS[roi]
            if dilation:
                # Dilate the segmentation.
                # Useful to make sure the landmarks do not disappear during registration...
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        for k in [-1, 0, 1]:
                            seg[coord_vox[0]+i, coord_vox[1]+j, coord_vox[2]+k] = LMKS_LABELS[roi]
        seg_nii = nib.Nifti1Image(seg, self.affine, self.header)
        nib.save(seg_nii, save_path)

    @classmethod
    def average(cls, lmks_list, target_GA, right_left_symmetry):
        coord = {
            roi: None for roi in LMKS_NAMES
        }
        # TODO: we assume that all lmks have the same affine...
        #  that might not be true for other data
        for roi in LMKS_NAMES:
            w_sum = 0.
            for lmks in lmks_list:
                if lmks.coord[roi] is not None:
                    w = gaussian_kernel(lmks.ga, target_ga=target_GA, std=GA_STD)
                    w_sum += w
                    if coord[roi] is None:  # initialization
                        coord[roi] = w * lmks.coord[roi]
                    else:
                        coord[roi] += w * lmks.coord[roi]
                    if right_left_symmetry:
                        flip_lmks = Landmarks.flip(lmks)
                        if flip_lmks.coord[roi] is not None:
                            coord[roi] += w * flip_lmks.coord[roi]
                            w_sum += w
            # Normalization
            if coord[roi] is not None:
                coord[roi] /= w_sum
        ave_lmks = cls(
            coord=coord,
            affine=lmks_list[0].affine,
            header=lmks_list[0].header,
            ga=target_GA,
        )
        return ave_lmks

    @classmethod
    def flip(cls, lmks, tmp=None):
        delete_tmp = False
        if tmp is None:
            tmp = 'tmp_flip_lmks'
            delete_tmp = True
        while os.path.exists(tmp):
            tmp = '%s%d' % (tmp, random.randint(1, 100))
        if not os.path.exists(tmp):
            os.mkdir(tmp)
        lmks_path = os.path.join(tmp, 'tmp_ori_lmks.nii.gz')
        lmks.save_as_seg(lmks_path)
        lmks_nii = nib.load(lmks_path)
        lmks_np = lmks_nii.get_fdata().astype(np.uint8)

        # Flip geometrically the landmarks
        flip_lmks_np = lmks_np[::-1, :, :]

        # Permute right and left landmarks
        out_lmks_np = np.zeros_like(flip_lmks_np)
        out_lmks_np[flip_lmks_np == LMKS_LABELS["Anterior Horn of the Right Lateral Ventricle"]] = \
            LMKS_LABELS["Anterior Horn of the Left Lateral Ventricle"]
        out_lmks_np[flip_lmks_np == LMKS_LABELS["Anterior Horn of the Left Lateral Ventricle"]] = \
            LMKS_LABELS["Anterior Horn of the Right Lateral Ventricle"]
        out_lmks_np[flip_lmks_np == LMKS_LABELS["Posterior Tectum edge"]] = \
            LMKS_LABELS["Posterior Tectum edge"]
        out_lmks_np[flip_lmks_np == LMKS_LABELS["Left Cerebellar-Brainstem Junction"]] = \
            LMKS_LABELS["Right Cerebellar-Brainstem Junction"]
        out_lmks_np[flip_lmks_np == LMKS_LABELS["Right Cerebellar-Brainstem Junction"]] = \
            LMKS_LABELS["Left Cerebellar-Brainstem Junction"]
        out_lmks_np[flip_lmks_np == LMKS_LABELS["Left Deep Grey Border at Anterior CSP line"]] = \
            LMKS_LABELS["Right Deep Grey Border at Anterior CSP line"]
        out_lmks_np[flip_lmks_np == LMKS_LABELS["Right Deep Grey Border at Anterior CSP line"]] = \
            LMKS_LABELS["Left Deep Grey Border at Anterior CSP line"]
        out_lmks_np[flip_lmks_np == LMKS_LABELS["Left Deep Grey Border at Posterior CSP Line"]] = \
            LMKS_LABELS["Right Deep Grey Border at Posterior CSP line"]
        out_lmks_np[flip_lmks_np == LMKS_LABELS["Right Deep Grey Border at Posterior CSP line"]] = \
            LMKS_LABELS["Left Deep Grey Border at Posterior CSP Line"]
        out_lmks_np[flip_lmks_np == LMKS_LABELS["Right Deep Grey Border at FOM"]] = \
            LMKS_LABELS["Left Deep Grey Border at FOM"]
        out_lmks_np[flip_lmks_np == LMKS_LABELS["Left Deep Grey Border at FOM"]] = \
            LMKS_LABELS["Right Deep Grey Border at FOM"]

        # Save the flipped landmarks segmentation
        flip_lmks_nii = nib.Nifti1Image(out_lmks_np, lmks.affine, lmks.header)
        flip_lmks_path = os.path.join(tmp, 'tmp_lmks.nii.gz')
        nib.save(flip_lmks_nii, flip_lmks_path)

        out_lmks = Landmarks.fromseg(flip_lmks_path, ga=lmks.ga)

        # Delete the tmp folder
        if delete_tmp:
            os.system('rm -r %s' % tmp)

        return out_lmks

    def __copy__(self):
        return Landmarks(self.coord, self.affine, self.header, ga=self.ga, rotation=self.rotation)

    def __str__(self):
        desc = 'Coordinates in mm:\n'
        for roi in LMKS_NAMES:
            desc += '%s %s\n' % (roi, str(self.coord[roi]))
        return desc


if __name__ == '__main__':
    study_name = 'UZL00066_Study1'  # rotation to apply
    # study_name = 'UZL00066_Study2'  # no rotation to apply
    study_folder = os.path.join(LEUVEN_MMC, study_name)
    study_srr_path = os.path.join(study_folder, 'srr_template.nii.gz')
    lmks = Landmarks.fromexcel(study_name, study_srr_path)
    print(lmks)
    lmks.save_as_seg('test_%s_lmks.nii.gz' % study_name)
