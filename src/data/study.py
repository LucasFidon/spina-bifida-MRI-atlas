import os
import numpy as np
import nibabel as nib
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from src.registration.tools import register, register_affine, warp_img
from src.data.landmarks import Landmarks, transform
from src.utils.definitions import UCLH_MMC, UCLH_MMC_2, LEUVEN_MMC, MEAN_INTENSITY, STD_INTENSITY, GA_STD
from src.utils.maths import gaussian_kernel

SUPPORTED_OPERATION_STATUS = ['notoperated', 'operated']
SKULL_STRIPPING = False
N_ITER_MASK_POST = 2  # number of iterations for dilation and erosion of the mask in average

def find_srr_path(study_n):
    srr_path = None
    for data_folder in [UCLH_MMC, UCLH_MMC_2, LEUVEN_MMC]:
        srr_path = os.path.join(data_folder, study_n, 'srr.nii.gz')
        if os.path.exists(srr_path):
            return srr_path
        srr_path = os.path.join(data_folder, study_n, 'srr_template.nii.gz')
        if os.path.exists(srr_path):
            return srr_path
    raise ValueError("SRR %s not found in %s" % (study_n, str([UCLH_MMC, UCLH_MMC_2, LEUVEN_MMC])))

def find_mask_path(srr_path):
    study_folder = os.path.dirname(srr_path)
    mask_path = os.path.join(study_folder, 'mask.nii.gz')
    if not os.path.exists(mask_path):
        mask_path = os.path.join(
            study_folder, 'srr_template_mask.nii.gz')
    if not os.path.exists(mask_path):
        mask_path = os.path.join(
            study_folder, 'output_mask.nii.gz')
    if not os.path.exists(mask_path):
        mask_path = os.path.join(
            study_folder, 'output_affine_mask.nii.gz')
        assert os.path.exists(mask_path), "Mask not found in %s" % study_folder
    return mask_path

def find_segmentation_path(srr_path):
    study_folder = os.path.dirname(srr_path)
    seg_path = os.path.join(study_folder, 'srr_template_parcellation_softmax_autoseg.nii.gz')
    if not os.path.exists(seg_path):
        seg_path = os.path.join(study_folder, 'srr_parcellation_softmax_autoseg.nii.gz')
    if not os.path.exists(seg_path):
        seg_path = os.path.join(study_folder, 'parcellation_softmax.nii.gz')
    if not os.path.exists(seg_path):
        seg_path = os.path.join(study_folder, 'output_parcellation_softmax.nii.gz')
    if not os.path.exists(seg_path):
        seg_path = os.path.join(study_folder, 'output_affine_parcellation_softmax.nii.gz')
    assert os.path.exists(seg_path), "Softmax parcellation not found in %s" % study_folder
    return seg_path

def _preprocess(srr_path, mask_path, lmks, save_folder, seg_path=None):
    out_srr_path = os.path.join(save_folder, 'srr.nii.gz')
    out_mask_path = os.path.join(save_folder, 'mask.nii.gz')
    if seg_path is None:
        out_seg_path = None
    else:
        out_seg_path = os.path.join(save_folder, 'parcellation_softmax.nii.gz')
    # Apply the same rotation as Lizzy if needed
    if lmks.rotation is not None:
        # print('Apply rotation %s to %s' % (lmks.rotation, srr_path))
        # Rotate the SRR
        # Use ITK-Snap command line tool as Lizzy used ITK-Snap
        cmd_c3d = 'c3d %s %s -interpolation Linear -reslice-itk %s -o %s' % \
            (srr_path, srr_path, lmks.rotation, out_srr_path)
        os.system(cmd_c3d)
        # Rotate the mask
        cmd_c3d = 'c3d %s %s -interpolation NearestNeighbor -reslice-itk %s -o %s' % \
            (mask_path, mask_path, lmks.rotation, out_mask_path)
        os.system(cmd_c3d)
        if seg_path is not None:
            seg_proba_nii = nib.load(seg_path)
            seg_proba = seg_proba_nii.get_fdata().astype(np.float32)
            out_seg_list = []
            # Rotate the channel one by one... this is so retarded...
            for c in range(seg_proba.shape[3]):
                seg_c = seg_proba[:, :, :, c]
                seg_c_nii = nib.Nifti1Image(seg_c, seg_proba_nii.affine, seg_proba_nii.header)
                seg_c_path = os.path.join(save_folder, 'tmp_seg_c.nii.gz')
                nib.save(seg_c_nii, seg_c_path)
                out_seg_c_path = os.path.join(save_folder, 'tmp_rotated_seg_c.nii.gz')
                cmd_c3d = 'c3d %s %s -interpolation Linear -reslice-itk %s -o %s' % \
                    (seg_c_path, seg_c_path, lmks.rotation, out_seg_c_path)
                os.system(cmd_c3d)
                out_seg_list.append(nib.load(out_seg_c_path).get_fdata().astype(np.float32))
                os.system('rm %s' % seg_c_path)
                os.system('rm %s' % out_seg_c_path)
            # Restack the channels
            out_seg_proba = np.stack(out_seg_list, axis=-1)
            out_mask = nib.load(out_mask_path).get_fdata().astype(np.uint8)
            out_mask = binary_dilation(out_mask, iterations=5).astype(np.uint8)
            out_seg_proba[out_mask == 0, :] = 0
            out_seg_proba[out_mask == 0, 0] = 1
            # out_seg_proba /= np.sum(out_seg_proba, axis=-1)
            out_seg_proba_nii = nib.Nifti1Image(out_seg_proba, seg_proba_nii.affine, seg_proba_nii.header)
            nib.save(out_seg_proba_nii, out_seg_path)
        lmks.rotation = None
    else:
        cmd = 'cp %s %s' % (srr_path, out_srr_path)
        os.system(cmd)
        cmd = 'cp %s %s' % (mask_path, out_mask_path)
        os.system(cmd)
        if seg_path is not None:
            cmd = 'cp %s %s' % (seg_path, out_seg_path)
            os.system(cmd)
    return out_srr_path, out_mask_path, out_seg_path


class Study:
    def __init__(self, study_name, srr_path, mask_path, lmks, seg_path=None, op_status=None):
        self.name = study_name
        self.srr_path = srr_path
        self.mask_path = mask_path
        self.lmks = lmks
        self.seg_path = seg_path
        self.study_folder = os.path.dirname(srr_path)
        if op_status is not None:
            assert op_status in SUPPORTED_OPERATION_STATUS, \
                "Found unknown operation status %s. %s are supported" % (op_status, str(SUPPORTED_OPERATION_STATUS))
            self.op_status = op_status

    @classmethod
    def from_study_name(cls, study_name, save_folder, df=None):
        def get_op_status():
            data_s = df[df['File Name/Study'] == study_name]
            op_code = int(data_s['Unnamed: 1'].values[0])
            if op_code == 0:
                return 'notoperated'
            elif op_code == 1:
                return 'operated'
            else:
                raise ValueError('Unknown operation code', op_code)

        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        # Set SRR path
        srr_path = find_srr_path(study_name)

        # Get mask path
        mask_path = find_mask_path(srr_path)

        # Get the landmarks
        lmks = Landmarks.fromexcel(
            study_name=study_name, srr_path=srr_path, df=df)
        lmks_path = os.path.join(save_folder, 'lmks.nii.gz')
        lmks.save_as_seg(lmks_path)

        # Get the segmentation path
        seg_path = find_segmentation_path(srr_path)

        # Run the preprocessing
        # We don't want to load all the volumes in memory
        # so we save the SRR in a new folder (save_folder) after preprocessing
        pre_srr, pre_mask, pre_seg = _preprocess(srr_path, mask_path, lmks, save_folder, seg_path=seg_path)

        # get the operation status
        op_status = get_op_status()

        return cls(study_name, pre_srr, pre_mask, lmks, seg_path=pre_seg, op_status=op_status)

    @property
    def ga(self):
        ga = self.lmks.ga
        if ga is None:
            print('ga is None for', self.name)
        return ga

    @property
    def nifti(self):
        vol_nii = nib.load(self.srr_path)
        return vol_nii

    @property
    def volume(self):
        vol = self.nifti.get_fdata().astype(np.float32)
        return vol

    @property
    def mask(self):
        mask = nib.load(self.mask_path).get_fdata().astype(np.uint8)
        return mask

    @property
    def seg_proba(self):
        if self.seg_path is None:
            return None
        else:
            seg_proba = nib.load(self.seg_path).get_fdata().astype(np.float32)
            return seg_proba

    @property
    def seg(self):
        seg_proba = self.seg_proba
        if seg_proba is None:
            return None
        else:
            seg = np.argmax(seg_proba, axis=-1)
            return seg

    @property
    def affine(self):
        return self.nifti.affine

    @classmethod
    def register_linear(cls, study1, study2, out_name, out_folder, **kwargs):
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        out_srr, out_mask, out_lmks, out_seg, _ = register_affine(
            study1.srr_path,
            study2.srr_path,
            save_folder=out_folder,
            mask1_path=study1.mask_path,
            mask2_path=study2.mask_path,
            lmks1=study1.lmks,
            seg1_path=study1.seg_path,
            **kwargs
        )
        out = cls(
            study_name=out_name,
            srr_path=out_srr,
            mask_path=out_mask,
            lmks=out_lmks,
            seg_path=out_seg,
            op_status=study1.op_status,
        )
        return out

    @classmethod
    def register(cls, study1, study2, out_name, out_folder, **kwargs):
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        register(study1.srr_path, study2.srr_path, save_folder=out_folder,
                 mask1_path=study1.mask_path, mask2_path=study2.mask_path,
                 seg1_path=study1.seg_path, seg2_path=study2.seg_path,
                 lmks1=study1.lmks, lmks2=study2.lmks, **kwargs)
        out_srr = os.path.join(out_folder, 'output.nii.gz')
        out_mask = os.path.join(out_folder, 'output_mask.nii.gz')
        out_seg = find_segmentation_path(out_srr)
        out_lmks_path = os.path.join(out_folder, 'output_lmks.nii.gz')
        out_lmks = Landmarks.fromseg(seg_path=out_lmks_path, ga=study1.ga)
        out = cls(
            study_name=out_name,
            srr_path=out_srr,
            mask_path=out_mask,
            lmks=out_lmks,
            seg_path=out_seg,
            op_status=study1.op_status,
        )
        return out

    @classmethod
    def warp(cls, study, trans, out_folder):
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        warp_srr_path = os.path.join(out_folder, 'srr.nii.gz')
        warp_img(
            img=study.srr_path,
            trans=trans,
            ref_img=study.srr_path,
            save_path=warp_srr_path,
        )
        warp_mask_path = os.path.join(out_folder, 'mask.nii.gz')
        warp_img(
            img=study.mask_path,
            trans=trans,
            ref_img=study.mask_path,
            save_path=warp_mask_path,
            is_seg=True,
        )
        if study.seg_path is None:
            warp_seg_path = None
        else:
            warp_seg_path = os.path.join(out_folder, 'parcellation_softmax.nii.gz')
            warp_img(
                img=study.seg_path,
                trans=trans,
                ref_img=study.seg_path,
                save_path=warp_seg_path,
                is_proba_seg=True,
            )
        warp_lmks = transform(
            lmks=study.lmks,
            trans=trans,
            ref_img=study.srr_path,
        )
        warp_lmks_path = os.path.join(out_folder, 'lmks.nii.gz')
        warp_lmks.save_as_seg(warp_lmks_path)
        out = cls(
            study_name=study.name,
            srr_path=warp_srr_path,
            mask_path=warp_mask_path,
            lmks=warp_lmks,
            seg_path=warp_seg_path,
            op_status=study.op_status,
        )
        return out

    @classmethod
    def average(cls, study_list, name, save_folder, target_GA, right_left_symmetry=True):
        def load_preprocess(study):
            volume = study.volume
            mask = study.mask
            # Clip the high intensity values
            p999 = np.percentile(volume, 99.9)
            volume[volume > p999] = p999
            # Set the mean and the std of the foreground
            volume_for = volume[mask > 0]
            volume -= np.mean(volume_for)
            volume *= STD_INTENSITY / np.std(volume_for)
            volume += MEAN_INTENSITY
            return volume

        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        w_sum = 0  # normalization term for the weighted sum

        # Average the landmarks
        lmks_list = [s.lmks for s in study_list]
        ave_lmks = Landmarks.average(
            lmks_list=lmks_list,
            target_GA=target_GA,
            right_left_symmetry=right_left_symmetry,
        )
        ave_lmks_path = os.path.join(save_folder, 'lmks.nii.gz')
        ave_lmks.save_as_seg(ave_lmks_path)

        # Average the masks
        ave_mask = 0.
        for s in study_list:
            mask_np = s.mask
            w = gaussian_kernel(s.ga, target_ga=target_GA, std=GA_STD)
            w_sum += w
            ave_mask += w * mask_np
        ave_mask /= w_sum
        if right_left_symmetry:
            ave_mask = 0.5 * (ave_mask + ave_mask[::-1, :, :])
        ave_mask = (ave_mask > 0.5).astype(np.uint8)
        ave_mask_nii = nib.Nifti1Image(ave_mask, study_list[0].affine)
        print('Save mask in ')
        ave_mask_path = os.path.join(save_folder, 'mask.nii.gz')
        print(ave_mask_path)
        nib.save(ave_mask_nii, ave_mask_path)

        # Average the images
        ave_vol = 0.
        for s in study_list:
            vol_np = load_preprocess(s)
            w = gaussian_kernel(s.ga, target_ga=target_GA, std=GA_STD)
            ave_vol += w * vol_np
        ave_vol /= w_sum
        if right_left_symmetry:
            ave_vol = 0.5 * (ave_vol + ave_vol[::-1, :, :])
        dilated_mask = binary_dilation(ave_mask, iterations=N_ITER_MASK_POST).astype(np.uint8)
        if SKULL_STRIPPING:
            ave_vol[dilated_mask == 0] = 0
        # Save the SRR
        ave_nii = nib.Nifti1Image(ave_vol, study_list[0].affine)
        ave_srr_path = os.path.join(save_folder, 'srr.nii.gz')
        nib.save(ave_nii, ave_srr_path)

        # Average the segmentations
        ave_proba_seg = 0
        seg_exists = False
        for s in study_list:
            proba_seg = s.seg_proba
            if proba_seg is not None:
                seg_exists = True
                w = gaussian_kernel(s.ga, target_ga=target_GA, std=GA_STD)
                ave_proba_seg += w * proba_seg
        if seg_exists:
            ave_proba_seg /= w_sum
            if right_left_symmetry:
                ave_proba_seg = 0.5 * (ave_proba_seg + ave_proba_seg[::-1, :, :])
            # Compatibility with mask / tackle border effects
            ave_proba_seg[dilated_mask == 0, :] = 0
            ave_proba_seg[dilated_mask == 0, 0] = 1
            eroded_mask = binary_erosion(ave_mask, iterations=N_ITER_MASK_POST).astype(np.uint8)
            ave_proba_seg[eroded_mask == 1, 0] = 0
            denom = np.sum(ave_proba_seg[eroded_mask == 1, :], axis=1)
            ave_proba_seg[eroded_mask == 1, :] /= denom[:, None]
            # Save the proba map
            ave_seg_nii = nib.Nifti1Image(ave_proba_seg, study_list[0].affine)
            ave_seg_path = os.path.join(save_folder, 'parcellation_softmax.nii.gz')
            nib.save(ave_seg_nii, ave_seg_path)
            # Compute and save the argmax segmentation
            argmax_seg = np.argmax(ave_proba_seg, axis=-1).astype(np.uint8)
            argmax_seg_nii = nib.Nifti1Image(argmax_seg, study_list[0].affine)
            argmax_seg_path = os.path.join(save_folder, 'parcellation.nii.gz')
            nib.save(argmax_seg_nii, argmax_seg_path)
        else:
            ave_seg_path = None

        average = cls(
            study_name=name,
            srr_path=ave_srr_path,
            mask_path=ave_mask_path,
            lmks=ave_lmks,
            seg_path=ave_seg_path,
            op_status=None,
        )
        return average

    @classmethod
    def flip(cls, study, name, save_folder):

        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        vol_np = study.volume
        mask_np = study.mask
        seg_np = study.seg_proba
        aff = study.affine

        # Perform the naive right left flip
        # Volume
        flip_vol = vol_np[::-1, :, :]
        flip_vol_nii = nib.Nifti1Image(flip_vol, aff)
        flip_vol_path = os.path.join(save_folder, 'srr.nii.gz')
        nib.save(flip_vol_nii, flip_vol_path)
        # Mask
        flip_mask = mask_np[::-1, :, :]
        flip_mask_nii = nib.Nifti1Image(flip_mask, aff)
        flip_mask_path = os.path.join(save_folder, 'mask.nii.gz')
        nib.save(flip_mask_nii, flip_mask_path)
        # Landmarks
        flip_lmks = Landmarks.flip(study.lmks, tmp=save_folder)
        # Segmentation
        if seg_np is not None:
            flip_seg = seg_np[::-1, :, :, :]
            flip_seg_nii = nib.Nifti1Image(flip_seg, aff)
            flip_seg_path = os.path.join(save_folder, 'parcellation_softmax.nii.gz')
            nib.save(flip_seg_nii, flip_seg_path)
        else:
            flip_seg_path = None

        # # Perform rigid registration to study
        # # to make sure the flipped image is correctly aligned to the original image
        # flip_srr_path, flip_mask_path, flip_lmks, flip_seg_path, _ = register_affine(
        #     img1_path=flip_vol_path_tmp,
        #     img2_path=study.srr_path,
        #     save_folder=save_folder,
        #     mask1_path=flip_mask_path_tmp,
        #     mask2_path=study.mask_path,
        #     lmks1=flip_lmks_tmp,
        #     seg1_path=flip_seg_path_tmp,
        #     rig_only=True,
        # )
        out = cls(
            study_name=name,
            srr_path=flip_vol_path,
            mask_path=flip_mask_path,
            lmks=flip_lmks,
            seg_path=flip_seg_path,
            op_status=study.op_status,
        )

        # # Delete tmp files
        # os.system('rm %s' % flip_vol_path_tmp)
        # os.system('rm %s' % flip_mask_path_tmp)
        # if os.path.exists(flip_seg_path_tmp):
        #     os.system('rm %s' % flip_seg_path_tmp)

        return out
