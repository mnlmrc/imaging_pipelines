import numpy as np
import os
import nibabel as nb
import rsatoolbox as rsa
import nitools as nt

def calc_rdm_roi(mask, beta_img, res_img, Hem=None, roi=None, sn=None, unbalanced=False):
    """
    Calculates the representational dissimilarity matrix (RDM) for a specified region of interest (ROI)
    by preprocessing functional neuroimaging data and using the cross-validated Crossnobis method.

    Args:
        mask (str or nibabel.Nifti1Image): Path to the mask image or NIfTI image defining the region of interest.
        beta_img (str): Path to the CIFTI-2 file containing beta estimates for conditions.
        res_img (str): Path to the NIfTI file containing residual variance for beta estimates.
        Hem (str, optional): Hemisphere specification, included as metadata in the dataset descriptors.
        roi (str, optional): Identifier for the region of interest, included as metadata in the dataset descriptors.
        sn (int, optional): Subject number, included as metadata in the dataset descriptors.
        unbalanced (bool, optional): Whether to compute an unbalanced RDM for unbalanced designs. Defaults to False.

    Returns:
        np.ndarray: A representational dissimilarity matrix (RDM) computed across conditions.
    """

    beta_cifti = nb.load(beta_img)
    res_nifti = nb.load(res_img)

    beta_nifti = nt.volume_from_cifti(beta_cifti, struct_names = ['CortexLeft', 'CortexRight'])
    coords = nt.get_mask_coords(mask)

    betas = nt.sample_image(beta_nifti, coords[0], coords[1], coords[2], interpolation=0).T

    res = nt.sample_image(res_nifti, coords[0], coords[1], coords[2], interpolation=0)

    betas_prewhitened = betas / np.sqrt(res)

    betas_prewhitened = np.array(betas_prewhitened)
    betas_prewhitened = betas_prewhitened[:, ~np.all(np.isnan(betas_prewhitened), axis=0)]

    reginfo = np.char.split(beta_cifti.header.get_axis(0).name, sep='.')
    conds = [r[0] for r in reginfo]
    run = [r[1] for r in reginfo]
    dataset = rsa.data.Dataset(
        betas_prewhitened,
        channel_descriptors={
            'channel': np.array(['vox_' + str(x) for x in range(betas_prewhitened.shape[-1])])},
        obs_descriptors={'conds': conds,
                         'run': run},
        descriptors={'ROI': roi, 'Hem': Hem, 'sn': sn}
    )
    # remove_mean removes the mean ACROSS VOXELS for each condition
    if unbalanced:
        rdm = rsa.rdm.calc_rdm_unbalanced(dataset, method='crossnobis', descriptor='conds', cv_descriptor='run')
    else:
        rdm = rsa.rdm.calc_rdm(dataset, method='crossnobis', descriptor='conds', cv_descriptor='run', remove_mean=False)

    return rdm