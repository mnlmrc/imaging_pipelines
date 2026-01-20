import glob
import os
from os import PathLike

import nitools as nt
from nitools import spm
import numpy as np
import Functional_Fusion.atlas_map as am
import nibabel as nb
import pandas as pd


def make_cifti_betas(masks: list,
                     struct: list,
                     betas: list = None,
                     path_glm: PathLike = None,
                     row_axis: np.ndarray | pd.Series = None):
    """

    :param path_glm:
    :param masks:
    :param struct:
    :param betas:
    :param row_axis:
    :return:
    """

    if len(masks) != len(struct):
        raise ValueError(f"Length mismatch: you need to have one struct per mask")

    for i, (s, mask) in enumerate(zip(struct, masks)):
        atlas = am.AtlasVolumetric('region', mask, structure=s)

        if i == 0:
            brain_axis = atlas.get_brain_model_axis()
            coords = nt.get_mask_coords(mask)
        else:
            brain_axis += atlas.get_brain_model_axis()
            coords = np.concatenate((coords, nt.get_mask_coords(mask)), axis=1)

    if betas is None:
        if path_glm is None:
            raise ValueError("If betas is not provided, path_glm must be provided to retrieve betas using the SPM.mat file.")
        SPM = spm.SpmGlm(path_glm)  #
        SPM.get_info_from_spm_mat()
        betas, _, info = SPM.get_betas(coords)
    else:
        betas = nt.sample_images(betas, coords, use_dataobj=False)
        if row_axis is None:
            raise ValueError("If 'betas' is provided by user, 'row_axis' must also be specified.")

    if row_axis is None:
        reg_name = np.array([n.split('*')[0] for n in info['reg_name']])
        row_axis = nb.cifti2.ScalarAxis(reg_name.astype(str) + '.' + info['run_number'].astype(str))
    if isinstance(row_axis, pd.Series):
        row_axis = nb.cifti2.ScalarAxis(row_axis)
    if isinstance(row_axis, nb.cifti2.ScalarAxis):
        header = nb.Cifti2Header.from_axes((row_axis, brain_axis))
        cifti = nb.Cifti2Image(
            dataobj=betas,  # Stack them along the rows (adjust as needed)
            header=header,  # Use one of the headers (may need to modify)
        )
    return cifti


def make_cifti_contrasts(path_glm: PathLike,
                         masks: list,
                         struct: list,
                         regressors: pd.Series | list):
    """

    :param path_glm:
    :param masks:
    :param struct:
    :param regressors:
    :return:
    """
    if len(masks) != len(struct):
        raise ValueError(f"Length mismatch: you need to have one struct per mask")

    if isinstance(regressors, list):
        regressors = pd.Series(regressors)

    for i, (s, mask) in enumerate(zip(struct, masks)):
        atlas = am.AtlasVolumetric('region', mask, structure=s)

        if i == 0:
            brain_axis = atlas.get_brain_model_axis()
            coords = nt.get_mask_coords(mask)
        else:
            brain_axis += atlas.get_brain_model_axis()
            coords = np.concatenate((coords, nt.get_mask_coords(mask)), axis=1)

    contrasts = list()
    for regr, regressor in enumerate(regressors.unique()):
        vol = nb.load(os.path.join(path_glm, f'con_{regressor}.nii'))
        con = nt.sample_image(vol, coords[0], coords[1], coords[2], 0)
        contrasts.append(con)

    contrasts = np.asarray(contrasts, dtype=np.float32)

    row_axis = nb.cifti2.ScalarAxis(regressors.unique())

    header = nb.Cifti2Header.from_axes((row_axis, brain_axis))
    cifti = nb.Cifti2Image(
        dataobj=contrasts,  # Stack them along the rows (adjust as needed)
        header=header,  # Use one of the headers (may need to modify)
    )
    return cifti


def make_cifti_residuals(path_glm, masks, struct):

    SPM = spm.SpmGlm(path_glm)  #
    SPM.get_info_from_spm_mat()

    for i, (s, mask) in enumerate(zip(struct, masks)):
        atlas = am.AtlasVolumetric('region', mask, structure=s)

        if i == 0:
            brain_axis = atlas.get_brain_model_axis()
            coords = nt.get_mask_coords(mask)
        else:
            brain_axis += atlas.get_brain_model_axis()
            coords = np.concatenate((coords, nt.get_mask_coords(mask)), axis=1)

    res, _, info = SPM.get_residuals(coords)

    row_axis = nb.cifti2.SeriesAxis(1, 1, res.shape[0], 'second')
    header = nb.Cifti2Header.from_axes((row_axis, brain_axis))
    cifti = nb.Cifti2Image(
         dataobj=res,  # Stack them along the rows (adjust as needed)
         header=header,  # Use one of the headers (may need to modify)
    )

    return cifti