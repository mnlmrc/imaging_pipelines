import glob
import os

import nitools as nt
from nitools import spm
import numpy as np
import Functional_Fusion.atlas_map as am
import nibabel as nb
import pandas as pd


def make_cifti_betas(path_glm, masks, struct, betas=None, row_axis=None):
    """
    Generates a CIFTI file containing beta coefficients for specific regions of interest (ROIs)
    based on a General Linear Model (GLM). It integrates spatial information, structures, and
    beta values extracted from an SPM-based GLM, producing a functional brain mapping file
    compatible with CIFTI standards.

    Parameters:
    - path_glm (str):
        The file path to the folder containing the SPM.mat file, which represents the configuration
        and results of the SPM GLM analysis.

    - masks (list of str):
        A list of file paths to ROI masks. Each mask corresponds to a spatially defined region
        of interest to extract beta values.

    - struct (list of str):
        A list of CIFTI brain structures corresponding to each mask. Examples include `CortexRight`,
        `CortexLeft`, or `Cerebellum`.

    - betas (optional, list of str):
        A list of file name corresponding to the beta coefficients extracted for each regressor
        in the first-level GLM. If not provided the function looks for SPM.mat file in path_glm.

    - row_axis (optional, None or nb.cifti2.Axis):
        Defines the rows (e.g., beta labels) of the resulting CIFTI file. If not provided, it is
        automatically constructed based on regressor names extracted from the GLM and their
        corresponding run numbers.

    Returns:
    - cifti (nibabel.Cifti2Image):
        A CIFTI2 image containing the beta values. Rows represent GLM regressors, while the combined
        brain mask axes define the spatial dimensions.

    Workflow:
    1. GLM Initialization:
        The function initializes the SPM GLM using the path_glm and retrieves metadata from
        the provided SPM.mat.

    2. Atlas & Mask Handling:
        For each structure-mask pair, an atlas is constructed using the AtlasVolumetric class.
        Spatial brain model axes and voxel coordinates are extracted for all masks:
        - For the first mask, both the brain axis and voxel coordinates are initialized.
        - Subsequent masks append their information to the existing brain model axis.

    3. Beta Value Extraction:
        Using the atlas-defined voxel coordinates, beta coefficients for all regressors are
        extracted from the GLM.

    4. Row Axis Construction (Optional):
        If row_axis is not explicitly provided, a scalar axis is created with regressor names
        and corresponding run numbers.

    5. CIFTI File Generation:
        Combines the row axis (regressors) and brain model axis (spatial dimension) into a
        CIFTI2 header. The beta values are then stored as a data object in the resulting
        CIFTI2 image.

    Example:
    >>> path_to_glm = "/path/to/glm_folder"
    >>> roi_masks = ["/path/to/mask1.nii", "/path/to/mask2.nii"]
    >>> roi_structures = ["CortexRight", "CortexLeft"]
    >>>
    >>> # Generate CIFTI file containing beta values
    >>> cifti_image = make_cifti_betas(path_to_glm, roi_masks, roi_structures)
    >>>
    >>> # Save the resulting CIFTI file
    >>> cifti_image.to_filename("output_betas.dscalar.nii")

    Notes:
    - Ensure that the SPM.mat file is correctly configured and contains valid GLM results.
    - Mask files must align spatially with the GLM for accurate voxel extraction.
    - The CIFTI file can be processed further for visualization or statistical analysis in
      compatible neuroimaging software.
    """

    for i, (s, mask) in enumerate(zip(struct, masks)):
        atlas = am.AtlasVolumetric('region', mask, structure=s)

        if i == 0:
            brain_axis = atlas.get_brain_model_axis()
            coords = nt.get_mask_coords(mask)
        else:
            brain_axis += atlas.get_brain_model_axis()
            coords = np.concatenate((coords, nt.get_mask_coords(mask)), axis=1)

    if betas is None:
        SPM = spm.SpmGlm(path_glm)  #
        SPM.get_info_from_spm_mat()
        betas, _, info = SPM.get_betas(coords)
    else:
        betas = nt.sample_images(betas, coords, use_dataobj=False)
        if row_axis is None:
            raise ValueError("If 'betas' is provided, 'row_axis' must also be specified.")

    if row_axis is None:
        reg_name = np.array([n.split('*')[0] for n in info['reg_name']])
        row_axis = nb.cifti2.ScalarAxis(reg_name.astype(str) + '.' + info['run_number'].astype(str))
    else:
        row_axis = nb.cifti2.ScalarAxis(row_axis)

    header = nb.Cifti2Header.from_axes((row_axis, brain_axis))
    cifti = nb.Cifti2Image(
        dataobj=betas,  # Stack them along the rows (adjust as needed)
        header=header,  # Use one of the headers (may need to modify)
    )
    return cifti


def make_cifti_contrasts(path_glm, masks, struct, regressors=None):
    for i, (s, mask) in enumerate(zip(struct, masks)):
        atlas = am.AtlasVolumetric('region', mask, structure=s)

        if i == 0:
            brain_axis = atlas.get_brain_model_axis()
            coords = nt.get_mask_coords(mask)
        else:
            brain_axis += atlas.get_brain_model_axis()
            coords = np.concatenate((coords, nt.get_mask_coords(mask)), axis=1)

    contrasts = list()
    for regr, regressor in enumerate(regressors):
        vol = nb.load(os.path.join(path_glm, f'con_{regressor}.nii'))
        con = nt.sample_image(vol, coords[0], coords[1], coords[2], 0)
        contrasts.append(con)

    contrasts = np.array(contrasts)

    row_axis = nb.cifti2.ScalarAxis(regressors)

    header = nb.Cifti2Header.from_axes((row_axis, brain_axis))
    cifti = nb.Cifti2Image(
        dataobj=contrasts,  # Stack them along the rows (adjust as needed)
        header=header,  # Use one of the headers (may need to modify)
    )
    return cifti

