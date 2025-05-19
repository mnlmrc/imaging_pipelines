import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import nibabel as nb
import nitools as nt
import numpy as np
from nitools import spm


def get_timeseries_in_voxels(path_glm, masks, struct):
    """

    Args:
        path_glm (str):
        masks (list): Must be non-overlapping voxels
        struct (list):

    Returns:

    """

    SPM = spm.SpmGlm(path_glm)
    SPM.get_info_from_spm_mat()

    for i, (s, mask) in enumerate(zip(struct, masks)):
        if isinstance(mask, str):
            atlas = am.AtlasVolumetric('native', mask, structure=s)
        else:
            raise Exception('mask must be the path to a .nii file')

        if i == 0:
            brain_axis = atlas.get_brain_model_axis()
        else:
            brain_axis += atlas.get_brain_model_axis()

    coords = nt.affine_transform_mat(brain_axis.voxel.T, brain_axis.affine)

    # get raw time series in roi
    y_raw = nt.sample_images(SPM.rawdata_files, coords, use_dataobj=False)

    # rerun glm
    _, info, y_filt, y_hat, y_adj, _ = SPM.rerun_glm(y_raw)

    row_axis = nb.cifti2.SeriesAxis(1, 1, y_filt.shape[0], 'second')

    header = nb.Cifti2Header.from_axes((row_axis, brain_axis))

    cifti_yraw = nb.Cifti2Image(ataobj=y_raw,header=header,)
    cifti_yfilt = nb.Cifti2Image(dataobj=y_filt, header=header,)
    cifti_yhat = nb.Cifti2Image(dataobj=y_hat,header=header,)
    cifti_yadj = nb.Cifti2Image(dataobj=y_adj,header=header,)

    return cifti_yraw, cifti_yfilt, cifti_yhat, cifti_yadj

def get_timeseries_in_parcels(path_glm, masks, rois, struct, timeseries):

    for i, (s, mask, roi) in enumerate(zip(struct, masks, rois)):
        atlas = am.AtlasVolumetric('native', mask, structure=s)

        if i == 0:
            label_vec, _ = atlas.get_parcel(roi)
            parcel_axis = atlas.get_parcel_axis()
        else:
            label_vec = np.concatenate((label_vec, atlas.get_parcel(roi)[0] + label_vec.max()), axis=0)
            parcel_axis += atlas.get_parcel_axis()

    data = timeseries.get_fdata()
    parcel_data, label = ds.agg_parcels(data, label_vec)

    row_axis = nb.cifti2.SeriesAxis(1, 1, parcel_data.shape[0], 'second')

    header = nb.Cifti2Header.from_axes((row_axis, parcel_axis))
    cifti_parcel = nb.Cifti2Image(parcel_data, header=header)

    return cifti_parcel

def cut_timeseries_at_onsets(path_glm, masks, rois, struct, timeseries, at=None):
    for i, (s, mask, roi) in enumerate(zip(struct, masks, rois)):
        atlas = am.AtlasVolumetric('native', mask, structure=s)

        if i == 0:
            label_vec, _ = atlas.get_parcel(roi)
            parcel_axis = atlas.get_parcel_axis()
        else:
            label_vec = np.concatenate((label_vec, atlas.get_parcel(roi)[0] + label_vec.max()), axis=0)
            parcel_axis += atlas.get_parcel_axis()

    data = timeseries.get_fdata()

    y_cut = spm.cut(data, 10, at, 20).mean(axis=0)

    parcel_data, label = ds.agg_parcels(y_cut, label_vec)

    row_axis = nb.cifti2.SeriesAxis(-10, 1, parcel_data.shape[0], 'second')

    header = nb.Cifti2Header.from_axes((row_axis, parcel_axis))
    cifti_parcel = nb.Cifti2Image(parcel_data, header=header)

    return cifti_parcel