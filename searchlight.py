import AnatSearchlight.searchlight as sl
from typing import Optional
import nibabel as nb
import numpy as np
from os import PathLike

def searchlight_surf(
        white: list,
        pial: list,
        mask: list | PathLike | nb.Nifti2Image,
        savedir: str,
        roi_mask: Optional[list] = [None, None],
        maxradius: Optional[int] = None, # in mm
        maxvoxels: Optional[int] = None
):

    structs, Hem = ['CortexLeft', 'CortexRight'], ['L', 'R']
   
    if isinstance(mask, list):
        mask_vol0 = nb.load(mask[0])
        mask_array = np.zeros_like(mask_vol0.get_fdata())
        for m in mask:
            mask_vol = nb.load(m)
            mask_array += mask_vol.get_fdata()
        
        mask_array = mask_array > 0

        voxel_mask = nb.Nifti2Image(mask_array, affine=mask_vol0.affine, header=mask_vol0.header)
            
    if isinstance(mask, PathLike):
        voxel_mask = nb.load(mask)

    for s, (struct, H) in enumerate(zip(structs, Hem)):
        Searchlight = sl.SearchlightSurface(struct)
        surf = [pial[s], white[s]]
        Searchlight.define(surf, voxel_mask, roi=roi_mask[s], maxradius=maxradius, maxvoxels=maxvoxels)
        print(f'{H} hemisphere, average size of the searchlight: {np.nanmean(Searchlight.radius):.1f} mm')
        print(f'{H} hemisphere, average number of voxels: {np.nanmean(Searchlight.nvoxels):.1f}')
        Searchlight.save(f'{savedir}/searchlight.{H}.h5')


### Some mvpa functions to perform in searchlight:



