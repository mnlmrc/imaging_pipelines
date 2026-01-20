import AnatSearchlight.searchlight as sl
from typing import Optional
import nibabel as nb

def searchlight_surf(
        white: list,
        pial: list,
        mask: str,
        savedir: str,
        roi_mask: Optional[list] = [None, None],
        radius: Optional[int] = None, # in mm
):

    structs, Hem = ['CortexLeft', 'CortexRight'], ['L', 'R']

    voxel_mask = nb.load(mask)

    for s, (struct, H) in enumerate(zip(structs, Hem)):
        Searchlight = sl.SearchlightSurface(struct)
        surf = [pial[s], white[s]]
        Searchlight.define(surf, voxel_mask, roi=roi_mask[s], maxradius=radius,)
        Searchlight.save(f'{savedir}/searchlight.{H}.h5')


### Some mvpa functions to perform in searchlight:



