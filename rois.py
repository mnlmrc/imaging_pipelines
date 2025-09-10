import nibabel as nb
import numpy as np
import Functional_Fusion.atlas_map as am
import os
from os import  PathLike

class SurfRois():
    def __init__(self,
                 atlas_name: str,
                 white: list,
                 pial: list,
                 mask: PathLike,
                 atlas_dir: PathLike,
                 rois_dir: PathLike,
                 space: str='fs32k'
                 ):
        """

        :param atlas_name:
        :param white:
        :param pial:
        :param mask:
        :param atlas_dir:
        :param rois_dir:
        :param space:
        """
        self.atlas_name = atlas_name
        self.white = white
        self.pial = pial
        self.mask = mask
        self.atlas_dir = atlas_dir
        self.rois_dir = rois_dir

        # create saving directory if it doesn exist
        os.makedirs(self.rois_dir, exist_ok=True)

        self.atlas, _ = am.get_atlas(space)
        self.Hem = ['L', 'R']

    def make_rois(self, exclude):

        for h, H in enumerate(self.Hem):
            atlas_path = os.path.join(self.atlas_dir, f'{self.atlas_name}.32k.{self.Hem[h]}.label.gii')

            # get labels, excluding blank (i.e., outside ROI)
            g_atlas = nb.load(atlas_path)
            labels = {
                ele.key: getattr(ele, 'label', '')
                for ele in g_atlas.labeltable.labels
            }

            amap = list()
            for nlabel, label in enumerate(labels.values()):
                print(f'making ROI: {label}, {H}')

                atlas_hem = self.atlas.get_hemisphere(h)
                subatlas = atlas_hem.get_subatlas_image(atlas_path, nlabel)

                amap_tmp = am.AtlasMapSurf(subatlas.vertex[0], self.white[h], self.pial[h], self.mask)
                amap_tmp.build()

                # add roi name
                amap_tmp.name = label

                # add number of voxels
                amap_tmp.n_voxels = len(np.unique(amap_tmp.vox_list))

                amap.append(amap_tmp)

            print('excluding voxels...')
            amap = am.exclude_overlapping_voxels(amap, exclude=exclude)

            roiMasks = []
            for amap_tmp in amap: # save a niftti file for each roi
                print(f'saving ROI {amap_tmp.name}, {H}')
                _ = amap_tmp.save_as_image(os.path.join(self.rois_dir, f'{self.atlas_name}.{H}.{amap_tmp.name}.nii'))
                if len(amap_tmp.name) > 0:
                    roiMasks.append(os.path.join(self.rois_dir, f'{self.atlas_name}.{H}.{amap_tmp.name}.nii'))

            # save a single nifti with all ROI (per hemisphere)
            am.parcel_combine(roiMasks, os.path.join(self.rois_dir, f'{self.atlas_name}.{H}.nii'))

    def make_hemispheres(self):
        amap = []
        for h, H in enumerate(self.Hem):
            atlas_hem = self.atlas.get_hemisphere(h)

            amap_tmp = am.AtlasMapSurf(atlas_hem.vertex[0], self.white[h], self.pial[h], self.mask)

            print(f'building hemisphere: {H}')
            amap_tmp.build()

            # add hem name
            amap_tmp.name = H

            # add number of voxels
            amap_tmp.n_voxels = len(np.unique(amap_tmp.vox_list))

            amap.append(amap_tmp)

        amap = am.exclude_overlapping_voxels(amap, exclude=[(0, 1)])
        for amap_tmp, H in zip(amap, self.Hem):
            print(f'saving hemisphere {amap_tmp.name}')
            _ = amap_tmp.save_as_image(os.path.join(self.rois_dir, f'Hem.{H}.nii'))

def make_cerebellum(atlas_path, space, labels, rois, deform, mask, out_path):
        atlas, _ = am.get_atlas(space)
        Hem = ['L', 'R']
        for H in Hem:
            amap = []
            for roi in rois[H]:
                label_value = labels.index(roi)
                subatlas = atlas.get_subatlas_image(atlas_path, label_value)
                amap_tmp = am.AtlasMapDeform(subatlas.world, deform, mask)
                amap_tmp.build(interpolation=1)  # Using Trilinear interpolation (0 for nearest neighbor, 2 for smoothing)

                # add hem name
                amap_tmp.name = roi

                # add number of voxels
                amap_tmp.n_voxels = len(np.unique(amap_tmp.vox_list))

                amap.append(amap_tmp)

            roiMasks = []
            for amap_tmp in amap:
                print(f'saving ROI {amap_tmp.name}, {H}')
                _ = amap_tmp.save_as_image(
                    os.path.join(out_path, f'cerebellum.{H}.{amap_tmp.name}.nii'))
                if len(amap_tmp.name) > 0:
                    roiMasks.append(os.path.join(out_path, f'cerebellum.{H}.{amap_tmp.name}.nii'))

            am.parcel_combine(roiMasks, os.path.join(out_path, f'cerebellum.{H}.nii'))