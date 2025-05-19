import nibabel as nb
import numpy as np
import Functional_Fusion.atlas_map as am
import os

class Rois():
    def __init__(self, atlas_name, white, pial, mask, atlas_dir, rois_dir):
        """

        :param atlas: str, atlas name
        :param white: list, paths to left and right white surfaces
        :param pial: list, paths to left and right pial surfaces
        :param mask: str, path to mask.nii in glm dir
        :param atlas_dir:
        :param rois_dir:
        """
        self.atlas_name = atlas_name
        self.white = white
        self.pial = pial
        self.mask = mask
        self.atlas_dir = atlas_dir
        self.rois_dir = rois_dir

        # create saving directory if it doesn exist
        os.makedirs(self.rois_dir, exist_ok=True)

        self.atlas, _ = am.get_atlas('fs32k')
        self.Hem = ['L', 'R']

    def make_rois(self, exclude):

        for h, H in enumerate(self.Hem):

            g_atlas = nb.load(os.path.join(self.atlas_dir, f'{self.atlas_name}.32k.{self.Hem[h]}.label.gii'))

            labels = {
                ele.key: getattr(ele, 'label', '')
                for ele in g_atlas.labeltable.labels
            }

            amap = list()
            for nlabel, label in enumerate(labels.values()):
                print(f'making ROI: {label}, {H}')

                atlas_hem = self.atlas.get_hemisphere(h)
                subatlas = atlas_hem.get_subatlas_image(os.path.join(self.atlas_dir,
                                                                     f'{self.atlas_name}.32k.{H}.label.gii'), nlabel)

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
            for amap_tmp in amap:
                print(f'saving ROI {amap_tmp.name}, {H}')
                mask_out = amap_tmp.save_as_image(os.path.join(self.rois_dir, f'{self.atlas_name}.{H}.{amap_tmp.name}.nii'))
                if len(amap_tmp.name) > 0:
                    roiMasks.append(os.path.join(self.rois_dir, f'{self.atlas_name}.{H}.{amap_tmp.name}.nii'))

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
            mask_out = amap_tmp.save_as_image(os.path.join(self.rois_dir, f'Hem.{H}.nii'))
