import sys
import nitools as nt
import numpy as np
import surfAnalysisPy as surf
from pathlib import Path
import os
import nibabel as nb
import pyvista as pv
import matplotlib.pyplot as plt

# ROOT =  Path().resolve().parent
# sys.path.append(str(ROOT))

def load_border_vertices_xml(filepath):
    vertices = []
    inside_vertices_block = False
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if "<Vertices>" in line:
                inside_vertices_block = True
                line = line.replace("<Vertices>", "")
            if inside_vertices_block:
                if "</Vertices>" in line:
                    line = line.replace("</Vertices>", "")
                    inside_vertices_block = False
                if line:
                    numbers = [int(x) for x in line.split()]
                    vertices.extend(numbers)
    return np.array(vertices)

def plot_surf(fig, ax, surf_data, H, vmin=-10, vmax=10, cmap='viridis', col=0, thresh=.01, title=None,
              overlay='overlay'):

    Hem = ['L', 'R']
    h = Hem.index(H)

    atlasDir = 'atlases/'

    surf = nb.load(os.path.join(atlasDir, f'fs_LR.32k.{H}.inflated.surf.gii'))
    coords = surf.darrays[0].data
    faces = surf.darrays[1].data.astype(np.uint32)  # pyvista requires uint32
    faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32).flatten()
    sulc = nt.get_gifti_data_matrix(nb.load(os.path.join(atlasDir, 'fs_LR.32k.LR.sulc.dscalar.gii')))

    if isinstance(surf_data, nb.Cifti2Image):
        column_names = surf_data.header.get_axis(0).name
        giftis = nt.split_cifti_to_giftis(surf_data, type='func', column_names=column_names)
        data = nt.get_gifti_data_matrix(giftis[h])[:, col]
    elif isinstance(surf_data, nb.GiftiImage):
        data = nt.get_gifti_data_matrix(surf_data)[:, col]
    if isinstance(surf_data, np.ndarray):
        data = surf_data.copy() if overlay == 'rgb' else surf_data.copy()[:, col]

    if H == 'L':
        sulc = sulc[:len(data)]
    else:
        sulc = sulc[len(data):]

    if thresh is not None:
        mask = (data > thresh) | (data < -thresh)
        data = data.copy()
        data[~mask] = np.nan

    mesh = pv.PolyData(coords, faces)
    mesh.point_data["sulc"] = sulc
    mesh.point_data[overlay] = data

    border_verts = load_border_vertices_xml(os.path.join(atlasDir, f'fs_LR.32k.{H}.border'))
    border = coords[border_verts]
    s = np.nanpercentile(np.abs(sulc), 98)

    p = pv.Plotter(window_size=(7200, 7200), off_screen=True)
    p.add_mesh(mesh, scalars="sulc", cmap="Greys", lighting=True, clim=[-s, s], show_scalar_bar=False)
    p.add_mesh(mesh,
               scalars=overlay,
               cmap=cmap if overlay=='overlay' else None,
               rgb=overlay=='rgb',
               clim=[vmin, vmax],
               opacity=1,  # <-- lets sulc show through
               lighting=True,
               show_scalar_bar=False)
    p.add_points(border[::3], color='k', point_size=40, render_points_as_spheres=True)
    p.set_background("white")
    if H == 'L':
        p.view_vector((-.8, 0, 1))
    elif H == 'R':
        p.view_vector((.8, 0, 1))
    p.show(screenshot='tmp.png', jupyter_backend='none',)
    p.close()
    img = plt.imread('tmp.png')
    os.remove('tmp.png')
    h, w = img.shape[:2]
    pad = 220  # number of pixels to keep around the center
    cropped_img = img[h // 2 - 1920:h // 2 + 1500, w // 2 - 3000:w // 2 + 3000]
    ax.imshow(cropped_img)
    ax.axis('off')
    ax.set_title(title, fontsize=20, pad=18)

    return fig, ax