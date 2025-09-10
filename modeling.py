import numpy as np
import PcmPy as pcm
import nibabel as nb
import os
from os import PathLike
import nitools as nt
from joblib import Parallel, delayed, parallel_backend
import pickle
import time
import errno
import AnatSearchlight.searchlight as sl

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

def find_model(M, name):
    if type(M) == str:
        f = open(M, 'rb')
        M = pickle.load(f)
    if type(M) == list:
        for m in M:
            if m.name == name:
                return m, M.index(m)
        if m == M[-1]:
            raise Exception(f'Model name not found')

def normalize_G(G):
    return (G - G.mean()) / G.std()

def normalize_Ac(Ac):
    for a in range(Ac.shape[0]):
        tr = np.trace(Ac[a] @ Ac[a].T)
        Ac[a] = Ac[a] / np.sqrt(tr)
    return Ac

def prewhiten(betas, res, lam=0.1, eps=1e-8):
    """
    betas: (n_cond, V)
    res:   (V,) ResMS  OR  residuals as (T, V) or (V, T)
    Returns: betas_wh, keep_mask
    """
    n_cond, V = betas.shape
    keep = np.ones(V, dtype=bool)

    # univariate prewhitening
    if res.ndim == 1:
        r = res.astype(float)
        bad = ~np.isfinite(r) | np.isclose(r, 0.0, atol=1e-6) | np.isnan(betas).all(axis=0)
        keep &= ~bad
        scale = np.sqrt(np.clip(r[keep], eps, None))
        return betas[:, keep] / scale

    # multivariate prewhitening
    R = res
    if R.shape == (V, R.shape[1]):     # (V, T)
        R = R.T                        # -> (T, V)
    if R.shape[1] != V:
        raise ValueError("Residuals do not match number of voxels in betas.")

    # drop bad voxels
    bad = ~np.isfinite(R).all(axis=0) | np.isclose(R.var(axis=0), 0.0, atol=1e-10) | np.isnan(betas).all(axis=0)
    keep &= ~bad
    R = R[:, keep]
    B = betas[:, keep]

    T = R.shape[0] - 1
    Sigma = (R.T @ R) / T

    # regularisation
    if lam and lam > 0:
        mu = np.mean(np.diag(Sigma))
        Sigma = (1 - lam) * Sigma + lam * mu * np.eye(Sigma.shape[0])

    w, U = np.linalg.eigh(Sigma)
    w = np.clip(w, eps, None)
    W = (U * (1.0 / np.sqrt(w))) @ U.T   # Σ^{-1/2}

    return B @ W


def calc_prewhitened_betas(betas: str | nb.Cifti2Image | nb.nifti1.Nifti1Image,
                           residuals: str | nb.Cifti2Image | nb.nifti1.Nifti1Image,
                           mask = str | nb.nifti1.Nifti1Image | np.ndarray,
                           struct_names=['CortexLeft', 'CortexRight'],
                           lam: float=0.1, eps: float=1e-8):
    """
    Get pre-whitened betas from ROI to submit to RSA/PCM
    Args:
        cifti_img:
        res_img:
        roi_img:
        struct_names:
        reg_mapping:
        reg_interest:

    Returns:

    """
    if isinstance(mask, str):
        mask = nb.load(roi_img)
    if isinstance(mask, nb.nifti1.Nifti1Image):
        coords = nt.get_mask_coords(mask)
    if isinstance(mask, np.ndarray):
        assert (mask.ndim==2) & (mask.shape[0]==3), "if mask is an array it must have shape (3, P)"
        coords = mask

    if isinstance(betas, str):
        cifti_img = nb.load(betas)
    if isinstance(betas, nb.Cifti2Image):
        beta_img = nt.volume_from_cifti(betas, struct_names=struct_names)
    if isinstance(betas, nb.nifti1.Nifti1Image):
        beta_img = betas
    betas = nt.sample_image(beta_img, coords[0], coords[1], coords[2], interpolation=0).T

    if isinstance(residuals, str):
        residuals = nb.load(residuals)
    if isinstance(residuals, nb.Cifti2Image):
        residuals = nt.volume_from_cifti(residuals, struct_names=struct_names)
    if isinstance(residuals, nb.nifti1.Nifti1Image):
        res = nt.sample_image(residuals, coords[0], coords[1], coords[2], interpolation=0)

    n_cond, V = betas.shape
    keep = np.ones(V, dtype=bool)

    # univariate prewhitening
    if res.ndim == 1:
        print('Doing univariate prewhitening...')
        r = res.astype(float)
        bad = ~np.isfinite(r) | np.isclose(r, 0.0, atol=1e-6) | np.isnan(betas).all(axis=0)
        keep &= ~bad
        scale = np.sqrt(np.clip(r[keep], eps, None))
        return betas[:, keep] / scale

    # multivariate prewhitening
    print('Doing multivariate prewhitening...')
    R = res
    if R.shape == (V, R.shape[1]):  # (V, T)
        R = R.T  # -> (T, V)
    if R.shape[1] != V:
        raise ValueError("Residuals do not match number of voxels in betas.")

    # drop bad voxels
    bad = ~np.isfinite(R).all(axis=0) | np.isclose(R.var(axis=0), 0.0, atol=1e-10) | np.isnan(betas).all(axis=0)
    keep &= ~bad
    R = R[:, keep]
    B = betas[:, keep]

    T = R.shape[0] - 1
    Sigma = (R.T @ R) / T

    # regularisation
    if lam and lam > 0:
        mu = np.mean(np.diag(Sigma))
        Sigma = (1 - lam) * Sigma + lam * mu * np.eye(Sigma.shape[0])

    w, U = np.linalg.eigh(Sigma)
    w = np.clip(w, eps, None)
    W = (U * (1.0 / np.sqrt(w))) @ U.T  # Σ^{-1/2}

    return B @ W

class PcmRois():
    """
    Class to perform PCM within a set of region of interest in parallel
    """

    def __init__(self,
                 sns: list,
                 M: list,
                 glm_path: PathLike,
                 cifti_img: str,
                 res_img: str,
                 roi_path: PathLike,
                 structnames: list = ['CortexLeft'],
                 regressor_mapping: dict = None,
                 regr_interest: list = None,
                 n_jobs: int = 12,
                 batch_size: int = 8):
        # def __init__(self, sns=None, M=None, glm_path=None, cifti_img=None, res_img='ResMS.nii', roi_path=None,
        #              roi_imgs=None, regressor_mapping=None, struct_names=['CortexLeft', 'CortexRight'],
        #              regr_of_interest=None, cond_order=None, n_jobs=16):
        self.snS = sns  # participants ids
        self.M = M  # pcm models to fit
        self.glm_path = glm_path  # path to cifti_img
        self.cifti_img = cifti_img  # name of cifti_img
        self.roi_path = roi_path  # path to individual roi masks, which must be named <atlas_name>.<H>.<roi>.nii
        self.roi_imgs = roi_imgs  # name of roi files to use as masks, e.g. ROI.L.M1.nii or cerebellum.L.nii
        self.regressor_mapping = regressor_mapping  # dict, maps name of regressors to numbers to control in which order conditions appear in the G matrix
        self.regr_of_interest = regr_of_interest  # indexes from regressor mapping of the regressors we want to include in the analysis
        self.cond_order = cond_order # order in which conditions should appear in the G matrix
        self.res_img = res_img
        self.struct_names = struct_names
        self.n_jobs = n_jobs

    def _make_dataset_within(self, sn, centre):
        print(f'making dataset...{sn} - ROI: {centre}')

        # load betas
        cifti_img = nb.load(os.path.join(self.glm_path, sn, self.cifti_img), mmap=False)
        beta_img = nt.volume_from_cifti(cifti_img, struct_names=[self.structnames])

        # load residuals
        res_img = nb.load(os.path.join(self.glm_path, sn, self.res_img))
        if isinstance(res_img, nb.Cifti2Image):
            res_img = nt.volume_from_cifti(res_img, struct_names=[self.structnames])

        # load reginfo
        reginfo = np.char.split(cifti_img.header.get_axis(0).name, sep='.')
        cond_vec = np.array([r[0] for r in reginfo])
        part_vec = np.array([int(r[1]) for r in reginfo])

        if self.regressor_mapping is not None:
            cond_vec = np.vectorize(self.regressor_mapping.get)(cond_vec)

        # do prewhitening
        betas_prewhitened = calc_prewhitened_betas(betas=beta_img, residuals=res_img, mask=coords)

        # make PCM dataset
        obs_des = {'cond_vec': cond_vec,
                   'part_vec': part_vec}
        if self.regr_interest is not None:
            idx = np.isin(cond_vec, self.regr_interest)
            betas_prewhitened = betas_prewhitened[idx]
            obs_des = {'cond_vec': cond_vec[idx],
                       'part_vec': part_vec[idx]}
        Y = pcm.dataset.Dataset(betas_prewhitened, obs_descriptors=obs_des)

        # est 2nd moment
        G_obs, _ = pcm.est_G_crossval(Y.measurements,
                                      Y.obs_descriptors['cond_vec'],
                                      Y.obs_descriptors['part_vec'],
                                      X=pcm.matrix.indicator(Y.obs_descriptors['part_vec']) #if demean else None
                                      )
        return Y, G_obs

    def _make_roi_dataset_within(self, roi_img, sn):
        print(f'making dataset...subj{sn} - {roi_img}')
        betas_prewhitened, obs_des = calc_prewhitened_betas(glm_path=self.glm_path + '/' + f'subj{sn}',
                                                            cifti_img='beta.dscalar.nii',
                                                            res_img=self.res_img,
                                                            roi_path=self.roi_path,
                                                            roi_img=f'subj{sn}' + '/' + roi_img,
                                                            struct_names=self.struct_names,
                                                            reg_mapping=self.regressor_mapping,
                                                            reg_interest=self.regr_of_interest, )
        Y = pcm.dataset.Dataset(betas_prewhitened, obs_descriptors=obs_des)
        G_obs, _ = pcm.est_G_crossval(Y.measurements,
                                      Y.obs_descriptors['cond_vec'],
                                      Y.obs_descriptors['part_vec'],
                                      X=pcm.matrix.indicator(Y.obs_descriptors['part_vec']) #if demean else None
                                     )
        return Y, G_obs

    def _make_dataset_between(self, roi_img):
        G_obs = list()
        Y = list()
        for s, sn in enumerate(self.snS):
            print(f'making dataset...subj{sn} - {roi_img}')
            Y_tmp, G_obs_tmp = self._make_dataset_within(sn, roi_img)
            Y.append(Y_tmp)
            G_obs.append(G_obs_tmp)
        G_obs = np.array(G_obs)

        return Y, G_obs

    def _fit_model_to_dataset(self, Y):
        T_in, theta_in = pcm.fit_model_individ(Y, self.M, fit_scale=False, verbose=True, fixed_effect='block')
        T_cv, theta_cv = pcm.fit_model_group_crossval(Y, self.M, fit_scale=True, verbose=True, fixed_effect='block')
        T_gr, theta_gr = pcm.fit_model_group(Y, self.M, fit_scale=True, verbose=True, fixed_effect='block')

        return T_in, theta_in, T_cv, theta_cv, T_gr, theta_gr

    def run_pcm_in_roi(self, roi_img):
        Y, G_obs = self._make_dataset_between(roi_img)
        T_in, theta_in, T_cv, theta_cv, T_gr, theta_gr = self._fit_model_to_dataset(Y)

        return G_obs, T_in, theta_in, T_cv, theta_cv, T_gr, theta_gr

    def run_parallel_pcm_across_rois(self):
        ##Parallel processing of rois
        with parallel_backend("loky"):  # use threading for debug in PyCharm, for run use loky
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.run_pcm_in_roi)(roi)
                for roi in self.roi_imgs
            )

        # for roi in self.roi_imgs:
        #     self.run_pcm_in_roi(roi)

        results = self._extract_results_from_parallel_process(results,
                                      field_names=['G_obs', 'T_in', 'theta_in', 'T_cv', 'theta_cv', 'T_gr', 'theta_gr'])
        return results

    def fit_model_family_across_rois(self, model, basecomp=None, comp_names=None):
        with parallel_backend("loky"):  # use threading for debug in PyCharm, for run use loky
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.fit_model_family_in_roi)(roi, model, basecomp, comp_names)
                for roi in self.roi_imgs
            )
        results = self._extract_results_from_parallel_process(results, ['T', 'theta'])
        return results

    def fit_model_family_in_roi(self, roi_img, model, basecomp=None, comp_names=None):
        M, _ = find_model(self.M, model)
        if isinstance(M, pcm.ComponentModel):
            G = M.Gc
            MF = pcm.model.ModelFamily(G, comp_names=comp_names, basecomponents=basecomp)
        elif isinstance(M, pcm.FeatureModel):
            MF = pcm.model.ModelFamily(M, comp_names=comp_names, basecomponents=basecomp)
        Y, _ = self._make_roi_dataset(roi_img)
        T, theta = pcm.fit_model_individ(Y, MF, verbose=True, fixed_effect='block', fit_scale=False)

        return T, theta

    def _extract_results_from_parallel_process(self, results, field_names):
        res_dict = {key: [] for key in ['roi_img'] + field_names}
        for r, result in enumerate(results):
            if len(result) != len(field_names):
                raise ValueError(f"Expected {len(field_names)} values, got {len(result)} at index {r}")
            res_dict['roi_img'].append(self.roi_imgs[r])
            for key, value in zip(field_names, result):
                res_dict[key].append(value)
        return res_dict


def _retry_eagain(callable_, *args, retries=12, delay=0.25, **kwargs):
    """Retry a callable on transient EAGAIN (errno 11) with exponential backoff."""
    for i in range(retries):
        try:
            return callable_(*args, **kwargs)
        except (BlockingIOError, OSError) as e:
            e_no = getattr(e, "errno", None)
            if e_no in (errno.EAGAIN, 11) and i < retries - 1:
                time.sleep(delay * (2 ** i))
                continue
            raise  # not EAGAIN, or out of retries


class PcmSearchlight():
    def __init__(self,
                 sns: list,
                 M: list,
                 glm_path: PathLike,
                 cifti_img: str,
                 res_img: str,
                 searchlight_path: PathLike,
                 structnames: list=['CortexLeft'],
                 regressor_mapping: dict=None,
                 regr_interest: list=None,
                 n_jobs: int=16,
                 batch_size: int=8):

        self.M = M # list of models to fit
        self.sns = sns # participants ids

        # paths
        self.glm_path = glm_path
        self.searchlight_path = searchlight_path

        # img names
        self.cifti_img = cifti_img
        self.res_img = res_img

        # struct names (for cifti imgs) and hemispheres
        self.structnames = structnames
        if self.structnames=='CortexLeft':
            self.H = 'L'
        if self.structnames=='CortexRight':
            self.H = 'R'

        self.regressor_mapping = regressor_mapping
        self.regr_interest = regr_interest

        # load betas, residuals and seachlights once
        # self.coords = self._load_cortical_searchlight()
        self.n_centre = 32492
        # self.betas, self.residuals, self.cond_vec, self.part_vec = self._load_betas_and_residuals()

        # parallel analysis setup
        self.n_jobs = n_jobs
        self.batch_size = batch_size

    def _make_dataset_within(self, sn, centre):
        print(f'making dataset...{sn} - centre: {centre}')

        # load betas
        cifti_img = nb.load(os.path.join(self.glm_path, sn, self.cifti_img), mmap=False)
        beta_img = nt.volume_from_cifti(cifti_img, struct_names=[self.structnames])

        # load residuals
        res_img = nb.load(os.path.join(self.glm_path, sn, self.res_img), mmap=False)
        if isinstance(res_img, nb.Cifti2Image):
            res_img = nt.volume_from_cifti(res_img, struct_names=[self.structnames])

        # load reginfo
        reginfo = np.char.split(cifti_img.header.get_axis(0).name, sep='.')
        cond_vec = np.array([r[0] for r in reginfo])
        part_vec = np.array([int(r[1]) for r in reginfo])

        if self.regressor_mapping is not None:
            cond_vec = np.vectorize(self.regressor_mapping.get)(cond_vec)

        # load searchlight
        searchlight = sl.load(os.path.join(self.searchlight_path, sn, f'searchlight.{self.H}.h5'), single=True, idx=centre)
        shape = searchlight.shape
        affine = searchlight.affine
        vn = searchlight.voxlist
        v_idx = searchlight.voxel_indx
        coords = nt.affine_transform_mat(v_idx[:, vn], affine)

        # do prewhitening
        betas_prewhitened = calc_prewhitened_betas(betas=beta_img, residuals=res_img, mask=coords)

        # make PCM dataset
        obs_des = {'cond_vec': cond_vec,
                   'part_vec': part_vec}
        if self.regr_interest is not None:
            idx = np.isin(cond_vec, self.regr_interest)
            betas_prewhitened = betas_prewhitened[idx]
            obs_des = {'cond_vec': cond_vec[idx],
                       'part_vec': part_vec[idx]}
        Y = pcm.dataset.Dataset(betas_prewhitened, obs_descriptors=obs_des)

        # est 2nd moment
        G_obs, _ = pcm.est_G_crossval(Y.measurements,
                                      Y.obs_descriptors['cond_vec'],
                                      Y.obs_descriptors['part_vec'],
                                      X=pcm.matrix.indicator(Y.obs_descriptors['part_vec']) #if demean else None
                                      )
        return Y, G_obs

    def _make_dataset_between(self, centre):
        G_obs = list()
        Y = list()
        for sn in self.sns:
            print(f'making dataset...{centre}')
            Y_tmp, G_obs_tmp = _retry_eagain(self._make_dataset_within, sn, centre)
            Y.append(Y_tmp)
            G_obs.append(G_obs_tmp)
        G_obs = np.array(G_obs)
        return Y, G_obs

    def _fit_model_to_dataset(self, Y):
        T_in, theta_in = pcm.fit_model_individ(Y, self.M, fit_scale=False, verbose=True, fixed_effect='block')
        T_cv, theta_cv = pcm.fit_model_group_crossval(Y, self.M, fit_scale=True, verbose=True, fixed_effect='block')
        T_gr, theta_gr = pcm.fit_model_group(Y, self.M, fit_scale=True, verbose=True, fixed_effect='block')
        return T_in, theta_in, T_cv, theta_cv, T_gr, theta_gr

    def _run_searchlight(self, centre):
        Y, G_obs = self._make_dataset_between(centre)
        try:
            T_in, theta_in, T_cv, theta_cv, T_gr, theta_gr = self._fit_model_to_dataset(Y)
            good = 1
        except:
            T_in, theta_in, T_cv, theta_cv, T_gr, theta_gr = [], [], [], [], [], []
            good = 0
        return G_obs, T_in, theta_in, T_cv, theta_cv, T_gr, theta_gr, good

    def run_seachlight_parallel(self):
        with parallel_backend("loky"):  # use threading for debug in PyCharm, for run use loky
            results = Parallel(n_jobs=self.n_jobs, batch_size=self.batch_size)(
                delayed(self._run_searchlight)(centre)
                for centre in range(self.n_centre)
            )
        G_obs, T_in, theta_in, T_cv, theta_cv, T_gr, theta_gr, good = map(list, zip(*results))
        return G_obs, T_in, theta_in, T_cv, theta_cv, T_gr, theta_gr, good




