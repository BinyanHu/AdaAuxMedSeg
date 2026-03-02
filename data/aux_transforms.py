from functools import partial
from itertools import permutations

import numpy as np
import tensorflow as tf
from absl import logging
from scipy import ndimage

from utils import get_nd_spatial_shape

from .transforms import Map, RandomMap, Transform

NEED_LABEL_TASKS = sorted(['seg', 'seg1', 'seg2', 'sdmin', 'sdmout', 'contour', 'segn'])
NEED_LABEL_AUX_TASKS = sorted(['sdmin', 'sdmout', 'contour'])
CONTRASTIVE_AUX_TASKS = sorted(['moco', 'vicreg'])


class BatchUnique(Transform):
    def __init__(self, batch_size: int, name=None):
        super().__init__(random=False, batch=False, name=name)
        # NOTE per-worker batch size
        self.batch_size = batch_size

    def call(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        def mark_duplicated(batch, example):
            id_ = example['id']
            if tf.reduce_any(batch == id_):
                id_ = tf.constant('')
            else:
                if tf.size(batch) == self.batch_size - 1:
                    batch = tf.constant([], dtype=tf.string)
                else:
                    batch = tf.concat([batch, [id_]], axis=0)
            example['id'] = id_
            return batch, example

        return dataset.scan(
            initial_state=tf.constant([], dtype=tf.string),
            scan_func=mark_duplicated,
        ).filter(lambda example: example['id'] != '')

    def get_config(self):
        return {
            **super().get_config(),
            'batch_size': self.batch_size,
        }


def center_from_bbox(bbox, spacing) -> tf.Tensor:
    with tf.name_scope('center_from_bbox'):
        bbox = tf.cast(bbox, tf.float32)
        shape = bbox.shape
        ndim = bbox.shape[-1] // 2
        bbox = tf.reshape(bbox, (*shape[:-1], 2, ndim))
        center = tf.reduce_mean(bbox, axis=-2)
        center *= spacing
        return center


class RandomAffineCropMulti(RandomAffineCrop):
    def __init__(
        self,
        crops,
        max_translation_mm=0.,  # NOTE in mm
        **kwargs,
    ):
        '''

        Parameters
        ----------
        max_translation_mm: float, in millimeter
            crop with the same center if == 0; not used if < 0.
        '''
        # if max_translation_mm >= 0:
        if 'return_center' in kwargs:
            assert kwargs['return_center'] == 'mm'
        else:
            kwargs['return_center'] = 'mm'
        super().__init__(**kwargs)
        self.crops = crops
        self.max_translation_mm = max_translation_mm

    def map_func(self, example: dict[str, tf.Tensor]):
        ndim = self.ndim
        return_center = self.return_center
        return_spacing = self.return_spacing

        # NOTE Set unit spacing for non-spacing dataset
        if f'{self.canonical}/spacing' not in example:
            unit_spacing = tf.ones([ndim], tf.float64)
            for key in self.keys + ('label',):
                if key.startswith('label/'):
                    continue
                logging.info('Setting unit spacing for: %s', key)
                example[f'{key}/spacing'] = unit_spacing
            del key

        canon_shape = get_nd_spatial_shape(example[self.canonical])
        canon_origin = example[f'{self.canonical}/origin']
        canon_spacing = example[f'{self.canonical}/spacing']
        if self.iso_threshold > 0:
            canon_isotropic = tf.reduce_max(canon_spacing) / tf.reduce_min(canon_spacing) < self.iso_threshold
        else:
            canon_isotropic = True

        canon_bbox_rel = self.get_bbox_rel_in_canon(example)

        def _transform_single_example(translation_mm=None):
            # NOTE skip `bboxes`
            canon_2_dst = self.random_affine_matrix(canon_shape, canon_spacing, canon_isotropic, canon_bbox_rel, translation_mm)
            dst_2_canon = tf.linalg.inv(canon_2_dst)

            dst_example = self.multikey_transform(example.copy(), dst_2_canon, canon_isotropic)
            # print(list(dst_example))  # NOTE 'label' is still there.
            for key in list(dst_example.keys()):
                if key.startswith('label'):
                    dst_example.pop(key)

            if return_center:
                canon_src_center = affine((self.dst_shape - 1) / 2, dst_2_canon)
                if return_center == 'pix':
                    dst_example['center'] = canon_src_center
                elif return_center == 'mm':
                    # NOTE Coord (mm) in src image from which the patch center cropped.
                    dst_example['center'] = canon_origin + canon_spacing * canon_src_center
                else:
                    raise NotImplementedError(return_center)

            if return_spacing == 'src':
                dst_example['spacing'] = canon_spacing
            elif return_spacing == 'dst':
                dst_example['spacing'] = affine(canon_spacing, canon_2_dst)
            elif return_spacing:
                raise NotImplementedError

            return dst_example

        canon_example = _transform_single_example()
        # NOTE Coord (mm) of center of src image
        canon_src_center_mm = canon_origin + canon_spacing * (tf.cast(canon_shape - 1, tf.float64) / 2)
        canon_translation_mm = canon_src_center_mm - canon_example['center']
        dst_examples = [canon_example]
        for _ in range(1, self.crops):
            if self.max_translation_mm >= 0:
                # NOTE reject sample within unit-sphere
                rel_translation_mm = tf.random.uniform([ndim], -1., 1., dtype=tf.float64)
                while tf.norm(rel_translation_mm) > 1:
                    rel_translation_mm = tf.random.uniform([ndim], -1., 1., dtype=tf.float64)
                rel_translation_mm *= self.max_translation_mm
                translation_mm = canon_translation_mm + rel_translation_mm
            else:
                translation_mm = None
            dst_examples.append(_transform_single_example(translation_mm))

        return {
            key: tf.stack([dst_example[key] for dst_example in dst_examples])
            for key in canon_example.keys()
        }

    def call(self, dataset: tf.data.Dataset):
        return super().call(dataset).unbatch()

    def get_config(self):
        return {
            'crops': self.crops,
            'max_translation_mm': self.max_translation_mm,
            **super().get_config(),
        }


class RandomRubikCube(RandomMap):
    def __init__(self, ndim, num_perms, flip, mask, **kwargs):
        super().__init__(keys=['image'], batch=False, **kwargs)
        self.ndim = ndim
        if ndim == 2:
            self.all_perms = tf.convert_to_tensor(list(permutations(range(4), 4)))
            self.num_perms = len(self.all_perms)
        else:
            assert ndim == 3
            self.num_perms = num_perms
            perm_path = f'research/selfsl/permutations_hamming_max_{num_perms}.npy'
            self.all_perms = tf.convert_to_tensor(np.load(perm_path))
            logging.info("Loaded %s from '%s'", self.all_perms.shape, perm_path)
        self.flip = flip
        self.mask = mask

    def map_func(self, example):
        ndim = self.ndim
        image = example['image']
        spatial_shape = np.array(image.shape[:-1])
        assert len(spatial_shape) == ndim
        num_patches = 2 ** ndim
        patch_shape = spatial_shape // 2
        assert (patch_shape * 2 == spatial_shape).all()
        num_perms = self.num_perms
        flip = self.flip
        mask = self.mask

        new_example = {}
        # new_example['id'] = example['id']

        # To patches
        patches = []
        if ndim == 2:
            h, w = patch_shape
            for y in range(2):
                y0 = y * h
                y1_ = y0 + h
                for x in range(2):
                    x0 = x * w
                    x1_ = x0 + w
                    patches.append(image[y0: y1_, x0: x1_])
        else:
            d, h, w = patch_shape
            for z in range(2):
                z0 = z * d
                z1_ = z0 + d
                for y in range(2):
                    y0 = y * h
                    y1_ = y0 + h
                    for x in range(2):
                        x0 = x * w
                        x1_ = x0 + w
                        patches.append(image[z0: z1_, y0: y1_, x0: x1_])
        patches = tf.stack(patches)  # [4/8(D)HWC]

        # Shuffle patches
        i_perm = tf.random.uniform([], maxval=num_perms, dtype=tf.int32)
        perm = self.all_perms[i_perm]
        # print(perm)
        patches = tf.gather(patches, perm)
        new_example['label/perm'] = i_perm
        # print(patches)

        # patches = tf.unstack(patches)

        # Rotate (actually flip) patches
        if flip > 0:
            flippings = tf.random.uniform([num_patches, flip]) > 0.5
            patches = tf.map_fn(
                lambda args: self.random_flip(*args),
                [patches, flippings],
                fn_output_signature=patches.dtype,
            )
            new_example['label/flip'] = flippings
            # print(patches)

        # Mask patches
        if mask:
            mask_indicators = tf.random.uniform([num_patches]) > 0.5
            patches = tf.map_fn(
                lambda args: self.random_mask(*args),
                [patches, mask_indicators],
                fn_output_signature=patches.dtype,
            )
            new_example['label/mask'] = mask_indicators
            # print(patches)

        new_example['image'] = patches

        return new_example

    def random_flip(self, patch, indicator):
        assert self.flip > 0
        axes = tf.where(indicator)[:, 0]
        if self.flip != self.ndim:
            assert (self.flip == 2) and (self.ndim == 3)
            axes += 1
        # tf.print(indicator, '->', axes)
        return tf.reverse(patch, axes)

    def random_mask(self, patch, indicator):
        if indicator:
            mask = tf.random.uniform(patch.shape) > 0.5
            patch = tf.where(mask, patch, tf.constant(0, patch.dtype))
        return patch

    def get_config(self):
        return {
            **super().get_config(),
            'num_perms': self.num_perms,
            'flip': self.flip,
            'mask': self.mask,
        }


class RKBReshape(Map):
    def __init__(self, **kwargs):
        super().__init__(keys=['image', 'label/flip', 'label/mask'], batch=True, **kwargs)

    def map_func(self, example):
        for key in self.keys:
            if key not in example:
                continue
            x = example[key]
            shape = x.shape
            x = tf.reshape(x, [shape[0] * shape[1], *shape[2:]])
            example[key] = x
        return example


class MultiCutout(RandomMap):
    def __init__(
        self,
        size,
        num=1,
        bound=False,
        pad_value=0,
        keys=('image',), dst_keys=('image',), mask_key=None,
        **kwargs
    ):
        super().__init__(keys=keys, batch=False, **kwargs)
        self.size = size
        self.num = num
        self.bound = bound
        self.pad_value = pad_value
        assert len(dst_keys) == len(keys)
        self.dst_keys = dst_keys
        self.mask_key = mask_key

    def map_func(self, example):
        mask_key = self.mask_key
        for key, dst_key in zip(self.keys, self.dst_keys):
            image = example[key]
            results = image.cutout(
                image,
                pad_size=self.size // 2,
                num=self.num,
                fill_value=self.pad_value,
                bound=self.bound,
                return_mask=bool(mask_key),
            )
            if mask_key:
                example[dst_key], example[mask_key] = results
            else:
                example[dst_key] = results
        return example

    def get_config(self):
        return {
            **super().get_config(),
            'dst_keys': self.dst_keys,
            'mask_key': self.mask_key,
            'size': self.size,
            'bound': self.bound,
            'num': self.num,
            'pad_value': self.pad_value,
        }


class MorphologyContour(Map):
    def __init__(self, size: int, drop_label, padding=0, **kwargs):
        super().__init__(keys='label', batch=False, **kwargs)
        self.size = size
        self.drop_label = drop_label
        self.padding = padding

    # def map_func(self, example):
    #     label = example['label']
    #     label = tf.cast(label, tf.bool)

    #     example[f'label/contour'] = tf.stack([
    #         _tf_morphology_contour(label[..., i_class], self.size)
    #         for i_class in range(label.shape[-1])
    #     ], axis=-1)

    #     if self.drop_label:
    #         example.pop('label')
    #     return example

    def map_func(self, example: dict[str, tf.Tensor]):
        padding = self.padding

        label_keys = [
            key
            for key in example.keys()
            if (key == 'label') or (key.startswith('label/') and not key.endswith(('origin', 'spacing')))
        ]
        # TODO handle onehot labels
        for key in label_keys:
            label = tf.cast(example[key], tf.bool)
            if padding:
                label = tf.pad(label, [[padding, padding]] * (label.shape.rank - 1) + [[0, 0]])
            example[key.replace('label', 'contour')] = tf.stack([
                _tf_morphology_contour(label[..., i_class], self.size)
                for i_class in range(label.shape[-1])
            ], axis=-1)

        # NOTE Should not modify the info for label.
        origin = example.get('label/origin', None)
        if origin is not None:
            origin = tf.identity(origin)
            spacing = example.get('label/spacing', None)
            padding_mm = padding
            if spacing is not None:
                spacing = tf.identity(spacing)
                example['contour/spacing'] = spacing
                padding_mm = padding * spacing
            origin -= padding_mm
            example['contour/origin'] = origin

        if self.drop_label:
            for key in label_keys:
                example.pop(key)
        return example

    def get_config(self):
        return {
            **super().get_config(),
            'size': self.size,
            'drop_label': self.drop_label,
            'padding': self.padding,
        }


def _tf_morphology_contour(label, size: int):
    src_shape = label.shape
    label = tf.numpy_function(
        partial(morphology_contour, size=size),
        [label],
        tf.bool,
        stateful=False,
    )
    label.set_shape(src_shape)
    return label


def morphology_contour(mask, size=1):
    '''morphology contour of `mask`

    According to [medpy](https://github.com/loli/medpy/blob/66265de8aedf6259feac00b897a22d0cf173d2e2/medpy/metric/binary.py#L1212), contour is obtained by `mask - dilate(mask)`.

    Parameters
    ----------
    mask : ndarray, bool, (HW) or (DHW)
        a binary mask with spatial dimensions only

    Returns
    -------
    ndarray, bool, (HW) or (DHW)
    '''
    # erode_size = size // 2
    # dilate_size = size - erode_size
    dilate_size = size // 2
    erode_size = size - dilate_size
    # print('dilate', dilate_size, '- erode', erode_size, )
    # structure = np.ones([3]*mask.ndim, bool)  # NOTE 8-connectivity
    if dilate_size > 0:
        dilated = ndimage.binary_dilation(mask, iterations=dilate_size)
    else:
        dilated = mask
    if erode_size > 0:
        eroded = ndimage.binary_erosion(mask, iterations=erode_size)
    else:
        eroded = mask
    contour = dilated ^ eroded
    return contour


class SurfaceDistance(Map):
    def __init__(self, mode: str, drop_label, padding=None, **kwargs):
        super().__init__(keys='label', **kwargs)
        self.mode = mode
        self.drop_label = drop_label
        self.padding = padding

    def map_func(self, example):
        mode = self.mode
        task = 'sdm'+mode
        padding = self.padding

        # tf.print(example['id'])

        label_keys = [
            key
            for key in example.keys()
            if (key == 'label') or (key.startswith('label/') and not key.endswith(('origin', 'spacing')))
        ]
        for key in label_keys:
            label = tf.cast(example[key], tf.bool)
            if padding is not None:
                padding = np.array(padding)
                if padding.size == 1:
                    full_padding = np.full([label.shape.rank - 1, 2], padding)
                else:
                    assert padding.size == len(label.shape) - 1
                    full_padding = np.stack([padding, padding], axis=1)
                full_padding = full_padding.tolist() + [[0, 0]]
                print('full_padding', full_padding)
                # src_label_shape = tf.shape(label)
                label = tf.pad(label, full_padding)
                # tf.print(key, src_label_shape, '->', tf.shape(label))
            example[key.replace('label', f'sdm{mode}')] = tf.stack([
                _tf_surface_distance_map(label[..., i_class], mode)
                for i_class in range(label.shape[-1])
            ], axis=-1)

        # NOTE Should not modify the info for label.
        origin = example.get('label/origin', None)
        if origin is not None:
            origin = tf.identity(origin)
            spacing = example.get('label/spacing', None)
            padding_mm = padding
            if spacing is not None:
                spacing = tf.identity(spacing)
                example[f'{task}/spacing'] = spacing
                padding_mm = padding * spacing
            origin -= padding_mm
            example[f'{task}/origin'] = origin

        if self.drop_label:
            for key in label_keys:
                example.pop(key)
        return example

    def get_config(self):
        return {
            **super().get_config(),
            'mode': self.mode,
            'drop_label': self.drop_label,
            'padding': self.padding,
        }


def _tf_surface_distance_map(label, mode):
    src_shape = label.shape
    label = tf.numpy_function(
        partial(surface_distance_map, mode=mode),
        [label],
        tf.float64,
        stateful=False,
    )
    label.set_shape(src_shape)
    label = tf.cast(label, tf.float32)
    return label


def surface_distance_map(mask, mode='both'):
    if np.any(mask):
        if mode == 'in':
            sdm = ndimage.distance_transform_edt(mask)
        elif mode == 'out':
            sdm = ndimage.distance_transform_edt(~mask)
        else:
            assert mode == 'both'
            # sdm = ndimage.distance_transform_edt(~mask) - (ndimage.distance_transform_edt(mask) - 1)
            sdm = ndimage.distance_transform_edt(~mask) - ndimage.distance_transform_edt(mask)
    else:
        sdm = np.zeros(shape=mask.shape, dtype=float)
    return sdm
