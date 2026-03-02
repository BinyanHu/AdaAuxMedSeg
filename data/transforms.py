from typing import Sequence

import tensorflow as tf


class Transform:
    def __init__(self, random=False, batch=None, name=None):
        self.random = random
        self._batch = batch
        self.name = name

    @property
    def batch(self):
        return self._batch

    @batch.setter
    def batch(self, new_batch):
        if self._batch is None:
            self._batch = new_batch
        elif new_batch != self._batch:
            raise ValueError(f'Attempting to turn `{self}.batch=={self._batch}` to `{new_batch}`')

    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return self.call(dataset)

    def call(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        raise NotImplementedError

    def get_config(self):
        return {
            'random': self.random,
            'batch': self.batch,
            'name': self.name,
        }


class Map(Transform):
    def __init__(self, keys: str | Sequence[str], parallel=True, **kwargs):
        super().__init__(**kwargs)
        self.keys = convert_to_tuple(keys)
        self.parallel = parallel

    def call(self, dataset: tf.data.Dataset):
        return dataset.map(
            self.map_func,
            num_parallel_calls=tf.data.AUTOTUNE if self.parallel else None,
        )

    def map_func(self, example):
        raise NotImplementedError

    def get_config(self):
        return {
            **super().get_config(),
            'parallel': self.parallel,
            'keys': self.keys,
        }


class ToCategorical(Map):
    def __init__(
        self,
        depth,
        background=True,
        smoothing=0.,
        axis=-1,
        new_axis=True,
        dtype=tf.float32,
        keys='label',
        to_multi_label=False,
        multi_label_suffixes=(),
        **kwargs
    ):
        super().__init__(keys, **kwargs)
        self.depth = depth
        self.background = background
        self.smoothing = smoothing
        self.axis = axis
        self.new_axis = new_axis
        self.dtype = dtype
        if to_multi_label:
            assert multi_label_suffixes
        self.to_multi_label = to_multi_label
        self.multi_label_suffixes = multi_label_suffixes

        if smoothing:
            delta = smoothing / depth
            off_value = delta
            on_value = 1 - smoothing + delta
        else:
            off_value = 0
            on_value = 1
        self.on_value = tf.cast(on_value, self.dtype)
        self.off_value = tf.cast(off_value, self.dtype)

    def map_func(self, example):
        for key in self.keys:
            if key in example:
                label = example[key]
                if self.depth == 1:
                    assert label.shape[-1] == 1
                    label = tf.cast(label, self.dtype)
                    if self.new_axis:
                        label = label[..., None]
                else:
                    if not self.new_axis:
                        label = tf.squeeze(label, axis=self.axis)
                    depth = int(self.depth)
                    if not self.background:
                        label -= 1
                        depth -= 1
                    label = tf.one_hot(
                        label,
                        depth,
                        on_value=self.on_value, off_value=self.off_value,
                        axis=self.axis,
                        dtype=self.dtype
                    )
                if self.to_multi_label:
                    assert label.shape[self.axis] == len(self.multi_label_suffixes)
                    for i_suffix, suffix in enumerate(self.multi_label_suffixes):
                        curr_label = example[f'{key}/{suffix}'] = tf.gather(label, [i_suffix], axis=self.axis)
                        print(f'after to categorical: {key}/{suffix}: {curr_label}')
                    example.pop(key)
                else:
                    example[key] = label
                    print(f'after to categorical: {label}')
        return example

    def get_config(self):
        return {
            **super().get_config(),
            'depth': self.depth,
            'background': self.background,
            'smoothing': self.smoothing,
            'axis': self.axis,
            'new_axis': self.new_axis,
            'dtype': self.dtype,
            'to_multi_label': self.to_multi_label,
            'multi_label_suffixes': self.multi_label_suffixes,
        }


class ToChannelsFirst(Map):
    def __init__(self, keys=None, batch=True, **kwargs):
        '''
        keys : defaults to all elements
        '''
        super().__init__(keys=keys, batch=batch, name=None, **kwargs)

    def map_func(self, example):
        min_ndim = 2 if self.batch else 1
        # processed_keys = []
        for key in (self.keys or example):
            x = example[key]
            ndim = x.shape.rank
            if ndim > min_ndim:
                # processed_keys.append(key)
                if self.batch:
                    # (0 2 1), (0 3 1 2), (0 4 1 2 3)
                    perm = [0, ndim-1, *range(1, ndim-1)]
                else:
                    perm = [ndim-1, *range(ndim - 1)]
                x = tf.transpose(x, perm=perm)
                example[key] = x
        # self.keys = tuple(processed_keys)
        return example


class DropModality(Map):
    def map_func(self, example: dict):
        for k in self.keys:
            example.pop(k, None)
            for suffix in ['origin', 'spacing', 'mean', 'std']:
                example.pop(k+'/'+suffix, None)
        return example


class RandomMap(Map):
    def __init__(self, keys, seed=None, **kwargs):
        super().__init__(keys=keys, random=True, **kwargs)
        self.seed = seed

    def get_config(self):
        return {
            **super().get_config(),
            'seed': self.seed,
        }


class RandomFlip(RandomMap):
    def __init__(self, prob=0.5, return_flipping=False, keys: str | Sequence[str] = 'image', **kwargs):
        super().__init__(keys=keys, batch=False, **kwargs)
        self.prob = prob
        self.return_flipping = return_flipping

    def map_func(self, example):
        ndim = example[self.keys[0]].shape.rank - 1
        flippings = tf.random.uniform([ndim]) > np.broadcast_to(self.prob, ndim).astype(float)
        flip_axes = tf.where(flippings)[:, 0]
        for key in self.keys:
            example[key] = tf.reverse(example[key], flip_axes)
        if self.return_flipping:
            example['flipping'] = flippings
        return example

    def get_config(self):
        return {
            **super().get_config(),
            'prob': self.prob,
            'return_flipping': self.return_flipping,
        }
