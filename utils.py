from itertools import chain, combinations
from math import prod
from pprint import pformat
from typing import Sequence

import tensorflow as tf
import tf_keras
from absl import logging
from tensorflow import Tensor, Variable
from tensorflow.python.util.object_identity import Reference
from tf_keras import Model
from tf_keras.layers import Layer
from tf_keras.src.mixed_precision.autocast_variable import AutoCastVariable


def unique_variables(variables: list[tf.Variable]) -> list[tf.Variable]:
    return [ref.deref() for ref in dict.fromkeys(var.ref() for var in variables)]  # NOTE Use `dict.fromkeys` to preserve order.


def batch_get_values(variables: list[Variable]) -> list[Tensor]:
    return [tf.convert_to_tensor(v) for v in variables]


def batch_assign_variables(variables: list[Variable], values: list[Variable | Tensor]):
    for var, val in zip(variables, values, strict=True):
        if val.dtype != var.dtype:
            val = tf.cast(val, var.dtype)
        var.assign(val)


def aggregate_models_from_updates(
    base_vals: list[Tensor | Variable],
    modelwise_update: list[list[Tensor | Variable]],
    agg_rates: list[Tensor | Variable],
):
    vals: list[Tensor] = []
    for base_val, modelwise_var in zip(base_vals, zip(*modelwise_update, strict=True), strict=True):
        vals.append(base_val + linear_combine(modelwise_var, agg_rates))
    return vals


def assign_aggregate_models_from_updates(
    dst_vars: list[Variable],
    src_base_params: list[Variable | Tensor],
    src_modelwise_update: list[list[Tensor | Variable]],
    agg_rates: list[Tensor | Variable],
):
    batch_assign_variables(
        dst_vars,
        aggregate_models_from_updates(src_base_params, src_modelwise_update, agg_rates)
    )


def aggregate_models(
    modelwise_params: list[list[Tensor | Variable]],
    agg_rates: list[Tensor | Variable],
):
    vals: list[Tensor] = []
    for modelwise_var in zip(*modelwise_params, strict=True):
        vals.append(linear_combine(modelwise_var, agg_rates))
    return vals


def assign_aggregate_models(
    dst_vars: list[Variable],
    src_modelwise_params: list[list[Tensor | Variable]],
    agg_rates: list[Tensor | Variable],
):
    batch_assign_variables(
        dst_vars,
        aggregate_models(src_modelwise_params, agg_rates)
    )


def update_ema_model(model, ema_model, rate):
    with tf.name_scope('update_ema_model'):
        for var, ema_var in zip(model.variables, ema_model.variables, strict=True):
            ema_var.assign(ema_var + (1-rate) * (var - ema_var))


def variable_core_name(var: Variable | str) -> str:
    if isinstance(var, str):
        name = var
    else:
        name = var.name
    return name.split(':')[0]


def clone_variables(src_vars: list[Variable], suffix=''):
    return [Variable(var, name=variable_core_name(var)+suffix) for var in src_vars]


def vector_from_variables(variables: list[Variable]) -> Tensor:
    return tf.concat([tf.reshape(var, shape=(-1,)) for var in variables], axis=0)


def values_from_vector(vector: Tensor, ref_vars: list[Variable]):
    vals = []
    index = 0
    for var in ref_vars:
        shape = var.shape
        size = prod(shape)
        vals.append(tf.reshape(vector[index:index+size], shape))
        index += size
    assert index == prod(vector.shape)
    return vals


def get_variable_attr(var: str | Variable):
    if isinstance(var, str):
        name = var
    else:
        name = var.name
    return variable_core_name(name.split('/')[-1])


def iter_layers(x: Layer):
    ''' Iterates the "most basic" layer components.
    '''
    # NOTE bool(generator) does not mean empty
    layers = list(x._flatten_layers(recursive=True, include_self=False))
    if layers:
        for layer in layers:
            if list(layer._flatten_layers(recursive=False, include_self=False)):
                continue
            yield layer
    else:
        yield x


def model_ref_2_entry(model: Model):
    ref_2_entry: dict[Reference, tuple[Layer, str, Variable]] = {}
    for layer in iter_layers(model):
        for var in layer.variables:
            ref_2_entry[var.ref()] = (layer, get_variable_attr(var), var)
    return ref_2_entry


def set_model_with_values(
    entries: dict[Reference, tuple[Layer, str, Variable]] | list[tuple[Layer, str, Variable]],
    vals: dict[Reference, Tensor] | list[Tensor],
    mode='replace',
):
    if isinstance(vals, dict):
        assert isinstance(vals, dict)
        entries = [entries[ref] for ref in vals]
        vals = list(vals.values())
    else:
        assert isinstance(entries, list)
        assert isinstance(vals, list)
    assert len(entries) == len(vals)
    n = 0
    for (layer, attr, var), val in zip(entries, vals, strict=True):
        assert var.shape == val.shape
        if mode == 'replace':
            if isinstance(var, AutoCastVariable):
                val = tf.cast(val, tf.float16)
            setattr(layer, attr, val)
        else:
            assert mode == 'assign'
            var.assign(val)
        n += 1
    logging.info(f'{mode.title()} {n} variables.')


def set_model_with_src_variables(
    entries: dict[Reference, tuple[Layer, str, Variable]] | list[tuple[Layer, str, Variable]],
):
    if isinstance(entries, dict):
        entries = list(entries.values())
    else:
        assert isinstance(entries, list)
    for layer, attr, var in entries:
        setattr(layer, attr, var)


def linear_combine(vals: Sequence[Tensor | Variable], rates: Sequence[Tensor]) -> Tensor:
    # assert common(v.shape for v in vals) is not None
    dtype = vals[0].dtype
    if tf.is_tensor(rates):
        rates = tf.unstack(rates)
    return tf.add_n(v * tf.cast(r, dtype) for v, r in zip(vals, rates, strict=True))


def l2norm(xs: Sequence[Tensor | Variable]) -> Tensor:
    ''' (Global) l2 norm.
    '''
    return tf.math.sqrt(tf.add_n([tf.reduce_sum(x ** 2) for x in xs]))


def l2norm_from_list(x, y):
    return l2norm([xx - yy for xx, yy in zip(x, y, strict=True)])


def off_diag(x: Tensor):
    shape = x.shape
    assert shape[-2] == shape[-1]
    C = x.shape[-1]
    return tf.reshape(tf.reshape(tf.reshape(x, [-1])[:-1], [C-1, C+1])[:, 1:], [-1])


def log_object(obj):
    if hasattr(obj, "get_config"):
        logging.info("Built %s with:\n%s", type(obj).__name__, pformat(obj.get_config()))
    else:
        logging.info("Built %s with:\n%s", type(obj).__name__, obj)


def get_channel_axis(data_format=None):
    data_format = data_format or tf_keras.backend.image_data_format()
    return -1 if (data_format == "channels_last") else 1


def get_spatial_axes(ndim, channel_axis=None):
    axes = set(range(1, ndim))
    channel_axis = channel_axis or get_channel_axis()
    if channel_axis < 0:
        channel_axis += ndim
    axes.remove(channel_axis)
    return tuple(axes)


def get_nd_spatial_shape(image):
    ''' Spatial shape.
    '''
    if isinstance(image, dict):
        shape = image['shape']
        if shape.shape.rank == 2:  # NOTE batched shapes (?,3) (3D)
            assert shape.shape[1] == 3
            D = tf.shape(shape)[0]
            HWC = shape[0]
            shape = tf.concat([[D], HWC[:-1]], axis=0)
        else:
            shape = shape[:-1]
    else:
        shape = tf.shape(image)[:-1]
    return shape
