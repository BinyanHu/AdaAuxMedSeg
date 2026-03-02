import abc

import tensorflow as tf
import tf_keras

from utils import get_channel_axis, get_spatial_axes

__all__ = [
    'SegmentationLoss',
    'DenseSegmentLoss',
    'DiceLoss',
    'soft_dice',
    'soft_confusion',
    'mxe_loss',
    'probs_from_logits',
]


class SegmentationLoss(tf.losses.Loss, metaclass=abc.ABCMeta):
    def __init__(
        self,
        from_sparse,
        from_logits,
        multi_label=None,
        axis=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if multi_label:
            from_sparse = False  # NOTE does not matter but simplifies process
        self.from_sparse = from_sparse
        self.from_logits = from_logits
        self.multi_label = multi_label
        self.axis = axis or get_channel_axis()

    def get_config(self):
        return {
            **super().get_config(),
            'from_sparse': self.from_sparse,
            'from_logits': self.from_logits,
            'multi_label': self.multi_label,
            'axis': self.axis,
        }


def _one_hot_from_sparse(
    y_true,
    num_classes,
    axis: int,
    smooth: float = 0.0,
    dtype=tf.float32
):
    if num_classes == 1:
        y_true_onehot = tf.clip_by_value(tf.cast(y_true, dtype), smooth, 1-smooth)
    else:
        if smooth:
            off_value = smooth / num_classes
            on_value = 1.0 - off_value * (num_classes-1)
        else:
            off_value = 0.0
            on_value = 1.0
        y_true_onehot = tf.one_hot(
            tf.cast(tf.squeeze(y_true, axis), tf.int32),
            depth=num_classes,
            on_value=on_value,
            off_value=off_value,
            axis=axis,
            dtype=dtype
        )

    return y_true_onehot


class DenseSegmentLoss(SegmentationLoss, metaclass=abc.ABCMeta):
    def __init__(
        self,
        from_sparse,
        from_logits,
        multi_label=None,
        axis=None,
        background: bool = False,
        **kwargs
    ):
        super().__init__(
            from_sparse=from_sparse,
            from_logits=from_logits,
            multi_label=multi_label,
            axis=axis,
            **kwargs
        )
        self.background = background

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        ''' Convert y_true to onehot form (one hot) and send to `self.call_from_onehot`

        Returns
        -------
        tf.Tensor:
            Loss values per sample.
        '''
        # Infer num_classes from `y_pred`
        num_classes = y_pred.shape[self.axis]

        trim_background = (not self.multi_label) and (not self.background) and (num_classes > 1)

        # Prepare `y_true_onehot`
        if self.from_sparse:
            if trim_background:
                # NOTE It's OK when y_true is unsigned because `num_classes` limits wrong behaviour.
                y_true -= 1
                num_classes -= 1
            y_true_onehot = _one_hot_from_sparse(
                y_true,
                num_classes=num_classes,
                axis=self.axis,
                dtype=tf.float32
            )
        else:
            y_true_onehot = y_true
            if trim_background:
                if self.axis == 1:
                    y_true_onehot = y_true_onehot[:, 1:]
                else:
                    y_true_onehot = y_true_onehot[..., 1:]
            y_true_onehot = tf.cast(y_true_onehot, tf.float32)

        # Prepare `y_pred`
        y_pred = tf.cast(y_pred, tf.float32)
        return self.call_from_onehot(y_true_onehot, y_pred)

    @abc.abstractmethod
    def call_from_onehot(self, y_true, y_pred):
        '''
        Parameters
        ----------
        y_true : (NDHWC)
            the onehot (one hot) label
        y_pred : (NDHWC)
            the logits
        '''
        raise NotImplementedError

    def get_config(self):
        return {
            **super().get_config(),
            'background': self.background,
        }


def probs_from_logits(logits, axis, multi_label):
    if multi_label or (logits.shape[axis] == 1):
        probs = tf.nn.sigmoid(logits)
    else:
        probs = tf.nn.softmax(logits, axis)
    return probs


def soft_confusion(y_true, y_pred, axis):
    '''get tn, fn, fp, tp matrix as input of `metric_from_confusion`

    Parameters
    ----------
    y_true : tf.Tensor, [0, 1], onehot
    y_pred : tf.Tensor, [0, 1], prob

    Returns
    -------
    Tuple[Tensor], 4 * [(N, C)]
                            y_pred
                (tn,    fn, Negative
                 fp,    tp) Positive
        y_true  False   True
    '''
    with tf.name_scope('soft_confusion'):
        spatial_axes = get_spatial_axes(y_pred.shape.rank, channel_axis=axis)  # HW of (NHWC), DHW of (NDHWC)
        true_pos, true_neg = y_true, (1 - y_true)
        pred_pos, pred_neg = y_pred, (1 - y_pred)
        tns = tf.reduce_sum(true_neg * pred_neg, axis=spatial_axes)  # (N, C)
        fns = tf.reduce_sum(true_pos * pred_neg, axis=spatial_axes)  # (N, C)
        fps = tf.reduce_sum(true_neg * pred_pos, axis=spatial_axes)  # (N, C)
        tps = tf.reduce_sum(true_pos * pred_pos, axis=spatial_axes)  # (N, C)
        return tns, fns, fps, tps


def soft_dice(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    *,
    axis: int,
    multi_label: None | bool = None,
    from_logits: bool,
    background: bool = False,
    epsilon: float = None,
    classwise: bool = False,
):
    ''' softmax/sigmoid + Dice Loss
    '''
    with tf.name_scope('soft_dice'):
        epsilon = epsilon or tf_keras.backend.epsilon()
        assert epsilon >= 0
        spatial_axes = get_spatial_axes(y_true.shape.rank, channel_axis=axis)
        num_classes = y_true.shape[axis]
        pred_num_channels = y_pred.shape[axis]

        if from_logits:
            y_pred = probs_from_logits(y_pred, axis, multi_label)

        if ((num_classes == 1) or multi_label) and background:
            assert pred_num_channels == num_classes, f'Expected {num_classes} vs. given {pred_num_channels}'
            tns, fns, fps, tps = soft_confusion(y_true, y_pred, axis)
            fp_fns = fps + fns
            foreground_dices = (2 * tps + epsilon) / (2 * tps + fp_fns + epsilon)
            background_dices = (2 * tns + epsilon) / (2 * tns + fp_fns + epsilon)
            dices = tf.concat([background_dices, foreground_dices], axis=1)
        else:
            if (pred_num_channels > 1) and (not multi_label) and (not background):
                # 'trim_background' case
                assert pred_num_channels == num_classes+1, f'Expected {num_classes+1} vs. given {pred_num_channels}'
                if axis == 1:
                    y_pred = y_pred[:, 1:]
                else:
                    y_pred = y_pred[..., 1:]
            tps = tf.reduce_sum(y_true * y_pred, axis=spatial_axes)
            tp_fns = tf.reduce_sum(y_true, axis=spatial_axes)
            tp_fps = tf.reduce_sum(y_pred, axis=spatial_axes)
            dices = (2 * tps + epsilon) / (tp_fns + tp_fps + epsilon)  # foreground_dices

        if classwise:
            return dices
        else:
            return tf.reduce_mean(dices, axis=1)


class DiceLoss(DenseSegmentLoss):
    ''' softmax/sigmoid + Dice Loss (DL)
    '''

    def __init__(
        self,
        from_sparse=True,
        from_logits=True,
        multi_label: None | bool = None,
        axis=None,
        background: bool = False,
        gamma: float = 1.,
        classwise: bool = False,
        epsilon: float = None,
        **kwargs
    ):
        super().__init__(
            from_sparse=from_sparse,
            from_logits=from_logits,
            multi_label=multi_label,
            axis=axis,
            background=background,
            **kwargs
        )
        assert gamma >= 1
        self.gamma = gamma
        self.classwise = classwise
        self.epsilon = epsilon

    def call_from_onehot(self, y_true, y_pred):
        losses = 1 - soft_dice(
            y_true, y_pred,
            axis=self.axis,
            from_logits=self.from_logits,
            background=self.background,
            multi_label=self.multi_label,
            classwise=True,
            epsilon=self.epsilon,
        )
        if self.gamma > 1:
            losses **= self.gamma
        if self.classwise:
            return losses
        else:
            return tf.reduce_mean(losses, axis=1)

    def get_config(self):
        return {
            **super().get_config(),
            'multi_label': self.multi_label,
            'gamma': self.gamma,
            'classwise': self.classwise,
            'epsilon': self.epsilon,
        }


def mxe_loss(y_true, y_pred, loss_id):
    with tf.name_scope(loss_id+'_loss'):
        assert y_true.shape == y_pred.shape
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        if loss_id == 'mae':
            losses = tf.abs(y_true - y_pred)
        elif loss_id == 'mse':
            losses = tf.math.squared_difference(y_true, y_pred)
        else:
            raise NotImplementedError(loss_id)
        return tf.reduce_mean(losses)
