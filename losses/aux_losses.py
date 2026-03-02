import tensorflow as tf
import tf_keras
from tf_keras import Model

from losses import mxe_loss
from utils import off_diag, update_ema_model


def handle_contour_loss(examples, outputs, metrics, loss_func):
    loss = tf.reduce_mean(loss_func(examples['contour'], outputs))
    metrics['loss/contour'] = loss
    return loss


def handle_sdm_loss(labels, preds, task, metrics, loss_id, focal, div_rate):
    with tf.name_scope('sdm_loss'):
        labels = labels / div_rate
        print('SDM /', div_rate)
        if focal:
            if focal == 'label':
                masks = (labels > 0)
            elif focal == 'union':
                masks = (labels > 0) | (preds > 0)
            else:
                raise NotImplementedError
            # tf.print('focused # pixels', tf.reduce_sum(tf.cast(masks, tf.int32)))
            labels = labels[masks]
            preds = preds[masks]
        else:
            preds = tf.nn.relu(preds)
        loss = mxe_loss(labels, preds, loss_id)
        metrics[f'loss/{task}'] = loss
        return loss


def handle_restore_loss(labels, preds, task, metrics, masks, loss_id):
    with tf.name_scope('restore_loss'):
        if masks is not None:
            masks = tf.broadcast_to(masks, labels.shape)
            labels = labels[masks]
            preds = preds[masks]
        loss = mxe_loss(labels, preds, loss_id)
        metrics[f'loss/{task}'] = loss
        return loss


def handle_rkb_loss(examples, outputs, metrics, flip: int, mask: bool):
    with tf.name_scope('rkb_loss'):
        loss_list = []
        for subtask in ['perm', 'flip', 'mask']:
            if (subtask == 'flip') and (flip == 0):
                continue
            if (subtask == 'mask') and (not mask):
                continue
            labels = examples[f'label/{subtask}']
            logits = outputs[f'pred/{subtask}']
            # NOTE different loss for each subtask
            if subtask == 'perm':
                with tf.name_scope('perm_loss'):
                    print(subtask, 'sparse_categorical_crossentropy', labels, logits)
                    loss = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
                    accuracy = tf.metrics.sparse_categorical_accuracy(labels, logits)
            elif subtask == 'flip':
                with tf.name_scope('flip_loss'):
                    labels = tf.reshape(labels, [-1])
                    logits = tf.reshape(logits, [-1])
                    print(subtask, 'binary_crossentropy', labels, logits)
                    loss = tf.reduce_mean(tf.losses.binary_crossentropy(labels, logits, from_logits=True))
                    accuracy = tf.metrics.binary_accuracy(tf.cast(labels, tf.float32), logits, threshold=0)
            else:
                assert subtask == 'mask'
                with tf.name_scope('mask_loss'):
                    logits = tf.reshape(logits, [-1])
                    print(subtask, 'binary_crossentropy', labels, logits)
                    loss = tf.reduce_mean(tf.losses.binary_crossentropy(labels, logits, from_logits=True))
                    accuracy = tf.metrics.binary_accuracy(tf.cast(labels, tf.float32), logits, threshold=0)
            loss_list.append(loss)
            metrics[f'loss/{subtask}'] = loss
            metrics[f'accuracy/{subtask}'] = accuracy
        return tf.add_n(loss_list) / len(loss_list)


def handle_moco_loss(xs, model: list[Model], outputs, tape, metrics, t=1):
    with tf.name_scope('moco_loss'):
        CTX: tf.distribute.ReplicaContext = tf.distribute.get_replica_context()

        xs_1, xs_2 = xs
        model, ema_model, predictor = model

        outputs_1, outputs_2 = outputs
        qs_1 = predictor(outputs_1['projection'], training=True)
        qs_2 = predictor(outputs_2['projection'], training=True)

        with tape.stop_recording():
            update_ema_model(model, ema_model, 0.99)
            ks_1 = ema_model(xs_1, training=True)['projection']
            ks_2 = ema_model(xs_2, training=True)['projection']

        ks_1, ks_2 = CTX.all_gather([ks_1, ks_2], axis=0)
        losses = tf.concat([cos_info_nce(ks_1, qs_2, metrics, t), cos_info_nce(ks_2, qs_1, metrics, t)], axis=0)
        return tf.reduce_mean(losses)


def cos_info_nce(k, q, metrics, t=1):
    '''
    k: true/label/target/momentum
    q: pred

    TODO consider B * n_workers in distributed learning
    '''
    with tf.name_scope('cos_info_nce'):
        assert k.shape == q.shape
        B = k.shape[0]
        k = tf.math.l2_normalize(tf.cast(k, tf.float32), axis=1)
        q = tf.math.l2_normalize(tf.cast(q, tf.float32), axis=1)
        logits = tf.einsum('nc,mc->nm', q, k) / t  # (len(q), len(k)), e.g., (256,1024)
        labels = tf.range(B) + B * 0  # NOTE i_worker in distributed learning.
        losses = t * tf_keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        metrics['accuracy/moco'] = tf_keras.metrics.sparse_categorical_accuracy(labels, logits)
        return losses


def handle_vicreg_loss(outputs, metrics, inv_rate, var_rate, cov_rate):
    with tf.name_scope('vicreg_loss'):
        sum_rate = inv_rate + var_rate + cov_rate
        inv_rate /= sum_rate
        var_rate /= sum_rate
        cov_rate /= sum_rate

        CTX: tf.distribute.ReplicaContext = tf.distribute.get_replica_context()

        x, y = CTX.all_gather([outputs[0]['projection'], outputs[1]['projection']], axis=0)
        mean_x, var_x = tf.nn.moments(x, axes=[0])
        mean_y, var_y = tf.nn.moments(y, axes=[0])
        x -= mean_x
        y -= mean_y

        loss_list = []
        # Invariance/Sim/repr_loss
        if inv_rate:
            with tf.name_scope('invariance'):
                inv_losses = tf.losses.mse(x, y)
                inv_loss = tf.reduce_mean(inv_losses)
                metrics['loss/invariance'] = inv_loss
                loss_list.append(inv_rate * inv_loss)
        # Variance
        if var_rate:
            with tf.name_scope('variance'):
                stds = tf.concat([tf.sqrt(var_x + 0.0001), tf.sqrt(var_y + 0.0001)], axis=0)
                var_losses = tf.nn.relu(1 - stds)
                var_loss = tf.reduce_mean(var_losses)
                metrics['loss/variance'] = var_loss
                loss_list.append(var_rate * var_loss)
        # Covariance
        # NOTE Different from Barlow: VICReg cov(x,x) and cov(y,y) vs. Barlow cov(x,y)
        if cov_rate:
            with tf.name_scope('covariance'):
                cov_loss = covariance_loss(x, y)
                metrics['loss/covariance'] = cov_loss
                loss_list.append(cov_rate * cov_loss)
        return tf.add_n(loss_list)


def covariance_loss(x, y):
    '''

    TODO consider B * n_workers in distributed learning
    '''
    assert x.shape == y.shape
    B, M = x.shape
    cov_x = tf.matmul(x, x, transpose_a=True) / (B - 1)
    cov_y = tf.matmul(y, y, transpose_a=True) / (B - 1)
    return tf.reduce_sum(tf.square(off_diag(cov_x))) / M + tf.reduce_sum(tf.square(off_diag(cov_y))) / M
