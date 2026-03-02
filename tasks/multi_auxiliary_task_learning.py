import re
from pprint import pformat, pprint
from typing import Any

import numpy as np
import tensorflow as tf
import tf_keras as keras
from absl import flags, logging
from absl.flags import FLAGS
from tensorflow import Tensor, Variable
from tensorflow.python.util.object_identity import Reference
from tf_keras import Model
from tf_keras.layers import BatchNormalization

from data import (CONTRASTIVE_AUX_TASKS, NEED_LABEL_AUX_TASKS, BatchUnique, DropModality, MedicalSegmentDataLoader, MorphologyContour, MultiCutout,
                  RandomAffineCropMulti, RandomFlip, RandomRubikCube, RKBReshape, SetImageAsLabel, SurfaceDistance, ToCategorical, ToChannelsFirst)
from losses import DiceLoss, handle_contour_loss, handle_moco_loss, handle_restore_loss, handle_rkb_loss, handle_sdm_loss, handle_vicreg_loss
from nn import MLP, EncoderProjector, MultiBranchUNet2D, MultiBranchUNet3D, RKBNet
from tasks import META, TRAIN, MedicalImageSegmentation
from utils import get_variable_attr, iter_layers, log_object


class MultiAuxiliaryTaskLearning(MedicalImageSegmentation):
    task_2_model: dict[str, Model | list[Model]]

    @classmethod
    def define_flags(cls):
        super().define_flags()
        flags.DEFINE_list('aux_tasks', [],
                          help='Name of auxiliary tasks.')
        flags.DEFINE_bool('aux_only', False,
                          help='Only perform auxiliary task.')
        # Restorative
        flags.DEFINE_string('restore_loss', 'mse',
                            help='Loss function for restorative tasks.')
        # AE
        flags.DEFINE_integer('ae_split_stages', 5,
                             lower_bound=0, upper_bound=5,
                             help='# split stages for AE.')
        # MAE
        flags.DEFINE_float('mae_cutout_shape', 1/4,
                           lower_bound=0, upper_bound=1,
                           help='Cutout shape.')
        flags.DEFINE_integer('mae_cutout_num', 1,
                             lower_bound=1,
                             help='# cutout patches per image.')
        flags.DEFINE_bool('mae_skip_connection', True,
                          help='Use skip connection to construct UNet.')
        flags.DEFINE_integer('mae_split_stages', 0,
                             lower_bound=0, upper_bound=5,
                             help='# split stages for MAE.')
        # SDM
        flags.DEFINE_integer('sdmin_split_stages', 0,
                             lower_bound=0, upper_bound=5,
                             help='# split stages for SDM-in.')
        flags.DEFINE_integer('sdmout_split_stages', 0,
                             lower_bound=0, upper_bound=5,
                             help='# split stages for SDM-out.')
        flags.DEFINE_float('sdmin_div', 1.,
                           lower_bound=0,
                           help='Divisor for normalizing in-SDM.')
        flags.DEFINE_float('sdmout_div', 100.,
                           lower_bound=0,
                           help='Divisor for normalizing out-SDM.')
        flags.DEFINE_enum('sdm_focal', 'union',
                          enum_values=['', 'label', 'union'],
                          help='Focus on foreground & erroneous regions.')
        flags.DEFINE_string('sdm_loss', 'mae',
                            help='Loss function for SDM prediction tasks.')
        flags.DEFINE_integer('contour_split_stages', 0,
                             lower_bound=0, upper_bound=5,
                             help='# split stages for Contour.')
        # RKB
        flags.DEFINE_bool('rkb_allow_flip', False,
                          help='Allow flip as DA in RKB.')
        flags.DEFINE_list('rkb_shape', [72, 72],
                          help='Image patch (after grid crop) shape for RKB.')
        flags.DEFINE_integer('rkb_perms', 24,
                             help='# perms in RKB task. Only valid for 3D data.')
        flags.DEFINE_integer('rkb_flip', 0,
                             help='Flip ? axes in RKB? Must be adapted to specified for different data.')
        flags.DEFINE_bool('rkb_mask', False,
                          help='Use mask op and task in RKB.')
        # Contrastive
        flags.DEFINE_float('cl_oversample', 1,
                           lower_bound=0, upper_bound=1,
                           help='Around-target oversampling rate for contrastive learning tasks.')
        flags.DEFINE_list('moco_shape', [128, 128],
                          help='Image patch shape for MoCo.')
        flags.DEFINE_list('vicreg_shape', [128, 128],
                          help='Image patch shape for VICReg.')
        flags.DEFINE_float('moco_max_translation_mm', 50.,
                           lower_bound=0.,
                           help='Maximum translation of the centers of a pair of patches from the same image.')
        flags.DEFINE_float('vicreg_max_translation_mm', 50.,
                           lower_bound=0.,
                           help='Maximum translation of the centers of a pair of patches from the same image.')
        flags.DEFINE_integer('cl_batch_size', 32,
                             help='# image *pairs* per batch.')
        flags.DEFINE_integer('cl_proj_channels', 1024,
                             help='# hidden and output units in projectors.')
        flags.DEFINE_float('cl_temper', 1,
                           help='Temperature for contrastive learning.s')
        flags.DEFINE_float('inv_rate', 25.,
                           help='Invariance regularization loss coefficient.')
        flags.DEFINE_float('var_rate', 25.,
                           help='Variance regularization loss coefficient.')
        flags.DEFINE_float('cov_rate', 1.,
                           help='Covariance regularization loss coefficient.')
        # Optimization
        flags.DEFINE_integer('select_n', 3,
                             help='Number of tasks to select.')
        flags.DEFINE_integer('select_freq', 1,
                             lower_bound=1,
                             help='When `select_alg=="comb" or "combseg"`, select tasks every ? epochs.')
        flags.DEFINE_enum('meta_alg', '',
                          enum_values=['', 'greedygrid', 'grad'],
                          help='Algorithm for meta search.')
        flags.DEFINE_integer('meta_freq', 1,
                             help='Interval for training meta weights. 0 means disable meta training (for debug). Negative value means epochs.')
        flags.DEFINE_integer('meta_steps', 1,
                             help='Steps for weight aggregation. Only valid when `meta_alg==grad`.')
        flags.DEFINE_float('meta_lr', 0.01,
                           lower_bound=0, upper_bound=1,
                           help='lr for agg optimizer')

    @classmethod
    def parse_flags(cls):
        AUX_ONLY = FLAGS.aux_only
        AUX_TASKS: list[str] = FLAGS.aux_tasks
        assert AUX_TASKS == sorted(AUX_TASKS)
        cls.N_AUX = len(AUX_TASKS)
        cls.HAS_CL = HAS_CL = bool(set(AUX_TASKS) & set(CONTRASTIVE_AUX_TASKS))

        if AUX_ONLY:
            if 'data' in FLAGS.modes:
                FLAGS.modes = ['train', 'data']
            else:
                FLAGS.modes = ['train']
            CONFIGS.saved_model_dir = 'saved_model-ssl'
            if HAS_CL:
                FLAGS.batch_size = FLAGS.cl_batch_size  # NOTE For multi-task pre-training

        super().parse_flags()

        if 'sdmin' in AUX_TASKS:
            assert FLAGS.sdmin_div > 0
        if 'sdmout' in AUX_TASKS:
            assert FLAGS.sdmout_div > 0
        if 'rkb' in AUX_TASKS:
            assert FLAGS.rkb_shape
            FLAGS.rkb_shape = list(map(int, FLAGS.rkb_shape))
        if HAS_CL:
            FLAGS.moco_shape = list(map(int, FLAGS.moco_shape))
            FLAGS.vicreg_shape = list(map(int, FLAGS.vicreg_shape))

        ALL_TASKS = list(AUX_TASKS)
        if not AUX_ONLY:
            ALL_TASKS = ['seg', *ALL_TASKS]
        cls.ALL_TASKS = ALL_TASKS
        cls.N_TASKS = len(ALL_TASKS)

    def build_data(self, mode_2_instruction):
        if self.META_ALG == 'grad':  # needs generator meta set.
            mode_2_instruction[META] = self.VALID_INSTR
        super().build_data(mode_2_instruction)

    def treat_as_train_data(self, mode: str):
        if mode.startswith(META):
            return True
        else:
            return super().treat_as_train_data(mode)

    def _transform_data(self, subloader: MedicalSegmentDataLoader):
        '''
        VALID set is defined as a static set of validation data, used for selection.
        META  set is defined as a generator of validation data, used for training.
        Both data can be augmented, but VALID always generates the same samples, but META generates infinite different samples.
        '''
        transform_seg_train_data = super().transform_train_data
        transform_seg_train_data(subloader, aug=self.META_AUG)
        return subloader

    def transform_train_data(self, subloader: MedicalSegmentDataLoader):
        ALL_TASKS = self.ALL_TASKS
        MODALITIES = FLAGS.modalities

        aug_ops = self.AUG.split('+')
        assert 'std' not in aug_ops
        assert 'batch' not in aug_ops
        rxxc_op = aug_ops[0]
        is_cc = (rxxc_op == 'cc')
        assert re.match('r|c.*c', rxxc_op)  # NOTE Only use the r.*c and ignore other augs for certain tasks.
        has_flip = (aug_ops[1] == 'f')

        # NOTE Redundant seg loader will be removed after building seg model.
        for task in (ALL_TASKS if 'seg' in ALL_TASKS else ['seg', *ALL_TASKS]):
            if (task != 'seg') and task.startswith('seg'):  # Aux seg tasks share the same loader
                continue
            task_loader = subloader[f'train_{task}']

            if task in NEED_LABEL_AUX_TASKS:
                if not task_loader.multi_label:
                    task_loader.transform(ToCategorical(
                        num_fg_classes=task_loader.num_fg_classes,
                        background=False,
                        new_axis=False,
                        dtype=tf.bool,
                        to_multi_label=True,
                        multi_label_suffixes=task_loader.class_names[1:],
                    ))
                    task_loader.class_names = task_loader.class_names[1:]
                    task_loader.class_colors = task_loader.class_colors[1:]
                if task == 'contour':
                    task_loader.transform(MorphologyContour(size=2, drop_label=False, padding=2))
                elif task == 'sdmin':
                    task_loader.transform(SurfaceDistance(mode='in', drop_label=False, padding=2))
                elif task == 'sdmout':
                    # assert all(s == self.SHAPE[0] for s in self.SHAPE)
                    if task_loader.ndim == 3:
                        # padding = int(self.SHAPE[0] // 2 + self.AUG_TRANSLATE)
                        padding = (np.array(self.SHAPE) // 2 + self.AUG_TRANSLATE).astype(int)
                    else:
                        # padding = int(self.SHAPE[0] + self.AUG_TRANSLATE)
                        padding = (np.array(self.SHAPE) + self.AUG_TRANSLATE).astype(int)
                    task_loader.transform(SurfaceDistance(mode='out', drop_label=False, padding=padding))
                else:
                    raise NotImplementedError(task)

            if self.DATA_CACHE:
                task_loader.cache()
            task_loader.repeat().shuffle(self.SHUFFLE_BUFFER)
            # rxxc
            assert not is_cc
            if task in CONTRASTIVE_AUX_TASKS:
                task_loader.transform(
                    BatchUnique(self.CL_BATCH_SIZE),
                    RandomAffineCropMulti(**self.get_deform_crop_params(rxxc_op, task_loader)),
                )
            else:
                task_loader.transform(self.get_data_transformation(rxxc_op, task_loader))
            # if task not in NEED_LABEL_TASKS:
            if not task.startswith('seg'):
                task_loader.transform(DropModality(keys=self.get_label_keys(task_loader)))
            # flip
            do_flip = has_flip
            if task == 'rkb':
                if has_flip and not FLAGS.rkb_allow_flip:
                    do_flip = False
            if do_flip:
                flip_keys = list(MODALITIES)
                if task.startswith('seg'):
                    flip_keys.append('label')
                if task in NEED_LABEL_AUX_TASKS:
                    flip_keys.append(task)
                task_loader.transform(RandomFlip(keys=flip_keys))
            self.standardize_data(task_loader)
            # post ops
            if task in ['ae', 'mae']:
                task_loader.transform(SetImageAsLabel(keys='image', label_key='label'))
                if task == 'mae':
                    task_loader.transform(MultiCutout(
                        size=(FLAGS.mae_cutout_shape * np.array(self.SHAPE)).round().astype(int),  
                        bound=task_loader.ndim == 3,  # too large chance to yield small patch on 3D
                        num=FLAGS.mae_cutout_num,
                        mask_key='label/mask',
                    ))
            elif task == 'rkb':
                task_loader.transform(
                    RandomRubikCube(
                        ndim=task_loader.ndim,
                        num_perms=FLAGS.rkb_perms,
                        flip=FLAGS.rkb_flip,
                        mask=FLAGS.rkb_mask,
                    )
                )
            # batch
            if task == 'rkb':
                task_loader.batch(self.BATCH_SIZE)
                task_loader.transform(RKBReshape())
            elif task in CONTRASTIVE_AUX_TASKS:
                task_loader.batch(self.CL_BATCH_SIZE * 2)
            else:
                task_loader.batch(self.BATCH_SIZE)

        subloader.transform(ToChannelsFirst())

        return subloader

    def get_deform_crop_params(self, id_: str, subloader):
        mode = subloader.mode
        task = mode.replace('train_', '')
        # if (not mode.startswith(TRAIN)) or (task in NEED_LABEL_TASKS):
        if (not mode.startswith(TRAIN)) or task.startswith('seg'):
            params = super().get_deform_crop_params(id_, subloader)
            # if task in NEED_LABEL_AUX_TASKS:
            #     class_names = subloader.class_names
            #     num_classes = len(class_names)
            #     params['label_like_keys'] = ['label', task]
            #     params['keys'] += [f'{task}/{c}' for c in class_names]
            #     params['sparse'] += [task == 'contour'] * num_classes
            #     params['order'] += [0] * num_classes
            #     params['pad_value'] += [0] * num_classes
            #     params['feature'] += [None] * num_classes
        else:
            params = MedicalImageTask.get_deform_crop_params(self, id_, subloader)
            if task in NEED_LABEL_AUX_TASKS:
                params['label_like_keys'] = ['label', task]
                if subloader.multi_label:
                    fg_class_names = subloader.fg_class_names
                    num_labels = len(fg_class_names)
                    params['keys'] += [key.replace('label', task) for key in self.get_label_keys(subloader)]
                else:
                    num_labels = 1
                    params['keys'].append(task)
                params['sparse'] += [task == 'contour'] * num_labels
                params['order'] += [0] * num_labels
                params['pad_value'] += [0] * num_labels
                params['feature'] += [None] * num_labels
            elif task == 'rkb':
                if not FLAGS.rkb_allow_flip:
                    params['flip_prob'] = 0
                params['translate_jitter'] = 0
                params['dst_shape'] = np.array(FLAGS.rkb_shape) * 2
            elif task in CONTRASTIVE_AUX_TASKS:
                params['crops'] = 2
                params['oversample'] = FLAGS.cl_oversample
                if task == 'moco':
                    params['dst_shape'] = FLAGS.moco_shape
                    params['max_translation_mm'] = FLAGS.moco_max_translation_mm
                elif task == 'vicreg':
                    params['dst_shape'] = FLAGS.vicreg_shape
                    params['max_translation_mm'] = FLAGS.vicreg_max_translation_mm
                else:
                    raise NotImplementedError(task)
        return params

    def get_data_options(self, mode):
        options = super().get_data_options(mode)
        if self.HAS_CL:
            # NOTE Ensure paired order
            options.deterministic = True
        return options

    def get_mode_2_steps(self):
        mode_2_steps = {}
        for mode, subloader in self.data_loader.items():
            if mode is TRAIN:
                if 'train_seg' in subloader:
                    steps = self.TRAIN_STEPS or subloader['train_seg'].steps_per_epoch
                else:
                    assert FLAGS.aux_only
                    if self.HAS_CL:
                        steps = max(1, round(subloader.num_examples / self.CL_BATCH_SIZE))
                    else:
                        steps = subloader.steps_per_epoch
            elif mode == META:
                META_STEPS = self.META_STEPS
                steps = META_STEPS if META_STEPS >= 0 else ((-META_STEPS) * mode_2_steps[TRAIN])
            else:
                steps = subloader.steps_per_epoch
            mode_2_steps[mode] = steps
        return mode_2_steps

    def get_model(
        self,
        task: str,
        backbone_params: dict[str, Any], unet_params: dict[str, Any],
        share_vars_from: None | nn.UNet2D,
    ) -> Model | list[Model]:
        data_loader = self.data_loader[self.mode]
        N_CHANNELS = data_loader.num_channels
        NFGCLASS = data_loader['train_seg'].nfgclass
        CL_PROJ_CHANNELS = FLAGS.cl_proj_channels

        backbone_name = backbone_params['name']
        unet_name = unet_params['name']

        logging.info(f'Building model for {task}.')

        if not hasattr(self, 'aux_2_shared_refs'):
            self.aux_2_shared_refs: dict[str, list[Reference]] = {}

        # NOTE 'seg' must be in this list if not `AUX_ONLY`
        if task.startswith('seg'):  # Create a seg model
            if task == 'segn':
                model = share_vars_from
            else:
                assert share_vars_from is None
                model = super().get_model(backbone_params, unet_params)
                self.BACKBONE_N_VARS = len(model.backbone.variables)
            if self.AUX_ONLY:
                self.data_loader[TRAIN].pop('train_seg')

        else:  # Create seg-aux siamese models
            curr_backbone_name = f'{backbone_name}_{task}'
            curr_backbone_params = backbone_params | dict(name=curr_backbone_name)
            curr_unet_name = f'{unet_name}_{task}'
            curr_unet_params = unet_params | dict(name=curr_unet_name)
            if task.startswith('sdm') or (task == 'contour'):
                model = super().get_model(
                    curr_backbone_params,
                    curr_unet_params | dict(
                        head_channels=NFGCLASS,
                    )
                )
            elif task == 'ae':
                model = super().get_model(
                    curr_backbone_params,
                    curr_unet_params | dict(
                        fuse=None,
                        head_channels=N_CHANNELS,
                    )
                )
            elif task == 'mae':
                model = super().get_model(
                    curr_backbone_params,
                    curr_unet_params | dict(
                        fuse='concat' if FLAGS.mae_skip_connection else None,
                        head_channels=N_CHANNELS,
                    )
                )
            elif task == 'rkb':
                model = self.get_rkb_model(
                    curr_backbone_params,
                    dict(
                        num_perms=FLAGS.rkb_perms,
                        flip=FLAGS.rkb_flip,
                        mask=FLAGS.rkb_mask,
                        name=curr_backbone_name
                    )
                )
            elif task == 'moco':
                MOCO_SHAPE = FLAGS.moco_shape
                model_params = dict(
                    hidden_channels=CL_PROJ_CHANNELS,
                    head_channels=256,
                    head_norm=True,
                    name=curr_backbone_name,
                )
                model = self.get_cl_model(
                    curr_backbone_params, model_params,
                    input_shape=MOCO_SHAPE
                )
                # EMA model
                ema_model = self.get_cl_model(
                    curr_backbone_params, model_params,
                    input_shape=MOCO_SHAPE
                )
                # Predictor
                predictor = self.get_moco_predictor(input_shape=model.output_shape['projection'])
            elif task == 'vicreg':
                model = self.get_cl_model(
                    curr_backbone_params,
                    dict(
                        hidden_channels=[CL_PROJ_CHANNELS, CL_PROJ_CHANNELS],
                        head_channels=CL_PROJ_CHANNELS,
                        head_norm=False,
                        name=curr_backbone_name,
                    ),
                    input_shape=FLAGS.vicreg_shape,
                )
            else:
                raise NotImplementedError(task)

            # NOTE Replace shared vars in the aux model
            if share_vars_from:
                if task == 'ae':
                    split_stages = FLAGS.ae_split_stages
                elif task == 'mae':
                    split_stages = FLAGS.mae_split_stages
                elif task == 'sdmin':
                    split_stages = FLAGS.sdmin_split_stages
                elif task == 'sdmout':
                    split_stages = FLAGS.sdmout_split_stages
                elif task == 'contour':
                    split_stages = FLAGS.contour_split_stages
                else:
                    split_stages = 5

                shared_refs = []
                shared_layers = list(iter_layers(share_vars_from.backbone))
                aux_layers = list(iter_layers(model.backbone))
                if hasattr(model, 'decoder'):
                    for module in share_vars_from.decoder:
                        shared_layers += list(iter_layers(module))
                    for module in model.decoder:
                        aux_layers += list(iter_layers(module))
                for seg_layer, aux_layer in zip(shared_layers, aux_layers):
                    for seg_var in seg_layer.variables:
                        seg_var_name = seg_var.name
                        if seg_var_name.startswith('u'):
                            i_stage = int(seg_var_name[1])
                            if i_stage < split_stages:
                                break
                        else:
                            assert seg_var_name.startswith('s'), seg_var_name
                        attr = get_variable_attr(seg_var_name)
                        aux_var: tf.Variable = getattr(aux_layer, attr)
                        assert aux_var.shape == seg_var.shape, f'Different var shape seg {seg_var.shape} vs. aux {aux_var.shape}'
                        setattr(aux_layer, attr, seg_var)
                        logging.info(f'Replaced variable: {aux_var.name} <- {seg_var_name}')
                        shared_refs.append((seg_var._variable if hasattr(seg_var, '_variable') else seg_var).ref())
                self.aux_2_shared_refs[task] = shared_refs
                logging.info(f'{task} shares {len(shared_refs)} variables with seg.')

        # Return built models
        if task == 'moco':
            # NOTE Moco EMA model's var need to be assigned late because of the moco model's vars will be replaced from seg.
            for var, ema_var in zip(model.variables, ema_model.variables, strict=True):
                ema_var.assign(var)
            return [model, ema_model, predictor]
        else:
            return model

    def build_optimizer(self):
        super().build_optimizer()

        self.task_2_do_train = {task: Variable(True, trainable=False, name=f'do_train/{task}') for task in self.ALL_TASKS}

        if self.META_ALG == 'grad':
            self.meta_optimizer = keras.optimizers.Adam(
                learning_rate=FLAGS.meta_lr,
            )
            log_object(self.meta_optimizer)

    def get_loss_params(self, loss_id='', train_loader=None):
        if self.has_train_mode:
            train_loader = self.data_loader[TRAIN]
            if FLAGS.aux_only:
                if 'train_contour' in train_loader:
                    return super().get_loss_params(loss_id=loss_id, train_loader=train_loader['train_contour'], label_key='contour')
                else:
                    return {}
            else:
                return super().get_loss_params(loss_id=loss_id, train_loader=train_loader['train_seg'])
        else:
            return super().get_loss_params(loss_id=loss_id, train_loader=train_loader)

    def on_mode_begin(self):
        super().on_mode_begin()
        if self.has_train_process and self.META_ALG == 'grad':
            self.mode_2_data_iter[META] = iter(self.mode_2_dataset[META])
        if self.has_train_process and 'contour' in self.ALL_TASKS:
            self.contour_loss = DiceLoss(
                from_sparse=False,
                from_logits=True,
                multi_label=True,
                background=True,
            )
            log_object(self.contour_loss)

    def forward_task(self, task: str, examples, model: Model | list[Model], tape) -> Tensor | list[Tensor]:
        print('Task:', task)
        pprint(examples)

        images = examples['image']
        metrics = {}

        # Forward
        if task in CONTRASTIVE_AUX_TASKS:
            xs_1, xs_2 = images[::2], images[1::2]
            print('Inputs:\n'+pformat([xs_1, xs_2]))
            if isinstance(model, list):
                _model = model[0]
            else:
                _model = model
            outputs = [_model(xs_1, training=True), _model(xs_2, training=True)]
        else:
            print('Inputs:\n'+pformat(images))
            outputs = model(images, training=True)
        outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)
        print('Outputs:\n'+pformat(outputs))

        # Loss
        if task.startswith('seg'):
            loss = self.handle_seg_loss(examples, outputs, metrics=metrics, data_loader=self.data_loader[TRAIN]['train_seg'])
            if task == 'segn':
                loss = -loss
        elif task == 'contour':
            loss = handle_contour_loss(examples, outputs, metrics, self.contour_loss)
        elif task.startswith('sdm'):
            if task == 'sdmin':
                div_rate = FLAGS.sdmin_div
            elif task == 'sdmout':
                div_rate = FLAGS.sdmout_div
            else:
                raise ValueError(task)
            loss = handle_sdm_loss(
                examples[task], outputs, task, metrics,
                loss_id=FLAGS.sdm_loss, focal=FLAGS.sdm_focal, div_rate=div_rate,
            )
        elif task == 'ae':
            loss = handle_restore_loss(
                examples['label'], outputs, task, metrics, masks=None,
                loss_id=FLAGS.restore_loss,
            )
        elif task == 'mae':
            loss = handle_restore_loss(
                examples['label'], outputs, task, metrics, masks=examples['label/mask'],
                loss_id=FLAGS.restore_loss,
            )
        elif task == 'rkb':
            loss = handle_rkb_loss(
                examples, outputs, metrics,
                flip=FLAGS.rkb_flip, mask=FLAGS.rkb_mask
            )
        elif task == 'moco':
            loss = handle_moco_loss(
                [xs_1, xs_2], model, outputs, tape, metrics,
                t=FLAGS.cl_temper
            )
        elif task == 'vicreg':
            loss = handle_vicreg_loss(
                outputs, metrics,
                inv_rate=FLAGS.inv_rate, var_rate=FLAGS.var_rate, cov_rate=FLAGS.cov_rate
            )
        else:
            raise NotImplementedError(task)

        self.update_metrics(values=metrics)

        print('Loss:\n'+pformat(loss), end='\n\n', flush=True)
        return loss

    def get_multi_branch_unet(self, backbone_params, unet_params):
        if self.UNET_CLASS is nn.UNet2D:
            model_class = MultiBranchUNet2D
        else:
            model_class = MultiBranchUNet3D
        pprint(unet_params)
        model = model_class(
            backbone=self.get_backbone(**backbone_params),
            **unet_params,
        )
        if self.verbose:
            self.log_model(model)
        return model

    def get_rkb_model(self, backbone_params, model_params):
        NDIM = self.data_loader.ndim
        BATCH_SIZE = FLAGS.batch_size
        RKB_SHAPE = FLAGS.rkb_shape

        input_spec = backbone_params['input_spec']
        # NOTE Override shape as the shape inferred from dataloader maybe incorrect
        shape = list(input_spec.shape)
        shape[1:] = RKB_SHAPE
        input_spec = tf.TensorSpec(shape=shape, dtype=input_spec.dtype, name=input_spec.name)
        batch_size = BATCH_SIZE * (2 ** NDIM)
        model = RKBNet(
            backbone=self.get_backbone(**(backbone_params | dict(input_spec=input_spec, batch_size=batch_size))),
            **model_params,
        )
        if self.verbose:
            self.log_model(model)
        return model

    def get_cl_model(self, backbone_params, model_params, input_shape=None):
        CL_BATCH_SIZE = FLAGS.cl_batch_size

        input_spec = backbone_params['input_spec']
        if input_shape:
            # NOTE Override shape as the shape inferred from dataloader maybe incorrect
            shape = list(input_spec.shape)
            shape[1:] = input_shape
            input_spec = tf.TensorSpec(shape=shape, dtype=input_spec.dtype, name=input_spec.name)

        model = EncoderProjector(
            backbone=self.get_backbone(**(backbone_params | dict(input_spec=input_spec, batch_size=CL_BATCH_SIZE))),
            **model_params,
        )
        if self.verbose:
            self.log_model(model)
        return model

    def get_moco_predictor(self, input_shape):
        CL_PROJ_CHANNELS = FLAGS.cl_proj_channels
        with nn.normalization_scope(BatchNormalization):
            moco_predictor = MLP([CL_PROJ_CHANNELS, 256], head_bias=False, head_norm=False, name='moco_predictor')
        moco_predictor.build(input_shape)
        if self.verbose:
            self.log_model(moco_predictor)
        return moco_predictor
