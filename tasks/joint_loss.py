from pprint import pformat

import tensorflow as tf
from tf_keras import Model

from . import TRAIN, MultiAuxiliaryTaskLearning


class JointLoss(MultiAuxiliaryTaskLearning):
    def build_model(self):
        seg_loader = self.data_loader[TRAIN]['train_seg']
        backbone_params, unet_params = self.get_model_params(
            input_spec=seg_loader.get_output_specs('image'),
            batch_size=seg_loader.strict_batch_size,
            head_channels=self.get_head_channels(seg_loader)
        )

        self.task_2_model: dict[str, Model | list[Model]] = {}
        self.model = self.get_model('seg', backbone_params, unet_params, share_vars_from=None)

        for task in self.ALL_TASKS:
            if task == 'seg':
                model = self.model
            else:
                assert self.model
                model = self.get_model(task, backbone_params, unet_params, share_vars_from=self.model)
            self.task_2_model[task] = model

    def build_train_step_func(self):
        self.mode_2_data_iter[TRAIN] = tf.nest.map_structure(iter, self.mode_2_dataset[TRAIN])
        self.mode_2_step_func[TRAIN] = tf.function(self.train_step)

    def train_step(self):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            self.watch_all_model_vars(tape)
            taskwise_loss = self.get_taskwise_loss(tape)
            if self.AUX_ONLY:
                loss = tf.add_n([r * l for r, l in zip(self.AUX_RATES, taskwise_loss, strict=True)])
            else:
                loss = tf.add_n([taskwise_loss[0]]+[r * l for r, l in zip(self.AUX_RATES, taskwise_loss[1:], strict=True)])

        self.minimize(tape=tape, loss=loss, var_list=tape.watched_variables())

        print(flush=True)

    def watch_all_model_vars(self, tape: tf.GradientTape):
        for task, model in self.task_2_model.items():
            if isinstance(model, list):
                assert task == 'moco'
                tape.watch(model[0].trainable_variables)
                tape.watch(model[2].trainable_variables)
            else:
                tape.watch(model.trainable_variables)

    def get_taskwise_loss(self, tape: tf.GradientTape):
        mode_2_data_iter = self.mode_2_data_iter[TRAIN]

        taskwise_loss: list[tf.Tensor] = []
        for task, model in self.task_2_model.items():
            mode = f'train_{"seg" if task.startswith("seg") else task}'
            with tape.stop_recording():
                examples = next(mode_2_data_iter[mode])
            examples = tf.nest.map_structure(tf.stop_gradient, examples)
            curr_loss = self.forward_task(task, examples, model, tape)
            taskwise_loss.append(curr_loss)

        print('Task-wise loss:\n'+pformat(taskwise_loss))
        return taskwise_loss
