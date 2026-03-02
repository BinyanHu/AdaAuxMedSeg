from itertools import combinations, zip_longest

import tensorflow as tf
from absl import flags, logging
from absl.flags import FLAGS
from research.utils import get_variable_attr, set_model_with_src_variables, set_model_with_values
from tensorflow import Tensor, Variable
from tensorflow.python.util.object_identity import Reference
from tf_keras.layers import Layer

from tasks import META, TEST, TRAIN, VALID
from utils import iter_layers, unique_variables, variable_core_name

from . import ParameterAggregationBased


class Ours(ParameterAggregationBased):
    ZIP_TRAIN_DATASETS = False

    @classmethod
    def define_flags(cls):
        super().define_flags()
        flags.DEFINE_list('agg_epochs', [],
                          help='Epochs to agg models.')

    @classmethod
    def parse_flags(cls):
        super().parse_flags()
        FLAGS.agg_epochs = list(map(int, FLAGS.agg_epochs))

    def build_model(self):
        ALL_TASKS = self.ALL_TASKS

        seg_loader = self.data_loader[TRAIN]['train_seg']
        backbone_params, unet_params = self.get_model_params(
            input_spec=seg_loader.get_output_specs('image'),
            batch_size=seg_loader.strict_batch_size,
            head_channels=self.get_head_channels(seg_loader)
        )

        self.task_2_model = {
            task: self.get_model(task, backbone_params, unet_params, share_vars_from=None)
            for task in ALL_TASKS
        }
        self.model = self.task_2_model['seg']

    def build_optimizer(self):
        self.build_transferable_var_shortcuts()
        self.sync_transferable_vars_vals()
        super().build_optimizer()
        self.build_agg_vars()
        self.meta_optimizer.build(self.agg_vars_list)
        ALL_TASKS = self.ALL_TASKS
        self.varwise_taskwise_var_stale: list[list[Variable]] = []
        self.task_2_varwise_var_stale: dict[str, list[Variable]] = {task: [] for task in ALL_TASKS}
        self.varwise_taskwise_delta: list[list[Variable]] = []
        self.task_2_varwise_delta: dict[str, list[Variable]] = {task: [] for task in ALL_TASKS}
        for taskwise_var in self.varwise_taskwise_var:
            taskwise_var_stale = []
            taskwise_delta = []
            for task, var in zip(self.ref_2_tasks[taskwise_var[0].ref()], taskwise_var, strict=True):
                var_core_name = variable_core_name(var.name)
                var_stale = Variable(tf.convert_to_tensor(var), trainable=False, name='stale_'+var_core_name)
                taskwise_var_stale.append(var_stale)
                self.task_2_varwise_var_stale[task].append(var_stale)
                delta = Variable(tf.zeros_like(var), trainable=False, name='delta_'+var_core_name)
                taskwise_delta.append(delta)
                self.task_2_varwise_delta[task].append(delta)
            self.varwise_taskwise_var_stale.append(taskwise_var_stale)
            self.varwise_taskwise_delta.append(taskwise_delta)
        self.task_2_do_agg = {task: Variable(True, trainable=False, name=f'do_agg/{task}') for task in ALL_TASKS}

    def build_transferable_var_shortcuts(self):
        # NOTE If same layer type and same `var.shape` then can aggregate.
        # NOTE Never aggregate head.

        ALL_TASKS = self.ALL_TASKS
        # NOTE If possible, enable transfer from past seg model.
        AGG_RATE_COUPLED = (self.AGG_NORM == 'softmax')

        self.varwise_taskwise_var: list[list[Variable]] = []
        self.task_2_varwise_var: dict[str, list[Variable]] = {task: [] for task in ALL_TASKS}
        self.ref_2_tasks: dict[Reference, list[str]] = {}
        self.seg_ref_2_entry: dict[Reference, tuple[Layer, str, Variable]] = {}
        taskwise_model = [model[0] if isinstance(model, list) else model for model in self.task_2_model.values()]
        for taskwise_layer in zip_longest(*[iter_layers(model) for model in taskwise_model]):
            seg_layer = taskwise_layer[0]
            if seg_layer is None:
                return
            if not seg_layer.variables:
                continue
            curr_tasks, curr_taskwise_layer = [], []
            for task, layer in zip(ALL_TASKS, taskwise_layer, strict=True):
                if type(layer) is type(seg_layer):
                    curr_tasks.append(task)
                    curr_taskwise_layer.append(layer)
            for taskwise_var in zip(*[layer.variables for layer in curr_taskwise_layer], strict=True):
                seg_var = taskwise_var[0]
                seg_var_name = seg_var.name
                seg_ref = seg_var.ref()
                self.seg_ref_2_entry[seg_ref] = (seg_layer, get_variable_attr(seg_var), seg_var)

                taskwise_var = list(filter(None, taskwise_var))
                if seg_var_name[0] in 'su':
                    taskwise_can_agg = [var.shape == seg_var.shape for var in taskwise_var]
                    if (seg_var_name[0] == 'u') and 'ae' in curr_tasks:
                        # NOTE Not reasonable to share AE decoder params.
                        taskwise_can_agg[curr_tasks.index('ae')] = False
                else:
                    assert seg_var_name.startswith('head'), seg_var_name
                    taskwise_can_agg = [task.startswith('seg') for task in curr_tasks]
                if AGG_RATE_COUPLED and (sum(taskwise_can_agg) == 1):
                    # print('no more aux to agg transfer from.')
                    break
                self.ref_2_tasks[seg_ref] = []
                self.varwise_taskwise_var.append([])
                for task, var, can_agg in zip(curr_tasks, taskwise_var, taskwise_can_agg, strict=True):
                    if can_agg:
                        self.ref_2_tasks[seg_ref].append(task)
                        self.varwise_taskwise_var[-1].append(var)
                        self.task_2_varwise_var[task].append(var)
                logging.info(f'Transferable var {seg_var_name} among {self.ref_2_tasks[seg_ref]}.')

    def sync_transferable_vars_vals(self):
        logging.info('Sync %d var vals.', len(self.varwise_taskwise_var))
        # NOTE Redundant step if restore from `ckpt`. Only valid on initialization.
        # NOTE If task-related model is a list, their var sharing already handled in `build_model`.
        for taskwise_var in self.varwise_taskwise_var:
            seg_var = taskwise_var[0]
            for var in taskwise_var[1:]:
                var.assign(seg_var)
                logging.info(f'Sync var val: {var.name} <- {seg_var.name}')
        if 'moco' in self.task_2_model:
            logging.info('Sync Moco var vals.')
            model, ema_model, _ = self.task_2_model['moco']
            for var, ema_var in zip(model.variables, ema_model.variables, strict=True):
                ema_var.assign(var)

    def build_agg_vars(self):
        ALL_TASKS = self.ALL_TASKS
        AGG_GRAN = self.AGG_GRAN

        key_2_agg_vars: dict[str | tuple[str], list[Variable]] = {}
        key_2_tasks: dict[str | tuple[str], list[str]] = {}

        if AGG_GRAN == 'model':
            agg_vars = self._get_agg_vars('', ALL_TASKS)  # NOTE Create only once.
            task_combs = set(map(tuple, self.ref_2_tasks.values()))
            for tasks in task_combs:
                key_2_agg_vars[tasks] = [agg_var for task, agg_var in zip(ALL_TASKS, agg_vars) if task in tasks]
                key_2_tasks[tasks] = list(tasks)
        else:
            for tasks, taskwise_var in zip(self.ref_2_tasks.values(), self.varwise_taskwise_var):
                seg_var = taskwise_var[0]
                agg_key = self.get_agg_key(seg_var)
                if agg_key in key_2_agg_vars:  # Already created.
                    continue
                agg_vars = self._get_agg_vars(agg_key, tasks, seg_var)
                key_2_agg_vars[agg_key] = agg_vars
                key_2_tasks[agg_key] = tasks

        self.key_2_agg_vars = key_2_agg_vars
        self.key_2_tasks = key_2_tasks
        # NOTE For checkpointing and optimization
        # NOTE `tf.nest.flatten` does not preserve order, but sorts keys.
        self.agg_vars_list = unique_variables(sum(key_2_agg_vars.values(), []))

    def build_train_step_func(self):
        mode_2_data_iter = tf.nest.map_structure(iter, self.mode_2_dataset[TRAIN])
        self.mode_2_data_iter[TRAIN] = mode_2_data_iter

        def train_step(task: str):
            self.train_step(
                next(mode_2_data_iter[f'train_{"seg" if task.startswith("seg") else task}']),
                task
            )

        self.mode_2_step_func[TRAIN] = tf.function(train_step)

    def on_mode_begin(self):
        super().on_mode_begin()
        self.build_test_step_func()

        AGG_FREQ = self.AGG_FREQ
        assert AGG_FREQ < 0
        AGG_FREQ = -AGG_FREQ
        SELECT_FREQ = self.SELECT_FREQ
        agg_per_select = SELECT_FREQ / AGG_FREQ
        ALL_TASKS_EPOCHS = [round(i_search * agg_per_select) * AGG_FREQ for i_search in range(self.total_epochs // SELECT_FREQ)]
        EXAM_EPOCHS = [e+AGG_FREQ for e in ALL_TASKS_EPOCHS]
        self.ALL_TASKS_EPOCHS = ALL_TASKS_EPOCHS
        self.EXAM_EPOCHS = EXAM_EPOCHS
        logging.info(f'All task epochs: {self.ALL_TASKS_EPOCHS}')
        logging.info(f'Exam     epochs: {self.EXAM_EPOCHS}')

        self.epoch_2_comb_2_valid_loss: dict[int, dict[tuple[str, ...], Tensor]] = {}

        if self.epoch == 0:  # NOTE Otherwise already determined at end of prev epoch
            self.determine_train_tasks()

    def train_epoch(self):
        ALL_TASKS = self.ALL_TASKS

        train_step = self.mode_2_step_func[TRAIN]
        epoch = self.epoch

        init_epochs = tf.convert_to_tensor(epoch)
        train_tasks = [t for t in ALL_TASKS if self.task_2_do_train[t]]
        for task in train_tasks:
            logging.info(f'Training: {task}')
            epoch.assign(init_epochs)
            for _ in self.iter_epochs():
                for _ in self.iter_steps():
                    train_step(task)
                epoch.assign_add(1)
            self._update_delta(task)
        is_final_epoch = (epoch == self.total_epochs)
        iterations = tf.reduce_max(list(optimizer.iterations for optimizer in self.task_2_optimizer.values()))
        for optimizer in self.task_2_optimizer.values():
            optimizer.iterations.assign(iterations)

        if (epoch % self.META_FREQ == 0) or is_final_epoch:
            logging.info('Meta triggered.')
            self.meta_train()

        if (epoch % self.AGG_FREQ == 0) or is_final_epoch:
            logging.info('Agg triggered.')
            self.determine_agg_tasks()
            logging.info(f'Aggregating: {[t for t in ALL_TASKS if self.task_2_do_agg[t]]}')
            self.agg_model(training=False)
            self.determine_train_tasks()  # do at final epoch in case resume training
            self.meta_optimizer.iterations.assign(0)

    def train_step(self, examples, task):
        with tf.GradientTape() as tape:
            loss = self.forward_task(task, examples, self.task_2_model[task], tape)
        self.minimize(tape=tape, loss=loss, optimizer=self.task_2_optimizer[task], var_list=tape.watched_variables())
        print(flush=True)

    def determine_agg_tasks(self):
        '''

        NOTE Modifies `task_2_do_agg` on return to avoid agg harmful tasks.
        '''
        epoch_int = self.epoch_int

        ALL_TASKS = self.ALL_TASKS
        do_exam = epoch_int in self.EXAM_EPOCHS
        SELECT_N = self.SELECT_N

        task_2_do_train = self.task_2_do_train
        task_2_do_agg = self.task_2_do_agg
        assert any(task_2_do_agg.values())

        comb_2_valid_loss = {}

        def _get_valid_loss(comb: tuple[str, ...]):
            ''' Validation with lru-cache implementation
            NOTE Changes `self.task_2_do_agg`
            '''
            if comb in comb_2_valid_loss:
                return comb_2_valid_loss[comb]
            else:
                for task in ALL_TASKS:
                    task_2_do_agg[task].assign(task in comb)
                valid_loss: Tensor = self.run_valid()
                comb_2_valid_loss[comb] = valid_loss
                return valid_loss

        if do_exam:
            combs = [comb for comb in combinations(ALL_TASKS, r=SELECT_N) if 'seg' in comb]
            logging.info(f'Valid combination perf triggered (n={len(combs)}).')
            for comb in combs:
                valid_loss = _get_valid_loss(comb)
                tf.print(comb, valid_loss)
            self.epoch_2_comb_2_valid_loss[epoch_int] = comb_2_valid_loss

            # Determine agg tasks
            best_comb = min(comb_2_valid_loss.keys(), key=comb_2_valid_loss.get)
            logging.info(f'Best comb: {best_comb}')
            for task in ALL_TASKS:
                task_2_do_agg[task].assign(task in best_comb)
        else:  # Include all tasks
            for task in ALL_TASKS:
                task_2_do_agg[task].assign(task_2_do_train[task])

    def determine_train_tasks(self):
        epoch = self.epoch_int
        task_2_do_train = self.task_2_do_train
        task_2_do_agg = self.task_2_do_agg

        ALL_TASKS = self.ALL_TASKS
        SELECT_N = self.SELECT_N

        if epoch in self.ALL_TASKS_EPOCHS:
            logging.info('All tasks triggered.')
            selected_tasks = list(ALL_TASKS)
        elif epoch in self.EXAM_EPOCHS:
            logging.info('Selecting tasks based on comb perf.')
            comb_2_valid_loss = self.epoch_2_comb_2_valid_loss[list(self.epoch_2_comb_2_valid_loss)[-1]]
            selected_tasks = min([key for key in comb_2_valid_loss.keys() if len(key) == SELECT_N], key=comb_2_valid_loss.get)
            # NOTE No need to explicitly add 'seg' as `comb_2_valid_loss.keys` asserts including seg
        else:  # Use previous.
            selected_tasks = [t for t, do_train in task_2_do_train.items() if do_train]

        if 'seg' not in selected_tasks:
            selected_tasks = ['seg', *selected_tasks[:-1]]
        elif selected_tasks[0] != 'seg':
            selected_tasks.remove('seg')
            selected_tasks.insert(0, 'seg')
        logging.info(f'Selected tasks: {selected_tasks}')
        for task in ALL_TASKS:
            selected = task in selected_tasks
            task_2_do_train[task].assign(selected)
            task_2_do_agg[task].assign(selected)

    def meta_train(self):
        for _ in range(self.mode_2_steps[META]):
            self.meta_step()

    @tf.function
    def meta_step(self):
        metrics = {}
        meta_examples = next(self.mode_2_data_iter[META])
        agg_var_list = self.agg_vars_list
        meta_optimizer = self.meta_optimizer
        with tf.GradientTape(watch_accessed_variables=False) as meta_tape:
            meta_tape.watch(agg_var_list)
            self.set_agg_vals()
            meta_losses = tf.reduce_mean(self.loss(
                meta_examples['label'],
                self.model(meta_examples['image'], training=False)
            ), axis=-1)
            print('meta_losses', meta_losses)
            metrics['loss/dice_loss/meta'] = meta_losses
            loss = tf.reduce_mean(meta_losses)
            self.set_src_vars()

        with tf.name_scope('minimize'):
            raw_grad_and_vars = self.get_raw_grad_and_vars(
                tape=meta_tape, loss=loss,
                optimizer=meta_optimizer,
                var_list=agg_var_list,
            )
            for task in self.ALL_TASKS:
                if self.task_2_do_train[task]:
                    curr_raw_grad_and_vars = [(grad, var) for grad, var in raw_grad_and_vars if f'_{task}' in var.name]
                    self.minimize_from_raw_grads(
                        raw_grad_and_vars=curr_raw_grad_and_vars,
                        optimizer=meta_optimizer,
                        weight_decay=self.META_WD,
                        decouple_weight_decay=True,
                    )
                    meta_optimizer.iterations.assign_sub(1)
            meta_optimizer.iterations.assign_add(1)
        self.update_metrics(values=metrics)

    def _update_delta(self, task):
        for var, var_stale, delta in zip(
            self.task_2_varwise_var[task], self.task_2_varwise_var_stale[task], self.task_2_varwise_delta[task], strict=True
        ):
            delta.assign(var - var_stale)

    def set_agg_vals(self, mode='replace'):
        ''' Replace seg model params <- aggregated tensors.

        Calls `self.agg_model(training=True)`
        '''
        seg_ref_2_new_val = self.agg_model(training=True)
        set_model_with_values(self.seg_ref_2_entry, seg_ref_2_new_val, mode=mode)

    def set_src_vars(self):
        set_model_with_src_variables(self.seg_ref_2_entry)

    @tf.function
    def agg_model(self, training: bool):
        ''' Agg task params according to `task_2_do_agg`
        '''
        ref_2_new_val = self._get_ref_2_new_val()

        if not training:  # NOTE Sync new val to all task models.
            for [
                (seg_ref, new_val),
                taskwise_var,
                taskwise_var_stale,
                taskwise_delta
            ] in zip(
                ref_2_new_val.items(),
                self.varwise_taskwise_var,
                self.varwise_taskwise_var_stale,
                self.varwise_taskwise_delta,
                strict=True,
            ):
                for var, var_stale, delta in zip(taskwise_var, taskwise_var_stale, taskwise_delta, strict=True):
                    var.assign(new_val)
                    var_stale.assign(new_val)
                    delta.assign(tf.zeros_like(delta))

        if training:
            return ref_2_new_val

    @tf.function
    def _get_ref_2_new_val(self):
        task_2_do_agg = self.task_2_do_agg
        # for task, do_agg in task_2_do_agg.items():
        #     if do_agg:
        #         tf.print('agg', 'training' if training else 'test', task)

        key_2_taskwise_agg_rates = self.get_key_2_taskwise_agg_rates()

        ref_2_new_val: dict[Reference, Tensor] = {}
        for [
            (seg_ref, tasks),
            taskwise_var,
            taskwise_var_stale,
            taskwise_delta
        ] in zip(
            self.ref_2_tasks.items(),
            self.varwise_taskwise_var,
            self.varwise_taskwise_var_stale,
            self.varwise_taskwise_delta,
            strict=True,
        ):
            assert taskwise_var[0].ref() == seg_ref
            seg_var_stale = taskwise_var_stale[0]
            taskwise_agg_rate = key_2_taskwise_agg_rates[self.get_agg_key(seg_ref.deref())]
            # NOTE Partial aggregation, different from super method
            new_val = tf.convert_to_tensor(seg_var_stale)  # stale seg val
            for task, delta, agg_rate in zip(tasks, taskwise_delta, taskwise_agg_rate, strict=True):
                if task_2_do_agg[task]:
                    new_val += agg_rate * delta
            ref_2_new_val[seg_ref] = new_val

        return ref_2_new_val

    def run_valid(self) -> Tensor:
        prev_mode = self.mode
        self.mode = VALID
        step_func = self.mode_2_step_func[TEST]
        metric = self._mean_metrics['loss_valid/dice_loss']
        metric.reset_state()

        # NOTE Valid with tmp agg weights
        self.set_agg_vals(mode='assign')
        data_iter = iter(self.mode_2_dataset[VALID])
        for _ in self.iter_steps(self.mode_2_steps[VALID]):
            step_func(next(data_iter), VALID, do_loss=True, do_metrics=False, tb_vis=False)
        # self.set_src_vars()  # No need because of later `self.agg_model(training=False)`

        metric_val = metric.result()
        metric.reset_state()
        self.mode = prev_mode
        return metric_val
