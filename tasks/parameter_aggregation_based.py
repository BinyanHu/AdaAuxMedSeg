import numpy as np
import tensorflow as tf
import tf_keras
from absl import flags, logging
from absl.flags import FLAGS
from tensorflow import Tensor, Variable
from tensorflow.python.util.object_identity import Reference
from tf_keras import Model

from tasks import TRAIN
from utils import linear_combine, unique_variables, variable_component, variable_core_name

from . import MultiAuxiliaryTaskLearning


class ParameterAggregationBased(MultiAuxiliaryTaskLearning):
    ZIP_TRAIN_DATASETS = False

    @classmethod
    def define_flags(cls):
        super().define_flags()
        flags.DEFINE_integer('agg_freq', 1,
                             help='# steps(+)/epochs(-) for weight aggregation. 0 means adaptive freq.')
        flags.DEFINE_string('agg_norm', 'softmax',
                            help='How to normalize agg vars to sum 1.')
        flags.DEFINE_enum('agg_gran', 'model',
                          enum_values=['model' 'block'],
                          help='Granularity of knowledge transfer.')

    def build_optimizer(self):
        ''' Build task-wise optimizers and track variables.
        
        Subclass is responsible for adding tracked weights for each item in `task_2_optimizer`.
        '''
        super().build_optimizer()
        task_2_optimizer: dict[str, tf_keras.optimizers.Optimizer] = {'seg': self.optimizer}
        for task, model in self.task_2_model.items():
            if task in task_2_optimizer:
                continue
            optimizer = self.get_optimizer(self.get_lr())
            optimizer.build(self._get_flatten_trainable_variables(model))
            task_2_optimizer[task] = optimizer
        self.task_2_optimizer = task_2_optimizer
        self.iterations_last_meta_or_agg = Variable(0, trainable=False, dtype=tf.int64, name='iterations_last_meta_or_agg')

    def get_agg_key(self, var: Variable | str) -> str | tuple[str]:
        AGG_GRAN = self.AGG_GRAN
        if AGG_GRAN == 'model':
            agg_key = tuple(self.ref_2_tasks[var.ref()])  # NOTE agg_key is task comb
        else:
            agg_key = variable_component(var, AGG_GRAN)
        return agg_key

    def _get_agg_vars(self, site, tasks, seg_var=None):
        AUX_TASKS = FLAGS.aux_tasks
        AGG_NORM = FLAGS.agg_norm
        AUX_RATES = FLAGS.aux_rates

        shape = []
        init_rates = np.array([1.] + [rate for aux, rate in zip(AUX_TASKS, AUX_RATES) if aux in tasks], np.float32)
        init_rates /= init_rates.sum()
        if AGG_NORM == '':
            init_weights = init_rates
        elif AGG_NORM == 'softmax':
            init_weights = np.log(init_rates)
            init_weights = init_weights.clip(-5, 5)
            init_weights -= init_weights.mean()  # NOTE numerical stability
        else:
            raise NotImplementedError(AGG_NORM)
        init_weights = init_weights.clip(-5, 5)
        name_prefix = 'agg'
        if site:
            name_prefix += f'_{site}'
        return [
            Variable(np.full(shape, init_weight), dtype=tf.float32, trainable=True, name=f'{name_prefix}_{task}')
            for task, init_weight in zip(tasks, init_weights)
        ]

    def get_key_2_taskwise_agg_rates(self):
        ''' Query each `agg_rates` by either a task combination (model) or a key (block, layer, op, var)
        '''
        with tf.name_scope('get_key_2_taskwise_agg_rates'):
            key_2_taskwise_agg_rates: dict[str | tuple[str], list[Tensor]] = {}
            for key, agg_vars in self.key_2_agg_vars.items():
                key_2_taskwise_agg_rates[key] = self.get_agg_rates(agg_vars)
            # tf.print(key_2_taskwise_agg_rates)
            return key_2_taskwise_agg_rates

    def get_agg_rates(self, agg_vars: list[Variable]) -> list[Tensor]:
        AGG_NORM = FLAGS.agg_norm
        with tf.name_scope('get_agg_rates'):
            task_2_do_train = self.task_2_do_train
            taskwise_do_train = tf.stack([
                task_2_do_train[variable_core_name(agg_var).split('_')[-1]]
                for agg_var in agg_vars
            ])
            agg_vars = tf.stack(agg_vars, axis=0)
            if AGG_NORM == '':
                agg_rates = agg_vars
            elif AGG_NORM == 'softmax':
                exps = tf.exp(agg_vars)
                agg_rates = exps / tf.reduce_sum(exps[taskwise_do_train], axis=0)
            else:
                raise NotImplementedError(AGG_NORM)
            return tf.unstack(agg_rates, axis=0)

    def _get_flatten_trainable_variables(self, model: Model | list[Model]):
        variables = []
        if isinstance(model, list):
            for model in model:
                variables += model.trainable_variables
        else:
            variables += model.trainable_variables
        variables = unique_variables(variables)
        return variables

    def sync_task_train(self):
        ''' Train aux models to catch up with the prim.
        '''
        ALL_TASKS = self.ALL_TASKS

        train_step = self.mode_2_step_func[TRAIN]
        to_iterations = self.optimizer.iterations.value()
        for task in ALL_TASKS:
            n_steps = to_iterations - self.task_2_optimizer[task].iterations
            if n_steps > 0:
                for _ in tf.range(n_steps):
                    train_step(task)

    @tf.function(reduce_retracing=True)
    def agg_model(self, training: bool) -> None | dict[Reference, Tensor]:
        key_2_taskwise_agg_rates = self.get_key_2_taskwise_agg_rates()

        ref_2_new_val: dict[Reference, Tensor] = {}
        num_vars = 0
        num_slots = 0
        for (ref, tasks), taskwise_var in zip(self.ref_2_tasks.items(), self.varwise_taskwise_var, strict=True):
            seg_var = taskwise_var[0]
            assert seg_var.ref() == ref
            agg_key = self.get_agg_key(seg_var)
            agg_rates = key_2_taskwise_agg_rates[agg_key]
            print(agg_key, [r.shape for r in agg_rates], '*', [variable_core_name(w) for w in taskwise_var])
            new_val = linear_combine(taskwise_var, agg_rates)
            if training:
                # Return the differentiable new values.
                ref_2_new_val[ref] = new_val
                num_vars += 1
            else:
                # Sync all models.
                for var in taskwise_var:
                    var.assign(new_val)
                num_vars += 1
        logging.info('Aggregated %d vars.', num_vars)
        logging.info('Aggregated %d optimizer vars.', num_slots)

        if training:
            return ref_2_new_val
