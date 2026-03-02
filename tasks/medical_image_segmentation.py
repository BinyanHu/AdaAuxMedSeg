import tensorflow as tf
from absl import flags, logging
from absl.flags import FLAGS

TRAIN = 'train'
VALID = 'valid'
TEST = 'test'
META = 'meta'


class MedicalImageSegmentation:
    @classmethod
    def define_flags(cls):
        ...

    @classmethod
    def parse_flags(cls):
        ...

    def run(self):
        self.epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.build_data()
        self.build_model()
        self.build_loss()
        self.build_optimizer()
        self.run_train()

    def build_data(self):
        ...

    def build_model(self):
        ...

    def build_optimizer(self):
        ...

    def build_loss(self):
        ...

    def run_train(self):
        TOTAL_EPOCHS = self.total_epochs
        TEST_FREQ = self.TEST_FREQ

        epoch = self.epoch

        while epoch < TOTAL_EPOCHS:
            log = self.run_train_core()
            self.handle_log(log)

            if TEST_FREQ and ((epoch % TEST_FREQ == 0) or (epoch >= TOTAL_EPOCHS)):
                self.mode = TEST
                self.run_test()
                self.mode = TRAIN

    def run_train_core(self):
        ...

    def run_test(self):
        ...

    def handle_log(log):
        ...

    def __getattr__(self, attr):
        __dict__ = self.__dict__
        if attr in __dict__:
            return __dict__[attr]
        elif attr.isupper():
            return getattr(FLAGS, attr.lower())
        else:
            return self.__getattribute__(attr)
