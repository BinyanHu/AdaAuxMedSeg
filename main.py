import numpy as np
from absl import app

from tasks import Ours

np.set_printoptions(precision=2)


task_class = Ours
task_class.define_flags()


def main():
    task_class.parse_flags()
    task = task_class()
    task.run()


app.run(main)
