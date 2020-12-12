from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import pandas as pd
import numpy as np
import argparse
import random
import sys
import os
from learning_core.rotation_utils import compute_geodesic_distance_from_two_matrices
from learning_core.rotation_utils import rotation_matrix_from_quaternion
from learning_core.rotation_utils import gen_rot_quaternion
from attitude_determination.wahba_solutions import TFWahbaSolutions

# Ignore excessive warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='Test Scenarios')

# Required arguments
parser.add_argument('--algorithm', type=str, required=True, help="Which algorithm to be used")
parser.add_argument('--iterations', type=int, default=0, help="Number of iterations executed by Newthon's method")
parser.add_argument('--samples', type=int, default=1, help="Number of samples generated")
parser.add_argument('--test_case', type=str, required=True, help="Which test case to run")
parser.add_argument('--performance', action='store_true', help='Count floating point operations per second (FLOPS)')
parser.add_argument('--csv_file', type=str, required=True, help='Path to where the .csv file will be saved')

arguments = parser.parse_args(sys.argv[1:])


class TestCases:

    def __call__(self, num_samples=1, case='case1'):

        case_fn = getattr(self, case)

        self.true_attitude = np.array([[0.352, 0.864, 0.360],
                                       [-0.864, 0.152, 0.480],
                                       [0.360, -0.480, 0.800]])

        return case_fn(num_samples)

    def apply_noise(self, vectors, stddevs):

        batch_size = vectors.shape[0]

        noise = tf.zeros([batch_size, 0, 3])
        for std in stddevs:
            temp = tf.random.normal([batch_size, 1, 3], 0, std)
            noise = tf.concat([noise, temp], axis=1)

        return vectors + noise

    def apply_rotation(self, vectors, attitudes):

        vectors = tf.cast(vectors, tf.float32)
        attitudes = tf.cast(attitudes, tf.float32)

        rotated = tf.matmul(attitudes, vectors, transpose_b=True)

        return tf.transpose(rotated, [0, 2, 1])

    def build_data(self, ref_vectors, stddev):

        body_vectors = self.apply_rotation(ref_vectors, self.true_attitude)

        body_vectors = self.apply_noise(body_vectors, stddev)

        return ref_vectors, body_vectors, stddev

    def case1(self, num_samples):

        ref_vectors = tf.convert_to_tensor([[[1., 0., 0.],
                                             [0., 1., 0.],
                                             [0., 0., 1.]]])

        ref_vectors = tf.tile(ref_vectors, [num_samples, 1, 1])

        return self.build_data(ref_vectors, [1e-6, 1e-6, 1e-6])

    def case2(self, num_samples):

        ref_vectors = tf.convert_to_tensor([[[1., 0., 0.],
                                             [0., 1., 0.]]])

        ref_vectors = tf.tile(ref_vectors, [num_samples, 1, 1])

        return self.build_data(ref_vectors, [1e-6, 1e-6])

    def case3(self, num_samples):

        ref_vectors = tf.convert_to_tensor([[[1., 0., 0.],
                                             [0., 1., 0.],
                                             [0., 0., 1.]]])

        ref_vectors = tf.tile(ref_vectors, [num_samples, 1, 1])

        return self.build_data(ref_vectors, [0.01, 0.01, 0.01])

    def case4(self, num_samples):

        ref_vectors = tf.convert_to_tensor([[[1., 0., 0.],
                                             [0., 1., 0.]]])

        ref_vectors = tf.tile(ref_vectors, [num_samples, 1, 1])

        return self.build_data(ref_vectors, [0.01, 0.01])

    def case5(self, num_samples):

        ref_vectors = tf.convert_to_tensor([[[0.6, 0.8, 0.],
                                             [0.8, -0.6, 0.]]])

        ref_vectors = tf.tile(ref_vectors, [num_samples, 1, 1])

        return self.build_data(ref_vectors, [1e-6, 0.01])

    def case6(self, num_samples):

        ref_vectors = tf.convert_to_tensor([[[1., 0., 0.],
                                             [1., 0.01, 0.],
                                             [1., 0., 0.01]]])

        ref_vectors = tf.tile(ref_vectors, [num_samples, 1, 1])

        return self.build_data(ref_vectors, [1e-6, 1e-6, 1e-6])

    def case7(self, num_samples):

        ref_vectors = tf.convert_to_tensor([[[1., 0., 0.],
                                             [1., 0.01, 0.]]])

        ref_vectors = tf.tile(ref_vectors, [num_samples, 1, 1])

        return self.build_data(ref_vectors, [1e-6, 1e-6])

    def case8(self, num_samples):

        ref_vectors = tf.convert_to_tensor([[[1., 0., 0.],
                                             [1., 0.01, 0.],
                                             [1., 0., 0.01]]])

        ref_vectors = tf.tile(ref_vectors, [num_samples, 1, 1])

        return self.build_data(ref_vectors, [0.01, 0.01, 0.01])

    def case9(self, num_samples):

        ref_vectors = tf.convert_to_tensor([[[1., 0., 0.],
                                             [1., 0.01, 0.]]])

        ref_vectors = tf.tile(ref_vectors, [num_samples, 1, 1])

        return self.build_data(ref_vectors, [0.01, 0.01])

    def case10(self, num_samples):

        ref_vectors = tf.convert_to_tensor([[[1., 0., 0.],
                                             [.96, 0.28, 0.],
                                             [.96, 0., 0.28]]])

        ref_vectors = tf.tile(ref_vectors, [num_samples, 1, 1])

        return self.build_data(ref_vectors, [1e-6, 0.01, 0.01])

    def case11(self, num_samples):

        ref_vectors = tf.convert_to_tensor([[[1., 0., 0.],
                                             [.96, 0.28, 0.]]])

        ref_vectors = tf.tile(ref_vectors, [num_samples, 1, 1])

        return self.build_data(ref_vectors, [1e-6, 0.01])

    def case12(self, num_samples):

        ref_vectors = tf.convert_to_tensor([[[1., 0., 0.],
                                             [.96, 0.28, 0.]]])

        ref_vectors = tf.tile(ref_vectors, [num_samples, 1, 1])

        return self.build_data(ref_vectors, [0.01, 1e-6])


def angular_error(true_attitude, pred_attitude):

    norm = tf.linalg.norm(pred_attitude - true_attitude)

    arg = tf.clip_by_value(norm / tf.sqrt(8.), -1., 1.)

    return 2 * tf.asin(arg)


def count_flops(function, input_signatures):

    concrete_func = function.get_concrete_function(*input_signatures)

    frozen_func = convert_variables_to_constants_v2(concrete_func)
    graph_def = frozen_func.graph.as_graph_def()

    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')

        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)

        return flops.total_float_ops


def wahba_loss(y_true, y_pred, stddevs: list) -> tf.Tensor:
    """ Loss function proposed by Grace Wahba in [Wahba1965].

    Args:
        y_true: True body vectors.
        y_pred: Predicted body vectors.

    Returns:
        Cost corresponding to the difference between the true and predicted body vectors.

    References:
        - [Wahba1965] Wahba, Grace. "A least squares estimate of satellite attitude." SIAM review 7.3 (1965): 409-409.
    """

    stddevs = tf.convert_to_tensor(stddevs)

    # Equation 97 from [Shuster1981]
    sig_tot = 1. / tf.reduce_sum(1/stddevs**2)
    # Equation 96 from [Shuster1981]
    weights = sig_tot / stddevs**2

    error = weights * tf.norm(y_true - y_pred, axis=1)

    return 0.5 * tf.reduce_mean(error)


def evaluate(arguments):

    results = pd.DataFrame(columns=['test_case', 'algorithm', 'phi_error', 'loss'])

    if not os.path.exists(arguments.csv_file):
        results.to_csv(arguments.csv_file, index=False)

    algorithm_name = arguments.algorithm

    iterations = arguments.iterations
    if iterations < 0:
        raise ValueError("The number of iterations must be greater or equal to zero")

    num_samples = arguments.samples
    if num_samples < 1:
        raise ValueError("The number of samples must be greater than zero")

    test_cases = TestCases()

    ref_vectors, body_vectors, stddevs = test_cases(num_samples, arguments.test_case)

    solution = TFWahbaSolutions(algorithm_name)

    angle_errors = []
    losses = []
    for index in range(num_samples):

        # if algorithm_name == 'svd':
        attitude_result = solution(body_vectors[index], ref_vectors[index], stddevs, iterations)
        # else:
        # pred_quaternion, A, loss = solution(body_vectors[index], ref_vectors[index], stddevs, iterations)
        # attitude_result = rotation_matrix_from_quaternion([pred_quaternion])
        pred_vectors = test_cases.apply_rotation([ref_vectors[index]], attitude_result)

        # print(f"body[{index}]: {body_vectors[index]}")
        # print(f"pred[{index}]: {pred_vectors[0]}")

        losses.append(wahba_loss(body_vectors[index], pred_vectors[0], stddevs))
        # losses.append(loss)

        angle_errors.append(angular_error(test_cases.true_attitude, attitude_result))

    if arguments.performance:

        if algorithm_name == 'svd' or algorithm_name == 'q_method':
            input_signatures = [tf.TensorSpec(body_vectors.shape[1:], tf.float32),
                                tf.TensorSpec(ref_vectors.shape[1:], tf.float32),
                                tf.TensorSpec([len(stddevs)], tf.float32)]
        else:
            input_signatures = [tf.TensorSpec(body_vectors.shape[1:], tf.float32),
                                tf.TensorSpec(ref_vectors.shape[1:], tf.float32),
                                tf.TensorSpec([len(stddevs)], tf.float32),
                                tf.TensorSpec([], tf.int32)]

        print(f"FLOPS: {count_flops(solution.algorithm, input_signatures)}")

    mean_angle = tf.reduce_mean(angle_errors).numpy()
    mean_loss = tf.reduce_mean(losses).numpy()

    # print(f"mean angle error (rad): {mean_angle}")
    # print(f"mean loss: {mean_loss}")

    data = [arguments.test_case, algorithm_name, mean_angle, mean_loss]

    results.loc[0] = data

    results.to_csv(arguments.csv_file, mode='a', header=False, index=False)


if __name__ == "__main__":
    evaluate(arguments)
