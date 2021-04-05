import numpy as np
import tensorflow as tf


class TestCases:

    def __init__(self):

        self._true_attitude = np.array([[0.352, 0.864, 0.360],
                                        [-0.864, 0.152, 0.480],
                                        [0.360, -0.480, 0.800]])

    def __call__(self, num_samples=1, case='case1'):

        case_fn = getattr(self, case)

        return case_fn(num_samples)

    @property
    def true_attitude(self):
        return self._true_attitude

    @true_attitude.setter
    def true_attitude(self, values):
        self._true_attitude = values

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

        body_vectors = self.apply_rotation(ref_vectors, self._true_attitude)

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
