import numpy as np


class TestCases:

    def __init__(self):

        self._true_attitude = np.array([[0.352, 0.864, 0.360],
                                        [-0.864, 0.152, 0.480],
                                        [0.360, -0.480, 0.800]])

    @property
    def true_attitude(self):
        return self._true_attitude

    @true_attitude.setter
    def true_attitude(self, values):
        values = np.array(values)
        assert values.shape == (3, 3), "The attitude matrix must be an array of shape (3, 3)"

        self._true_attitude = values

    def apply_noise(self, vectors, stddevs):

        batch_size = vectors.shape[0]

        noise = np.zeros([batch_size, 0, 3])
        for std in stddevs:
            temp = np.random.normal(0, std, [batch_size, 1, 3])
            noise = np.concatenate([noise, temp], axis=1)

        return vectors + noise

    def apply_rotation(self, vectors, attitudes):

        vectors = np.asarray(vectors, dtype="float32")
        attitudes = np.asarray(attitudes, dtype="float32")

        vectors = np.transpose(vectors, [0, 2, 1])

        rotated = attitudes @ vectors

        return np.transpose(rotated, [0, 2, 1])

    def build_data(self, ref_vectors, stddev, num_samples=1):

        ref_vectors = np.asarray(ref_vectors)

        if ref_vectors.ndim < 3:
            ref_vectors = ref_vectors[np.newaxis, ...]

        ref_vectors = np.tile(ref_vectors, [num_samples, 1, 1])

        body_vectors = self.apply_rotation(ref_vectors, self._true_attitude)

        body_vectors = self.apply_noise(body_vectors, stddev)

        return ref_vectors, body_vectors, stddev

    def __call__(self, num_samples=1, case='case1'):

        if case == "case1":
            ref_vectors = [[1., 0., 0.],
                           [0., 1., 0.],
                           [0., 0., 1.]]
            stddevs = [1e-6, 1e-6, 1e-6]

        elif case == "case2":
            ref_vectors = [[1., 0., 0.],
                           [0., 1., 0.]]
            stddevs = [1e-6, 1e-6]

        elif case == "case3":
            ref_vectors = [[1., 0., 0.],
                           [0., 1., 0.],
                           [0., 0., 1.]]
            stddevs = [0.01, 0.01, 0.01]

        elif case == "case4":
            ref_vectors = [[1., 0., 0.],
                           [0., 1., 0.]]
            stddevs = [0.01, 0.01]

        elif case == "case5":
            ref_vectors = [[0.6, 0.8, 0.],
                           [0.8, -0.6, 0.]]
            stddevs = [1e-6, 0.01]

        elif case == "case6":
            ref_vectors = [[1., 0., 0.],
                           [1., 0.01, 0.],
                           [1., 0., 0.01]]
            stddevs = [1e-6, 1e-6, 1e-6]

        elif case == "case7":
            ref_vectors = [[1., 0., 0.],
                           [1., 0.01, 0.]]
            stddevs = [1e-6, 1e-6]

        elif case == "case8":
            ref_vectors = [[1., 0., 0.],
                           [1., 0.01, 0.],
                           [1., 0., 0.01]]
            stddevs = [0.01, 0.01, 0.01]

        elif case == "case9":
            ref_vectors = [[1., 0., 0.],
                           [1., 0.01, 0.]]
            stddevs = [0.01, 0.01]

        elif case == "case10":
            ref_vectors = [[1., 0., 0.],
                           [.96, 0.28, 0.],
                           [.96, 0., 0.28]]
            stddevs = [1e-6, 0.01, 0.01]

        elif case == "case11":
            ref_vectors = [[1., 0., 0.],
                           [.96, 0.28, 0.]]
            stddevs = [1e-6, 0.01]

        elif case == "case12":
            ref_vectors = [[1., 0., 0.],
                           [.96, 0.28, 0.]]
            stddevs = [0.01, 1e-6]

        return self.build_data(ref_vectors, stddevs, num_samples=num_samples)
