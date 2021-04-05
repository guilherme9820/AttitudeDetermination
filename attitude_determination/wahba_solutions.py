import sys
from scipy.special import softmax
import tensorflow as tf
import numpy as np
from typing import Union
from typing import Tuple


class TFWahbaSolutions:
    """
        This class implements some algorithms used to solve the Wahba's problem [Wahba1965].
        The algorithms implemented are the following:
            - QUEST, ESOQ2, SVD, Q-Method and Neural Network.

        The Neural Network implementation just performs a forward pass through a model passed by
        the user as argument. Such algorithms are implemented using the Tensorflow 2 framework.

        References:
            - [Wahba1965] Wahba, Grace. "A least squares estimate of satellite attitude."
                          SIAM review 7.3 (1965): 409-409.
    """

    def __init__(self, algorithm='quest'):

        self.algorithm = algorithm

    def __call__(self, body_vectors: np.ndarray, ref_vectors: np.ndarray, **kwargs):

        return self.algorithm(body_vectors, ref_vectors, **kwargs)

    @property
    def algorithm(self):
        return self.__algorithm

    @algorithm.setter
    def algorithm(self, algorithm):
        self.__algorithm = getattr(self, algorithm)

    def rotation_matrix_from_quaternion(self, quaternion: tf.Tensor) -> tf.Tensor:
        """ Builds a Direction Cosine Matrix (DCM) given a quaternion.

        Args:
            quat: An array of size (4,) representing the quaternion.

        Returns:
            An array of shape (3, 3) containing the Direction Cosine Matrix.
        """

        # Unit quaternion rotation matrices computatation
        xx = quaternion[0]*quaternion[0]
        yy = quaternion[1]*quaternion[1]
        zz = quaternion[2]*quaternion[2]
        xy = quaternion[0]*quaternion[1]
        xz = quaternion[0]*quaternion[2]
        yz = quaternion[1]*quaternion[2]
        xw = quaternion[0]*quaternion[3]
        yw = quaternion[1]*quaternion[3]
        zw = quaternion[2]*quaternion[3]

        matrix = tf.convert_to_tensor([[1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw],
                                       [2*xy + 2*zw,  1-2*xx-2*zz, 2*yz-2*xw],
                                       [2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy]])

        return matrix

    def undo_rotation(self, quat: tf.Tensor, axis: int = 0) -> tf.Tensor:
        """ Undo rotations done by sequential rotations (QUEST) and initial \
            rotations (ESOQ2) by permuting and changing signs of the quaternion.

            Obs.: The permutations done by this method were given by multiplying
                  the quaternion by a unit quaternion pointing to the desired axis.
                  For example:
                        If one wants to undo a x-axis rotation, he must do:
                            q = [q1, q2, q3, q4] * [1, 0, 0, 0] -> [q4, q3, -q2, -q1]
                        If one wants to undo a y-axis rotation, he must do:
                            q = [q1, q2, q3, q4] * [0, 1, 0, 0] -> [-q3, q4, q1, -q2]
                        If one wants to undo a z-axis rotation, he must do:
                            q = [q1, q2, q3, q4] * [0, 0, 1, 0] -> [q2, -q1, q4, -q3]

                   The permutation given by the original papers result in wrong rotations.
        Args:
            quat: An array of size (4,) representing the quaternion.
            axis: Desired axis of rotation. Defaults to 0.

        Returns:
            The rotated quaternion by a given axis.
        """

        if axis == 0:
            return tf.convert_to_tensor([quat[3], quat[2], -quat[1], -quat[0]])
        elif axis == 1:
            return tf.convert_to_tensor([-quat[2], quat[3], quat[0], -quat[1]])
        elif axis == 2:
            return tf.convert_to_tensor([quat[1], -quat[0], quat[3], -quat[2]])
        else:
            return quat

    def _quest_core(self, B: tf.Tensor, num_iterations: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """ QUEST core computation.

        Args:
            B: Profile matrix.
            num_iterations: Number of Newton-Raphson iterations.

        Returns:
            An unnormalized quaternion and its scalar absolute value.

        References:
            - [Shuster1981] Shuster, M.D. and Oh, S.D. "Three-Axis Attitude Determination
                from Vector Observations," Journal of Guidance and Control, Vol.4, No.1,
                Jan.-Feb. 1981, pp. 70-77.
            - [Markley2000] Markley, F. Landis, and Daniele Mortari. "New developments
                in quaternion estimation from vector observations." (2000).
        """

        S = B + tf.transpose(B)

        z = tf.convert_to_tensor([B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]])

        Sz = tf.convert_to_tensor([z[0]*S[0, 0] + z[1]*S[0, 1] + z[2]*S[0, 2],
                                   z[0]*S[1, 0] + z[1]*S[1, 1] + z[2]*S[1, 2],
                                   z[0]*S[2, 0] + z[1]*S[2, 1] + z[2]*S[2, 2]])

        SSz = tf.convert_to_tensor([Sz[0]*S[0, 0] + Sz[1]*S[0, 1] + Sz[2]*S[0, 2],
                                    Sz[0]*S[1, 0] + Sz[1]*S[1, 1] + Sz[2]*S[1, 2],
                                    Sz[0]*S[2, 0] + Sz[1]*S[2, 1] + Sz[2]*S[2, 2]])

        # Parameters of characeristic equation (eq. 63) from [Shuster1981]
        sigma = B[0, 0] + B[1, 1] + B[2, 2]  # equations 44 and 63 are equivalent

        K1 = S - sigma*tf.eye(3)

        K = tf.convert_to_tensor([[K1[0, 0], K1[0, 1], K1[0, 2], z[0]],
                                  [K1[1, 0], K1[1, 1], K1[1, 2], z[1]],
                                  [K1[2, 0], K1[2, 1], K1[2, 2], z[2]],
                                  [z[0], z[1], z[2], sigma]])

        kappa = S[0, 0]*S[1, 1]+S[1, 1]*S[2, 2]+S[2, 2]*S[0, 0] \
            - S[0, 1]*S[1, 0]-S[1, 2]*S[2, 1]-S[2, 0]*S[0, 2]

        kappa2 = tf.linalg.trace(tf.transpose(K) * tf.linalg.det(K))

        # (eq. 71) from [Shuster1981]
        b = -2*sigma + kappa - (z[0]**2 + z[1]**2 + z[2]**2)
        c = -kappa2
        d = tf.linalg.det(K)

        # Newton-Raphson method (eq. 70) from [Shuster1981]
        lam = 1.
        for i in range(num_iterations):
            phi = lam**4 + b * lam**2 + c * lam + d
            phi_prime = 4 * lam**3 + 2 * b * lam + c

            lam -= (phi / phi_prime)

        # Equations 66 and 68 from [Shuster1981]
        alpha = lam**2 - sigma**2 + kappa
        beta = lam - sigma
        gamma = alpha*(lam + sigma) - tf.linalg.det(S)
        X = alpha*z + beta*Sz + SSz

        # Optimal Quaternion (eq. 69) from [Shuster1981].
        # If implemented following the papers the code does not
        # work very well, in order to work we must use the
        # quaternion inverse
        quaternions = tf.convert_to_tensor([-X[0], -X[1], -X[2], gamma])

        return quaternions, tf.abs(gamma)

    @ tf.function
    def quest(self,
              body_vectors: Union[np.ndarray, tf.Tensor],
              ref_vectors: Union[np.ndarray, tf.Tensor],
              stddevs: list,
              num_iterations: int = 0,
              **kwargs) -> tf.Tensor:
        """ QUaternion ESTimator (QUEST).

            Args:
                body_vectors: (N, 3) array of measurement vectors, where N is the number of measurements.
                ref_vectors: (N, 3) array of reference vectors, where N is the number of measurements.
                stddevs: A list of size N containing the measurement standard deviations for each
                         measurement vector.
                num_iterations: Number of iterations done by Newton-Raphson method.

            Raises:
                ValueError: Raises an error if the arguments have different shapes.

            Returns:
                Array of size (4,) containing the estimated quaternion.

            References:
                - [Shuster1981] Shuster, M.D. and Oh, S.D. "Three-Axis Attitude Determination from Vector
                                Observations," Journal of Guidance and Control, Vol.4, No.1, Jan.-Feb.
                                1981, pp. 70-77.
                - [Markley1999] Markley, F. Landis, and Daniele Mortari. "How to estimate attitude from
                                vector observations." (1999).

        """

        ref_vectors = tf.cast(ref_vectors, tf.float32)
        body_vectors = tf.cast(body_vectors, tf.float32)

        if body_vectors.shape != ref_vectors.shape:
            raise ValueError("body_vectors and ref_vectors are not the same size")

        # Equation 97 from [Shuster1981]
        sig_tot = 1. / tf.reduce_sum(1/tf.convert_to_tensor(stddevs)**2)
        # Equation 96 from [Shuster1981]
        weights = sig_tot / tf.reshape(stddevs, [-1, 1])**2

        body_vectors = tf.transpose(body_vectors)

        # Vectorized form of equation 38 from [Shuster1981]
        B = tf.matmul(body_vectors, ref_vectors * weights)

        # First quaternion estimative
        quaternions, qual = self._quest_core(B, num_iterations)
        axis = 3

        # Sequential rotations, where the quaternion scalar value must
        # be smaller than a qmin. According to [Markley1999] a qmin=0.1
        # is enough to avoid loss of significance
        if qual < 0.1:
            _B = B * tf.constant([1., -1., -1.])
            quaternions, qual = self._quest_core(_B, num_iterations)
            axis = 0

        if qual < 0.1:
            _B = B * tf.constant([-1., 1., -1.])
            quaternions, qual = self._quest_core(_B, num_iterations)
            axis = 1

        if qual < 0.1:
            _B = B * tf.constant([-1., -1., 1.])
            quaternions, qual = self._quest_core(_B, num_iterations)
            axis = 2

        quaternions = self.undo_rotation(quaternions, axis=axis)

        quaternions = tf.linalg.normalize(quaternions)[0]

        optimal_matrix = self.rotation_matrix_from_quaternion(quaternions)

        return optimal_matrix

    @ tf.function
    def esoq2(self,
              body_vectors: Union[np.ndarray, tf.Tensor],
              ref_vectors: Union[np.ndarray, tf.Tensor],
              stddevs: list,
              num_iterations: int = 0,
              **kwargs) -> tf.Tensor:
        """
            Second EStimator of the Optimal Quaternion (ESOQ2).

        Args:
            body_vectors (np.ndarray): (N, 3) array of measurement vectors, where N is the number of measurements.
            ref_vectors (np.ndarray): (N, 3) array of reference vectors, where N is the number of measurements.
            stddevs: A list of size N containing the measurement standard deviations for each
                        measurement vector.
            num_iterations: Number of iterations done by Newton-Raphson method.

        Raises:
            ValueError: Raises an error if the arguments have different shapes.

        Returns:
            Array of size (3, 3) containing the optimal attitude matrices.

        References:
            - [Shuster1981] Shuster, M.D. and Oh, S.D. "Three-Axis Attitude Determination from Vector
                            Observations," Journal of Guidance and Control, Vol.4, No.1, Jan.-Feb.
                            1981, pp. 70-77.
            - [Markley2000] Markley, F. Landis, and Daniele Mortari. "New developments in quaternion
                            estimation from vector observations." (2000).
            - [Mortari2000] Mortari, Daniele. "Second estimator of the optimal quaternion." Journal
                            of Guidance, Control, and Dynamics 23.5 (2000): 885-888.
        """

        ref_vectors = tf.cast(ref_vectors, tf.float32)
        body_vectors = tf.cast(body_vectors, tf.float32)

        if body_vectors.shape != ref_vectors.shape:
            raise ValueError("body_vectors and ref_vectors are not the same size")

        # Equation 97 from [Shuster1981]
        sig_tot = 1. / tf.reduce_sum(1/tf.convert_to_tensor(stddevs)**2)
        # Equation 96 from [Shuster1981]
        weights = sig_tot / tf.reshape(stddevs, [-1, 1])**2

        body_vectors = tf.transpose(body_vectors)

        # Vectorized form of equation 38 from [Shuster1981]
        _B = tf.matmul(body_vectors, ref_vectors * weights)

        # Initial rotation verification as described in [Markley2000]
        axis = tf.argmin([_B[0, 0],
                          _B[1, 1],
                          _B[2, 2],
                          _B[0, 0] + _B[1, 1] + _B[2, 2]])

        if axis == 0:
            B = _B * tf.constant([1., -1., -1.])
        elif axis == 1:
            B = _B * tf.constant([-1., 1., -1.])
        elif axis == 2:
            B = _B * tf.constant([-1., -1., 1.])
        else:
            B = _B

        S = B + tf.transpose(B)

        z = tf.convert_to_tensor([B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]])

        # Parameters of characeristic equation (eq. 63) from [Shuster1981]
        sigma = B[0, 0] + B[1, 1] + B[2, 2]  # equations 44 and 63 are equivalent

        K1 = S - sigma*tf.eye(3)

        K = tf.convert_to_tensor([[K1[0, 0], K1[0, 1], K1[0, 2], z[0]],
                                  [K1[1, 0], K1[1, 1], K1[1, 2], z[1]],
                                  [K1[2, 0], K1[2, 1], K1[2, 2], z[2]],
                                  [z[0], z[1], z[2], sigma]])

        # Adjugate of S
        kappa = S[0, 0]*S[1, 1]+S[1, 1]*S[2, 2]+S[2, 2]*S[0, 0] \
            - S[0, 1]*S[1, 0]-S[1, 2]*S[2, 1]-S[2, 0]*S[0, 2]

        kappa2 = tf.linalg.trace(tf.transpose(K) * tf.linalg.det(K))

        # (eq. 71) from [Shuster1981]
        b = -2*sigma + kappa - (z[0]**2 + z[1]**2 + z[2]**2)
        c = -kappa2
        d = tf.linalg.det(K)

        # Newton-Raphson method (eq. 70) from [Shuster1981]
        lam = 1.
        for _ in range(num_iterations):
            phi = lam**4 + b * lam**2 + c * lam + d
            phi_prime = 4 * lam**3 + 2 * b * lam + c

            lam -= (phi / phi_prime)

        m1 = z[0]**2
        m2 = z[1]**2
        m3 = z[2]**2
        m4 = z[0]*z[1]
        m5 = z[0]*z[2]
        m6 = z[1]*z[2]

        # Equation 67 from [Markley2000]
        M = (lam - sigma) * ((lam + sigma) * tf.eye(3) - S)
        M -= tf.convert_to_tensor([[m1, m4, m5],
                                   [m4, m2, m6],
                                   [m5, m6, m3]])

        ma = M[0, 0]
        mb = M[1, 1]
        mc = M[2, 2]
        mx = M[0, 1]
        my = M[0, 2]
        mz = M[1, 2]

        # Equation 11 from [Mortari2000]
        adj_M = tf.convert_to_tensor([[mb*mc - mz**2, my*mz - mx*mc, mx*mz - my*mb],
                                      [my*mz - mx*mc, ma*mc - my**2, mx*my - mz*ma],
                                      [mx*mz - my*mb, mx*my - mz*ma, ma*mb - mx**2]])

        # Gets the index of the column with maximum norm according to [Markley2000]
        index_n = tf.argmax(tf.linalg.norm(adj_M, axis=1))

        y = adj_M[:, index_n]

        # with open("quat_y.txt", "a") as f:
        #     np.savetxt(f, tf.reshape(y, [1, -1]).numpy())

        # Equation 75 from [Markley2000]
        X = (lam - sigma) * y
        gamma = z[0]*y[0] + z[1]*y[1] + z[2]*y[2]

        quaternions = tf.convert_to_tensor([-X[0], -X[1], -X[2], gamma])

        quaternions = self.undo_rotation(quaternions, axis=axis)

        quaternions = tf.linalg.normalize(quaternions)[0]

        optimal_matrix = self.rotation_matrix_from_quaternion(quaternions)

        return optimal_matrix

    @ tf.function
    def svd(self,
            body_vectors: Union[np.ndarray, tf.Tensor],
            ref_vectors: Union[np.ndarray, tf.Tensor],
            stddevs: list,
            **kwargs) -> tf.Tensor:
        """ Singular Value Decomposition method by [Markley1988].

            Args:
                body_vectors: (N, 3) array of measurement vectors, where N is the number of measurements.
                ref_vectors: (N, 3) array of reference vectors, where N is the number of measurements.
                stddevs: A list of size N containing the measurement standard deviations for each
                         measurement vector.

            Returns:
                Array of size (3, 3) containing the optimal attitude matrices.

            References:
                - [Shuster1981] Shuster, M.D. and Oh, S.D. "Three-Axis Attitude Determination from Vector
                                Observations," Journal of Guidance and Control, Vol.4, No.1, Jan.-Feb.
                                1981, pp. 70-77.
                - [Markley1988] Markley, F. Landis. "Attitude determination using vector observations and
                                the singular value decomposition." Journal of the Astronautical Sciences
                                36.3 (1988): 245-258.
        """

        ref_vectors = tf.cast(ref_vectors, tf.float32)
        body_vectors = tf.cast(body_vectors, tf.float32)

        if body_vectors.shape != ref_vectors.shape:
            raise ValueError("body_vectors and ref_vectors are not the same size")

        # Equation 97 from [Shuster1981]
        sig_tot = 1. / tf.reduce_sum(1. / tf.convert_to_tensor(stddevs)**2)
        # Equation 96 from [Shuster1981]
        weights = sig_tot / tf.reshape(stddevs, [-1, 1])**2

        body_vectors = tf.transpose(body_vectors)

        # Vectorized form of equation 38 from [Shuster1981]
        B = tf.matmul(body_vectors, ref_vectors * weights)

        s, u, v = tf.linalg.svd(B)

        s3 = tf.linalg.det(u) * tf.linalg.det(v)

        diag = tf.convert_to_tensor([1., 1., s3])

        optimal_matrix = tf.matmul(u * diag, v, transpose_b=True)

        return optimal_matrix

    @ tf.function
    def q_method(self,
                 body_vectors: Union[np.ndarray, tf.Tensor],
                 ref_vectors: Union[np.ndarray, tf.Tensor],
                 stddevs: list,
                 **kwargs) -> tf.Tensor:
        """ Q-Method by [Keat1977].

            Args:
                body_vectors: (N, 3) array of measurement vectors, where N is the number of measurements.
                ref_vectors: (N, 3) array of reference vectors, where N is the number of measurements.
                stddevs: A list of size N containing the measurement standard deviations for each
                         measurement vector.

            Returns:
                Array of size (3, 3) containing the optimal attitude matrices.

            References:
                - [Shuster1981] Shuster, M.D. and Oh, S.D. "Three-Axis Attitude Determination from Vector
                                Observations," Journal of Guidance and Control, Vol.4, No.1, Jan.-Feb.
                                1981, pp. 70-77.
                - [Keat1977] Keat, J. Analysis of least-squares attitude determination routine DOAOP, Computer
                             Sciences Corporation Report CSC. TM-77/6034, February, 1977.
        """

        ref_vectors = tf.cast(ref_vectors, tf.float32)
        body_vectors = tf.cast(body_vectors, tf.float32)

        if body_vectors.shape != ref_vectors.shape:
            raise ValueError("body_vectors and ref_vectors are not the same size")

        # Equation 97 from [Shuster1981]
        sig_tot = 1. / tf.reduce_sum(1/tf.convert_to_tensor(stddevs)**2)
        # Equation 96 from [Shuster1981]
        weights = sig_tot / tf.reshape(stddevs, [-1, 1])**2

        body_vectors = tf.transpose(body_vectors)

        # Vectorized form of equation 38 from [Shuster1981]
        B = tf.matmul(body_vectors, ref_vectors * weights)

        S = B + tf.transpose(B) - tf.eye(3) * tf.linalg.trace(B)

        z = tf.convert_to_tensor([B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]])
        z = z[:, tf.newaxis]

        temp1 = tf.concat([S, z], axis=-1)

        temp2 = tf.concat([tf.transpose(z), [[tf.linalg.trace(B)]]], axis=-1)

        K = tf.concat([temp1, temp2], axis=0)

        eigen_values, eigen_vectors = tf.linalg.eigh(K)

        index = tf.argmax(eigen_values)

        quaternions = eigen_vectors[:, index]

        quaternions = tf.concat([-quaternions[:3], [quaternions[-1]]], axis=0)

        quaternions = tf.linalg.normalize(quaternions)[0]

        optimal_matrix = self.rotation_matrix_from_quaternion(quaternions)

        return optimal_matrix

    @tf.function
    def nn(self,
           body_vectors: Union[np.ndarray, tf.Tensor],
           ref_vectors: Union[np.ndarray, tf.Tensor],
           model,
           **kwargs) -> tf.Tensor:
        """ Neural Network.

            Args:
                body_vectors: (N, 3) array of measurement vectors, where N is the number of measurements.
                ref_vectors: (N, 3) array of reference vectors, where N is the number of measurements.
                model: A function that implements the model. Such that the model __call__ method performs
                       the forward pass and accepts an input argument and the keyword 'training' specifying
                       the training phase.

            Returns:
                Array of size (3, 3) containing the optimal attitude matrices.
        """

        ref_vectors = tf.cast(ref_vectors, tf.float32)
        body_vectors = tf.cast(body_vectors, tf.float32)

        if body_vectors.shape != ref_vectors.shape:
            raise ValueError("body_vectors and ref_vectors are not the same size")

        obs = body_vectors.shape[0]
        weights = tf.ones([obs, 1]) / obs

        # Vectorized form of equation 38 from [Shuster1981]
        profile_matrix = tf.matmul(body_vectors, ref_vectors * weights, transpose_a=True)
        profile_matrix = tf.reshape(profile_matrix, [-1, 9, 1])

        optimal_rotation = model(profile_matrix, training=False)

        return optimal_rotation[0]
