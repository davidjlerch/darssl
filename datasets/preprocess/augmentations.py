import numpy as np


class Normalize3D:
    """
    Code base from https://github.com/kennymckormick/pyskl:
    PreNormalize for NTURGB+D 3D keypoints (x, y, z).
    Codes adapted from https://github.com/lshiwjx/2s-AGCN.
    """

    def unit_vector(self, vector):
        """Returns the unit vector of the vector. """
        if np.linalg.norm(vector) == 0.:
            return np.asarray([0, 0, 1])
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """Returns the angle in radians between vectors 'v1' and 'v2'. """
        if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
            return 0
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def rotation_matrix(self, axis, theta):
        """Return the rotation matrix associated with counterclockwise rotation
        about the given axis by theta radians."""
        if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
            return np.eye(3)
        axis = np.asarray(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    def __init__(self, zaxis=[0, 20], xaxis=[8, 4], align_spine=True,
                 align_center=True, scale=None, dataset: str = "NTU"):
        self.zaxis = zaxis
        self.xaxis = xaxis
        self.align_spine = align_spine
        self.align_center = align_center
        self.scale = scale
        self.dataset = dataset
        print(self.scale)

    def __call__(self, results):
        skeleton = results['keypoint']
        total_frames = results.get('total_frames', skeleton.shape[1])
        T, V, C = skeleton.shape
        if T != total_frames:
            total_frames = T
        # assert T == total_frames
        if skeleton.sum() == 0:
            results['body_center'] = np.zeros(3)
            return results

        # index0 = [i for i in range(T) if not np.all(np.isclose(skeleton[i], 0))]
        # skeleton = skeleton[np.array(index0)]

        T_new = skeleton.shape[1]

        if self.dataset == "NTU" or self.dataset == 'PKU' or self.dataset == 'NUCLA':
            main_body_center = skeleton[0, 1].copy()
            results['body_center'] = main_body_center
        elif self.dataset == "DAA":
            main_body_center = skeleton[0, 0].copy()
            results['body_center'] = main_body_center
            # main_body_center = np.mean(skeleton[0], axis=0)
        else:
            raise NotImplemented
            # main_body_center = skeleton[0, -1].copy()
        if self.align_center:
            mask = ((skeleton != 0).sum(-1) > 0)[..., None]
            skeleton = (skeleton - main_body_center) * mask
        if self.align_spine:
            if self.dataset == "NTU" or self.dataset == 'PKU' or self.dataset == 'NUCLA':
                joint_bottom = skeleton[0, self.zaxis[0]]
                joint_top = skeleton[0, self.zaxis[1]]
            elif self.dataset == "DAA" or self.dataset == "NTU_DAA":
                joint_bottom = np.mean(skeleton[:, self.zaxis[0]], axis=0)
                joint_top = np.mean(skeleton[:, self.zaxis[1]], axis=0)

            axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
            angle = self.angle_between(joint_top - joint_bottom, [0, 0, 1])
            matrix_z = self.rotation_matrix(axis, angle)
            skeleton = np.einsum('bcd,kd->bck', skeleton, matrix_z)
            if self.dataset == "NTU" or self.dataset == 'PKU' or self.dataset == 'NUCLA':
                joint_rshoulder = skeleton[0, self.xaxis[0]]
                joint_lshoulder = skeleton[0, self.xaxis[1]]
            elif self.dataset == "DAA" or self.dataset == "NTU_DAA":
                joint_rshoulder = np.mean(skeleton[:, self.xaxis[0]], axis=0)
                joint_lshoulder = np.mean(skeleton[:, self.xaxis[1]], axis=0)

            axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            angle = self.angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            matrix_x = self.rotation_matrix(axis, angle)
            skeleton = np.einsum('bcd,kd->bck', skeleton, matrix_x)

        if self.scale is not None:
            size = np.linalg.norm(np.median(skeleton[:, self.zaxis[1]], axis=0) -
                                  np.median(skeleton[:, self.zaxis[0]], axis=0))
            if size > 0:
                skeleton *= self.scale / np.amax(np.abs(skeleton))
                # print(np.amax(skeleton), np.amax(skeleton_))

        results['keypoint'] = skeleton
        results['total_frames'] = T_new
        results['body_center'] = main_body_center
        return results


class RandomMasking:
    def random_masking(self, x):
        if len(self.augmentations) > 0:
            if 'frames' in self.augmentations:
                # mask frames
                frames_mask = np.random.choice([0, 1], (x.shape[0],), p=[self.augmentations['frames'],
                                                                         1 - self.augmentations['frames']])
                frames_mask = np.clip(frames_mask, 0, 1)
                frames_mask = np.expand_dims(np.expand_dims(frames_mask, -1), -1)

                x *= np.repeat(np.repeat(frames_mask, x.shape[1], axis=1), x.shape[2], axis=2)

            if 'joints' in self.augmentations:
                # mask joints
                joints_mask1 = np.random.choice([0, 1], (x.shape[1],), p=[self.augmentations['joints'],
                                                                          1 - self.augmentations['joints']])
                joints_mask1 = np.clip(joints_mask1, 0, 1)
                joints_mask1 = np.expand_dims(np.expand_dims(joints_mask1, -1), 0)

                x *= np.repeat(np.repeat(joints_mask1, x.shape[-1], axis=-1), x.shape[0], axis=0)

        return x

    def __init__(self, augmentations={}):
        self.augmentations = augmentations

    def __call__(self, results):
        skeleton = results['input']
        results['input'] = self.random_masking(skeleton)
        return results


class RandomAdditiveNoise:
    def random_noise(self, x):
        if self.prob > 0.:
            if self.dist == 'UNI':
                # uniformal distributed noise
                joints_noise = np.random.rand(x.shape)
                # center noise and add mean
                joints_noise -= 0.5 + self.mean
                # scale noise to std
                joints_noise *= np.sqrt(12) * self.std / 2

            elif self.dist == 'NORMAL':
                # normal distributed noise
                joints_noise = np.random.normal(self.mean, self.std, size=x.shape)
            noise_mask = np.random.choice([0, 1], x.shape, p=[self.prob, 1 - self.prob])
            joints_noise *= noise_mask
            x += joints_noise
        return x

    def __init__(self, dist, prob=0.0, mean=0.0, std=0.001):
        assert dist in ['UNI', 'NORMAL'], f'Invalid noise distribution type: {dist}'

        self.dist = dist
        self.mean = mean
        self.std = std
        self.prob = prob

    def __call__(self, results):
        skeleton = results['input']
        results['input'] = self.random_noise(skeleton)
        return results


class RandomRot:
    def __init__(self, theta=0.3):
        self.theta = theta

    def _rot3d(self, theta):
        cos, sin = np.cos(theta), np.sin(theta)
        rx = np.array([[1, 0, 0], [0, cos[0], sin[0]], [0, -sin[0], cos[0]]])
        ry = np.array([[cos[1], 0, -sin[1]], [0, 1, 0], [sin[1], 0, cos[1]]])
        rz = np.array([[cos[2], sin[2], 0], [-sin[2], cos[2], 0], [0, 0, 1]])

        rot = np.matmul(rz, np.matmul(ry, rx))
        return rot

    def _rot2d(self, theta):
        cos, sin = np.cos(theta), np.sin(theta)
        return np.array([[cos, -sin], [sin, cos]])

    def __call__(self, results):
        skeleton = results['keypoint']
        T, V, C = skeleton.shape

        if np.all(np.isclose(skeleton, 0)):
            return results

        assert C in [2, 3]
        if C == 3:
            theta = np.random.uniform(-self.theta, self.theta, size=3)
            rot_mat = self._rot3d(theta)
        elif C == 2:
            theta = np.random.uniform(-self.theta)
            rot_mat = self._rot2d(theta)
        results['input'] = np.einsum('ab,tvb->tva', rot_mat, skeleton)

        return results


class RandomScale:
    def __init__(self, scale=0.2):
        assert isinstance(scale, tuple) or isinstance(scale, float)
        self.scale = scale

    def __call__(self, results):
        skeleton = results['keypoint']
        scale = self.scale
        if isinstance(scale, float):
            scale = (scale, ) * skeleton.shape[-1]
        assert len(scale) == skeleton.shape[-1]
        scale = 1 + np.random.uniform(-1, 1, size=len(scale)) * np.array(scale)
        results['keypoint'] = skeleton * scale
        return results


class RandomGaussianNoise:
    def __init__(self, sigma=0.01, base='frame', shared=False):
        assert isinstance(sigma, float)
        self.sigma = sigma
        self.base = base
        self.shared = shared
        assert self.base in ['frame', 'video']
        if self.base == 'frame':
            assert not self.shared

    def __call__(self, results):
        skeleton = results['keypoint']
        T, V, C = skeleton.shape
        skeleton = skeleton.reshape(-1, V, C)
        ske_min, ske_max = skeleton.min(axis=1), skeleton.max(axis=1)
        # MT * C
        flag = ((ske_min ** 2).sum(axis=1) > 1e-4)
        # MT
        if self.base == 'frame':
            norm = np.linalg.norm(ske_max - ske_min, axis=1) * flag
            # MT
        elif self.base == 'video':
            assert np.sum(flag)
            ske_min, ske_max = ske_min[flag].min(axis=0), ske_max[flag].max(axis=0)
            # C
            norm = np.linalg.norm(ske_max - ske_min)
            norm = np.array([norm] * T) * flag
        # MT * V
        if self.shared:
            noise = np.random.randn(V) * self.sigma
            noise = np.stack([noise] * T)
            noise = (noise.T * norm).T
            random_vec = np.random.uniform(-1, 1, size=(C, V))
            random_vec = random_vec / np.linalg.norm(random_vec, axis=0)
            random_vec = np.concatenate([random_vec] * T, axis=-1)
        else:
            noise = np.random.randn(T, V) * self.sigma
            noise = (noise.T * norm).T
            random_vec = np.random.uniform(-1, 1, size=(C, T * V))
            random_vec = random_vec / np.linalg.norm(random_vec, axis=0)
            # C * MTV
        random_vec = random_vec * noise.reshape(-1)
        # C * MTV
        random_vec = (random_vec.T).reshape(T, V, C)
        results['keypoint'] = skeleton.reshape(T, V, C) + random_vec
        return results


class PreNormalize3D:
    def __init__(self):
        pass

    def __call__(self, results):
        return results

    def downsample(self, data_numpy, step, random_sample=True):
        # input: C,T,V,M
        begin = np.random.randint(step) if random_sample else 0
        return data_numpy[begin::step, :, :]

    def temporal_slice(self, data_numpy, step):
        # input: C,T,V,M
        T, V, M = data_numpy.shape
        return data_numpy.reshape(T / step, step, V, M).transpose(
            (0, 1, 3, 2, 4)).reshape(T / step, V, step * M)

    def mean_subtractor(self, data_numpy, mean):
        # input: C,T,V,M
        # naive version
        if mean == 0:
            return
        T, V, M = data_numpy.shape
        valid_frame = (data_numpy != 0).sum(axis=2).sum(axis=1) > 0
        begin = valid_frame.argmax()
        end = len(valid_frame) - valid_frame[::-1].argmax()
        data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
        return data_numpy

    def auto_padding(self, data_numpy, size, random_pad=False):
        T, V, M = data_numpy.shape
        if T < size:
            begin = np.random.randint(0, size - T) if random_pad else 0
            data_numpy_paded = np.zeros((size, V, M))
            data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
            return data_numpy_paded
        else:
            return data_numpy

    def random_choose(self, data_numpy, size, auto_pad=True):
        # input: C,T,V,M
        T, V, M = data_numpy.shape
        if T == size:
            return data_numpy
        elif T < size:
            if auto_pad:
                return self.auto_padding(data_numpy, size, random_pad=True)
            else:
                return data_numpy
        else:
            begin = np.random.randint(0, T - size)
            return data_numpy[:, begin:begin + size, :, :]

    def random_move(self, data_numpy,
                    angle_candidate=[-10., -5., 0., 5., 10.],
                    scale_candidate=[0.9, 1.0, 1.1],
                    transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                    move_time_candidate=[1]):
        # input: C,T,V,M
        T, V, M = data_numpy.shape
        move_time = np.random.choice(move_time_candidate)
        node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
        node = np.append(node, T)
        num_node = len(node)

        A = np.random.choice(angle_candidate, num_node)
        S = np.random.choice(scale_candidate, num_node)
        T_x = np.random.choice(transform_candidate, num_node)
        T_y = np.random.choice(transform_candidate, num_node)

        a = np.zeros(T)
        s = np.zeros(T)
        t_x = np.zeros(T)
        t_y = np.zeros(T)

        # linspace
        for i in range(num_node - 1):
            a[node[i]:node[i + 1]] = np.linspace(
                A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
            s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                                 node[i + 1] - node[i])
            t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                                   node[i + 1] - node[i])
            t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                                   node[i + 1] - node[i])

        theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                          [np.sin(a) * s, np.cos(a) * s]])  # xuanzhuan juzhen

        # perform transformation
        for i_frame in range(T):
            xy = data_numpy[0:2, i_frame, :, :]
            new_xy = np.dot(theta[:, :, i_frame], xy.reshape(1, -1))
            new_xy[0] += t_x[i_frame]
            new_xy[1] += t_y[i_frame]  # pingyi bianhuan
            data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

        return data_numpy

    def random_shift(self, data_numpy):
        # input: C,T,V,M
        T, V, M = data_numpy.shape
        data_shift = np.zeros(data_numpy.shape)
        valid_frame = (data_numpy != 0).sum(axis=2).sum(axis=1) > 0
        begin = valid_frame.argmax()
        end = len(valid_frame) - valid_frame[::-1].argmax()

        size = end - begin
        bias = np.random.randint(0, T - size)
        data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

        return data_shift
