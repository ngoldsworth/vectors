import numpy as np
import astropy.units as u


class Vector3D:
    def __init__(self,
                 q0: np.ndarray or u.Quantity,
                 q1: np.ndarray or u.Quantity,
                 q2: np.ndarray or u.Quantity,
                 coordinate_system: str = 'cartesian',
                 ):
        """

        :param coordinate_system: `cartesian` for [x, y, z]. `spherical` for [r, phi, theta] where r is the distance
        from the origin, phi is the angle from the x-axis, and theta is the angle from the z-axis. This follows the
        notation in Griffiths. Any angles are assumed to be in radians
        """
        if len(set([q.shape for q in [q0, q1, q2]])) != 1:
            raise ValueError('All three coordinate arrays must be the same shape')

        if coordinate_system == 'cartesian':
            self.magnitude = np.sqrt(np.square(q0) + np.square(q1) + np.square(q2))
            self._x = q0
            self._y = q1
            self._z = q2

        elif coordinate_system == 'spherical':
            # where r = q0, phi = q1, theta = q2, in Griffith's notation
            self.magnitude = q0
            self._x = q0 * np.cos(q1) * np.sin(q2)
            self._y = q0 * np.sin(q1) * np.sin(q2)
            self._z = q0 * np.cos(q2)

        elif coordinate_system == 'cylindrical':
            # where s = q0, phi = q1, z = q3, in Griffith's notation
            self.magnitude = np.sqrt(q0 ** 2 + q2 ** 2)
            self._x = q0 * np.cos(q1)
            self._y = q0 * np.sin(q1)
            self._z = q2

        else:
            raise ValueError('Must choose one of following coordinate systems: `cartesian`, `cylindrical`, `spherical')

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def s(self):
        return np.sqrt(self._x ** 2 + self._y ** 2)

    @property
    def phi(self):
        """
        :return: phi value for the vector in spherical or cylindrical coordinates. Given as angle between 0 and 2pi
        """
        angles = np.arctan2(self._x / self._y)
        angles[angles < 0] = angles[angles < 0] + 2 * np.pi
        return angles

    @property
    def theta(self):
        return np.arccos(self._z / self.magnitude)

    def normalize(self):
        norm_x = self._x / self.magnitude
        norm_y = self._y / self.magnitude
        norm_z = self._z / self.magnitude

        return Vector3D(norm_x, norm_y, norm_z, coordinate_system='cartesian')

    def copy(self):
        return Vector3D(self._x.copy, self._y.copy, self._z.copy(), coordinate_system='cartesian')

    def __neg__(self):
        return Vector3D(-1 * self._x.copy(), -1 * self._y.copy(), -1 * self._z.copy(), coordinate_system='cartesian')

    def __add__(self, other):
        sumx = self._x + other.x()
        sumy = self._y + other.y()
        sumz = self._z + other.z()
        return Vector3D(sumx, sumy, sumz, coordinate_system='cartesian')

    def __sub__(self, other):
        return self.__add__(-other)

    def __mul__(self, other: int or float):
        return Vector3D(
            other * self._x,
            other * self._y,
            other * self._z,
            coordinate_system='cartesian'
        )

    def __truediv__(self, other: float or int):
        return self.__mul__(self, 1/other)


def inner_product(u: Vector3D, v: Vector3D):
    return v.x * u.x + v.y * u.y + v.z * u.z


def outer_product(u: Vector3D, v: Vector3D):
    """
    Standard 3D cross product. Order matters, this assumes u cross v.
    :param u:
    :param v:
    :return: u x v
    """
    cx = u.y * v.z - u.z * v.y
    cy = u.z * v.x - u.x * v.z
    cz = u.x * v.y - u.y * v.x
    return Vector3D(cx, cy, cz, coordinate_system='cartesian')


if __name__ == '__main__':
    one = np.ones(1)
    zero = np.zeros(1)
    U = Vector3D(one, zero, zero)
    V = Vector3D(zero, one, zero)

    print(outer_product(U, V).z)
