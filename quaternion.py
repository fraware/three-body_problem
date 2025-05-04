#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quaternion implementation for three-body problem analysis.

This module provides a complete implementation of quaternion algebra operations
required for quaternionic regularization of the three-body problem.
"""

import numpy as np
from typing import Union, Tuple, List

class Quaternion:
    """
    Quaternion class implementing quaternion algebra operations.

    A quaternion is represented as q = q0 + q1*i + q2*j + q3*k, where
    i^2 = j^2 = k^2 = ijk = -1.
    """

    def __init__(self, q0: float = 0.0, q1: float = 0.0, q2: float = 0.0, q3: float = 0.0):
        """
        Initialize a quaternion q = q0 + q1*i + q2*j + q3*k.

        Args:
            q0: Real scalar part
            q1: i coefficient (first imaginary component)
            q2: j coefficient (second imaginary component)
            q3: k coefficient (third imaginary component)
        """
        self.q0 = float(q0)
        self.q1 = float(q1)
        self.q2 = float(q2)
        self.q3 = float(q3)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Quaternion':
        """
        Create a quaternion from a numpy array with 4 elements.

        Args:
            arr: Array with 4 elements [q0, q1, q2, q3]

        Returns:
            A new Quaternion instance
        """
        if len(arr) != 4:
            raise ValueError("Array must have exactly 4 elements")
        return cls(arr[0], arr[1], arr[2], arr[3])

    @classmethod
    def from_scalar(cls, scalar: float) -> 'Quaternion':
        """
        Create a quaternion from a scalar (real number).

        Args:
            scalar: Real number

        Returns:
            A new Quaternion instance with only the scalar part
        """
        return cls(scalar, 0.0, 0.0, 0.0)

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> 'Quaternion':
        """
        Create a pure quaternion from a 3D vector.

        Args:
            vector: 3D vector [x, y, z] to be converted to a pure quaternion

        Returns:
            A new pure quaternion i*x + j*y + k*z
        """
        if len(vector) != 3:
            raise ValueError("Vector must have exactly 3 elements")
        return cls(0.0, vector[0], vector[1], vector[2])

    def to_array(self) -> np.ndarray:
        """
        Convert the quaternion to a numpy array.

        Returns:
            Numpy array [q0, q1, q2, q3]
        """
        return np.array([self.q0, self.q1, self.q2, self.q3])

    def scalar_part(self) -> float:
        """
        Get the scalar part of the quaternion.

        Returns:
            The scalar part q0
        """
        return self.q0

    def vector_part(self) -> np.ndarray:
        """
        Get the vector part of the quaternion.

        Returns:
            The vector part [q1, q2, q3]
        """
        return np.array([self.q1, self.q2, self.q3])

    def is_pure(self, tol: float = 1e-10) -> bool:
        """
        Check if the quaternion is a pure quaternion (scalar part is zero).

        Args:
            tol: Tolerance for floating-point comparison

        Returns:
            True if the quaternion is pure, False otherwise
        """
        return abs(self.q0) < tol

    def is_unit(self, tol: float = 1e-10) -> bool:
        """
        Check if the quaternion is a unit quaternion (norm is 1).

        Args:
            tol: Tolerance for floating-point comparison

        Returns:
            True if the quaternion has unit norm, False otherwise
        """
        return abs(self.norm() - 1.0) < tol

    def conjugate(self) -> 'Quaternion':
        """
        Calculate the conjugate of the quaternion.

        The conjugate of q = q0 + q1*i + q2*j + q3*k is q* = q0 - q1*i - q2*j - q3*k.

        Returns:
            A new Quaternion representing the conjugate
        """
        return Quaternion(self.q0, -self.q1, -self.q2, -self.q3)

    def norm_squared(self) -> float:
        """
        Calculate the squared norm of the quaternion.

        The squared norm is q0^2 + q1^2 + q2^2 + q3^2.

        Returns:
            The squared norm as a float
        """
        return self.q0**2 + self.q1**2 + self.q2**2 + self.q3**2

    def norm(self) -> float:
        """
        Calculate the norm (magnitude) of the quaternion.

        The norm is sqrt(q0^2 + q1^2 + q2^2 + q3^2).

        Returns:
            The norm as a float
        """
        return np.sqrt(self.norm_squared())

    def normalize(self) -> 'Quaternion':
        """
        Normalize the quaternion to have unit norm.

        Returns:
            A new Quaternion with unit norm
        """
        n = self.norm()
        if n < 1e-15:
            raise ValueError("Cannot normalize a zero quaternion")
        return Quaternion(self.q0/n, self.q1/n, self.q2/n, self.q3/n)

    def inverse(self) -> 'Quaternion':
        """
        Calculate the inverse of the quaternion.

        The inverse of q is q^(-1) = q* / |q|^2.

        Returns:
            A new Quaternion representing the inverse
        """
        n_squared = self.norm_squared()
        if n_squared < 1e-15:
            raise ValueError("Cannot invert a zero quaternion")
        return Quaternion(
            self.q0 / n_squared,
            -self.q1 / n_squared,
            -self.q2 / n_squared,
            -self.q3 / n_squared
        )

    def exponential(self) -> 'Quaternion':
        """
        Calculate the exponential of the quaternion.

        For a quaternion q = a + v (where a is the scalar part and v is the vector part),
        exp(q) = exp(a) * (cos(|v|) + v/|v| * sin(|v|)).

        Returns:
            A new Quaternion representing the exponential
        """
        a = self.q0
        v = np.array([self.q1, self.q2, self.q3])
        v_norm = np.linalg.norm(v)

        exp_a = np.exp(a)

        if v_norm < 1e-15:
            return Quaternion(exp_a, 0.0, 0.0, 0.0)

        scale = exp_a * np.sin(v_norm) / v_norm

        return Quaternion(
            exp_a * np.cos(v_norm),
            v[0] * scale,
            v[1] * scale,
            v[2] * scale
        )

    def logarithm(self) -> 'Quaternion':
        """
        Calculate the natural logarithm of the quaternion.

        For a quaternion q = a + v (where a is the scalar part and v is the vector part),
        log(q) = log(|q|) + v/|v| * arccos(a/|q|).

        Returns:
            A new Quaternion representing the logarithm
        """
        q_norm = self.norm()

        if q_norm < 1e-15:
            raise ValueError("Cannot compute logarithm of a zero quaternion")

        a = self.q0
        v = np.array([self.q1, self.q2, self.q3])
        v_norm = np.linalg.norm(v)

        if v_norm < 1e-15:
            if a > 0:
                return Quaternion(np.log(a), 0.0, 0.0, 0.0)
            else:
                # For negative real quaternions, the logarithm is not unique
                # We choose the principal branch
                return Quaternion(np.log(-a), np.pi, 0.0, 0.0)

        theta = np.arccos(min(max(a / q_norm, -1.0), 1.0))
        scale = theta / v_norm

        return Quaternion(
            np.log(q_norm),
            v[0] * scale,
            v[1] * scale,
            v[2] * scale
        )

    def power(self, exponent: float) -> 'Quaternion':
        """
        Calculate the quaternion raised to a real power.

        For a quaternion q and real number t, q^t = exp(t * log(q)).

        Args:
            exponent: Real number exponent

        Returns:
            A new Quaternion representing q^exponent
        """
        return (self.logarithm() * exponent).exponential()

    def rotate_vector(self, v: np.ndarray) -> np.ndarray:
        """
        Rotate a 3D vector using this quaternion as a rotation operator.

        For a unit quaternion q and a vector v, the rotated vector is q * v * q^(-1),
        where v is treated as a pure quaternion.

        Args:
            v: 3D vector to be rotated

        Returns:
            The rotated vector
        """
        if not self.is_unit(tol=1e-5):
            q_unit = self.normalize()
        else:
            q_unit = self

        v_quat = Quaternion(0.0, v[0], v[1], v[2])
        q_inv = q_unit.conjugate()  # For unit quaternions, conjugate equals inverse

        # Apply the rotation: q * v * q^(-1)
        rotated = q_unit * v_quat * q_inv

        return rotated.vector_part()

    @staticmethod
    def rotation_quaternion(axis: np.ndarray, angle: float) -> 'Quaternion':
        """
        Create a quaternion representing a rotation around an axis.

        Args:
            axis: 3D unit vector representing the rotation axis
            angle: Rotation angle in radians

        Returns:
            A new Quaternion representing the rotation
        """
        axis = np.asarray(axis)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-15:
            raise ValueError("Rotation axis cannot be a zero vector")

        # Normalize the axis
        axis = axis / axis_norm

        half_angle = angle / 2.0
        sin_half = np.sin(half_angle)

        return Quaternion(
            np.cos(half_angle),
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half
        )

    # Arithmetic operations
    def __add__(self, other: Union['Quaternion', float, int]) -> 'Quaternion':
        """Addition operator."""
        if isinstance(other, (float, int)):
            return Quaternion(self.q0 + other, self.q1, self.q2, self.q3)
        return Quaternion(
            self.q0 + other.q0,
            self.q1 + other.q1,
            self.q2 + other.q2,
            self.q3 + other.q3
        )

    def __radd__(self, other: Union[float, int]) -> 'Quaternion':
        """Right addition operator."""
        return self.__add__(other)

    def __sub__(self, other: Union['Quaternion', float, int]) -> 'Quaternion':
        """Subtraction operator."""
        if isinstance(other, (float, int)):
            return Quaternion(self.q0 - other, self.q1, self.q2, self.q3)
        return Quaternion(
            self.q0 - other.q0,
            self.q1 - other.q1,
            self.q2 - other.q2,
            self.q3 - other.q3
        )

    def __rsub__(self, other: Union[float, int]) -> 'Quaternion':
        """Right subtraction operator."""
        return Quaternion(other - self.q0, -self.q1, -self.q2, -self.q3)

    def __mul__(self, other: Union['Quaternion', float, int]) -> 'Quaternion':
        """Multiplication operator."""
        if isinstance(other, (float, int)):
            return Quaternion(
                self.q0 * other,
                self.q1 * other,
                self.q2 * other,
                self.q3 * other
            )

        # Quaternion multiplication: q * p
        a, b, c, d = self.q0, self.q1, self.q2, self.q3
        e, f, g, h = other.q0, other.q1, other.q2, other.q3

        return Quaternion(
            a*e - b*f - c*g - d*h,  # scalar part
            a*f + b*e + c*h - d*g,  # i coefficient
            a*g - b*h + c*e + d*f,  # j coefficient
            a*h + b*g - c*f + d*e   # k coefficient
        )

    def __rmul__(self, other: Union[float, int]) -> 'Quaternion':
        """Right multiplication operator."""
        if isinstance(other, (float, int)):
            return Quaternion(
                self.q0 * other,
                self.q1 * other,
                self.q2 * other,
                self.q3 * other
            )
        return NotImplemented

    def __truediv__(self, other: Union['Quaternion', float, int]) -> 'Quaternion':
        """Division operator."""
        if isinstance(other, (float, int)):
            if abs(other) < 1e-15:
                raise ZeroDivisionError("Division by near-zero scalar")
            return Quaternion(
                self.q0 / other,
                self.q1 / other,
                self.q2 / other,
                self.q3 / other
            )

        # Quaternion division: q / p = q * p^(-1)
        return self * other.inverse()

    def __rtruediv__(self, other: Union[float, int]) -> 'Quaternion':
        """Right division operator."""
        # other / self = other * self^(-1)
        return Quaternion.from_scalar(other) * self.inverse()

    def __neg__(self) -> 'Quaternion':
        """Negation operator."""
        return Quaternion(-self.q0, -self.q1, -self.q2, -self.q3)

    def __pos__(self) -> 'Quaternion':
        """Positive operator."""
        return Quaternion(self.q0, self.q1, self.q2, self.q3)

    def __eq__(self, other: object) -> bool:
        """Equality operator."""
        if not isinstance(other, Quaternion):
            return NotImplemented
        return (
            abs(self.q0 - other.q0) < 1e-10 and
            abs(self.q1 - other.q1) < 1e-10 and
            abs(self.q2 - other.q2) < 1e-10 and
            abs(self.q3 - other.q3) < 1e-10
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"Quaternion({self.q0}, {self.q1}, {self.q2}, {self.q3})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        terms = []
        if abs(self.q0) > 1e-10 or (abs(self.q1) < 1e-10 and abs(self.q2) < 1e-10 and abs(self.q3) < 1e-10):
            terms.append(f"{self.q0:.6g}")
        if abs(self.q1) > 1e-10:
            terms.append(f"{self.q1:.6g}i")
        if abs(self.q2) > 1e-10:
            terms.append(f"{self.q2:.6g}j")
        if abs(self.q3) > 1e-10:
            terms.append(f"{self.q3:.6g}k")

        return " + ".join(terms).replace(" + -", " - ")


def test_quaternion_operations():
    """Test quaternion operations to verify correctness."""
    # Test addition
    q1 = Quaternion(1, 2, 3, 4)
    q2 = Quaternion(5, 6, 7, 8)
    q_sum = q1 + q2
    assert np.allclose(q_sum.to_array(), np.array([6, 8, 10, 12]))

    # Test scalar addition
    q3 = q1 + 5
    assert np.allclose(q3.to_array(), np.array([6, 2, 3, 4]))

    # Test multiplication
    q_prod = q1 * q2
    expected_prod = Quaternion(
        1*5 - 2*6 - 3*7 - 4*8,
        1*6 + 2*5 + 3*8 - 4*7,
        1*7 - 2*8 + 3*5 + 4*6,
        1*8 + 2*7 - 3*6 + 4*5
    )
    assert np.allclose(q_prod.to_array(), expected_prod.to_array())

    # Test scalar multiplication
    q4 = q1 * 2
    assert np.allclose(q4.to_array(), np.array([2, 4, 6, 8]))

    # Test conjugate
    q_conj = q1.conjugate()
    assert np.allclose(q_conj.to_array(), np.array([1, -2, -3, -4]))

    # Test norm
    assert abs(q1.norm() - np.sqrt(30)) < 1e-10

    # Test inverse and verify q * q^(-1) = 1
    q_inv = q1.inverse()
    q_id = q1 * q_inv
    assert np.allclose(q_id.to_array(), np.array([1, 0, 0, 0]), atol=1e-10)

    # Test normalization
    q_norm = q1.normalize()
    assert abs(q_norm.norm() - 1.0) < 1e-10

    # Test exponential and logarithm
    q5 = Quaternion(0, 1, 0, 0)  # Pure quaternion i
    q_exp = q5.exponential()
    assert np.allclose(q_exp.to_array(), np.array([np.cos(1), np.sin(1), 0, 0]), atol=1e-10)

    q_log = q_exp.logarithm()
    assert np.allclose(q_log.to_array(), q5.to_array(), atol=1e-10)

    # Test rotation
    q_rot = Quaternion.rotation_quaternion(np.array([0, 0, 1]), np.pi/2)
    v = np.array([1, 0, 0])
    v_rot = q_rot.rotate_vector(v)
    assert np.allclose(v_rot, np.array([0, 1, 0]), atol=1e-10)

    # Test power
    q6 = Quaternion(0, 1, 0, 0)  # i
    q_pow = q6.power(2)
    assert np.allclose(q_pow.to_array(), np.array([-1, 0, 0, 0]), atol=1e-10)  # i^2 = -1

    print("All quaternion tests passed!")


if __name__ == "__main__":
    # Run tests to verify the quaternion implementation
    test_quaternion_operations()
