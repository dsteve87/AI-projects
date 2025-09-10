import math
import numpy as np
from scipy.optimize import root_scalar

class Projector:
    def __init__(self, coeffs, intrinsics, image_center, verbose=False):
        self.coeffs = coeffs

        self.u0 = image_center[0]
        self.v0 = image_center[1]

        self.fx = intrinsics[0]# / 1000.
        self.fy = intrinsics[1]# / 1000.

        self.Z = 2.

        self.verbose = verbose

    def set_Z(self, Z):
        self.Z = Z

    # def solve_theta(self, phi):
        

    def __call__(self, u, v):
        x = (u - self.u0) / self.fx
        y = (v - self.v0) / self.fy

        r = math.sqrt(x**2 + y**2)

        if self.verbose:
            print("r: ", r)

        phi = math.atan2(y, x)
        if self.verbose:
            print("phi: ", math.degrees(phi))

        return self.get_loc(r, phi)
    
    ###############################################################################################################
    def image_to_3d(self, u, v, Z=None):
        """Convert pixel coordinates (u, v) to 3D coordinates (X, Y, Z) using fisheye projection."""
        if Z is not None:
            self.set_Z(Z)
        x = (u - self.u0) / self.fx
        y = (v - self.v0) / self.fy
        r = math.sqrt(x**2 + y**2)

        if self.verbose:
            print(f"[image_to_3d] r: {r}")

        phi = math.atan2(y, x)
        if self.verbose:
            print(f"[image_to_3d] phi (deg): {math.degrees(phi)}")

        # Solve for theta using root-finding
        k = self.coeffs

        def r_theta(theta):
            return sum(k[i] * theta**(2*i + 1) for i in range(len(k)))

        def f(theta):
            return r_theta(theta) - r

        try:
            result = root_scalar(f, bracket=[0, np.pi/2], method='brentq')
        except ValueError:
            if self.verbose:
                print(f"[image_to_3d] No theta found for r={r}")
            return None

        if not result.converged:
            if self.verbose:
                print(f"[image_to_3d] Root finding did not converge for r={r}")
            return None

        theta = result.root
        if self.verbose:
            print(f"[image_to_3d] theta (deg): {math.degrees(theta)}")

        X = self.Z * math.tan(theta) * math.cos(phi)
        Y = self.Z * math.tan(theta) * math.sin(phi)

        return np.array([X, Y, self.Z])

#############################################################################################
    def get_loc(self, r, phi):

        # Define the polynomial equation f(theta) = r(theta) - r_value
        k = self.coeffs

        def r_theta(theta):
            n = len(k)
            return sum(k[i] * theta**(2*i + 1) for i in range(n))

        def f(theta):
            return r_theta(theta) - r

        # Use a root-finding method
        try:
            result = root_scalar(f, bracket=[0, np.pi/2], method='brentq')  # bracket range can vary
        except ValueError as e:
            print(f'Failed to find theta for r = {r}')
            # raise ValueError('Failed to find theta') from e
            return None, None

        if not result.converged:
            raise ValueError(f'Failed to find theta for r = {r}')

        theta = result.root
        theta = math.atan2(math.sin(theta), math.cos(theta))
        if self.verbose:
            print("theta: ", theta)

        X = self.Z * math.tan(theta) * math.cos(phi)
        Y = self.Z * math.tan(theta) * math.sin(phi)

        if self.verbose:
            print("X: ", X)
            print("Y: ", Y)

        return X, Y