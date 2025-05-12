import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import json

class GroundPlaneCalibrator:
    """
    Calibrates camera intrinsics, radial distortion, and height from known floor markers,
    back-projects image pixels to (X, Z) on the ground plane,
    captures images, and can save/load parameters.
    """

    def __init__(self):
        # Intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.s = None  # skew
        # Radial distortion coefficients
        self.k1 = None
        self.k2 = None
        # Camera height above floor
        self.h = None

    def capture_image(self, save_path: str, camera_index: int = 0) -> np.ndarray:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            cap.open(camera_index)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Failed to capture from camera index {camera_index}")
        cv2.imwrite(save_path, frame)
        return frame

    def collect_data(self, image_path=None, image=None, num_points=6):
        if image is None:
            assert image_path is not None, "Provide image_path or image array"
            img_bgr = cv2.imread(image_path)
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title(f'Click {num_points} floor points (u, v)')
        pts = plt.ginput(num_points, timeout=0)
        plt.close(fig)
        return np.array(pts)

    def tune(self, world_pts, img_pts, initial_guess=None):
        """
        Fit intrinsics (fx, fy, cx, cy, s), radial distortion (k1, k2),
        and height h from world_pts (N×2) and img_pts (N×2).
        """
        world_pts = np.asarray(world_pts)
        img_pts = np.asarray(img_pts)

        def residuals(params):
            fx, fy, cx, cy, s, k1, k2, h = params
            res = []
            for (Xw, Zw), (u_obs, v_obs) in zip(world_pts, img_pts):
                # Camera coords of world point on floor (Yw=0): Xc=Xw, Yc=h, Zc=Zw
                x = Xw / Zw
                y = h / Zw
                # Radial distortion
                r2 = x*x + y*y
                factor = 1 + k1*r2 + k2*(r2**2)
                x_d = x * factor
                y_d = y * factor
                # Pixel projection
                u_pred = fx * x_d + s * y_d + cx
                v_pred = fy * y_d + cy
                res.append(u_pred - u_obs)
                res.append(v_pred - v_obs)
            return np.array(res)

        if initial_guess is None:
            w_img = img_pts[:,0].max()
            h_img = img_pts[:,1].max()
            # [fx, fy, cx, cy, s, k1, k2, h]
            initial_guess = np.array([w_img, h_img, w_img/2, h_img/2, 0.0, 0.0, 0.0, 1.0])

        result = least_squares(
            residuals, initial_guess,
            xtol=1e-12, ftol=1e-12, max_nfev=2000
        )
        (self.fx, self.fy, self.cx, self.cy,
         self.s, self.k1, self.k2, self.h) = result.x
        return result

    def infer(self, u, v):
        """
        Given a pixel (u, v) on the floor, returns its (X, Z).
        """
        # Build camera matrix and dist coeffs
        K = np.array([[self.fx, self.s, self.cx],
                      [0,      self.fy, self.cy],
                      [0,         0,     1    ]], dtype=np.float64)
        dist = np.array([self.k1, self.k2, 0.0, 0.0, 0.0], dtype=np.float64)
        # Undistort point to normalized coordinates
        pts = np.array([[[u, v]]], dtype=np.float32)
        undist = cv2.undistortPoints(pts, K, dist)  # shape (1,1,2)
        x_norm, y_norm = undist[0,0]
        # Intersect ray with floor Yc = h
        lam = self.h / y_norm
        X = lam * x_norm
        Z = lam
        return X, Z

    def save_params(self, filepath: str):
        params = {
            'fx': self.fx, 'fy': self.fy,
            'cx': self.cx, 'cy': self.cy,
            's': self.s, 'k1': self.k1, 'k2': self.k2,
            'h': self.h
        }
        with open(filepath, 'w') as f:
            json.dump(params, f)

    def load_params(self, filepath: str):
        with open(filepath, 'r') as f:
            params = json.load(f)
        self.fx = params['fx']
        self.fy = params['fy']
        self.cx = params['cx']
        self.cy = params['cy']
        self.s = params['s']
        self.k1 = params['k1']
        self.k2 = params['k2']
        self.h = params['h']





calibrator = GroundPlaneCalibrator()
# img_pts = calibrator.collect_data(image_path='abba.jpeg', num_points=7)
# world_pts = np.array([[0, 71],[54.4, 99],[-55.5,99],[42,145.7],[-42,145.7],[42,175.4],[-42,175.4]])
# calibrator.tune(world_pts, img_pts)
# calibrator.save_params('calib_params.json')


calibrator.load_params('calib_params.json')

X_obj, Z_obj = calibrator.infer(126, 215)

print(X_obj, Z_obj)
X_obj, Z_obj = calibrator.infer(519, 223)

print(X_obj, Z_obj)

X_obj, Z_obj = calibrator.infer(322, 465)

print(X_obj, Z_obj)


