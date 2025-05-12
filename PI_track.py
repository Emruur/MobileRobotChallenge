import cv2
import numpy as np
from tracker import MultiObjectTracker
from mapper import GroundPlaneCalibrator
from picamera2 import Picamera2
import picar_4wd as fc
import time



def expand_bbox(bbox, frame_shape, margin=0.2):
    x, y, w, h = bbox
    dw, dh = int(w * margin), int(h * margin)
    x0 = max(0, x - dw)
    y0 = max(0, y - dh)
    w0 = min(frame_shape[1] - x0, w + 2*dw)
    h0 = min(frame_shape[0] - y0, h + 2*dh)
    return (x0, y0, w0, h0)


class TrackMap:
    """
    Encapsulates background subtraction, multi-object tracking, and bird's-eye mapping.
    """
    def __init__(self,
                 calib_params_path: str,
                 map_size: tuple = (800, 600),
                 world_limits: tuple = ((-100.0, 100.0), (0.0, 200.0))):
        # Load calibrator
        self.calibrator = GroundPlaneCalibrator()
        self.calibrator.load_params(calib_params_path)
        # Initialize tracker
        self.tracker = MultiObjectTracker()
        # Bird's-eye parameters
        self.map_w, self.map_h = map_size
        (self.min_X, self.max_X), (self.min_Z, self.max_Z) = world_limits
        # compute scale
        scale_x = self.map_w / (self.max_X - self.min_X)
        scale_z = self.map_h / (self.max_Z - self.min_Z)
        self.map_scale = int(min(scale_x, scale_z))
        # origin in pixels: X=0 → center, Z=0 → bottom
        self.origin_x = self.map_w // 2
        self.origin_y = self.map_h - 1
        # Pre-build grid canvas
        self.grid = self._build_grid()
        
        

    def _build_grid(self):
        grid = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 255
        # vertical X lines
        for m in range(int(abs(self.max_X)) + 1):
            dx = int(m * self.map_scale)
            for sign in (+1, -1):
                x = self.origin_x + sign * dx
                if 0 <= x < self.map_w:
                    cv2.line(grid, (x, 0), (x, self.map_h-1), (220,220,220), 1)
        # horizontal Z lines
        for n in range(int(self.max_Z) + 1):
            dz = int(n * self.map_scale)
            y = self.origin_y - dz
            if 0 <= y < self.map_h:
                cv2.line(grid, (0, y), (self.map_w-1, y), (220,220,220), 1)
        return grid

    def detect(self, bg_frame: np.ndarray, frame: np.ndarray):
        """
        Capture background, detect and initialize trackers,
        return list of bboxes and their world coordinates.
        """
        self.tracker.set_background(bg_frame)
        bboxes = self.tracker.detect_no_shadow(frame)
        if len(bboxes) < self.tracker.max_objects:
            raise RuntimeError(
                f"Detected {len(bboxes)} objects, need {self.tracker.max_objects}.")
            
        padded = [expand_bbox(b, frame.shape, margin=0.2) for b in bboxes]
        self.tracker.init_trackers(frame, padded)
        coords = []
        for (x,y,w,h) in bboxes:
            u = x + w//2
            v = y + h
            X, Z = self.calibrator.infer(u, v)
            coords.append((X, Z))
        return bboxes, coords

    def update(self, frame: np.ndarray):
        """
        Update trackers on new frame; return list of tuples (ok, bbox, (X,Z) or None).
        """
        results = self.tracker.update(frame)
        output = []
        for ok, bbox in results:
            if not ok:
                output.append((False, None, None))
            else:
                x, y, w, h = bbox
                u = x + w//2
                v = y + h
                X, Z = self.calibrator.infer(u, v)
                output.append((True, (x, y, w, h), (X, Z)))
        return output

    def get_bev(self, frame: np.ndarray = None, draw_objects: bool = False):
        """
        Return bird's-eye view image; optionally overlay tracked objects.
        """
        bev = self.grid.copy()
        # draw camera
        cv2.circle(bev, (self.origin_x, self.origin_y), 8, (0,0,0), -1)
        # draw scale bar
        bar_len = self.map_scale * 10
        bar_start = (20, self.map_h - 30)
        bar_end = (20 + bar_len, self.map_h - 30)
        cv2.line(bev, bar_start, bar_end, (0,0,0), 2)
        cv2.putText(bev, "10 m", (bar_start[0], bar_start[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        if draw_objects and frame is not None:
            results = self.update(frame)
            colors = [(0,255,0),(0,0,255),(255,0,0),(0,255,255)]
            for idx, (ok, bbox, _) in enumerate(results):
                if not ok:
                    continue
                x, y, w, h = bbox
                X, Z = _
                mx = int(self.origin_x + X * self.map_scale)
                my = int(self.origin_y - Z * self.map_scale)
                mx = np.clip(mx, 0, self.map_w-1)
                my = np.clip(my, 0, self.map_h-1)
                color = colors[idx % len(colors)]
                cv2.circle(bev, (mx,my), 6, color, -1)
                cv2.putText(bev, str(idx+1), (mx+8, my+4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return bev

    def annotate_frame(self, frame: np.ndarray):
        """
        Draw bounding boxes, bottom-midpoint, and labels on the video frame.
        Returns the annotated frame and the track results.
        """
        results = self.update(frame)
        colors = [(0,255,0), (0,0,255), (255,0,0), (0,255,255)]
        for idx, (ok, bbox, coord) in enumerate(results):
            color = colors[idx % len(colors)]
            if not ok:
                cv2.putText(frame, f"Obj{idx+1} LOST", (10, 60 + 30*idx),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                continue
            x, y, w, h = bbox
            # draw bounding box and ID
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"Obj{idx+1}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            # bottom-midpoint
            u = x + w//2
            v = y + h
            cv2.circle(frame, (u, v), 5, color, -1)
            cv2.putText(frame, f"({u},{v})", (u+6, v-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame, results

