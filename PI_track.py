import cv2
import numpy as np
from tracker import MultiObjectTracker
from mapper import GroundPlaneCalibrator
from picamera2 import Picamera2

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
        bboxes = self.tracker.detect(frame)
        if len(bboxes) < self.tracker.max_objects:
            raise RuntimeError(
                f"Detected {len(bboxes)} objects, need {self.tracker.max_objects}.")
        self.tracker.init_trackers(frame, bboxes)
        coords = []
        for (x,y,w,h) in bboxes:
            u = x + w//2
            v = y + h
            X, Z = self.calibrator.infer(u, v)
            coords.append((X, Z))
        return bboxes, coords

    def update(self, frame: np.ndarray):
        """
        Update trackers on new frame; return world coords or None if lost.
        """
        results = self.tracker.update(frame)
        world_coords = []
        for ok, bbox in results:
            if not ok:
                world_coords.append(None)
            else:
                x, y, w, h = bbox
                u = x + w//2
                v = y + h
                X, Z = self.calibrator.infer(u, v)
                world_coords.append((X, Z))
        return world_coords

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
            results = self.tracker.update(frame)
            colors = [(0,255,0),(0,0,255),(255,0,0),(0,255,255)]
            for idx, (ok, bbox) in enumerate(results):
                if not ok:
                    continue
                x,y,w,h = bbox
                u = x + w//2
                v = y + h
                X, Z = self.calibrator.infer(u, v)
                mx = int(self.origin_x + X * self.map_scale)
                my = int(self.origin_y - Z * self.map_scale)
                mx = np.clip(mx, 0, self.map_w-1)
                my = np.clip(my, 0, self.map_h-1)
                color = colors[idx % len(colors)]
                cv2.circle(bev, (mx,my), 6, color, -1)
                cv2.putText(bev, str(idx+1), (mx+8, my+4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return bev


def main():
    tm = TrackMap('calib_params.json')
    with Picamera2() as camera:
        # PiCamera configuration
        camera.preview_configuration.main.size = (640, 480)
        camera.preview_configuration.main.format = "RGB888"
        camera.preview_configuration.align()
        camera.configure("preview")
        camera.start()

        state = 0  # 0: capture bg, 1: detect & init
        print("1) Point camera at empty scene and press SPACE to capture background.")
        bg_frame = None

        # Setup loop
        while True:
            frame = camera.capture_array()
            cv2.putText(frame, f"STEP {state}: press SPACE", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Setup", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                cv2.destroyAllWindows()
                return
            if key == 32:  # SPACE
                if state == 0:
                    bg_frame = frame.copy()
                    state = 1
                    print("Background captured.\n2) Place objects and press SPACE again.")
                else:
                    try:
                        bboxes, coords = tm.detect(bg_frame, frame)
                    except RuntimeError as e:
                        print(e)
                        continue
                    for i, (x,y,w,h) in enumerate(bboxes, 1):
                        print(f" Object {i}: x={x}, y={y}, w={w}, h={h}")
                    print("Trackers initialized. Entering live tracking…")
                    break

        cv2.destroyWindow("Setup")

        # Save initial BEV
        init_bev = tm.get_bev()
        cv2.imwrite("initial_map.png", init_bev)
        print("Saved initial BEV: initial_map.png")

        # Live tracking loop
        while True:
            frame = camera.capture_array()
            coords = tm.update(frame)
            bev = tm.get_bev(frame, draw_objects=True)
            for idx, coord in enumerate(coords):
                if coord is None:
                    print(f"[Obj{idx+1}] LOST")
                else:
                    X, Z = coord
                    print(f"[Obj{idx+1}] World (X={X:.2f}m, Z={Z:.2f}m)")

            cv2.imshow("Tracking", frame)
            cv2.imshow("Birds Eye View", bev)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
