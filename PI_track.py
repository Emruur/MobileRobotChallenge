# track_map.py

import cv2
import numpy as np
from tracker import MultiObjectTracker
from mapper import GroundPlaneCalibrator
from picamera2 import Picamera2

def main():
    # --- Load calibrator ---
    calibrator = GroundPlaneCalibrator()
    calibrator.load_params('calib_params.json')

    # --- Initialize tracker ---
    tracker = MultiObjectTracker()

    # --- Open Picamera2 ---
    with Picamera2() as camera:
        # match your original driver.py settings
        camera.preview_configuration.main.size = (640, 480)
        camera.preview_configuration.main.format = "RGB888"
        camera.preview_configuration.align()
        camera.configure("preview")
        camera.start()

        state = 0  # 0 = capture bg, 1 = detect & init trackers
        print("1) Point camera at empty scene and press SPACE to capture background.")

        # --- Setup phase: capture background & initial detection ---
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
                    tracker.set_background(frame)
                    state = 1
                    print("Background captured.\n2) Place objects and press SPACE again.")
                else:  # state == 1
                    bboxes = tracker.detect(frame)
                    if len(bboxes) < tracker.max_objects:
                        print(f"Detected {len(bboxes)} objects, need {tracker.max_objects}. Try again.")
                        continue
                    for i, (x,y,w,h) in enumerate(bboxes,1):
                        print(f" Object {i}: x={x}, y={y}, w={w}, h={h}")
                    tracker.init_trackers(frame, bboxes)
                    print("Trackers initialized. Entering live tracking…")
                    break

        cv2.destroyWindow("Setup")

        # --- Bird's-eye view setup ---
        map_scale = 50            # pixels per world meter
        map_w, map_h = 800, 600   # BEV image size
        origin_x = map_w // 2     # world X=0 → center
        origin_y = map_h - 1      # world Z=0 → bottom row

        # draw a grid every 1 meter
        grid = np.ones((map_h, map_w, 3), dtype=np.uint8) * 255
        for m in range(0, map_w//map_scale + 1):
            x = origin_x + m*map_scale
            if 0 <= x < map_w:
                cv2.line(grid, (x,0),(x,map_h-1), (220,220,220), 1)
            x = origin_x - m*map_scale
            if 0 <= x < map_w:
                cv2.line(grid, (x,0),(x,map_h-1), (220,220,220), 1)
        for n in range(0, map_h//map_scale + 1):
            y = origin_y - n*map_scale
            if 0 <= y < map_h:
                cv2.line(grid, (0,y),(map_w-1,y), (220,220,220), 1)

        colors = [(0,255,0), (0,0,255), (255,0,0), (0,255,255)]

        # --- Live tracking loop ---
        while True:
            frame = camera.capture_array()
            bev = grid.copy()
            results = tracker.update(frame)

            for idx, (ok, bbox) in enumerate(results):
                color = colors[idx % len(colors)]
                if ok:
                    x, y, w, h = bbox
                    # draw tracking box
                    cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                    cv2.putText(frame, f"Obj{idx+1}", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    # infer ground‐plane coordinate at bottom‐center of bbox
                    u = x + w // 2
                    v = y + h
                    X, Z = calibrator.infer(u, v)
                    print(f"[Obj{idx+1}] Image pt ({u},{v}) → World (X={X:.2f}m, Z={Z:.2f}m)")

                    # map to BEV pixels
                    mx = int(origin_x + X * map_scale)
                    my = int(origin_y - Z * map_scale)
                    mx = np.clip(mx, 0, map_w-1)
                    my = np.clip(my, 0, map_h-1)

                    # draw on BEV
                    cv2.circle(bev, (mx, my), 6, color, -1)
                    cv2.putText(bev, f"{idx+1}", (mx+8, my+4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    cv2.putText(frame, f"Obj{idx+1} LOST", (10, 60 + 30*idx),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Tracking", frame)
            cv2.imshow("Birds Eye View", bev)

            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
