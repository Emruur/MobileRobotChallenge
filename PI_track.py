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
        camera.preview_configuration.main.size = (640, 480)
        camera.preview_configuration.main.format = "RGB888"
        camera.preview_configuration.align()
        camera.configure("preview")
        camera.start()

        state = 0  # 0 = capture bg, 1 = detect & init trackers
        print("1) Point camera at empty scene and press SPACE to capture background.")

        # --- Setup: capture background & initial detection ---
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
                    for i, (x, y, w, h) in enumerate(bboxes, 1):
                        print(f" Object {i}: x={x}, y={y}, w={w}, h={h}")
                    tracker.init_trackers(frame, bboxes)
                    print("Trackers initialized. Entering live tracking…")
                    break

        cv2.destroyWindow("Setup")

        # --- Bird's-eye view parameters (dynamic) ---
        max_abs_X = 100.0   # world X ∈ [–100, +100] m
        max_Z     = 200.0   # world Z ∈ [  0,  200] m
        map_w, map_h = 800, 600

        # compute pixels‐per‐meter so both axes fit
        scale_x = map_w  / (2 * max_abs_X)   # px/m in X
        scale_z = map_h  / (    max_Z   )    # px/m in Z
        map_scale = int(min(scale_x, scale_z))

        origin_x = map_w // 2      # world X=0 → center of image
        origin_y = map_h - 1       # world Z=0 → bottom row

        print(f"Using map_scale = {map_scale} px/m")

        # draw a meter‐grid on a white canvas
        grid = np.ones((map_h, map_w, 3), dtype=np.uint8) * 255
        # vertical lines (X ticks)
        for m in range(0, int(max_abs_X)+1):
            dx = int(m * map_scale)
            # +X
            x1 = origin_x + dx
            if 0 <= x1 < map_w:
                cv2.line(grid, (x1, 0), (x1, map_h-1), (220,220,220), 1)
            # –X
            x2 = origin_x - dx
            if 0 <= x2 < map_w:
                cv2.line(grid, (x2, 0), (x2, map_h-1), (220,220,220), 1)
        # horizontal lines (Z ticks)
        for n in range(0, int(max_Z)+1):
            dz = int(n * map_scale)
            y = origin_y - dz
            if 0 <= y < map_h:
                cv2.line(grid, (0, y), (map_w-1, y), (220,220,220), 1)

        colors = [(0,255,0), (0,0,255), (255,0,0), (0,255,255)]

        # --- Save initial BEV with objects and camera dot ---
        init_bev = grid.copy()
        # draw camera at (0,0)
        cv2.circle(init_bev, (origin_x, origin_y), 8, (0,0,0), -1)
        # draw scale bar (10 m)
        bar_len = map_scale * 10
        bar_start = (20, map_h - 30)
        bar_end   = (20 + bar_len, map_h - 30)
        cv2.line(init_bev, bar_start, bar_end, (0,0,0), 2)
        cv2.putText(init_bev, "10 m", (bar_start[0], bar_start[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        # plot each detected object
        for idx, (x, y, w, h) in enumerate(bboxes):
            u = x + w//2
            v = y + h
            X, Z = calibrator.infer(u, v)
            mx = int(origin_x + X * map_scale)
            my = int(origin_y - Z * map_scale)
            mx = np.clip(mx, 0, map_w-1)
            my = np.clip(my, 0, map_h-1)
            cv2.circle(init_bev, (mx, my), 6, colors[idx % len(colors)], -1)
            cv2.putText(init_bev, str(idx+1), (mx+8, my+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[idx % len(colors)], 2)

        # save the first map to disk
        cv2.imwrite("initial_map.png", init_bev)
        print("Saved initial birds-eye map: initial_map.png")

        # --- Live tracking loop ---
        while True:
            frame = camera.capture_array()
            bev   = grid.copy()

            # draw camera on each frame
            cv2.circle(bev, (origin_x, origin_y), 8, (0,0,0), -1)
            # draw scale bar on each frame
            cv2.line(bev, bar_start, bar_end, (0,0,0), 2)
            cv2.putText(bev, "10 m", (bar_start[0], bar_start[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

            results = tracker.update(frame)

            for idx, (ok, bbox) in enumerate(results):
                color = colors[idx % len(colors)]
                if not ok:
                    cv2.putText(frame, f"Obj{idx+1} LOST",
                                (10, 60 + 30*idx),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, color, 2)
                    continue

                x, y, w, h = bbox
                # draw box & label
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                cv2.putText(frame, f"Obj{idx+1}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # bottom-midpoint
                u = x + w//2
                v = y + h
                cv2.circle(frame, (u,v), 5, color, -1)
                cv2.putText(frame, f"({u},{v})", (u+6, v-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # infer world coords
                X, Z = calibrator.infer(u, v)
                print(f"[Obj{idx+1}] Img pt=({u},{v}) → World (X={X:.2f}m, Z={Z:.2f}m)")

                # map to BEV px
                mx = int(origin_x + X * map_scale)
                my = int(origin_y - Z * map_scale)
                mx = np.clip(mx, 0, map_w-1)
                my = np.clip(my, 0, map_h-1)

                cv2.circle(bev, (mx, my), 6, color, -1)
                cv2.putText(bev, str(idx+1), (mx+8, my+4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Tracking", frame)
            cv2.imshow("Birds Eye View", bev)

            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
