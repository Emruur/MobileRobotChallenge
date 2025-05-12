import time
import cv2
import numpy as np
from math import atan2, degrees
from picamera2 import Picamera2
import picar_4wd as fc
from PI_track import TrackMap

# Configuration
CALIB_PARAMS = 'calib_params.json'
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# Alignment thresholds
ANGLE_THRESHOLD_DEG = 5.0   # degrees tolerance for alignment
ROTATE_STEP_S = 0.05         # duration for each rotation step
FORWARD_MOVE_S = 0.5      # duration to move forward after alignment
POWER_VAL = 10               # motor power
# Normal-offset path parameters
A_INITIAL = 50.0             # initial normal offset in meters
A_MIN = 1.0                  # minimum offset to stop iteration
A_FACTOR = 0.85               # shrink factor for next iteration



def compute_target(p1, p2, a):
    """
    Compute midpoint-normal target in world coords.
    """
    mid = ((p1[0] + p2[0]) / 2.0,
           (p1[1] + p2[1]) / 2.0)
    delta = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    norm = np.array([-delta[1], delta[0]])
    if np.linalg.norm(norm) < 1e-6:
        return mid
    norm_unit = norm / np.linalg.norm(norm)
    c1 = mid + a * norm_unit
    c2 = mid - a * norm_unit
    return tuple(c1) if np.linalg.norm(c1) < np.linalg.norm(c2) else tuple(c2)


def world_to_bev_px(world, tm):
    """
    Convert world (X,Z) to BEV pixel coords.
    """
    mx = int(tm.origin_x + world[0] * tm.map_scale)
    my = int(tm.origin_y - world[1] * tm.map_scale)
    return mx, my


def main():
    tm = TrackMap(CALIB_PARAMS)
    with Picamera2() as camera:
        camera.preview_configuration.main.size = (FRAME_WIDTH, FRAME_HEIGHT)
        camera.preview_configuration.main.format = "RGB888"
        camera.preview_configuration.align()
        camera.configure("preview")
        camera.start()

        # Setup phase
        state = 0
        bg_frame = None
        print("1) Point camera at empty scene and press SPACE to capture background.")
        while True:
            frame = camera.capture_array()
            cv2.putText(frame, f"STEP {state}: press SPACE", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Setup", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                if state == 0:
                    bg_frame = frame.copy()
                    state = 1
                    print("Background captured. Place objects and press SPACE again.")
                else:
                    bboxes, coords = tm.detect(bg_frame, frame)
                    print(f"Initialized trackers on {len(bboxes)} objects.")
                    break
            elif key in (27, ord('q')):
                print("Setup aborted.")
                camera.stop()
                cv2.destroyAllWindows()
                return
        cv2.destroyWindow("Setup")

        # initial world positions and offset
        p1, p2 = coords[0], coords[1]
        a = A_INITIAL
        stop = False
        print("Movement started. Press SPACE to abort at any time.")

        # main loop over offsets
        while a >= A_MIN and not stop:
            frame = camera.capture_array()
            results = tm.update(frame)
            # if objects lost before alignment, drive straight
            if len(results) < 2 or not (results[0][0] and results[1][0]):
                print("Objects lost: driving straight until SPACE.")
                while True:
                    frame = camera.capture_array()
                    bev = tm.get_bev(frame, draw_objects=True)
                    cv2.imshow("Camera", frame)
                    cv2.imshow("Birds Eye View", bev)
                    fc.forward(POWER_VAL)
                    if cv2.waitKey(1) & 0xFF == 32:
                        stop = True
                        break
                break

            # alignment loop
            while not stop:
                frame = camera.capture_array()
                results = tm.update(frame)
                # lost detection during alignment
                if len(results) < 2 or not (results[0][0] and results[1][0]):
                    print("Objects lost during alignment, proceeding straight.")
                    break
                # recalc target and error angle
                p1, p2 = results[0][2], results[1][2]
                target_world = compute_target(p1, p2, a)
                err_rad = atan2(target_world[0], target_world[1])
                err_deg = degrees(err_rad)
                # visualize BEV
                bev = tm.get_bev(frame, draw_objects=True)
                mid_world = ((p1[0] + p2[0]) / 2.0,
                             (p1[1] + p2[1]) / 2.0)
                mid_bev = world_to_bev_px(mid_world, tm)
                target_bev = world_to_bev_px(target_world, tm)
                cv2.arrowedLine(bev, mid_bev, target_bev, (255,0,0), 2)
                cv2.circle(bev, target_bev, 6, (0,0,255), -1)
                cv2.imshow("Birds Eye View", bev)
                
                for found, bbox, _ in results:
                    if not found:
                        continue
                    x, y, w, h = bbox
                    x1, y1 = x, y
                    x2, y2 = x + w, y + h
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    mid_pt = (int((x1+x2)/2), y2)
                    cv2.circle(frame, mid_pt, 5, (0,0,255), -1)
                # --------------------------------

                cv2.imshow("Camera", frame)

                if abs(err_deg) <= ANGLE_THRESHOLD_DEG:
                    break
                # small rotation step
                if err_deg > 0:
                    fc.turn_right(POWER_VAL)
                else:
                    fc.turn_left(POWER_VAL)
                time.sleep(ROTATE_STEP_S)
                fc.stop()
                # abort check
                if cv2.waitKey(1) & 0xFF == 32:
                    stop = True

            # forward motion
            if stop:
                break
            fc.forward(POWER_VAL)
            t0 = time.time()
            while time.time() - t0 < FORWARD_MOVE_S and not stop:
                frame = camera.capture_array()
                ## Shouldnt we also call update here
                tm.update(frame)
                bev = tm.get_bev(frame, draw_objects=True)
                cv2.imshow("Camera", frame)
                cv2.imshow("Birds Eye View", bev)
                if cv2.waitKey(1) & 0xFF == 32:
                    stop = True
            fc.stop()
            # shrink offset
            a *= A_FACTOR

        # final cleanup
        fc.stop()
        camera.stop()
        cv2.destroyAllWindows()
        print("Movement finished. Robot stopped.")

if __name__ == '__main__':
    main()
