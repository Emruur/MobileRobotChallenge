import time
import cv2
import numpy as np
from picamera2 import Picamera2
import picar_4wd as fc
from PI_track import TrackMap

# Configuration
CALIB_PARAMS = 'calib_params.json'
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CENTER_U = FRAME_WIDTH // 2
ANGLE_THRESHOLD_PX = 10      # pixel tolerance for alignment
ROTATE_STEP_S = 0.05         # duration for each rotation step
FORWARD_MOVE_S = 2.0         # duration to move forward after alignment
POWER_VAL = 50               # motor power
A_INITIAL = 50.0             # initial normal offset in meters
A_MIN = 1.0                  # minimum offset to stop iteration
A_FACTOR = 0.5               # shrink factor for next iteration


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
    # choose candidate closer to origin
    if np.linalg.norm(c1) < np.linalg.norm(c2):
        return tuple(c1)
    else:
        return tuple(c2)


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

        # initial object positions
        p1, p2 = coords[0], coords[1]
        a = A_INITIAL

        print("Movement started. Press SPACE to abort.")
        try:
            while a >= A_MIN:
                # compute world-target and its BEV pixel
                target_world = compute_target(p1, p2, a)
                target_bev = world_to_bev_px(target_world, tm)

                # alignment loop: rotate until midpoint aligns with target direction
                while True:
                    frame = camera.capture_array()
                    results = tm.update(frame)
                    # draw live BEV
                    bev = tm.get_bev(frame, draw_objects=True)
                    mid_world = ((results[0][2][0] + results[1][2][0]) / 2.0,
                                 (results[0][2][1] + results[1][2][1]) / 2.0)
                    mid_bev = world_to_bev_px(mid_world, tm)
                    cv2.arrowedLine(bev, mid_bev, target_bev, (255,0,0), 2)
                    cv2.circle(bev, target_bev, 6, (0,0,255), -1)
                    cv2.imshow("Birds Eye View", bev)
                    cv2.imshow("Camera", frame)

                    # pixel midpoint in image coordinates
                    u1 = results[0][1][0] + results[0][1][2]//2
                    u2 = results[1][1][0] + results[1][1][2]//2
                    mid_u = (u1 + u2) / 2.0
                    # estimate direction error sign by comparing mid_bev vector
                    # here we approximate using mid_u vs CENTER_U
                    err = mid_u - CENTER_U
                    if abs(err) <= ANGLE_THRESHOLD_PX:
                        break
                    # small rotation
                    if err > 0:
                        fc.turn_right(POWER_VAL)
                    else:
                        fc.turn_left(POWER_VAL)
                    time.sleep(ROTATE_STEP_S)
                    fc.stop()

                    # abort check
                    if cv2.waitKey(1) & 0xFF == 32:
                        raise KeyboardInterrupt

                # once aligned, move forward fixed duration
                fc.forward(POWER_VAL)
                start = time.time()
                while time.time() - start < FORWARD_MOVE_S:
                    # show live BEV and camera during forward motion
                    frame = camera.capture_array()
                    results = tm.update(frame)
                    bev = tm.get_bev(frame, draw_objects=True)
                    cv2.imshow("Birds Eye View", bev)
                    cv2.imshow("Camera", frame)
                    if cv2.waitKey(1) & 0xFF == 32:
                        raise KeyboardInterrupt
                fc.stop()

                # update object positions
                p1, p2 = results[0][2], results[1][2]
                # reduce offset
                a *= A_FACTOR

        except KeyboardInterrupt:
            print("Movement aborted by user.")
        finally:
            fc.stop()
            camera.stop()
            cv2.destroyAllWindows()
            print("Robot stopped.")

if __name__ == '__main__':
    main()
