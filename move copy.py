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
ANGLE_THRESHOLD_PX = 20     # pixels tolerance for alignment
FORWARD_DURATION_S = 0.1    # duration to move forward each cycle
ROTATE_DURATION_S = 0.05    # duration to rotate during adjustment
POWER_VAL = 50              # motor power
ADJUST_INTERVAL = 20        # frames between re-adjustments


def main():
    # Initialize TrackMap and camera
    tm = TrackMap(CALIB_PARAMS)
    with Picamera2() as camera:
        camera.preview_configuration.main.size = (FRAME_WIDTH, FRAME_HEIGHT)
        camera.preview_configuration.main.format = "RGB888"
        camera.preview_configuration.align()
        camera.configure("preview")
        camera.start()

        # -- Setup: capture background & initialize trackers --
        state = 0
        bg_frame = None
        print("1) Point camera at empty scene and press SPACE to capture background.")
        while True:
            frame = camera.capture_array()
            cv2.putText(frame, f"STEP {state}: press SPACE", (10, 30),
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

        frame_count = 0
        last_mid_u = CENTER_U  # start aiming at center
        print("Movement started. Press SPACE again to stop.")
        try:
            while True:
                frame = camera.capture_array()
                results = tm.update(frame)

                # update midpoint if both objects visible
                if len(results) >= 2 and results[0][0] and results[1][0]:
                    _, bbox1, _ = results[0]
                    _, bbox2, _ = results[1]
                    x1, y1 = bbox1[0] + bbox1[2] // 2, bbox1[1] + bbox1[3] // 2
                    x2, y2 = bbox2[0] + bbox2[2] // 2, bbox2[1] + bbox2[3] // 2

                    # Midpoint between objects
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2

                    # Line between objects
                    dx = x2 - x1
                    dy = y2 - y1

                    # Normal vector (perpendicular direction)
                    norm_dx = -dy
                    norm_dy = dx

                    # Normalize
                    norm_length = np.hypot(norm_dx, norm_dy)
                    norm_dx /= norm_length
                    norm_dy /= norm_length

                    # Decrease offset distance over time (starts large, shrinks)
                    max_offset = 100  # pixels
                    decay_rate = 0.98  # how fast it decays
                    offset = max_offset * (decay_rate ** frame_count)

                    # Target point along the normal
                    target_x = mid_x + norm_dx * offset
                    target_u = target_x  # horizontal coordinate

                    last_mid_u = target_u  # Use this as the new alignment goal


                # every ADJUST_INTERVAL frames, perform fine alignment loop
                error = ANGLE_THRESHOLD_PX +1
                if frame_count % ADJUST_INTERVAL == 0:
                    while abs(error) > ANGLE_THRESHOLD_PX:
                        if error > 0:
                            fc.turn_right(POWER_VAL)
                        else:
                            fc.turn_left(POWER_VAL)
                        time.sleep(ROTATE_DURATION_S)
                        fc.stop()
                        frame2 = camera.capture_array()
                        results2 = tm.update(frame2)
                        if len(results2) >= 2 and results2[0][0] and results2[1][0]:
                            _, b1, _ = results2[0]
                            _, b2, _ = results2[1]
                            u1 = b1[0] + b1[2] // 2
                            u2 = b2[0] + b2[2] // 2
                            last_mid_u = (u1 + u2) / 2
                        error = last_mid_u - CENTER_U
                else:
                    fc.forward(POWER_VAL)
                    time.sleep(FORWARD_DURATION_S)
                    fc.stop()

                frame_count += 1

                # Render and display camera and BEV
                bev = tm.get_bev(frame, draw_objects=True)
                cv2.imshow("Movement", frame)
                cv2.imshow("Birds Eye View", bev)

                # Check for SPACE to exit
                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # SPACE
                    print("SPACE pressed. Stopping movement.")
                    break
                elif key in (27, ord('q')):
                    print("Quit signal received. Exiting.")
                    break

        finally:
            fc.stop()
            camera.stop()
            cv2.destroyAllWindows()
            print("Movement finished. Robot stopped.")


if __name__ == '__main__':
    main()
