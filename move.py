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
POWER_VAL = 10              # motor power
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
            if cv2.waitKey(1) & 0xFF == 32:
                if state == 0:
                    bg_frame = frame.copy()
                    state = 1
                    print("Background captured. Place objects and press SPACE again.")
                else:
                    bboxes, coords = tm.detect(bg_frame, frame)
                    print(f"Initialized trackers on {len(bboxes)} objects.")
                    break
        cv2.destroyWindow("Setup")

        frame_count = 0
        last_mid_u = CENTER_U  # start aiming at center
        try:
            while True:
                frame = camera.capture_array()
                results = tm.update(frame)

                # update midpoint if both objects visible
                if len(results) >= 2 and results[0][0] and results[1][0]:
                    _, bbox1, _ = results[0]
                    _, bbox2, _ = results[1]
                    u1 = bbox1[0] + bbox1[2] // 2
                    u2 = bbox2[0] + bbox2[2] // 2
                    last_mid_u = (u1 + u2) / 2

                # compute current error
                error = last_mid_u - CENTER_U

                # every ADJUST_INTERVAL frames, perform fine alignment loop
                if frame_count % ADJUST_INTERVAL == 0:
                    # continue rotating in small increments until aligned
                    while abs(error) > ANGLE_THRESHOLD_PX:
                        if error > 0:
                            fc.turn_right(POWER_VAL)
                        else:
                            fc.turn_left(POWER_VAL)
                        time.sleep(ROTATE_DURATION_S)
                        fc.stop()
                        # recalc error using fresh frame
                        frame2 = camera.capture_array()
                        results2 = tm.update(frame2)
                        if len(results2) >= 2 and results2[0][0] and results2[1][0]:
                            _, b1, _ = results2[0]
                            _, b2, _ = results2[1]
                            u1 = b1[0] + b1[2] // 2
                            u2 = b2[0] + b2[2] // 2
                            last_mid_u = (u1 + u2) / 2
                        # update error even if objects lost
                        error = last_mid_u - CENTER_U
                else:
                    # move forward toward the aimed direction
                    fc.forward(POWER_VAL)
                    time.sleep(FORWARD_DURATION_S)
                    fc.stop()

                frame_count += 1

        finally:
            fc.stop()
            camera.stop()
            cv2.destroyAllWindows()
            print("Movement finished. Robot stopped.")


if __name__ == '__main__':
    main()
