import cv2
import numpy as np
def expand_bbox(bbox, frame_shape, margin=0.2):
        x, y, w, h = bbox
        dw, dh = int(w * margin), int(h * margin)
        x0 = max(0, x - dw)
        y0 = max(0, y - dh)
        w0 = min(frame_shape[1] - x0, w + 2*dw)
        h0 = min(frame_shape[0] - y0, h + 2*dh)
        return (x0, y0, w0, h0)
    
class MultiObjectTracker:
    def __init__(self,
                 max_objects: int = 2,
                 diff_thresh: int = 30,
                 min_area_ratio: float = 0.1,
                 solidity_thresh: float = 0.5,
                 mog2_history: int = 500,
                 mog2_varThreshold: float = 16,
                 y_drop: int = 30,
                 cr_cb_delta: int = 10):
        self.max_objects = max_objects
        self.diff_thresh = diff_thresh
        self.min_area_ratio = min_area_ratio
        self.solidity_thresh = solidity_thresh
        self.y_drop = y_drop
        self.cr_cb_delta = cr_cb_delta

        # 1) MOG2 background subtractor with shadow detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=mog2_history,
            varThreshold=mog2_varThreshold,
            detectShadows=True
        )
        
        params = cv2.legacy.TrackerCSRT_Params()
        params.use_hog               = True     # enable HOG features
        params.use_color_names       = True     # include color‐name features
        params.use_channel_weights   = True     # weight channels by importance
        params.filter_lr             = 0.02     # lower = more conservative update
        params.pca_learning_rate     = 0.15     # PCA subspace update speed
        params.template_size         = 200      # larger = more spatial context
        # you can also tweak:
        # params.gsl_sigma             = 1.0
        # params.hog_interp_factor     = 0.02
        # params.admm_iterations       = 4

        self.csrt_params = params

        # placeholders for static background
        self.bg = None
        self.ycrcb_bg = None
        self.trackers = []
        
    

    def set_background(self, bg_img: np.ndarray):
        """Store empty‐scene background and prime the MOG2 model."""
        self.bg = bg_img.copy()
        # prime the MOG2 with a few identical frames to stabilize to the empty scene
        for _ in range(5):
            self.bg_subtractor.apply(self.bg)
        # precompute YCrCb channels of the background
        self.ycrcb_bg = cv2.cvtColor(self.bg, cv2.COLOR_BGR2YCrCb).astype(np.int16)

    def detect_no_shadow(self, img: np.ndarray):
        """
        Detect up to max_objects foreground blobs in img vs stored bg,
        using only raw diff + morphological cleanup (no shadow removal).
        Returns list of (x, y, w, h) bounding boxes.
        """
        if self.bg is None:
            raise RuntimeError("Background not set. Call set_background() first.")

        # --- 1) raw abs-diff mask ---
        diff = cv2.absdiff(img, self.bg)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.diff_thresh, 255, cv2.THRESH_BINARY)

        # --- 2) morphological cleanup ---
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)

        # --- 3) contour find & filter ---
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) < self.max_objects:
            return []

        # measure area & solidity as before
        areas      = [cv2.contourArea(c) for c in cnts]
        hulls      = [cv2.convexHull(c) for c in cnts]
        hull_areas = [cv2.contourArea(h) for h in hulls]
        solidities = [a/h if h>0 else 0 for a, h in zip(areas, hull_areas)]

        # pick the largest blobs up to max_objects
        idxs   = sorted(range(len(cnts)), key=lambda i: areas[i], reverse=True)
        largest = areas[idxs[0]]
        bboxes = []

        for i in idxs:
            if len(bboxes) >= self.max_objects:
                break
            # optional: skip too-small or non-solid blobs
            if areas[i] < self.min_area_ratio * largest:
                continue
            if solidities[i] < self.solidity_thresh:
                continue
            x, y, w, h = cv2.boundingRect(cnts[i])
            bboxes.append((x, y, w, h))

        return bboxes

    def detect(self, img: np.ndarray):
        """
        Detect up to max_objects foreground blobs in img vs stored bg,
        using raw diff, MOG2 shadow mask, and YCrCb chroma test.
        Returns list of (x, y, w, h) bounding boxes.
        """
        if self.bg is None:
            raise RuntimeError("Background not set. Call set_background() first.")

        # --- 1) raw abs-diff mask ---
        diff = cv2.absdiff(img, self.bg)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.diff_thresh, 255, cv2.THRESH_BINARY)

        # --- 2) MOG2 shadow removal ---
        mog2_mask = self.bg_subtractor.apply(img)
        # mog2_mask == 127 → shadows, ==255 → fg
        _, fg_mask = cv2.threshold(mog2_mask, 128, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, fg_mask)

        # --- 3) YCrCb‐based shadow kill ---
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.int16)
        Ybg, Crbg, Cbbg = cv2.split(self.ycrcb_bg)
        Y,   Cr,   Cb   = cv2.split(ycrcb)

        shadow2 = ((Y < (Ybg - self.y_drop)) &
                   (np.abs(Cr - Crbg) < self.cr_cb_delta) &
                   (np.abs(Cb - Cbbg) < self.cr_cb_delta))
        mask[shadow2] = 0

        # --- 4) morphological cleanup ---
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)

        # --- 5) contour find & filter ---
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) < self.max_objects:
            return []

        areas      = [cv2.contourArea(c) for c in cnts]
        hulls      = [cv2.convexHull(c) for c in cnts]
        hull_areas = [cv2.contourArea(h) for h in hulls]
        solidities = [a/h if h>0 else 0 for a, h in zip(areas, hull_areas)]

        idxs    = sorted(range(len(cnts)), key=lambda i: areas[i], reverse=True)
        largest = areas[idxs[0]]
        bboxes  = []

        for i in idxs:
            if len(bboxes) >= self.max_objects:
                break
            if areas[i] < self.min_area_ratio * largest:
                continue
            if solidities[i] < self.solidity_thresh:
                continue
            x, y, w, h = cv2.boundingRect(cnts[i])
            bboxes.append((x, y, w, h))

        return bboxes

    def init_trackers(self, img: np.ndarray, bboxes: list):
        """Initialize one CSRT tracker per bbox."""
        self.trackers = []
        def make_csrt():
            if hasattr(cv2, 'TrackerCSRT_create'):
                return cv2.TrackerCSRT_create(self.csrt_params)
            return cv2.legacy.TrackerCSRT_create(self.csrt_params)

        for bbox in bboxes:
            t = make_csrt()
            t.init(img, bbox)
            self.trackers.append(t)

    def update(self, img: np.ndarray):
        """
        Update all trackers on a new frame.
        Returns list of (success: bool, bbox: (x,y,w,h) or None).
        """
        results = []
        for t in self.trackers:
            ok, box = t.update(img)
            if ok:
                box = tuple(map(int, box))
            else:
                box = None
            results.append((ok, box))
        return results


def main():
    cap = cv2.VideoCapture(0)  # or provide a video file path
    if not cap.isOpened():
        print("ERROR: Cannot open video source")
        return

    mot = MultiObjectTracker()
    state = 0  # 0 = capture bg, 1 = detect objects

    print("1) Point camera at empty scene and press SPACE to capture background.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, f"STEP {state}: press SPACE", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Setup", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            cap.release()
            cv2.destroyAllWindows()
            return
        if key == 32:  # SPACE
            if state == 0:
                mot.set_background(frame)
                state = 1
                print("Background captured.\n2) Place objects and press SPACE again.")
            elif state == 1:
                bboxes = mot.detect(frame)
                if len(bboxes) < mot.max_objects:
                    print("Couldn't detect enough objects; try again.")
                    continue
                for i, (x, y, w, h) in enumerate(bboxes, 1):
                    print(f" Object {i}: x={x}, y={y}, w={w}, h={h}")
                padded = [expand_bbox(b, frame.shape, margin=0.2) for b in raw_boxes]
                mot.init_trackers(frame, bboxes)
                print("Trackers initialized. Entering live tracking…")
                break

    cv2.destroyWindow("Setup")

    colors = [(0,255,0), (0,0,255), (255,0,0), (0,255,255)]
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = mot.update(frame)
        for idx, (ok, bbox) in enumerate(results):
            if ok:
                x, y, w, h = bbox
                color = colors[idx % len(colors)]
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                cv2.putText(frame, f"Obj{idx+1}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                cv2.putText(frame, f"Obj{idx+1} LOST", (10, 60 + 30*idx),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[idx % len(colors)], 2)

        cv2.imshow("Multi-Object Tracking", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
