import cv2
import mediapipe as mp
import numpy as np
import time
import subprocess
from collections import deque


# ============================================================
# WORK SLOUCH
# Privacy-first real-time posture awareness
# ============================================================

APP_NAME = "Work Slouch"

# ---------------- CAMERA ----------------
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ---------------- CALIBRATION ----------------
CALIBRATION_TIME = 8.0

# ---------------- ALERT ----------------
ALERT_DELAY = 3.0
NOTIFICATION_COOLDOWN = 15.0

# ---------------- SMOOTHING ----------------
SMOOTHING_FRAMES = 10

# ============================================================
# POSTURE THRESHOLDS
# ============================================================

# Head dropping downward
HEAD_DROP_THRESHOLD = 0.045
STRONG_HEAD_DROP = 0.080

# Forward head movement
FORWARD_HEAD_THRESHOLD = 0.060
STRONG_FORWARD_HEAD = 0.100

# Shoulder movement
SHOULDER_DROP_THRESHOLD = 0.030
STRONG_SHOULDER_DROP = 0.065

# Shoulder tilt
SHOULDER_TILT_THRESHOLD = 0.045
STRONG_SHOULDER_TILT = 0.080

# Very large changes
ABS_HEAD_DROP = 0.090
ABS_FORWARD_HEAD = 0.120
ABS_SHOULDER_TILT = 0.090

# Head-turn protection
HEAD_TURN_THRESHOLD = 0.20
HEAD_TURN_HOLD = 0.75


# ============================================================
# MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_median(values):
    """Return median safely."""
    if not values:
        return 0.0
    return float(np.median(values))


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def send_notification():
    """Send a macOS notification."""
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                'display notification "Please sit upright and relax your shoulders." '
                'with title "Work Slouch"'
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass


def get_landmark(landmarks, index):
    """Return a MediaPipe landmark."""
    return landmarks[index]


# ============================================================
# POSTURE MEASUREMENT
# ============================================================

def calculate_measurements(results):
    """
    Calculate normalized posture measurements.

    Returns:
        dictionary containing posture measurements
        or None if required landmarks are unavailable.
    """

    if results.pose_landmarks is None:
        return None

    lm = results.pose_landmarks.landmark

    nose = lm[mp_pose.PoseLandmark.NOSE]
    left_ear = lm[mp_pose.PoseLandmark.LEFT_EAR]
    right_ear = lm[mp_pose.PoseLandmark.RIGHT_EAR]

    left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    required = [
        nose,
        left_ear,
        right_ear,
        left_shoulder,
        right_shoulder,
    ]

    # Require reasonably visible landmarks
    if any(point.visibility < 0.45 for point in required):
        return None

    # --------------------------------------------------------
    # Shoulder geometry
    # --------------------------------------------------------

    shoulder_dx = right_shoulder.x - left_shoulder.x
    shoulder_dy = right_shoulder.y - left_shoulder.y

    shoulder_width = np.sqrt(
        shoulder_dx ** 2 + shoulder_dy ** 2
    )

    if shoulder_width < 0.03:
        return None

    shoulder_y = (
        left_shoulder.y + right_shoulder.y
    ) / 2.0

    shoulder_tilt = abs(
        left_shoulder.y - right_shoulder.y
    ) / shoulder_width

    # --------------------------------------------------------
    # Ear/head geometry
    # --------------------------------------------------------

    ear_x = (
        left_ear.x + right_ear.x
    ) / 2.0

    ear_y = (
        left_ear.y + right_ear.y
    ) / 2.0

    # Distance from head to shoulder line.
    #
    # In image coordinates, smaller ear_y means head is higher.
    head_height = (
        shoulder_y - ear_y
    ) / shoulder_width

    # --------------------------------------------------------
    # Head turn
    # --------------------------------------------------------

    head_turn_amount = abs(
        nose.x - ear_x
    ) / shoulder_width

    # --------------------------------------------------------
    # World coordinates
    # --------------------------------------------------------

    forward_head = 0.0

    if results.pose_world_landmarks is not None:

        world = results.pose_world_landmarks.landmark

        world_left_ear = world[
            mp_pose.PoseLandmark.LEFT_EAR
        ]

        world_right_ear = world[
            mp_pose.PoseLandmark.RIGHT_EAR
        ]

        world_left_shoulder = world[
            mp_pose.PoseLandmark.LEFT_SHOULDER
        ]

        world_right_shoulder = world[
            mp_pose.PoseLandmark.RIGHT_SHOULDER
        ]

        world_ear_z = (
            world_left_ear.z +
            world_right_ear.z
        ) / 2.0

        world_shoulder_z = (
            world_left_shoulder.z +
            world_right_shoulder.z
        ) / 2.0

        world_shoulder_width = np.sqrt(
            (
                world_right_shoulder.x -
                world_left_shoulder.x
            ) ** 2
            +
            (
                world_right_shoulder.y -
                world_left_shoulder.y
            ) ** 2
        )

        if world_shoulder_width > 0.01:
            forward_head = (
                world_shoulder_z -
                world_ear_z
            ) / world_shoulder_width

    return {
        "head_height": float(head_height),
        "head_y": float(ear_y),
        "shoulder_y": float(shoulder_y),
        "shoulder_width": float(shoulder_width),
        "shoulder_tilt": float(shoulder_tilt),
        "forward_head": float(forward_head),
        "head_turn": float(head_turn_amount),
    }


# ============================================================
# DRAWING
# ============================================================

def draw_text(
    frame,
    text,
    position,
    scale=0.7,
    thickness=2,
):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_panel(
    frame,
    x1,
    y1,
    x2,
    y2,
):
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (20, 25, 32),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.82,
        frame,
        0.18,
        0,
        frame,
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (90, 90, 90),
        1,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("WORK SLOUCH")
    print("Privacy-first posture awareness")
    print("=" * 60)

    print("\nOpening camera...")

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("\nERROR: Could not open camera.")
        print("Check macOS camera permission for Terminal/Python.")
        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT,
    )

    # --------------------------------------------------------
    # MediaPipe Pose
    # --------------------------------------------------------

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    )

    # --------------------------------------------------------
    # Calibration storage
    # --------------------------------------------------------

    calibration_values = {
        "head_height": [],
        "head_y": [],
        "shoulder_y": [],
        "shoulder_width": [],
        "shoulder_tilt": [],
        "forward_head": [],
    }

    baseline = None

    calibration_start = time.time()
    calibrated = False

    # --------------------------------------------------------
    # Smoothing buffers
    # --------------------------------------------------------

    head_drop_history = deque(
        maxlen=SMOOTHING_FRAMES
    )

    forward_history = deque(
        maxlen=SMOOTHING_FRAMES
    )

    shoulder_drop_history = deque(
        maxlen=SMOOTHING_FRAMES
    )

    tilt_history = deque(
        maxlen=SMOOTHING_FRAMES
    )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    poor_posture_start = None
    alert_active = False

    last_notification_time = 0.0

    head_turn_start = None
    head_turn_active = False

    posture_state = "CALIBRATING"

    last_measurements = None

    # ========================================================
    # LOOP
    # ========================================================

    while True:

        success, frame = cap.read()

        if not success:
            print("Unable to read camera frame.")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # ----------------------------------------------------
        # Process frame
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        results = pose.process(rgb)

        measurements = calculate_measurements(results)

        # ----------------------------------------------------
        # Calibration
        # ----------------------------------------------------

        if not calibrated:

            elapsed = time.time() - calibration_start

            if measurements is not None:

                calibration_values[
                    "head_height"
                ].append(
                    measurements["head_height"]
                )

                calibration_values[
                    "head_y"
                ].append(
                    measurements["head_y"]
                )

                calibration_values[
                    "shoulder_y"
                ].append(
                    measurements["shoulder_y"]
                )

                calibration_values[
                    "shoulder_width"
                ].append(
                    measurements["shoulder_width"]
                )

                calibration_values[
                    "shoulder_tilt"
                ].append(
                    measurements["shoulder_tilt"]
                )

                calibration_values[
                    "forward_head"
                ].append(
                    measurements["forward_head"]
                )

                last_measurements = measurements

            remaining = max(
                0.0,
                CALIBRATION_TIME - elapsed,
            )

            posture_state = "CALIBRATING"

            # ------------------------------------------------
            # Calibration UI
            # ------------------------------------------------

            draw_panel(
                frame,
                25,
                25,
                500,
                190,
            )

            draw_text(
                frame,
                "WORK SLOUCH",
                (45, 65),
                0.95,
                2,
            )

            draw_text(
                frame,
                "CALIBRATING...",
                (45, 105),
                0.75,
                2,
            )

            draw_text(
                frame,
                f"Keep your normal sitting position",
                (45, 140),
                0.55,
                1,
            )

            draw_text(
                frame,
                f"Starting in {remaining:.1f}s",
                (45, 172),
                0.55,
                1,
            )

            if elapsed >= CALIBRATION_TIME:

                if len(
                    calibration_values["head_height"]
                ) >= 15:

                    baseline = {
                        key: safe_median(values)
                        for key, values
                        in calibration_values.items()
                    }

                    calibrated = True

                    print("\nCalibration complete.")
                    print("Baseline measurements:")
                    print(baseline)

                else:

                    print(
                        "\nNot enough visible frames "
                        "for calibration. Restarting..."
                    )

                    calibration_start = time.time()

                    for values in calibration_values.values():
                        values.clear()

            # ------------------------------------------------
            # Draw skeleton during calibration
            # ------------------------------------------------

            if results.pose_landmarks:

                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

            cv2.imshow(
                "Work Slouch",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            continue

        # ====================================================
        # AFTER CALIBRATION
        # ====================================================

        poor_posture = False

        if measurements is not None:

            last_measurements = measurements

            # ------------------------------------------------
            # Calculate changes from user's baseline
            # ------------------------------------------------

            head_drop = (
                baseline["head_height"]
                - measurements["head_height"]
            )

            forward_change = abs(
                measurements["forward_head"]
                - baseline["forward_head"]
            )

            shoulder_drop = (
                measurements["shoulder_y"]
                - baseline["shoulder_y"]
            ) / max(
                baseline["shoulder_width"],
                0.01,
            )

            tilt_change = abs(
                measurements["shoulder_tilt"]
                - baseline["shoulder_tilt"]
            )

            # ------------------------------------------------
            # Smoothing
            # ------------------------------------------------

            head_drop_history.append(
                head_drop
            )

            forward_history.append(
                forward_change
            )

            shoulder_drop_history.append(
                shoulder_drop
            )

            tilt_history.append(
                tilt_change
            )

            smooth_head_drop = safe_median(
                list(head_drop_history)
            )

            smooth_forward = safe_median(
                list(forward_history)
            )

            smooth_shoulder_drop = safe_median(
                list(shoulder_drop_history)
            )

            smooth_tilt = safe_median(
                list(tilt_history)
            )

            # ------------------------------------------------
            # Head-turn detection
            # ------------------------------------------------

            current_head_turn = (
                measurements["head_turn"]
            )

            if current_head_turn > HEAD_TURN_THRESHOLD:

                if head_turn_start is None:
                    head_turn_start = time.time()

                if (
                    time.time() - head_turn_start
                    >= HEAD_TURN_HOLD
                ):
                    head_turn_active = True

            else:

                head_turn_start = None
                head_turn_active = False

            # ------------------------------------------------
            # Posture signals
            # ------------------------------------------------

            head_signal = (
                smooth_head_drop
                > HEAD_DROP_THRESHOLD
                or
                smooth_forward
                > FORWARD_HEAD_THRESHOLD
            )

            shoulder_signal = (
                smooth_shoulder_drop
                > SHOULDER_DROP_THRESHOLD
            )

            tilt_signal = (
                smooth_tilt
                > SHOULDER_TILT_THRESHOLD
            )

            strong_signals = sum(
                [
                    smooth_head_drop
                    > STRONG_HEAD_DROP,

                    smooth_forward
                    > STRONG_FORWARD_HEAD,

                    smooth_shoulder_drop
                    > STRONG_SHOULDER_DROP,

                    smooth_tilt
                    > STRONG_SHOULDER_TILT,
                ]
            )

            moderate_signals = sum(
                [
                    head_signal,
                    shoulder_signal,
                    tilt_signal,
                ]
            )

            absolute_signals = sum(
                [
                    smooth_head_drop
                    > ABS_HEAD_DROP,

                    smooth_forward
                    > ABS_FORWARD_HEAD,

                    smooth_tilt
                    > ABS_SHOULDER_TILT,
                ]
            )

            # ------------------------------------------------
            # Main posture decision
            # ------------------------------------------------

            poor_posture = (
                strong_signals >= 1
                or
                moderate_signals >= 2
                or
                absolute_signals >= 1
            )

            # Turning your head should not trigger an alert.
            if head_turn_active:
                poor_posture = False

        else:

            # No reliable landmarks
            poor_posture = False

            head_turn_active = False
            head_turn_start = None

        # ====================================================
        # ALERT TIMER
        # ====================================================

        if poor_posture:

            if poor_posture_start is None:
                poor_posture_start = time.time()

            poor_duration = (
                time.time()
                - poor_posture_start
            )

            if poor_duration >= ALERT_DELAY:

                alert_active = True

                if (
                    time.time()
                    - last_notification_time
                    >= NOTIFICATION_COOLDOWN
                ):

                    send_notification()

                    last_notification_time = (
                        time.time()
                    )

        else:

            poor_posture_start = None
            alert_active = False

        # ====================================================
        # POSTURE STATE
        # ====================================================

        if alert_active:

            posture_state = "POSTURE REMINDER"

        elif poor_posture:

            posture_state = "POSTURE CHECK"

        else:

            posture_state = "GOOD POSTURE"

        # ====================================================
        # DRAW SKELETON
        # ====================================================

        if results.pose_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=2,
                    circle_radius=3,
                ),
                mp_drawing.DrawingSpec(
                    color=(255, 255, 255),
                    thickness=2,
                    circle_radius=2,
                ),
            )

        # ====================================================
        # MAIN STATUS PANEL
        # ====================================================

        draw_panel(
            frame,
            25,
            25,
            480,
            215,
        )

        draw_text(
            frame,
            "WORK SLOUCH",
            (45, 65),
            0.95,
            2,
        )

        draw_text(
            frame,
            "AI POSTURE AWARENESS",
            (45, 95),
            0.50,
            1,
        )

        # Status
        if posture_state == "GOOD POSTURE":

            status_text = "GOOD POSTURE"
            status_color = (80, 220, 100)

        elif posture_state == "POSTURE CHECK":

            status_text = "POSTURE CHECK"
            status_color = (0, 200, 255)

        else:

            status_text = "POSTURE REMINDER"
            status_color = (0, 80, 255)

        cv2.putText(
            frame,
            status_text,
            (45, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            status_color,
            2,
            cv2.LINE_AA,
        )

        # Timer
        if poor_posture and poor_posture_start:

            duration = (
                time.time()
                - poor_posture_start
            )

            draw_text(
                frame,
                f"Checking: {duration:.1f}s / {ALERT_DELAY:.0f}s",
                (45, 175),
                0.48,
                1,
            )

        else:

            draw_text(
                frame,
                "Monitoring your normal sitting position",
                (45, 175),
                0.48,
                1,
            )

        # ====================================================
        # METRICS PANEL
        # ====================================================

        draw_panel(
            frame,
            25,
            235,
            480,
            445,
        )

        draw_text(
            frame,
            "POSTURE METRICS",
            (45, 270),
            0.65,
            2,
        )

        if last_measurements is not None:

            # Recalculate display values
            head_drop_display = (
                baseline["head_height"]
                - last_measurements["head_height"]
            )

            forward_display = abs(
                last_measurements["forward_head"]
                - baseline["forward_head"]
            )

            shoulder_display = (
                last_measurements["shoulder_y"]
                - baseline["shoulder_y"]
            ) / max(
                baseline["shoulder_width"],
                0.01,
            )

            tilt_display = abs(
                last_measurements["shoulder_tilt"]
                - baseline["shoulder_tilt"]
            )

            draw_text(
                frame,
                f"Head drop       {head_drop_display:+.3f}",
                (45, 315),
                0.52,
                1,
            )

            draw_text(
                frame,
                f"Forward head    {forward_display:.3f}",
                (45, 350),
                0.52,
                1,
            )

            draw_text(
                frame,
                f"Shoulder drop   {shoulder_display:+.3f}",
                (45, 385),
                0.52,
                1,
            )

            draw_text(
                frame,
                f"Shoulder tilt   {tilt_display:.3f}",
                (45, 420),
                0.52,
                1,
            )

        else:

            draw_text(
                frame,
                "Waiting for body landmarks...",
                (45, 320),
                0.52,
                1,
            )

        # ====================================================
        # HEAD TURN PANEL
        # ====================================================

        draw_panel(
            frame,
            25,
            465,
            480,
            555,
        )

        if head_turn_active:

            draw_text(
                frame,
                "HEAD TURN DETECTED",
                (45, 505),
                0.60,
                2,
            )

            draw_text(
                frame,
                "Posture alert temporarily paused",
                (45, 535),
                0.45,
                1,
            )

        else:

            draw_text(
                frame,
                "HEAD TURN: NORMAL",
                (45, 505),
                0.60,
                2,
            )

            draw_text(
                frame,
                "Normal movement is ignored",
                (45, 535),
                0.45,
                1,
            )

        # ====================================================
        # PRIVACY / CONTROLS
        # ====================================================

        draw_text(
            frame,
            "LOCAL PROCESSING • NO VIDEO SAVED",
            (25, 690),
            0.45,
            1,
        )

        draw_text(
            frame,
            "Press Q or ESC to quit",
            (960, 690),
            0.45,
            1,
        )

        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            "Work Slouch",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

    # ========================================================
    # CLEANUP
    # ========================================================

    print("\nStopping Work Slouch...")

    cap.release()
    pose.close()
    cv2.destroyAllWindows()

    print("Camera released.")
    print("Work Slouch closed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()