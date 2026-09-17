import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import threading


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Work Slouch",
    page_icon="🧍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #0b0f14;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.7;
        margin-bottom: 25px;
    }

    .status-good {
        padding: 18px;
        border-radius: 14px;
        background: rgba(46, 204, 113, 0.12);
        border: 1px solid rgba(46, 204, 113, 0.35);
        text-align: center;
        font-size: 24px;
        font-weight: 700;
    }

    .status-warning {
        padding: 18px;
        border-radius: 14px;
        background: rgba(241, 196, 15, 0.12);
        border: 1px solid rgba(241, 196, 15, 0.35);
        text-align: center;
        font-size: 24px;
        font-weight: 700;
    }

    .status-alert {
        padding: 18px;
        border-radius: 14px;
        background: rgba(231, 76, 60, 0.12);
        border: 1px solid rgba(231, 76, 60, 0.35);
        text-align: center;
        font-size: 24px;
        font-weight: 700;
    }

    .info-card {
        padding: 20px;
        border-radius: 14px;
        background: #121821;
        border: 1px solid #202936;
        margin-bottom: 15px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 14px;
        background: #121821;
        border: 1px solid #202936;
        text-align: center;
    }

    .small-text {
        opacity: 0.65;
        font-size: 14px;
    }

    section[data-testid="stSidebar"] {
        background: #0e141c;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧍 Work Slouch")
st.sidebar.caption("Stay aware. Sit better. Work smarter.")

page = st.sidebar.radio(
    "Navigation",
    [
        "Live Monitor",
        "Dashboard",
        "How It Works",
        "Benefits",
        "About",
    ],
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Privacy**

    Work Slouch does not intentionally record or save webcam video.

    The webcam stream is processed by the running application for posture awareness.
    """
)


# ============================================================
# MEDIA PIPE
# ============================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_value(value, default=0.0):
    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


def calculate_angle(a, b, c):
    """
    Calculate angle ABC.
    """

    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)

    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


# ============================================================
# POSTURE PROCESSOR
# ============================================================

class PostureProcessor(VideoProcessorBase):

    def __init__(self):

        # ----------------------------------------------------
        # MediaPipe
        # ----------------------------------------------------

        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )

        # ----------------------------------------------------
        # Calibration
        # ----------------------------------------------------

        self.calibration_duration = 8.0
        self.calibration_start = time.time()

        self.calibration_head_y = []
        self.calibration_shoulder_y = []
        self.calibration_shoulder_tilt = []
        self.calibration_forward = []

        self.calibrated = False

        self.base_head_y = None
        self.base_shoulder_y = None
        self.base_shoulder_tilt = None
        self.base_forward = None

        # ----------------------------------------------------
        # Smoothing
        # ----------------------------------------------------

        self.head_y_history = deque(maxlen=12)
        self.shoulder_y_history = deque(maxlen=12)
        self.tilt_history = deque(maxlen=12)
        self.forward_history = deque(maxlen=12)

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        self.status = "CALIBRATING"
        self.person_detected = False

        self.posture_score = 100
        self.head_drop = 0.0
        self.shoulder_drop = 0.0
        self.shoulder_tilt = 0.0
        self.forward_distance = 0.0
        self.head_turn = 0.0

        # ----------------------------------------------------
        # Alert
        # ----------------------------------------------------

        self.bad_posture_start = None
        self.last_alert = 0
        self.alert_delay = 3.0
        self.alert_cooldown = 15.0

        # ----------------------------------------------------
        # Thread safety
        # ----------------------------------------------------

        self.lock = threading.Lock()

    # ========================================================
    # RESET CALIBRATION
    # ========================================================

    def reset_calibration(self):

        with self.lock:

            self.calibration_start = time.time()

            self.calibration_head_y.clear()
            self.calibration_shoulder_y.clear()
            self.calibration_shoulder_tilt.clear()
            self.calibration_forward.clear()

            self.head_y_history.clear()
            self.shoulder_y_history.clear()
            self.tilt_history.clear()
            self.forward_history.clear()

            self.calibrated = False

            self.base_head_y = None
            self.base_shoulder_y = None
            self.base_shoulder_tilt = None
            self.base_forward = None

            self.status = "CALIBRATING"

    # ========================================================
    # MAIN FRAME PROCESSOR
    # ========================================================

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        if image is None or image.size == 0:
            return frame

        # ----------------------------------------------------
        # Flip horizontally so it behaves like a mirror
        # ----------------------------------------------------

        image = cv2.flip(image, 1)

        # ----------------------------------------------------
        # Convert to RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.pose.process(rgb)

        h, w = image.shape[:2]

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        cv2.rectangle(
            image,
            (0, 0),
            (w, 70),
            (10, 15, 22),
            -1,
        )

        cv2.putText(
            image,
            "WORK SLOUCH",
            (25, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            "Privacy-first posture awareness",
            (25, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (170, 180, 190),
            1,
            cv2.LINE_AA,
        )

        # ====================================================
        # NO PERSON
        # ====================================================

        if not results.pose_landmarks:

            self.person_detected = False
            self.status = "NO PERSON DETECTED"

            cv2.putText(
                image,
                "NO PERSON DETECTED",
                (25, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                "Move into the camera frame",
                (25, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            return frame.from_ndarray(image, format="bgr24")

        self.person_detected = True

        landmarks = results.pose_landmarks.landmark

        # ====================================================
        # LANDMARKS
        # ====================================================

        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

        left_shoulder = landmarks[
            mp_pose.PoseLandmark.LEFT_SHOULDER.value
        ]

        right_shoulder = landmarks[
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        ]

        left_ear = landmarks[
            mp_pose.PoseLandmark.LEFT_EAR.value
        ]

        right_ear = landmarks[
            mp_pose.PoseLandmark.RIGHT_EAR.value
        ]

        # ----------------------------------------------------
        # Visibility check
        # ----------------------------------------------------

        required = [
            nose,
            left_shoulder,
            right_shoulder,
            left_ear,
            right_ear,
        ]

        if min(safe_value(x.visibility) for x in required) < 0.35:

            self.status = "MOVE INTO VIEW"

            cv2.putText(
                image,
                "MOVE INTO VIEW",
                (25, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
            )

            return frame.from_ndarray(image, format="bgr24")

        # ====================================================
        # BASIC METRICS
        # ====================================================

        shoulder_y = (
            safe_value(left_shoulder.y)
            + safe_value(right_shoulder.y)
        ) / 2.0

        head_y = safe_value(nose.y)

        shoulder_tilt = abs(
            safe_value(left_shoulder.y)
            - safe_value(right_shoulder.y)
        )

        # ----------------------------------------------------
        # Forward head estimation
        #
        # MediaPipe world z is used here.
        # Smaller/greater direction can vary, so we use
        # relative change after calibration.
        # ----------------------------------------------------

        nose_world_z = safe_value(nose.z)

        shoulder_world_z = (
            safe_value(left_shoulder.z)
            + safe_value(right_shoulder.z)
        ) / 2.0

        forward_distance = nose_world_z - shoulder_world_z

        # ====================================================
        # HEAD TURN ESTIMATION
        # ====================================================

        left_ear_x = safe_value(left_ear.x)
        right_ear_x = safe_value(right_ear.x)
        nose_x = safe_value(nose.x)

        ear_center = (left_ear_x + right_ear_x) / 2.0
        ear_width = abs(right_ear_x - left_ear_x)

        if ear_width > 0.02:

            head_turn = abs(nose_x - ear_center) / ear_width

        else:

            head_turn = 0.0

        self.head_turn = head_turn

        # ====================================================
        # SMOOTHING
        # ====================================================

        self.head_y_history.append(head_y)
        self.shoulder_y_history.append(shoulder_y)
        self.tilt_history.append(shoulder_tilt)
        self.forward_history.append(forward_distance)

        smooth_head_y = float(np.median(self.head_y_history))
        smooth_shoulder_y = float(
            np.median(self.shoulder_y_history)
        )
        smooth_tilt = float(np.median(self.tilt_history))
        smooth_forward = float(
            np.median(self.forward_history)
        )

        # ====================================================
        # CALIBRATION
        # ====================================================

        elapsed = time.time() - self.calibration_start

        if not self.calibrated:

            self.calibration_head_y.append(smooth_head_y)
            self.calibration_shoulder_y.append(
                smooth_shoulder_y
            )
            self.calibration_shoulder_tilt.append(
                smooth_tilt
            )
            self.calibration_forward.append(
                smooth_forward
            )

            remaining = max(
                0,
                self.calibration_duration - elapsed,
            )

            self.status = f"CALIBRATING {remaining:.1f}s"

            cv2.putText(
                image,
                self.status,
                (25, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                "Sit naturally and look at the screen",
                (25, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            if elapsed >= self.calibration_duration:

                if len(self.calibration_head_y) >= 10:

                    self.base_head_y = float(
                        np.median(
                            self.calibration_head_y
                        )
                    )

                    self.base_shoulder_y = float(
                        np.median(
                            self.calibration_shoulder_y
                        )
                    )

                    self.base_shoulder_tilt = float(
                        np.median(
                            self.calibration_shoulder_tilt
                        )
                    )

                    self.base_forward = float(
                        np.median(
                            self.calibration_forward
                        )
                    )

                    self.calibrated = True

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
            )

            return frame.from_ndarray(
                image,
                format="bgr24",
            )

        # ====================================================
        # RELATIVE POSTURE METRICS
        # ====================================================

        self.head_drop = (
            smooth_head_y - self.base_head_y
        )

        self.shoulder_drop = (
            smooth_shoulder_y
            - self.base_shoulder_y
        )

        self.shoulder_tilt = max(
            0.0,
            smooth_tilt - self.base_shoulder_tilt
        )

        self.forward_distance = abs(
            smooth_forward - self.base_forward
        )

        # ====================================================
        # HEAD TURN SUPPRESSION
        # ====================================================

        head_is_turned = head_turn > 0.75

        # ====================================================
        # POSTURE CONDITIONS
        # ====================================================

        head_drop_bad = (
            self.head_drop > 0.045
        )

        strong_head_drop = (
            self.head_drop > 0.080
        )

        forward_bad = (
            self.forward_distance > 0.060
        )

        strong_forward = (
            self.forward_distance > 0.100
        )

        shoulder_bad = (
            self.shoulder_drop > 0.030
        )

        strong_shoulder = (
            self.shoulder_drop > 0.065
        )

        tilt_bad = (
            self.shoulder_tilt > 0.045
        )

        strong_tilt = (
            self.shoulder_tilt > 0.080
        )

        bad_signals = sum(
            [
                head_drop_bad,
                forward_bad,
                shoulder_bad or tilt_bad,
            ]
        )

        strong_signals = sum(
            [
                strong_head_drop,
                strong_forward,
                strong_shoulder or strong_tilt,
            ]
        )

        # ====================================================
        # POSTURE CLASSIFICATION
        # ====================================================

        if head_is_turned:

            self.status = "HEAD TURN"

            self.bad_posture_start = None

        elif strong_signals >= 2:

            self.status = "POSTURE ALERT"

        elif bad_signals >= 2:

            self.status = "POSTURE CHECK"

        else:

            self.status = "GOOD POSTURE"

            self.bad_posture_start = None

        # ====================================================
        # ALERT TIMER
        # ====================================================

        if (
            self.status in ["POSTURE CHECK", "POSTURE ALERT"]
            and not head_is_turned
        ):

            if self.bad_posture_start is None:

                self.bad_posture_start = time.time()

            bad_duration = (
                time.time()
                - self.bad_posture_start
            )

            if (
                bad_duration >= self.alert_delay
                and time.time() - self.last_alert
                >= self.alert_cooldown
            ):

                self.last_alert = time.time()

        else:

            self.bad_posture_start = None

        # ====================================================
        # POSTURE SCORE
        # ====================================================

        penalty = 0

        if self.head_drop > 0:
            penalty += min(
                35,
                self.head_drop * 400
            )

        if self.forward_distance > 0:
            penalty += min(
                35,
                self.forward_distance * 250
            )

        if self.shoulder_drop > 0:
            penalty += min(
                20,
                self.shoulder_drop * 250
            )

        if self.shoulder_tilt > 0:
            penalty += min(
                20,
                self.shoulder_tilt * 200
            )

        self.posture_score = int(
            np.clip(
                100 - penalty,
                0,
                100,
            )
        )

        # ====================================================
        # DRAW SKELETON
        # ====================================================

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
        )

        # ====================================================
        # STATUS PANEL
        # ====================================================

        if self.status == "GOOD POSTURE":

            status_color = (70, 200, 100)

        elif self.status == "HEAD TURN":

            status_color = (255, 190, 60)

        elif self.status == "POSTURE CHECK":

            status_color = (0, 190, 255)

        else:

            status_color = (60, 70, 230)

        panel_y1 = 90
        panel_y2 = 190

        overlay = image.copy()

        cv2.rectangle(
            overlay,
            (20, panel_y1),
            (350, panel_y2),
            (15, 20, 28),
            -1,
        )

        image = cv2.addWeighted(
            overlay,
            0.90,
            image,
            0.10,
            0,
        )

        cv2.putText(
            image,
            self.status,
            (40, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            status_color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            f"Posture score: {self.posture_score}/100",
            (40, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        # ====================================================
        # METRIC PANEL
        # ====================================================

        metric_y = h - 105

        cv2.rectangle(
            image,
            (20, metric_y),
            (w - 20, h - 15),
            (12, 17, 24),
            -1,
        )

        cv2.putText(
            image,
            f"Head drop: {self.head_drop:+.3f}",
            (40, metric_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            f"Forward: {self.forward_distance:.3f}",
            (240, metric_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            f"Shoulder tilt: {self.shoulder_tilt:.3f}",
            (440, metric_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            f"Head turn: {self.head_turn:.2f}",
            (690, metric_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        # ====================================================
        # CALIBRATION INDICATOR
        # ====================================================

        cv2.putText(
            image,
            "CALIBRATED",
            (w - 145, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (100, 220, 130),
            1,
            cv2.LINE_AA,
        )

        # ====================================================
        # RETURN PROCESSED FRAME
        # ====================================================

        return frame.from_ndarray(
            image,
            format="bgr24",
        )


# ============================================================
# LIVE MONITOR
# ============================================================

if page == "Live Monitor":

    st.markdown(
        '<div class="main-title">🧍 Work Slouch</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">Stay aware. Sit better. Work smarter.</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Allow camera access when your browser asks. "
        "Sit naturally for the first 8 seconds while Work Slouch calibrates."
    )

    rtc_configuration = RTCConfiguration(
        {
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        }
    )

    webrtc_ctx = webrtc_streamer(
        key="work-slouch-camera",
        video_processor_factory=PostureProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 20},
            },
            "audio": False,
        },
        async_processing=True,
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="info-card">
            <h3>🎯 Personalized</h3>
            <p>
            The first few seconds establish your natural sitting position
            as a personal baseline.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="info-card">
            <h3>👀 Head-turn aware</h3>
            <p>
            Temporary head turns are separated from sustained posture
            changes to reduce unnecessary warnings.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            """
            <div class="info-card">
            <h3>🔒 Privacy focused</h3>
            <p>
            The application does not intentionally record or save webcam
            video.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# DASHBOARD
# ============================================================

elif page == "Dashboard":

    st.markdown(
        '<div class="main-title">📊 Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">Understand what Work Slouch monitors.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            """
            <div class="metric-card">
            <h2>8s</h2>
            <p>Calibration</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="metric-card">
            <h2>3s</h2>
            <p>Alert delay</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            """
            <div class="metric-card">
            <h2>4</h2>
            <p>Posture signals</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:

        st.markdown(
            """
            <div class="metric-card">
            <h2>0</h2>
            <p>Recorded videos</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.subheader("Signals monitored")

    signals = [
        (
            "Head position",
            "Detects sustained changes in head height relative to your calibrated baseline.",
        ),
        (
            "Forward head",
            "Uses MediaPipe's 3D landmark information to estimate relative forward movement.",
        ),
        (
            "Shoulder alignment",
            "Checks changes in shoulder height and tilt.",
        ),
        (
            "Head direction",
            "Helps distinguish looking away from a sustained posture change.",
        ),
    ]

    for title, description in signals:

        st.markdown(
            f"""
            <div class="info-card">
            <h3>{title}</h3>
            <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# HOW IT WORKS
# ============================================================

elif page == "How It Works":

    st.markdown(
        '<div class="main-title">⚙️ How It Works</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">A simple computer-vision pipeline.</div>',
        unsafe_allow_html=True,
    )

    steps = [
        (
            "1",
            "Camera",
            "Your browser provides a live webcam stream to the application.",
        ),
        (
            "2",
            "Pose Detection",
            "MediaPipe detects body landmarks such as the nose, ears and shoulders.",
        ),
        (
            "3",
            "Calibration",
            "Work Slouch learns your natural sitting position for several seconds.",
        ),
        (
            "4",
            "Analysis",
            "Recent frames are smoothed to reduce reactions to individual noisy frames.",
        ),
        (
            "5",
            "Decision",
            "Multiple posture signals are considered before a posture warning appears.",
        ),
        (
            "6",
            "Reminder",
            "The interface provides an awareness status when sustained changes are detected.",
        ),
    ]

    for number, title, description in steps:

        st.markdown(
            f"""
            <div class="info-card">
            <h2>{number}. {title}</h2>
            <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# BENEFITS
# ============================================================

elif page == "Benefits":

    st.markdown(
        '<div class="main-title">✨ Benefits</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">Designed for long desk-work sessions.</div>',
        unsafe_allow_html=True,
    )

    benefits = [
        (
            "🧠 Awareness",
            "Provides real-time awareness of sustained changes in sitting position.",
        ),
        (
            "🎯 Personalized",
            "Uses your own initial posture as the reference rather than one fixed body position.",
        ),
        (
            "⏱️ Delayed warnings",
            "Short movements are less likely to immediately trigger a warning.",
        ),
        (
            "🔒 Privacy conscious",
            "The application does not intentionally store webcam recordings.",
        ),
        (
            "💻 Simple",
            "Runs through a browser with no specialized posture hardware.",
        ),
        (
            "🤖 AI-powered",
            "Uses computer vision and pose landmarks rather than manually placed sensors.",
        ),
    ]

    for title, description in benefits:

        st.markdown(
            f"""
            <div class="info-card">
            <h3>{title}</h3>
            <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.markdown(
        '<div class="main-title">ℹ️ About Work Slouch</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">A privacy-conscious computer vision project.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">

        <h2>Work Slouch</h2>

        <p>
        Work Slouch is a computer-vision awareness tool designed to help
        people notice sustained changes in their desk posture.
        </p>

        <h3>Technology</h3>

        <p>
        Python · Streamlit · OpenCV · MediaPipe · WebRTC
        </p>

        <h3>Privacy</h3>

        <p>
        Work Slouch does not intentionally record or save webcam video.
        The live stream is processed by the running application for
        posture analysis.
        </p>

        <h3>Important</h3>

        <p>
        Work Slouch is an awareness and reminder application. It is not
        a medical diagnostic system and should not be used as a substitute
        for professional medical advice.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )
