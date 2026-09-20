import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import av


# ============================================================
# WORK SLOUCH
# Real-time AI posture awareness
# ============================================================

APP_NAME = "Work Slouch"

CALIBRATION_TIME = 8.0
ALERT_DELAY = 3.0
NOTIFICATION_COOLDOWN = 15.0

SMOOTHING_FRAMES = 8

HEAD_DROP_THRESHOLD = 0.045
FORWARD_HEAD_THRESHOLD = 0.060
SHOULDER_DROP_THRESHOLD = 0.030
SHOULDER_TILT_THRESHOLD = 0.045

STRONG_HEAD_DROP = 0.080
STRONG_FORWARD_HEAD = 0.100
STRONG_SHOULDER_TILT = 0.080

ABS_HEAD_DROP = 0.090
ABS_FORWARD_HEAD = 0.120
ABS_SHOULDER_TILT = 0.090

HEAD_TURN_THRESHOLD = 0.20
HEAD_TURN_HOLD = 0.75


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
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(30, 100, 150, 0.15), transparent 30%),
            radial-gradient(circle at 90% 10%, rgba(50, 180, 130, 0.10), transparent 25%),
            #081018;
    }

    [data-testid="stSidebar"] {
        background: #0b141d;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .hero {
        padding: 20px 0 10px 0;
    }

    .hero-title {
        font-size: 46px;
        font-weight: 800;
        letter-spacing: -2px;
        margin-bottom: 4px;
    }

    .hero-subtitle {
        color: #9caab5;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .glass {
        background: rgba(17, 28, 38, 0.80);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 16px;
    }

    .metric-card {
        background: rgba(17, 28, 38, 0.90);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 18px;
        min-height: 110px;
    }

    .metric-label {
        color: #8fa0ad;
        font-size: 13px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #f4f7f9;
        font-size: 27px;
        font-weight: 750;
    }

    .status-good {
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.35);
        color: #4ade80;
        padding: 18px;
        border-radius: 16px;
        text-align: center;
        font-size: 25px;
        font-weight: 750;
    }

    .status-warning {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #fbbf24;
        padding: 18px;
        border-radius: 16px;
        text-align: center;
        font-size: 25px;
        font-weight: 750;
    }

    .status-danger {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.35);
        color: #f87171;
        padding: 18px;
        border-radius: 16px;
        text-align: center;
        font-size: 25px;
        font-weight: 750;
    }

    .info-box {
        background: rgba(15, 23, 32, 0.9);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 15px 18px;
        color: #aebbc4;
        line-height: 1.6;
    }

    .privacy {
        font-size: 12px;
        color: #71808b;
        padding-top: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Live Monitor"

if "calibrated" not in st.session_state:
    st.session_state.calibrated = False

if "baseline" not in st.session_state:
    st.session_state.baseline = None

if "calibration_start" not in st.session_state:
    st.session_state.calibration_start = None

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="font-size:28px;font-weight:800;">
        🧍 Work Slouch
        </div>
        <div style="color:#84929d;margin-bottom:25px;">
        Stay aware. Sit better. Work smarter.
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Live Monitor",
            "Dashboard",
            "How It Works",
            "Benefits",
            "About",
        ],
    )

    st.session_state.page = page

    st.markdown("---")

    st.markdown(
        """
        <div class="info-box">
        🔒 <b>Privacy First</b><br>
        Your webcam is processed through this local Streamlit application.
        No posture video is intentionally saved by Work Slouch.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def distance(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def calculate_measurements(results):

    if not results.pose_landmarks:
        return None

    lm = results.pose_landmarks.landmark

    nose = lm[mp_pose.PoseLandmark.NOSE]
    left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_ear = lm[mp_pose.PoseLandmark.LEFT_EAR]
    right_ear = lm[mp_pose.PoseLandmark.RIGHT_EAR]

    shoulder_width = distance(
        (left_shoulder.x, left_shoulder.y),
        (right_shoulder.x, right_shoulder.y),
    )

    if shoulder_width < 0.03:
        return None

    shoulder_y = (
        left_shoulder.y + right_shoulder.y
    ) / 2.0

    ear_x = (
        left_ear.x + right_ear.x
    ) / 2.0

    ear_y = (
        left_ear.y + right_ear.y
    ) / 2.0

    head_height = shoulder_y - ear_y

    shoulder_tilt = abs(
        left_shoulder.y - right_shoulder.y
    ) / shoulder_width

    nose_turn = abs(nose.x - ear_x) / shoulder_width

    world_z = None

    if results.pose_world_landmarks:

        world = results.pose_world_landmarks.landmark

        nose_w = world[mp_pose.PoseLandmark.NOSE]
        left_shoulder_w = world[
            mp_pose.PoseLandmark.LEFT_SHOULDER
        ]
        right_shoulder_w = world[
            mp_pose.PoseLandmark.RIGHT_SHOULDER
        ]

        shoulder_z = (
            left_shoulder_w.z +
            right_shoulder_w.z
        ) / 2.0

        world_z = nose_w.z - shoulder_z

    return {
        "head_height": head_height,
        "shoulder_y": shoulder_y,
        "shoulder_tilt": shoulder_tilt,
        "forward_head": world_z,
        "head_turn": nose_turn,
    }


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class PostureProcessor(VideoProcessorBase):

    def __init__(self):

        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )

        self.measurements = deque(
            maxlen=SMOOTHING_FRAMES
        )

        self.calibration_samples = []

        self.calibration_start = time.time()

        self.baseline = None

        self.status = "CALIBRATING"

        self.score = 100

        self.head_drop = 0
        self.forward_head = 0
        self.shoulder_tilt = 0

        self.bad_since = None
        self.last_alert = 0

        self.head_turn_since = None

        self.person_detected = False

    def get_median(self, key):

        values = [
            x[key]
            for x in self.measurements
            if x.get(key) is not None
        ]

        if not values:
            return None

        return float(np.median(values))

    def build_baseline(self):

        if len(self.calibration_samples) < 15:
            return

        keys = [
            "head_height",
            "shoulder_y",
            "shoulder_tilt",
            "forward_head",
        ]

        self.baseline = {}

        for key in keys:

            values = [
                x[key]
                for x in self.calibration_samples
                if x.get(key) is not None
            ]

            if values:
                self.baseline[key] = float(
                    np.median(values)
                )

        if "head_height" in self.baseline:
            self.calibrated = True

    def analyze(self, m):

        if self.baseline is None:
            return

        baseline = self.baseline

        self.head_drop = (
            baseline["head_height"]
            - m["head_height"]
        )

        self.shoulder_tilt = (
            m["shoulder_tilt"]
            - baseline["shoulder_tilt"]
        )

        if (
            m["forward_head"] is not None
            and baseline.get("forward_head") is not None
        ):
            self.forward_head = abs(
                m["forward_head"]
                - baseline["forward_head"]
            )
        else:
            self.forward_head = 0

        head_turning = (
            m["head_turn"] > HEAD_TURN_THRESHOLD
        )

        if head_turning:

            if self.head_turn_since is None:
                self.head_turn_since = time.time()

            turning_long_enough = (
                time.time() - self.head_turn_since
                > HEAD_TURN_HOLD
            )

        else:

            self.head_turn_since = None
            turning_long_enough = False

        # Ignore posture warnings while clearly turning
        # the head.
        if turning_long_enough:

            self.status = "HEAD TURN"
            self.score = 100
            self.bad_since = None
            return

        strong_signals = 0
        moderate_signals = 0
        absolute_signals = 0

        if self.head_drop >= STRONG_HEAD_DROP:
            strong_signals += 1
        elif self.head_drop >= HEAD_DROP_THRESHOLD:
            moderate_signals += 1

        if self.forward_head >= STRONG_FORWARD_HEAD:
            strong_signals += 1
        elif self.forward_head >= FORWARD_HEAD_THRESHOLD:
            moderate_signals += 1

        if self.shoulder_tilt >= STRONG_SHOULDER_TILT:
            strong_signals += 1
        elif self.shoulder_tilt >= SHOULDER_TILT_THRESHOLD:
            moderate_signals += 1

        if self.head_drop >= ABS_HEAD_DROP:
            absolute_signals += 1

        if self.forward_head >= ABS_FORWARD_HEAD:
            absolute_signals += 1

        if self.shoulder_tilt >= ABS_SHOULDER_TILT:
            absolute_signals += 1

        poor_posture = (
            strong_signals >= 1
            or moderate_signals >= 2
            or absolute_signals >= 1
        )

        severity = (
            strong_signals * 35
            + moderate_signals * 20
            + absolute_signals * 20
        )

        self.score = int(
            max(0, min(100, 100 - severity))
        )

        if poor_posture:

            if self.bad_since is None:
                self.bad_since = time.time()

            elapsed = (
                time.time() - self.bad_since
            )

            if elapsed >= ALERT_DELAY:

                self.status = "POSTURE ALERT"

                # macOS notification
                if (
                    time.time() - self.last_alert
                    > NOTIFICATION_COOLDOWN
                ):
                    try:
                        import subprocess

                        subprocess.Popen(
                            [
                                "osascript",
                                "-e",
                                'display notification '
                                '"Please sit upright and relax your shoulders." '
                                'with title "Work Slouch"'
                            ]
                        )

                    except Exception:
                        pass

                    self.last_alert = time.time()

            else:

                self.status = "POSTURE CHECK"

        else:

            self.status = "GOOD POSTURE"
            self.bad_since = None

    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        results = self.pose.process(
            image_rgb
        )

        self.person_detected = (
            results.pose_landmarks is not None
        )

        if results.pose_landmarks:

            measurements = calculate_measurements(
                results
            )

            if measurements:

                self.measurements.append(
                    measurements
                )

                # -----------------------------------------
                # CALIBRATION
                # -----------------------------------------

                if self.baseline is None:

                    elapsed = (
                        time.time()
                        - self.calibration_start
                    )

                    if elapsed <= CALIBRATION_TIME:

                        self.calibration_samples.append(
                            measurements
                        )

                        remaining = max(
                            0,
                            CALIBRATION_TIME - elapsed,
                        )

                        self.status = (
                            f"CALIBRATING {remaining:.1f}s"
                        )

                    else:

                        self.build_baseline()

                        if self.baseline is not None:
                            self.status = (
                                "GOOD POSTURE"
                            )

                # -----------------------------------------
                # POSTURE ANALYSIS
                # -----------------------------------------

                if self.baseline is not None:

                    smoothed = {}

                    for key in measurements:

                        value = self.get_median(
                            key
                        )

                        smoothed[key] = (
                            value
                            if value is not None
                            else measurements[key]
                        )

                    self.analyze(
                        smoothed
                    )

            # ---------------------------------------------
            # DRAW POSE
            # ---------------------------------------------

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    thickness=2,
                    circle_radius=2,
                ),
                mp_drawing.DrawingSpec(
                    thickness=2,
                    circle_radius=2,
                ),
            )

        else:

            self.status = "NO PERSON DETECTED"

        # =================================================
        # CAMERA OVERLAY
        # =================================================

        cv2.rectangle(
            image,
            (15, 15),
            (420, 95),
            (10, 18, 25),
            -1,
        )

        cv2.putText(
            image,
            "WORK SLOUCH",
            (30, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (240, 245, 248),
            2,
        )

        cv2.putText(
            image,
            self.status,
            (30, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (
                70,
                220,
                130
                if self.status == "GOOD POSTURE"
                else 80,
            ),
            2,
        )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )


# ============================================================
# PAGE: LIVE MONITOR
# ============================================================

if page == "Live Monitor":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                Work Slouch
            </div>
            <div class="hero-subtitle">
                Stay aware. Sit better. Work smarter.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="glass">
        <b>AI Posture Monitor</b><br>
        Your webcam is analyzed in real time using
        computer vision. The system first learns your
        normal sitting position and then detects
        significant posture changes.
        </div>
        """,
        unsafe_allow_html=True,
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

    ctx = webrtc_streamer(
        key="work-slouch-camera",
        video_processor_factory=PostureProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    st.markdown("### Live Status")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                Monitoring
                </div>
                <div class="metric-value">
                Real-Time
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                Calibration
                </div>
                <div class="metric-value">
                8 seconds
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                Alert Delay
                </div>
                <div class="metric-value">
                3 seconds
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                Processing
                </div>
                <div class="metric-value">
                Local
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if ctx.state.playing and ctx.video_processor:

        processor = ctx.video_processor

        st.markdown("### Current Posture")

        status = processor.status
        score = processor.score

        if status == "GOOD POSTURE":

            st.markdown(
                """
                <div class="status-good">
                ✓ GOOD POSTURE
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif status == "POSTURE ALERT":

            st.markdown(
                """
                <div class="status-danger">
                ⚠ POSTURE ALERT
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif status == "HEAD TURN":

            st.markdown(
                """
                <div class="status-warning">
                ↔ HEAD TURN DETECTED — IGNORED
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                f"""
                <div class="status-warning">
                {status}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Posture Score",
                f"{score}/100",
            )

        with c2:
            st.metric(
                "Head Drop",
                f"{processor.head_drop:.3f}",
            )

        with c3:
            st.metric(
                "Forward Head",
                f"{processor.forward_head:.3f}",
            )

        with c4:
            st.metric(
                "Shoulder Tilt",
                f"{processor.shoulder_tilt:.3f}",
            )

        st.markdown(
            """
            <div class="privacy">
            🔒 Webcam frames are processed by the local
            application. Work Slouch does not intentionally
            record or save your camera video.
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.info(
            "Click START above the camera to begin "
            "real-time posture monitoring."
        )


# ============================================================
# PAGE: DASHBOARD
# ============================================================

elif page == "Dashboard":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                Dashboard
            </div>
            <div class="hero-subtitle">
                Your posture awareness overview.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Monitoring Mode",
            "Real-Time",
        )

    with c2:
        st.metric(
            "Calibration",
            "Personalized",
        )

    with c3:
        st.metric(
            "Video Storage",
            "None",
        )

    st.markdown("### What Work Slouch Tracks")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="glass">
            <h3>🧠 Head Position</h3>
            Detects significant changes in head
            position compared with your calibrated
            sitting position.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="glass">
            <h3>↔️ Shoulder Alignment</h3>
            Observes changes in shoulder alignment
            to identify sustained asymmetry.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="glass">
            <h3>📐 Forward Head</h3>
            Uses 3D pose information to detect
            meaningful forward movement of the head.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="glass">
            <h3>⏱️ Smart Alerts</h3>
            A short delay helps avoid alerts caused
            by quick natural movements.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# PAGE: HOW IT WORKS
# ============================================================

elif page == "How It Works":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                How It Works
            </div>
            <div class="hero-subtitle">
                From webcam input to posture awareness.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    steps = [
        (
            "01",
            "Webcam",
            "The browser provides a live camera stream."
        ),
        (
            "02",
            "Pose Detection",
            "MediaPipe identifies important body landmarks."
        ),
        (
            "03",
            "Personal Calibration",
            "Your normal sitting position is learned."
        ),
        (
            "04",
            "Posture Analysis",
            "Multiple signals are analyzed together."
        ),
        (
            "05",
            "Smart Alert",
            "A reminder appears only after sustained poor posture."
        ),
    ]

    for number, title, description in steps:

        st.markdown(
            f"""
            <div class="glass">
                <div style="
                    font-size:14px;
                    color:#62d7ff;
                    font-weight:700;
                ">
                    {number}
                </div>
                <h2>{title}</h2>
                <div style="color:#9daab4;">
                    {description}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# PAGE: BENEFITS
# ============================================================

elif page == "Benefits":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                Built For Everyday Work
            </div>
            <div class="hero-subtitle">
                Simple posture awareness using hardware
                you already have.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    benefits = [
        (
            "🎓",
            "Students",
            "Useful during long study and coding sessions."
        ),
        (
            "💻",
            "Developers",
            "Designed for people who spend long periods at a computer."
        ),
        (
            "🏢",
            "Office Workers",
            "Provides gentle awareness during desk work."
        ),
        (
            "🏠",
            "Remote Workers",
            "Works with a normal computer webcam."
        ),
        (
            "🔒",
            "Privacy First",
            "Designed around local webcam processing."
        ),
        (
            "💰",
            "Affordable",
            "No special posture-tracking hardware is required."
        ),
    ]

    cols = st.columns(2)

    for i, (icon, title, text) in enumerate(benefits):

        with cols[i % 2]:

            st.markdown(
                f"""
                <div class="glass">
                    <div style="font-size:32px;">
                        {icon}
                    </div>
                    <h3>{title}</h3>
                    <div style="color:#9daab4;">
                        {text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# PAGE: ABOUT
# ============================================================

elif page == "About":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                About Work Slouch
            </div>
            <div class="hero-subtitle">
                Stay aware. Sit better. Work smarter.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="glass">

        <h2>What is Work Slouch?</h2>

        <p>
        Work Slouch is a privacy-first AI posture awareness
        application designed for people who spend significant
        time working at a computer.
        </p>

        <p>
        It uses computer vision to analyze body landmarks
        from a webcam and compare the current position with
        a personalized baseline.
        </p>

        <p>
        The system combines head position, forward-head
        movement and shoulder alignment instead of relying
        on a single measurement.
        </p>

        <h3>Technology</h3>

        <p>
        Python · OpenCV · MediaPipe · Streamlit · WebRTC
        </p>

        <h3>Privacy</h3>

        <p>
        The application is designed to run locally.
        Work Slouch does not intentionally store webcam
        recordings.
        </p>

        <p style="color:#71808b;">
        Work Slouch is an awareness and reminder tool,
        not a medical diagnostic system.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )