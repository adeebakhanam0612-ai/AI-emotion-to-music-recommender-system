import streamlit as st
from fer import FER
import cv2
import numpy as np
from PIL import Image

# Import music data
from backend.music_data import music_recommendations

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Emotion Music Recommender",
    page_icon="🎵",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

/* MAIN BACKGROUND */

.stApp {
    background: linear-gradient(
        135deg,
        #ff9a9e 0%,
        #fad0c4 25%,
        #a18cd1 50%,
        #fbc2eb 75%,
        #84fab0 100%
    );
    background-attachment: fixed;
}

/* MAIN CONTAINER */

.block-container {
    padding-top: 2rem;
}

/* HEADINGS */

h1 {
    color: #ffffff !important;
    text-shadow: 3px 3px 15px rgba(0,0,0,0.4);
    font-weight: bold;
}

h2, h3 {
    color: #111827 !important;
    font-weight: bold;
}

/* GLASS EFFECT BOX */

.glass {
    background: rgba(255,255,255,0.25);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.3);
    box-shadow: 0px 8px 32px rgba(0,0,0,0.15);
}

/* SONG CARDS */

.song-card {
    background: linear-gradient(
        to right,
        #667eea,
        #764ba2
    );
    padding: 15px;
    border-radius: 15px;
    margin-bottom: 12px;
    color: white;
    font-size: 18px;
    font-weight: bold;
    transition: 0.3s;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.2);
}

.song-card:hover {
    transform: scale(1.03);
}

/* EMOTION BOX */

.emotion-box {
    background: linear-gradient(
        to right,
        #ff512f,
        #dd2476
    );
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    color: white;
    font-size: 30px;
    font-weight: bold;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.3);
}

/* CONFIDENCE BOX */

.conf-box {
    background: linear-gradient(
        to right,
        #11998e,
        #38ef7d
    );
    padding: 15px;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-size: 24px;
    font-weight: bold;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.3);
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #667eea,
        #764ba2
    );
    color: white;
}

/* SIDEBAR TEXT */

section[data-testid="stSidebar"] * {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.markdown("""
<h1 style='text-align:center; font-size:55px;'>
🎵 AI Emotion Music Recommender
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align:center;
font-size:24px;
color:#ffffff;
font-weight:bold;
text-shadow: 0px 0px 10px rgba(255,255,255,0.8);'>
Detect emotions using AI and get mood-based music recommendations
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- SIDEBAR ----------------

st.sidebar.title("🎧 About Project")

st.sidebar.info("""
AI Emotion-to-Music Recommender System

Built using:
- Python
- Streamlit
- TensorFlow
- FER
- OpenCV
""")

st.sidebar.success("✨ Exhibition Ready AI Project")

# ---------------- DETECTOR ----------------

detector = FER(mtcnn=True)

# =========================================================
# IMAGE UPLOAD SECTION
# =========================================================

st.markdown("""
<div class="glass">
<h2>📸 Upload Image</h2>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Your Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    image_np = np.array(image)

    image_cv = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    result = detector.detect_emotions(image_cv)

    if len(result) > 0:

        emotions = result[0]["emotions"]

        top_emotion = max(
            emotions,
            key=emotions.get
        )

        confidence = emotions[top_emotion] * 100

        emoji_dict = {
            "happy": "😊",
            "sad": "😢",
            "angry": "😠",
            "neutral": "😐",
            "fear": "😨",
            "surprise": "😲",
            "disgust": "🤢"
        }

        emoji = emoji_dict.get(top_emotion, "🙂")

        # Emotion box

        st.markdown(f"""
        <div class="emotion-box">
        {emoji} {top_emotion.upper()}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confidence

        st.markdown(f"""
        <div class="conf-box">
        🎯 Confidence : {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Mood Meter

        st.subheader("📊 Mood Meter")

        for emotion, score in emotions.items():

            percentage = int(score * 100)

            st.write(f"### {emotion.upper()} : {percentage}%")

            st.progress(percentage)

        # Songs

        st.subheader("🎶 Recommended Songs")

        songs = music_recommendations.get(
            top_emotion,
            []
        )

        for song in songs:

            st.markdown(
                f"""
                <div class="song-card">
                🎵 {song['name']}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.video(song["video"])

    else:
        st.error("❌ No face detected")

# =========================================================
# CAMERA SECTION
# =========================================================

st.markdown("---")

st.markdown("""
<div class="glass">
<h2>🎥 Capture Live Photo</h2>
</div>
""", unsafe_allow_html=True)

camera = st.camera_input("Take a picture")

if camera is not None:

    file_bytes = np.asarray(
        bytearray(camera.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    result = detector.detect_emotions(img)

    if len(result) > 0:

        x, y, w, h = result[0]["box"]

        emotions = result[0]["emotions"]

        top_emotion = max(
            emotions,
            key=emotions.get
        )

        confidence = emotions[top_emotion] * 100

        cv2.rectangle(
            img,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            f"{top_emotion} ({confidence:.1f}%)",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        img_rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            img_rgb,
            caption="AI Processed Image",
            use_container_width=True
        )

        # Emotion

        st.markdown(f"""
        <div class="emotion-box">
        🎭 {top_emotion.upper()}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confidence

        st.markdown(f"""
        <div class="conf-box">
        🎯 Confidence : {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Mood meter

        st.subheader("📊 Mood Meter")

        for emotion, score in emotions.items():

            percentage = int(score * 100)

            st.write(f"### {emotion.upper()} : {percentage}%")

            st.progress(percentage)

        # Songs

        st.subheader("🎶 Recommended Songs")

        songs = music_recommendations.get(
            top_emotion,
            []
        )

        for song in songs:

            st.markdown(
                f"""
                <div class="song-card">
                🎵 {song['name']}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.video(song["video"])

    else:
        st.error("❌ No face detected")