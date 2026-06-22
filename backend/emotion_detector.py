from fer import FER
import cv2

# Import music data
from music_data import music_recommendations

# Load image
image = cv2.imread("../images/test.jpg")

# Check image
if image is None:
    print("Error: Image not found")
    exit()

# Initialize detector
detector = FER(mtcnn=True)

# Detect emotions
result = detector.detect_emotions(image)

# If face detected
if len(result) > 0:

    # Face box
    x, y, w, h = result[0]["box"]

    # Emotion scores
    emotions = result[0]["emotions"]

    # Highest emotion
    top_emotion = max(emotions, key=emotions.get)

    # Confidence percentage
    confidence = emotions[top_emotion] * 100

    print("\nDetected Emotion:")
    print(top_emotion)

    print("\nConfidence:")
    print(f"{confidence:.2f}%")

    # Get song recommendations
    songs = music_recommendations.get(top_emotion, [])

    print("\nRecommended Songs:")

    for song in songs:
        print("-", song)

    # Draw rectangle
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Emotion label
    text = f"{top_emotion} ({confidence:.1f}%)"

    cv2.putText(
        image,
        text,
        (x, y-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

else:
    print("No face detected")

# Show result window
cv2.imshow("AI Emotion Music System", image)

# Wait for key press
cv2.waitKey(0)

# Close window
cv2.destroyAllWindows()