import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from deep_translator import GoogleTranslator, exceptions
from spellchecker import SpellChecker
from gtts import gTTS
import tempfile
import time

# Load the trained model
model_dict = pickle.load(open(r'D:\pythonnnn\model.p', 'rb'))
model = model_dict['model']

# Define the labels dictionary
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Thank You',
    27: 'Nice To Meet You', 28: 'How are you', 29: 'our', 30: 'project',
    31: 'is', 32: ' ', 33: 'sign', 34: 'language'
}

# Define language options
languages = {
    'English': 'en',
    'French': 'fr',
    'Spanish': 'es',
    'German': 'de',
    'Hindi': 'hi',
    'Telugu': 'te',
    'Tamil': 'ta',
    'Kannada': 'kn',
    'Malayalam': 'ml'
}

# Initialize MediaPipe modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False)
pose = mp_pose.Pose()

# Spell checker
spell = SpellChecker()

# Initialize Streamlit app
st.title("Real-Time Sign Language Detection")
selected_language = st.selectbox("Select Language for Translation", list(languages.keys()))

# Move audio placeholder to the top
audio_placeholder = st.empty()
recognized_word_placeholder = st.empty()  # Placeholder for the current recognized word
sentence_placeholder = st.empty()  # Placeholder for the full sentence
translation_placeholder = st.empty()
video_placeholder = st.empty()

# Initialize buffer and variables
letter_buffer = []
word_buffer = []  # Buffer for recognized words
sentence_buffer = []  # Buffer for recognized sentence
recognized_word = ""
last_symbol_time = time.time()
last_word_time = time.time()

# Store the last recognized letter and its frame count
last_recognized_letter = None
letter_frame_count = 0
required_frame_count = 10  # Number of consecutive frames required to confirm a letter

# Sentence timeout
word_timeout = 4  # Time after which we consider a word complete
sentence_timeout = 8  # Time after which we consider the sentence complete

# Correct spelling of recognized word
def correct_spelling(sentence):
    corrected_sentence = ""
    words = sentence.split()
    for word in words:
        corrected_word = spell.correction(word)
        if corrected_word is None:
            corrected_word = word
        corrected_sentence += corrected_word + " "
    return corrected_sentence.strip()

# Speak the translation
def speak_translation(text, lang_code):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
        temp_file_path = temp_audio_file.name
        # Use gTTS for text-to-speech
        tts = gTTS(text=text, lang=lang_code)
        tts.save(temp_file_path)
        
        # Play the audio in Streamlit
        audio_placeholder.audio(temp_file_path)

# Show translation and play the audio
def show_translation(sentence, dest_language_code):
    # Correct spelling of the input sentence
    corrected_sentence = correct_spelling(sentence)
    if not corrected_sentence:
        st.warning("The sentence is empty after correction.")
        return

    try:
        # Translate the text
        translated_text = GoogleTranslator(source='auto', target=dest_language_code).translate(corrected_sentence)
        
        # Print the original and translated text for debugging
        print(f"Original Sentence: {corrected_sentence}")
        print(f"Translated Text: {translated_text}")

        # Check if the translation is in the same language as the input
        if translated_text.strip().lower() == corrected_sentence.strip().lower():
            st.warning("Translation seems to be in the same language as the input.")
        else:
            translation_placeholder.text(f"Translation: {translated_text}")
            speak_translation(translated_text, dest_language_code)

    except exceptions.TranslationNotFound as e:
        st.error(f"Translation error: The translation could not be found. {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

# Open the camera and process frames
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Error: Unable to open the camera.")
    st.stop()

# Streamlit button to stop the camera
stop_camera = st.button('Stop Camera')

while cap.isOpened():
    if stop_camera:
        st.write("Stopping the camera...")
        break
    
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Unable to read frame from the camera.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    hand_results = hands.process(frame_rgb)
    
    # Process face landmarks
    face_results = face_mesh.process(frame_rgb)
    
    # Process pose (shoulder detection)
    pose_results = pose.process(frame_rgb)

    # Drawing landmarks for hands
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Draw landmarks for each hand separately
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),  # Customize drawing for this hand
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)   # Customize drawing for this hand
            )
            
            # Processing each hand for recognition
            data_aux = []
            x_ = []
            y_ = []
            
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)
            
            # Normalize and collect data into data_aux
            for i in range(len(x_)):
                data_aux.append(x_[i] - min(x_))
                data_aux.append(y_[i] - min(y_))

            # Ensure we only collect the expected number of features (42)
            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Check if the current letter matches the last recognized letter
                if predicted_character == last_recognized_letter:
                    letter_frame_count += 1
                else:
                    last_recognized_letter = predicted_character
                    letter_frame_count = 1

                # Add letter to the recognized word if it has been stable for required frames
                if letter_frame_count >= required_frame_count:
                    letter_buffer.append(predicted_character)
                    recognized_word += predicted_character
                    
                    # Show the recognized word
                    recognized_word_placeholder.text(f"Recognized Word: {recognized_word}")
                    letter_frame_count = 0  # Reset frame count for next letter

    # Drawing simplified face landmarks
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, 
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )

    # Drawing shoulder landmarks (Pose detection)
    if pose_results.pose_landmarks:
        # Detecting and drawing shoulder landmarks
        landmarks = pose_results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Draw circles on the shoulders
        left_shoulder_pos = (int(left_shoulder.x * frame.shape[1]), int(left_shoulder.y * frame.shape[0]))
        right_shoulder_pos = (int(right_shoulder.x * frame.shape[1]), int(right_shoulder.y * frame.shape[0]))
        cv2.circle(frame, left_shoulder_pos, 10, (255, 0, 0), -1)
        cv2.circle(frame, right_shoulder_pos, 10, (255, 0, 0), -1)

    # Check for word completion after inactivity
    current_time = time.time()
    if current_time - last_symbol_time > word_timeout and recognized_word:
        word_buffer.append(recognized_word.strip())
        recognized_word = ""
        recognized_word_placeholder.text(f"Recognized Word: {recognized_word}")
        sentence_placeholder.text(f"Current Sentence: {' '.join(word_buffer)}")
        last_word_time = current_time

    # Check for sentence completion after a longer period of inactivity
    if current_time - last_word_time > sentence_timeout and word_buffer:
        # Translate the complete sentence
        sentence = ' '.join(word_buffer)
        show_translation(sentence, languages[selected_language])

        # Reset buffers
        word_buffer = []
        recognized_word = ""
        last_symbol_time = time.time()

    # Show the processed frame
    video_placeholder.image(frame, channels="BGR")

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
