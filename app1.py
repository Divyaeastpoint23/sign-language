import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import time
import collections

# Initialize MediaPipe components
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load letter detection model
letter_model_path = './model.p'
letter_model_dict = pickle.load(open(letter_model_path, 'rb'))
letter_model = letter_model_dict['model']

# Load word detection model
word_model_path = 'best_model1.keras'
word_model = load_model(word_model_path)

# Constants
ACTIONS = np.array(['hello', 'thanks', 'iloveyou', 'No', 'yes', 'All the best'])
SEQUENCE_LENGTH = 30

# Letter labels dictionary
LETTER_LABELS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 
    26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9',
    36: ' ', 37: '.'
}

def process_hand_landmarks(hand_landmarks):
    """Process hand landmarks for letter detection."""
    data_aux = []
    x_ = []
    y_ = []
    
    for landmark in hand_landmarks.landmark:
        x_.append(landmark.x)
        y_.append(landmark.y)
    
    for landmark in hand_landmarks.landmark:
        data_aux.append(landmark.x - min(x_))
        data_aux.append(landmark.y - min(y_))
    
    while len(data_aux) < 42:
        data_aux.append(0)
    return data_aux[:42]

def extract_holistic_keypoints(results):
    """Extract keypoints for word detection."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, left_hand, right_hand])

def letter_detection():
    """Function to detect individual letters in ASL."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access the camera.")
        return
    
    frame_width = 1920
    frame_height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    window_name = 'Letter Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, frame_width, frame_height)
    
    word_buffer = ""
    sentence_buffer = ""
    last_capture_time = 0
    capture_interval = 1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Add guidance box
        height, width = frame.shape[:2]
        box_size = int(min(width, height) * 0.7)
        center_x = width // 2
        center_y = height // 2
        cv2.rectangle(frame, 
                     (center_x - box_size//2, center_y - box_size//2),
                     (center_x + box_size//2, center_y + box_size//2), 
                     (0, 255, 0), 3)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        detected_char = '?'
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            prediction = letter_model.predict([np.asarray(process_hand_landmarks(hand_landmarks))])
            predicted_index = int(prediction[0])
            detected_char = LETTER_LABELS.get(predicted_index, '?')

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"Detected: {detected_char}", (10, 250), 
                       cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        
        # Display buffers
        cv2.putText(frame, f"Word: {word_buffer}", (10, 300), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Sentence: {sentence_buffer}", (10, 350), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        current_time = time.time()
        
        if key == ord(' '):
            if current_time - last_capture_time > capture_interval:
                word_buffer += detected_char
                last_capture_time = current_time
                flash = np.ones_like(frame) * 255
                cv2.imshow(window_name, flash)
                cv2.waitKey(50)
        elif key == 13:  # Enter key
            if word_buffer.strip():
                if sentence_buffer:
                    sentence_buffer += ' ' + word_buffer.strip()
                else:
                    sentence_buffer = word_buffer.strip()
                word_buffer = ""
        elif key == ord('q'):
            if word_buffer.strip():
                if sentence_buffer:
                    sentence_buffer += ' ' + word_buffer.strip()
                else:
                    sentence_buffer = word_buffer.strip()
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Detected Sentence: {sentence_buffer}")

def word_detection():
    """Function to detect ASL words/phrases."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access the camera.")
        return
    
    frame_width = 1920
    frame_height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    window_name = 'Word Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, frame_width, frame_height)
    
    sequence = collections.deque(maxlen=SEQUENCE_LENGTH)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        keypoints = extract_holistic_keypoints(results)
        sequence.append(keypoints)

        if len(sequence) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(sequence, axis=0)
            res = word_model.predict(input_data)
            action = ACTIONS[np.argmax(res)]
            cv2.putText(image, f'Action: {action}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(window_name, image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function to run the application."""
    while True:
        print("\nChoose Detection Mode:")
        print("1. Letter Detection")
        print("2. Word Detection")
        print("3. Exit")
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        
        if choice == '1':
            letter_detection()
        elif choice == '2':
            word_detection()
        elif choice == '3':
            print("Exiting the application.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()