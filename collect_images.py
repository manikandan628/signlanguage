import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 35  # Increased to accommodate 'Good Night'
dataset_size = 100

# Alphabet and additional messages
characters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              'Thank You', 'Nice To Meet You', 'How are you', 'our', 'project', 'is', ' ', 'sign','language']  # Added 'Good Night'

# Attempt to open the camera
cap = cv2.VideoCapture(0)  # Use the default camera (index 0)

# Check if the camera was opened successfully
if not cap.isOpened():
    print("Error: Unable to open the camera.")
    exit()

for j, current_character in enumerate(characters):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {} ({})'.format(j, current_character))

    print('Press "S" when ready.')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        message = 'Ready? Press "S"! To skip, press "N" :)'
        if current_character == 'Thank You':
            message = 'Thank You!'
        elif current_character == 'Nice To Meet You':
            message = 'Nice To Meet You!'
        elif current_character == 'How are you':
            message = 'How are you!'
        elif current_character == 'our':
            message = 'our!'
        elif current_character == 'project':
            message = 'project!'
        elif current_character == 'is':
            message = 'is!'
        elif current_character ==' ':
            message = ' !'
        elif current_character == 'sign':
            message = 'sign!'        
        elif current_character =='language':
            message = 'language!'    

        cv2.putText(frame, message, (50, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 155, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(25)
        if key == ord('s'):
            break
        elif key == ord('n'):
            print('Skipping class {} ({})'.format(j, current_character))
            break

    if key == ord('n'):
        continue  # Skip to the next character if 'N' was pressed

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the current character on the frame
        cv2.putText(frame, 'Character: {}'.format(current_character), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Skip taking pictures if 'N' is pressed
        key = cv2.waitKey(25)
        if key == ord('n'):
            print('Skipping pictures for class {} ({})'.format(j, current_character))
            break

        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()

