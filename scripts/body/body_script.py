# Import libraries
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to analyze movements based on landmarks
def calculate_movement(landmarks_curr, landsmarks_prev):
    # Initialize dictionary for distances 
    movement = {}
    
    # Body parts tracked
    body_parts = {
        'Left Eye': (mp_pose.PoseLandmark.LEFT_EYE.value),  # Left eye landmark (used for tracking head movement)
        'Right Eye': (mp_pose.PoseLandmark.RIGHT_EYE.value),  # Right eye landmark (used for tracking head movement)
        #'Nose ': (mp_pose.PoseLandmark.NOSE.value),  #Nose landmark (not in use)
        'Left Shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER.value,  # Left Shoulder
        'Right Shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER.value,  # Right Shoulder
        'Left Elbow': mp_pose.PoseLandmark.LEFT_ELBOW.value,  # Left Elbow
        'Right Elbow': mp_pose.PoseLandmark.RIGHT_ELBOW.value,  # Right Elbow
        'Left Wrist': mp_pose.PoseLandmark.LEFT_WRIST.value, # Left wrist
        'Right Wrist': mp_pose.PoseLandmark.RIGHT_WRIST.value, # Right wrist
        'Left Hip': mp_pose.PoseLandmark.LEFT_HIP.value,  # Left Hip
        'Right Hip': mp_pose.PoseLandmark.RIGHT_HIP.value,  # Right Hip
        'Left Knee': mp_pose.PoseLandmark.LEFT_KNEE.value,  # Left Knee
        'Right Knee': mp_pose.PoseLandmark.RIGHT_KNEE.value,  # Right Knee
        'Left Ankle': mp_pose.PoseLandmark.LEFT_ANKLE.value,  # Left Ankle
        'Right Ankle': mp_pose.PoseLandmark.RIGHT_ANKLE.value  # Right Ankle
    }
    
    # Calculate movement for each tracked body part
    for part, i in body_parts.items():
        pos_curr = [landmarks_curr[i].x, landmarks_curr[i].y]
        pos_prev = [landsmarks_prev[i].x, landsmarks_prev[i].y]
        distance = np.linalg.norm(np.array(pos_curr) - np.array(pos_prev))
        movement[part] = distance

    # Return distances dictionary
    return movement

# Function to categorize overall body movement
def categorize_movement(total_movement):
    if total_movement < 0.1:  # Low body movement observed during video capture (very little movement observed)
        return "Low", total_movement
    elif total_movement < 0.4:  # Medium body movement observed during video capture (regular movement observed)
        return "Medium", total_movement
    else:  # High body movement observed during video capture (excessive movement observed)
        return "High", total_movement
