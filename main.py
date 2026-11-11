import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageTk
import customtkinter as ctk
from typing import Optional
import threading
import time
from spotify_auth import SpotifyAuthManager
from custom_model import EmotionCNN

EMOTION_COLORS = {
    'happy': '#00D26A',      # Vibrant green
    'sad': '#4A90E2',        # Soft blue
    'angry': '#FF4757',      # Bright red
    'surprise': '#FFA502',   # Bright orange
    'neutral': '#A4B0BE'     # Clean cool gray
}
EMOTION_EMOJIS = {
    'happy': '😊 Happy',
    'sad': '😢 Sad',
    'angry': '😠 Angry',
    'surprise': '😲 Surprised',
    'neutral': '😐 Neutral'
}

class EmotionMusicPlayer(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Initialize Spotify
        self.spotify = SpotifyAuthManager()
        
        # Configure window
        self.title("Emotion Music Player - Custom CNN Model")
        self.geometry("800x600")
        
        # Load Custom Model (use CPU for ultra-fast, robust real-time inference)
        self.device = torch.device('cpu')
        self.model = EmotionCNN(num_classes=5).to(self.device)
        weights_path = os.path.join(os.path.dirname(__file__), 'custom_model_weights.pth')
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"Loaded custom model weights from {weights_path}")
        else:
            print("Warning: custom_model_weights.pth not found!")
        self.model.eval()
        
        # Load face cascade safely
        cascade_file = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
        if not os.path.exists(cascade_file):
            cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_file)
        if self.face_cascade.empty():
            print(f"Warning: Haar cascade could not be loaded from {cascade_file}")
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Initialize variables
        self.camera = cv2.VideoCapture(0)
        self.current_emotion: Optional[str] = None
        self.last_emotion_change = time.time()
        self.emotion_cooldown = 10  # Seconds between emotion changes
        self.running = True
        
        # Create GUI elements
        self.create_gui()
        
        # Schedule periodic camera update on the main thread via Tkinter's event loop
        self.after(50, self.update_camera)
    
    def create_gui(self):
        """Create the GUI elements"""
        # Configure grid for main window
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)
        
        # Camera frame
        self.camera_frame = ctk.CTkFrame(self, corner_radius=15)
        self.camera_frame.pack(pady=(15, 10), padx=20, fill="both", expand=True)
        
        self.camera_label = ctk.CTkLabel(self.camera_frame, text="")
        self.camera_label.pack(pady=10, padx=10, expand=True)
        
        # Live Emotion Badge directly under the webcam playback
        self.emotion_card = ctk.CTkFrame(self, fg_color="#1e1e24", corner_radius=12, border_width=2, border_color="#3a3a4c")
        self.emotion_card.pack(pady=(0, 15), padx=20, fill="x")
        
        self.emotion_title_label = ctk.CTkLabel(
            self.emotion_card,
            text="LIVE EMOTION",
            font=("Helvetica", 11, "bold"),
            text_color="#9e9eb3"
        )
        self.emotion_title_label.pack(pady=(8, 0))
        
        self.emotion_label = ctk.CTkLabel(
            self.emotion_card, 
            text="Analyzing...",
            font=("Helvetica", 24, "bold"),
            text_color="#00D26A"
        )
        self.emotion_label.pack(pady=(2, 4))
        
        self.music_status_label = ctk.CTkLabel(
            self.emotion_card,
            text="🎵 Music syncs on steady expression",
            font=("Helvetica", 12),
            text_color="#7a7a90"
        )
        self.music_status_label.pack(pady=(0, 8))
        
        # Control buttons
        self.control_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.control_frame.pack(pady=(0, 15), padx=20, fill="x")
        
        self.toggle_btn = ctk.CTkButton(
            self.control_frame,
            text="Stop Camera",
            font=("Helvetica", 13, "bold"),
            command=self.toggle_camera
        )
        self.toggle_btn.pack(side="left", padx=10)
        
        self.quit_btn = ctk.CTkButton(
            self.control_frame,
            text="Quit",
            font=("Helvetica", 13, "bold"),
            fg_color="#e03131",
            hover_color="#c92a2a",
            command=self.quit_app
        )
        self.quit_btn.pack(side="right", padx=10)
    
    def detect_emotion(self, frame):
        """Predict emotion using Custom CNN"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []
        if not self.face_cascade.empty():
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) == 0:
            # Fallback to full center frame crop if no face detected
            h, w = gray.shape
            face_crop = gray[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
        else:
            (x, y, w, h) = faces[0]
            face_crop = gray[y:y+h, x:x+w]
            
        pil_img = Image.fromarray(face_crop)
        t_img = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(t_img)
            pred_idx = torch.argmax(output, dim=1).item()
            return CLASS_NAMES[pred_idx]

    def update_camera(self):
        """Update camera feed and detect emotions safely via Tkinter loop"""
        if not self.running:
            return

        ret, frame = self.camera.read()
        if ret:
            try:
                # Continuous real-time detection on every camera frame
                emotion = self.detect_emotion(frame)
                
                # Dynamically update the displayed emotion badge on every frame
                display_text = EMOTION_EMOJIS.get(emotion.lower(), emotion.capitalize())
                badge_color = EMOTION_COLORS.get(emotion.lower(), "#00D26A")
                self.emotion_label.configure(text=display_text, text_color=badge_color)
                self.emotion_card.configure(border_color=badge_color)
                
                # Music playback change triggers only after cooldown
                current_time = time.time()
                if (emotion != self.current_emotion and 
                    current_time - self.last_emotion_change > self.emotion_cooldown):
                    self.current_emotion = emotion
                    self.last_emotion_change = current_time
                    self.music_status_label.configure(text=f"🎶 Now Playing music for: {emotion.capitalize()}")
                    print(f"Triggering Spotify playback for new emotion: {emotion}...")
                    threading.Thread(
                        target=self.spotify.play_tracks_by_mood,
                        args=(emotion,),
                        daemon=True
                    ).start()
            
            except Exception as e:
                print(f"Error detecting emotion: {str(e)}")
            
            disp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            disp_frame = cv2.resize(disp_frame, (640, 480))
            img = Image.fromarray(disp_frame)
            img_tk = ImageTk.PhotoImage(image=img)
            self.camera_label.configure(image=img_tk)
            self.camera_label.image = img_tk
            
        if self.running:
            self.after(30, self.update_camera)
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.toggle_btn.cget("text") == "Stop":
            self.toggle_btn.configure(text="Start")
            self.running = False
        else:
            self.toggle_btn.configure(text="Stop")
            self.running = True
            self.after(30, self.update_camera)
    
    def quit_app(self):
        """Clean up and quit application"""
        self.running = False
        if self.camera.isOpened():
            self.camera.release()
        self.quit()

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = EmotionMusicPlayer()
    app.mainloop()
