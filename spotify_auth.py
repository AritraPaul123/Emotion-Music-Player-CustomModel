import spotipy
from spotipy.oauth2 import SpotifyOAuth
from typing import Optional

class SpotifyAuthManager:
    def __init__(self):
        # Read Spotify API credentials from environment variables or config file
        try:
            # First try to read from config.py
            from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
            try:
                from config import SPOTIFY_REDIRECT_URI
                self.redirect_uri = SPOTIFY_REDIRECT_URI
            except ImportError:
                self.redirect_uri = "http://127.0.0.1:8888"
            self.client_id = SPOTIFY_CLIENT_ID
            self.client_secret = SPOTIFY_CLIENT_SECRET
        except ImportError:
            # If config.py doesn't exist, check environment variables
            import os
            self.client_id = os.getenv('SPOTIFY_CLIENT_ID')
            self.client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
            self.redirect_uri = os.getenv('SPOTIFY_REDIRECT_URI', 'http://127.0.0.1:8888')
            
            if not self.client_id or not self.client_secret:
                print("\nPlease set up your Spotify credentials by either:")
                print("1. Creating a config.py file with SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET")
                print("   OR")
                print("2. Setting environment variables SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET\n")
                raise ValueError("Missing Spotify credentials")
        
        self.scope = "user-modify-playback-state user-read-playback-state"
        
        self.sp: Optional[spotipy.Spotify] = None
    
    def authenticate(self) -> spotipy.Spotify:
        """Authenticate with Spotify and return the client."""
        try:
            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=self.redirect_uri,
                scope=self.scope
            ))
            return self.sp
        except Exception as e:
            raise Exception(f"Failed to authenticate with Spotify: {str(e)}")
    
    def get_client(self) -> Optional[spotipy.Spotify]:
        """Get the authenticated Spotify client."""
        return self.sp if self.sp else self.authenticate()

    def play_tracks_by_mood(self, mood: str) -> None:
        """Play tracks based on the detected mood."""
        if not self.sp:
            self.authenticate()
            
        # Mood to music mapping
        mood_queries = {
            "happy": "genre:pop mood:happy",
            "sad": "genre:indie mood:sad",
            "angry": "genre:rock mood:intense",
            "neutral": "genre:ambient mood:peaceful",
            "fear": "genre:classical mood:calm",
            "surprise": "genre:electronic mood:energetic",
            "disgust": "genre:jazz mood:sophisticated"
        }
        
        try:
            # Get available devices
            devices = self.sp.devices()
            
            # Find an active device or the first available device
            device_id = None
            for device in devices['devices']:
                if device['is_active']:
                    device_id = device['id']
                    break
            
            # If no active device found, use the first available device
            if not device_id and devices['devices']:
                device_id = devices['devices'][0]['id']
                # Transfer playback to this device
                self.sp.transfer_playback(device_id=device_id)
            
            if not device_id:
                print("No Spotify devices found. Please open Spotify on your computer or mobile device.")
                return
                
            # Search for tracks matching the mood
            query = mood_queries.get(mood.lower(), "genre:pop")
            results = self.sp.search(q=query, limit=10, type="track")
            
            if results and results["tracks"]["items"]:
                # Get track URIs
                track_uris = [track["uri"] for track in results["tracks"]["items"]]
                # Start playback on the selected device
                self.sp.start_playback(device_id=device_id, uris=track_uris)
        except Exception as e:
            print(f"Error playing tracks: {str(e)}")