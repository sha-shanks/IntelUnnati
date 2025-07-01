import streamlit as st
from PIL import Image
import io
import base64
import json
from pathlib import Path

import streamlit_chatbox 
import time
import simplejson as json

# Configure page
st.set_page_config(page_title="Profile Selection", layout="wide")

# Custom CSS for profile selection
st.markdown("""
<style>
    .profile-card {
        background-color: #141414;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px;
        cursor: pointer;
        transition: transform 0.2s;
        border: 2px solid transparent;
        color: white;
    }
    
    .profile-card:hover {
        transform: scale(1.05);
        border-color: white;
    }
    
    .profile-image {
        width: 120px;
        height: 120px;
        border-radius: 10px;
        object-fit: cover;
        margin-bottom: 10px;
    }
    
    .profile-name {
        font-size: 16px;
        font-weight: bold;
        color: #e5e5e5;
    }
    
    .add-profile-card {
        background-color: #333333;
        border: 2px dashed #666666;
        border-radius: 10px;
        padding: 40px 20px;
        text-align: center;
        margin: 10px;
        cursor: pointer;
        transition: all 0.2s;
        color: #999999;
    }
    
    .add-profile-card:hover {
        background-color: #444444;
        border-color: #999999;
        color: white;
    }
    
    .main-title {
        text-align: center;
        color: white;
        font-size: 48px;
        font-weight: 300;
        margin-bottom: 40px;
    }
    
    .stApp {
        background-color: #000000;
    }
    
    .sidebar-tab {
        background-color: #1a1a1a;
        border: none;
        color: #cccccc;
        padding: 15px 20px;
        margin: 5px 0;
        border-radius: 5px;
        cursor: pointer;
        transition: all 0.3s;
        width: 100%;
        text-align: left;
        font-size: 16px;
    }
    
    .sidebar-tab:hover {
        background-color: #333333;
        color: white;
    }
    
    .sidebar-tab.active {
        background-color: #e50914;
        color: white;
    }
    
    .content-area {
        background-color: #141414;
        padding: 30px;
        border-radius: 10px;
        color: white;
        min-height: 500px;
    }
            
    .about-area {
        background-color: #141414;
        padding: 30px;
        border-radius: 10px;
        color: white;
        min-height: 200px;
    }
    
    .user-header {
        background-color: #1a1a1a;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Local storage Fuction
def ensure_user_data_directory():
    """
    Create user_data directory if it doesn't exist
    """

    user_data_path = Path("user_data")
    user_data_path.mkdir(exist_ok=True)
    return user_data_path

def save_profile_to_storage(profile, index):
    """
    Save profile data to local storage
    """
    
    user_data_path = ensure_user_data_directory()
    
    # saving profile metadata
    profile_data = {
        'name': profile['name'],
        'has_image': profile['image'] is not None
    }
    
    with open(user_data_path / f"profile_{index}.json", 'w') as f:
        json.dump(profile_data, f)
    
    # saving image if it exists
    if profile['image']:
        profile['image'].save(user_data_path / f"profile_{index}.png")

def load_profiles_from_storage():
    """
    Load profiles from local storage
    """

    user_data_path = ensure_user_data_directory()
    profiles = []
    
    # Check for existing profiles
    for i in range(3):  # Maximum profile is 3
        profile_file = user_data_path / f"profile_{i}.json"
        if profile_file.exists():
            with open(profile_file, 'r') as f:
                profile_data = json.load(f)
            
            # Load image if exists
            image_file = user_data_path / f"profile_{i}.png"
            image = None
            if profile_data['has_image'] and image_file.exists():
                try:
                    image = Image.open(image_file)
                except Exception:
                    pass
            
            profiles.append({
                'name': profile_data['name'],
                'image': image
            })
    
    # Creating default user if no profile
    if not profiles:
        default_profile = {'name': 'Default User', 'image': None}
        profiles.append(default_profile)
        save_profile_to_storage(default_profile, 0)
    
    return profiles

def save_all_profiles():
    """
    Save all current profiles to storage
    """

    for i, profile in enumerate(st.session_state['profiles']):
        save_profile_to_storage(profile, i)

# Initialize session state
if 'profiles' not in st.session_state:
    st.session_state['profiles'] = load_profiles_from_storage()

if 'show_add_dialog' not in st.session_state:
    st.session_state['show_add_dialog'] = False

if 'selected_profile' not in st.session_state:
    st.session_state['selected_profile'] = None

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'profile_selection'

if 'active_tab' not in st.session_state:
    st.session_state['active_tab'] = 'chatbot_unnati'

# Helper functions
def image_to_base64(image):
    """
    Convert PIL image to base64 string
    """

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def create_profile_card(profile, index):
    """
    Create a profile card with image and name
    """

    if profile['image']:
        img_b64 = image_to_base64(profile['image'])
        card_html = f"""
        <div class="profile-card">
            <img src="{img_b64}" class="profile-image" alt="{profile['name']}">
            <div class="profile-name">{profile['name']}</div>
        </div>
        """

    else:
        card_html = f"""
        <div class="profile-card">
            <div style="width: 120px; height: 120px; background-color: #666666; 
                        border-radius: 10px; margin: 0 auto 10px auto; 
                        display: flex; align-items: center; justify-content: center;
                        font-size: 48px; color: white;">
                {profile['name'][0].upper()}
            </div>
            <div class="profile-name">{profile['name']}</div>
        </div>
        """
    return card_html

def add_profile(name, image):
    """
    Add a new profile to the list
    """

    if len(st.session_state['profiles']) < 3 and name.strip():
        new_profile = {'name': name.strip(), 'image': image}
        st.session_state['profiles'].append(new_profile)
        save_profile_to_storage(new_profile, len(st.session_state['profiles']) - 1)
        return True
    return False

# Main application logic
if st.session_state['current_page'] == 'profile_selection':
    # Profile Selection Page
    if not st.session_state['show_add_dialog']:
        # Main profile selection screen
        st.markdown('<h1 class="main-title">Who\'s learning?</h1>', unsafe_allow_html=True)
        
        # Calculate columns needed
        total_items = len(st.session_state['profiles']) + (1 if len(st.session_state['profiles']) < 3 else 0)
        cols_per_row = min(3, total_items)
        
        # Create profile grid
        cols = st.columns(cols_per_row)
        
        # Display existing profiles
        for i, profile in enumerate(st.session_state['profiles']):
            with cols[i % cols_per_row]:
                card_html = create_profile_card(profile, i)
                st.markdown(card_html, unsafe_allow_html=True)
                
                if st.button(f"Select {profile['name']}", key=f"select_{i}", use_container_width=True):
                    st.session_state['selected_profile'] = profile['name']
                    st.session_state['current_page'] = 'main_app'
                    st.rerun()
        
        # Add Profile button (if under limit)
        if len(st.session_state['profiles']) < 3:
            with cols[len(st.session_state['profiles']) % cols_per_row]:
                add_card_html = """
                <div class="add-profile-card">
                    <div style="font-size: 48px; margin-bottom: 10px;">+</div>
                    <div style="font-size: 16px; font-weight: bold;">Add Profile</div>
                </div>
                """
                st.markdown(add_card_html, unsafe_allow_html=True)
                
                if st.button("+ Add Profile", key="add_profile_btn", use_container_width=True):
                    st.session_state['show_add_dialog'] = True
                    st.rerun()

    else:
        # Add Profile dialog
        st.markdown('<h1 class="main-title">Add Profile</h1>', unsafe_allow_html=True)
        
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                # Profile name input
                profile_name = st.text_input(
                    "Profile Name",
                    placeholder="Enter profile name...",
                    max_chars=20
                )
                
                # Profile picture upload
                uploaded_file = st.file_uploader(
                    "Upload Profile Picture (Optional)",
                    type=['png', 'jpg', 'jpeg'],
                    help="Upload a square image for best results"
                )
                
                profile_image = None
                if uploaded_file is not None:
                    try:
                        profile_image = Image.open(uploaded_file)
                        # Resize image to square
                        size = min(profile_image.size)
                        profile_image = profile_image.crop((
                            (profile_image.width - size) // 2,
                            (profile_image.height - size) // 2,
                            (profile_image.width + size) // 2,
                            (profile_image.height + size) // 2
                        ))
                        profile_image = profile_image.resize((200, 200), Image.Resampling.LANCZOS)
                        
                        # Preview the image
                        st.image(profile_image, width=150, caption="Profile Picture Preview")
                    except Exception as e:
                        st.error("Error processing image. Please try a different file.")
                
                st.markdown("---")
                
                # Action buttons
                col_add, col_cancel = st.columns(2)
                
                with col_add:
                    if st.button("Add Profile", type="primary", use_container_width=True):
                        if profile_name.strip():
                            if add_profile(profile_name, profile_image):
                                st.success(f"Profile '{profile_name}' added successfully!")
                                st.session_state['show_add_dialog'] = False
                                st.rerun()
                            else:
                                st.error("Maximum 3 profiles allowed or invalid name.")
                        else:
                            st.error("Please enter a profile name.")
                
                with col_cancel:
                    if st.button("Cancel", use_container_width=True):
                        st.session_state['show_add_dialog'] = False
                        st.rerun()

elif st.session_state['current_page'] == 'main_app':
    # Main Application Page
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div class="user-header">
            <h3>Welcome, {st.session_state['selected_profile']}!</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Navigation")
        
        # Tab buttons
        if st.button("🪄 Chat with Unnati!", key="chatbot_unnati", use_container_width=True):
            st.session_state['active_tab'] = 'chatbot_unnati'
            st.rerun()
        
        if st.button("⚙️ Settings", key="settings_tab", use_container_width=True):
            st.session_state['active_tab'] = 'settings'
            st.rerun()

        if st.button("🧑‍💻 About", key="about_tab", use_container_width=True):
            st.session_state['active_tab'] = 'about'
            st.rerun()
        
        st.markdown("---")
        
        if st.button("← Back to Profiles", use_container_width=True):
            st.session_state['current_page'] = 'profile_selection'
            st.session_state['selected_profile'] = None
            st.rerun()
    
    # Main content
    if st.session_state['active_tab'] == 'chatbot_unnati':
        # --- Chat History Management Functions ---
        def get_profile_chat_dir(profile_name):
            # Sanitize profile name to create a valid directory name
            sanitized_name = "".join(c for c in profile_name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
            chat_dir = Path(f"user_data/{sanitized_name}_chats")
            chat_dir.mkdir(exist_ok=True)
            return chat_dir

        def save_chat_history(profile_name, chat_history, current_chat_file):
            chat_dir = get_profile_chat_dir(profile_name)
            # If it's a new chat, create a filename
            if not current_chat_file:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                # Try to get the first user message for a meaningful name
                first_user_message = next((msg['content'] for msg in chat_history if msg['role'] == 'user'), 'chat')
                # Sanitize the message for the filename
                filename_stem = "".join(c for c in first_user_message[:30] if c.isalnum() or c in (' ')).strip()
                if not filename_stem: # handle case where message has no alphanumeric chars
                    filename_stem = "chat"
                new_filename = f"{timestamp}_{filename_stem}.json"
                current_chat_file = chat_dir / new_filename
            
            try:
                with open(current_chat_file, 'w') as f:
                    json.dump(chat_history, f, indent=4)
                return current_chat_file
            except Exception as e:
                st.error(f"Error saving chat: {e}")
                return current_chat_file

        def load_chat_history(filepath):
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                st.error("Could not load chat history.")
                return []
            except Exception as e:
                st.error(f"Error loading chat: {e}")
                return []

        def list_chat_histories(profile_name):
            chat_dir = get_profile_chat_dir(profile_name)
            # List and sort files by modification time (newest first)
            return sorted(list(chat_dir.glob("*.json")), key=lambda f: f.stat().st_mtime, reverse=True)

        # --- Session State Initialization for Chat ---
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        if 'current_chat' not in st.session_state:
            st.session_state.current_chat = None # Stores the Path object of the current chat file
        if 'selected_prompt' not in st.session_state:
            st.session_state.selected_prompt = ""

        # --- UI Implementation ---
        st.markdown("<h1><center>🤖 Chat with Unnati</center></h1>", unsafe_allow_html=True)

        history_col, chat_col = st.columns([1, 3])

        # Left column for chat history
        with history_col:
            st.markdown("<h4><center>Chat History</center></h4>", unsafe_allow_html=True)
            
            if st.button("➕ New Chat", use_container_width=True):
                st.session_state.chat_messages = []
                st.session_state.current_chat = None
                st.session_state.selected_prompt = ""
                st.rerun()
                
            st.markdown("---")
            
            chat_files = list_chat_histories(st.session_state['selected_profile'])
            
            # Scrollable container for chat history
            with st.container(height=200):
                for chat_file in chat_files:
                    # Use file stem for a cleaner title
                    chat_title = chat_file.stem
                    if st.button(chat_title, key=chat_file.name, use_container_width=True):
                        st.session_state.current_chat = chat_file
                        st.session_state.chat_messages = load_chat_history(chat_file)
                        st.session_state.selected_prompt = ""
                        st.rerun()

        # Right column for the chatbox
        with chat_col:
            # This container holds the chat messages and will be scrollable
            with st.container(height=400):
                chat_box = streamlit_chatbox.ChatBox()
                # Display the chat history
                for msg in st.session_state.chat_messages:
                    if msg["role"] == "user":
                        chat_box.user_say(msg["content"])
                    elif msg["role"] == "assistant":
                        chat_box.ai_say(msg["content"])

            # Placeholder prompts are now at the bottom of the chat column
            placeholders = ["Class 10 Mathematics", "Class 10 Science", "Class 10 History", "Class 10 Geography", "Class 10 Civics", "Class 10 Economics"]
            
            # Callback to update prompt from dropdown
            def set_prompt_from_dropdown():
                st.session_state.selected_prompt = st.session_state.prompt_selector
            
            st.selectbox(
                "Select the subject:",
                placeholders,
                key="prompt_selector",
                on_change=set_prompt_from_dropdown
            )

        # Handle user input (from chat_input and dropdown) - MOVED THIS OUTSIDE of chat_col for pinning, DONT TOUCHHH!
        user_query = None
        if prompt := st.chat_input("Your message to Unnati..."):
            user_query = prompt
        elif st.session_state.selected_prompt:
            user_query = st.session_state.selected_prompt

        if user_query:
            # Add user message to state and display it
            st.session_state.chat_messages.append({"role": "user", "content": user_query})
            
            # Dummy bot response
            with st.spinner("Unnati is thinking..."):
                time.sleep(1) # Simulate work
                bot_response = f"I received: '{user_query}'. I am a dummy bot."
            
            # Add bot response to state
            st.session_state.chat_messages.append({"role": "assistant", "content": bot_response})
            
            # Save the updated chat history
            st.session_state.current_chat = save_chat_history(
                st.session_state['selected_profile'],
                st.session_state.chat_messages,
                st.session_state.current_chat
            )
            
            # Clear the selected prompt and rerun to update UI
            st.session_state.selected_prompt = ""
            st.rerun()

    elif st.session_state['active_tab'] == 'settings':
        st.markdown("""
        <div class="content-area">
            <h1>⚙️ Settings</h1>
            <p>Manage your preferences and account settings.</p>
            <h3>Account Settings</h3>
            <div style="background-color: #333; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <p><strong>Profile Name:</strong> {}</p>
            </div>
        </div>
        """.format(st.session_state['selected_profile']), unsafe_allow_html=True)

    elif st.session_state['active_tab'] == 'about':
        st.markdown("""
        <div class="about-area">
            <h1>🧑‍💻 About</h1>
            <p>Made with 💖 for Intel Unnati</p>
        </div>
        """, unsafe_allow_html=True)