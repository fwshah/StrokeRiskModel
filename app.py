import streamlit as st
import os
import sys

# --- 1. Global Streamlit Page Configuration ---
st.set_page_config(
    page_title="StrokeRisk AI System - Login",
    page_icon="🏥",
    layout="wide",                   # Keep wide layout for the app content
    initial_sidebar_state="collapsed" # Collapsed by default for login page
)

# --- 2. Global Styling Injection (Tailwind CSS & Fonts) ---
def inject_global_css_and_fonts():
    """
    Injects Tailwind CSS CDN and Google Fonts for consistent styling across the app.
    Includes comprehensive custom CSS to override Streamlit's default layout
    and ensure full-width content rendering for main app pages, while keeping
    the login page contained.
    """
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.gstatic.com/" crossorigin>
        <link
            rel="stylesheet"
            as="style"
            onload="this.rel='stylesheet'"
            href="https://fonts.googleapis.com/css2?display=swap&family=Noto+Sans%3Awght%40400%3B500%3B700%3B900&family=Public+Sans%3Awght%40400%3B500%3B700%3B900"
        />
        <script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
        <style>
            /* Universal reset for margins, paddings, borders, outlines, shadows */
            html, body, #root, .stApp, .stApp > header, .main, .main .block-container,
            div[data-testid^="st"], /* Target all Streamlit data-testid divs */
            div[data-testid^="st"] > div,
            div[data-testid^="st"] > div > div,
            div[data-testid^="st"] > div > div > div {
                margin: 0 !important;
                padding: 0 !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
                box-sizing: border-box !important; /* Include padding and border in element's total width and height */
            }

            /* Ensure the entire viewport is used and scrolling is prevented */
            html, body {
                width: 100%;
                height: 100%;
                overflow-x: hidden; /* Prevent horizontal scrolling */
                font-family: "Public Sans", "Noto Sans", sans-serif;
                background-color: #f8f9fa; /* Consistent background for the whole page */
            }

            /* Default Streamlit app container styling - reset to allow granular control */
            .stApp {
                max-width: none !important;
                padding-top: 0 !important;
                padding-right: 0 !important;
                padding-left: 0 !important;
                padding-bottom: 0 !important;
                overflow-x: hidden; /* Prevent horizontal scrollbar if content overflows */
            }

            /* Hide the default Streamlit "Made with Streamlit" footer and hamburger menu */
            footer {
                visibility: hidden;
                height: 0;
            }
            #MainMenu {
                visibility: hidden;
            }

            /* Ensure the iframe itself takes full available width for embedded HTML */
            iframe {
                width: 100% !important;
                height: 100% !important; /* Make iframe fill its parent height */
                border: none !important; /* Remove default iframe border */
                box-sizing: border-box !important;
            }

            /* --- Conditional Styling based on login state (handled by Python now) --- */
            /* These classes will be added/removed dynamically by Python to control width */
            .login-container-style {
                max-width: 400px !important; /* Contained width for login */
                padding: 2rem !important;
                margin: auto !important; /* Center the block */
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 100vh; /* Ensure it takes full height to center content vertically */
            }

            .full-width-container-style {
                width: 100% !important;
                padding: 0 !important;
                margin: 0 !important;
                max-width: none !important;
                box-sizing: border-box !important;
                display: flex;
                flex-direction: column;
                align-items: stretch;
            }

            /* Specific override for the sidebar to ensure it's hidden on login */
            .st-emotion-cache-1jmve0k { /* This targets the sidebar container */
                display: none !important;
            }

            /* Styling for Streamlit's native text input widgets to match theme on login page */
            div[data-testid="stTextInput"] label {
                font-weight: 500; /* font-medium */
                color: #4b5563; /* text-gray-700 */
                margin-bottom: 0.5rem !important; /* Ensure label has space below */
                display: block !important;
                text-align: left !important;
            }

            div[data-testid="stTextInput"] input {
                width: 100% !important;
                padding: 0.75rem !important; /* p-3 */
                border: 1px solid #d1d5db !important; /* border-gray-300 */
                border-radius: 0.375rem !important; /* rounded-md */
                background-color: #f1f2f4 !important; /* Match your input background */
                color: #121516 !important; /* Input text color */
            }

            /* Styling for Streamlit's native form submit button to match theme on login page */
            div[data-testid="stFormSubmitButton"] button {
                width: 100% !important;
                padding: 0.75rem !important; /* p-3 */
                background-color: #3f8abf !important; /* bg-blue-600 */
                color: white !important;
                border-radius: 0.375rem !important; /* rounded-md */
                font-weight: 600 !important; /* font-semibold */
                transition: background-color 0.2s !important;
            }

            div[data-testid="stFormSubmitButton"] button:hover {
                background-color: #3371a1 !important; /* bg-blue-700 */
            }

            /* Ensure the form elements within the Streamlit form are stacked correctly */
            div[data-testid="stForm"] > div > div {
                display: flex !important;
                flex-direction: column !important;
                gap: 1rem !important; /* Space between form elements */
            }

            /* Adjust spacing for the "Don't have access" text on login page */
            .contact-admin-text {
                margin-top: 1rem !important; /* Add space above this text */
                color: #6a7781; /* text-gray-600 */
                font-size: 0.875rem; /* text-sm */
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- 3. Session State Initialization for Backend Logic ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False # Track login status

# --- 4. Main Application Flow (Login Page) ---
if __name__ == "__main__":
    inject_global_css_and_fonts()

    # Hardcoded credentials for demonstration
    VALID_MEDIC_ID = "medic123"
    VALID_PASSWORD = "password123"

    # Check if already logged in
    if st.session_state['logged_in']:
        # Apply full-width styling for logged-in state
        st.markdown(
            """
            <style>
                .main .block-container {
                    width: 100% !important;
                    padding: 0 !important;
                    margin: 0 !important;
                    max-width: none !important;
                    box-sizing: border-box !important;
                    display: flex;
                    flex-direction: column;
                    align-items: stretch;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.switch_page("pages/01_Home.py") # Redirect to home if already logged in
    else:
        # Apply contained styling for login page
        st.markdown(
            """
            <style>
                .main .block-container {
                    max-width: 400px !important;
                    padding: 2rem !important;
                    margin: auto !important;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    min-height: 100vh;
                }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Display the login form
        st.markdown(
            """
            <div class="flex flex-col items-center justify-center min-h-screen bg-gray-50">
                <div class="bg-white p-8 rounded-lg shadow-md w-full max-w-sm text-center">
                    <h1 class="text-[#121516] text-2xl font-bold leading-tight tracking-[-0.015em] mb-6">Login</h1>
                    <p class="text-gray-600 mb-6">Please enter your MedicID and Password</p>
            """,
            unsafe_allow_html=True
        )

        # Use Streamlit's native input widgets for interactivity
        with st.form("login_form"):
            medic_id = st.text_input("MedicID", placeholder="Enter your MedicID")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            login_button = st.form_submit_button("Login")

            if login_button:
                if medic_id == VALID_MEDIC_ID and password == VALID_PASSWORD:
                    st.session_state['logged_in'] = True
                    st.success("Login successful! Redirecting...")
                    st.rerun() # Rerun to trigger the redirection
                else:
                    st.error("Invalid MedicID or Password. Please try again.")

        st.markdown(
            """
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
