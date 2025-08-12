import streamlit as st

def logout_page():
    """
    Handles the logout process by clearing session state and redirecting to the login page.
    """
    # Clear the 'logged_in' state to log the user out
    if 'logged_in' in st.session_state:
        del st.session_state['logged_in']
    
    st.info("You have been logged out. Redirecting to login page...")
    
    # It's good practice to stop further execution of the current script before redirecting
    st.stop() 

    # Redirect to the main application (app.py) which will now show the login screen
    # "app" refers to your main app.py file in the root directory
    st.switch_page("app") 

# This ensures the logout logic runs when the page is accessed
if __name__ == "__main__":
    logout_page()