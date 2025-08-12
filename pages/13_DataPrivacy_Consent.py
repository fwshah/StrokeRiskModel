import streamlit as st

def data_privacy_and_consent_page():
    """
    Renders the Data Privacy and Consent page.
    The header and footer HTML are embedded directly within this page.
    """
    # Define the header HTML content
    header_html_content = """
    <header class="flex items-center justify-between whitespace-nowrap border-b border-solid border-b-[#f1f2f4] px-10 py-3">
        <div class="flex items-center gap-4 text-[#121516]">
            <div class="size-4">
                <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path fill-rule="evenodd" clip-rule="evenodd" d="M24 4H6V17.3333V30.6667H24V44H42V30.6667V17.3333H24V4Z" fill="currentColor"></path>
                </svg>
            </div>
            <h2 class="text-[#121516] text-lg font-bold leading-tight tracking-[-0.015em]">StrokeRisk</h2>
        </div>
        <div class="flex flex-1 justify-end gap-6 flex-wrap">
            <div class="flex items-center gap-9">
                <a class="text-[#121516] text-sm font-medium leading-normal" href="pages/01_Home.py">Home</a>
                <a class="text-[#121516] text-sm font-medium leading-normal" href="pages/07_About_Us.py">About Us</a>
                <a class="text-[#121516] text-sm font-medium leading-normal" href="pages/02_Patient_Data_Entry.py">New Assessment</a>
                <a class="text-[#121516] text-sm font-medium leading-normal" href="pages/05_Patient_Records.py">Manage Patient Records</a>
            </div>
            <a href="pages/08_Practitioner_Profile.py" class="flex size-10 items-center justify-center rounded-full overflow-hidden">
                <div class="bg-center bg-no-repeat aspect-square bg-cover rounded-full size-10"
                    style='background-image: url("https://lh3.googleusercontent.com/aida-public/AB6AXuBwkktHQpR_bDtX78dae6gQMFY5NXBfpOazltOY9n2KDjANnnbfMSvj1VWr6h9sdxif9ERNDC4qiRzYq3h5DAPM8-R7pLtxTDUn1yabMKiR-M9aKohcwYezZ-0kO1omt02RpBEHBmpq9JeThPsCa8WcsleL1TSS668g3P_dtzNLUOWlc2lRl_IzpdsrsvTBBl7ypv9s3rjUgLBazzzNUdhIp345xspze6IOTt0Cntw09qi6KJ913o7nc9nSNJdeuqd8gyzf5dsfw1k");'>
                </div>
            </a>
            <a href="pages/13_Logout.py" class="flex min-w-[84px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-10 px-4 bg-[#3f8abf] text-white text-sm font-bold leading-normal tracking-[0.015em]">
                <span class="truncate">Log Out</span>
            </a>
        </div>
    </header>
    """

    # Define the footer HTML content 
    footer_html_content = """
    <footer class="flex justify-center">
        <div class="flex flex-1 flex-col">
            <footer class="flex flex-col gap-6 px-5 py-10 text-center @container">
                <div class="flex flex-wrap items-center justify-center gap-6 @[480px]:flex-row @[480px]:justify-around">
                    <a class="text-[#6a7781] text-base font-normal leading-normal min-w-40" href="pages/04_Model_Performance.py">Model Performance</a>
                    <a class="text-[#6a7781] text-base font-normal leading-normal min-w-40" href="pages/10_System_Settings.py">System Settings</a>
                    <a class="text-[#6a7781] text-base font-normal leading-normal min-w-40" href="pages/12_Help_and_Support.py">Help & Support</a>
                    <a class="text-[#6a7781] text-base font-normal leading-normal min-w-40" href="pages/13_Logout.py">Log Out</a>
                </div>
                <p class="text-[#6a7781] text-base font-normal leading-normal">© 2025 G4 Pulse. All rights reserved.</p>
            </footer>
        </div>
    </footer>
    """

    # The main content of your "Data Privacy and Consent" page
    # Added placeholder to Email Address input
    main_content_html = """
    <div class="px-40 flex flex-1 justify-center py-5">
        <div class="layout-content-container flex flex-col max-w-[960px] flex-1">
        <div class="flex flex-wrap justify-between gap-3 p-4">
            <div class="flex min-w-72 flex-col gap-3">
            <p class="text-[#121516] tracking-light text-[32px] font-bold leading-tight">Data Privacy and Consent</p>
            <p class="text-[#6a7781] text-sm font-normal leading-normal">
                Manage how patient data is used and ensure informed consent. We are committed to patient privacy and adhere to all relevant regulations.
            </p>
            </div>
        </div>
        <h2 class="text-[#121516] text-[22px] font-bold leading-tight tracking-[-0.015em] px-4 pb-3 pt-5">Automated Notifications</h2>
        <p class="text-[#121516] text-base font-normal leading-normal pb-3 pt-1 px-4">Enable automated reminders or notifications for follow-up appointments.</p>
        <div class="flex items-center gap-4 bg-white px-4 min-h-[72px] py-2 justify-between">
            <div class="flex flex-col justify-center">
            <p class="text-[#121516] text-base font-medium leading-normal line-clamp-1">Automated Notifications</p>
            <p class="text-[#6a7781] text-sm font-normal leading-normal line-clamp-2">Enable automated notifications for this patient</p>
        </div>
        <div class="shrink-0">
            <label
                class="relative flex h-[31px] w-[51px] cursor-pointer items-center rounded-full border-none bg-[#f1f2f4] p-0.5 has-[:checked]:justify-end has-[:checked]:bg-[#3f8abf]"
            >
                <div class="h-full w-[27px] rounded-full bg-white" style="box-shadow: rgba(0, 0, 0, 0.15) 0px 3px 8px, rgba(0, 0, 0, 0.06) 0px 3px 1px;"></div>
                <input type="checkbox" class="invisible absolute" />
            </label>
        </div>
        </div>
        <div class="flex max-w-[480px] flex-wrap items-end gap-4 px-4 py-3">
            <label class="flex flex-col min-w-40 flex-1">
            <p class="text-[#121516] text-base font-medium leading-normal pb-2">Preferred Method</p>
            <select
                class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 bg-[image:--select-button-svg] placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
            >
            <option value="one"></option>
            <option value="two">two</option>
            <option value="three">three</option>
            </select>
        </label>
        </div>
        <div class="flex max-w-[480px] flex-wrap items-end gap-4 px-4 py-3">
        <label class="flex flex-col min-w-40 flex-1">
            <p class="text-[#121516] text-base font-medium leading-normal pb-2">Timing of Reminders</p>
            <select
            class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 bg-[image:--select-button-svg] placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
            >
            <option value="one"></option>
            <option value="two">two</option>
            <option value="three">three</option>
            </select>
        </label>
        </div>
        <h2 class="text-[#121516] text-[22px] font-bold leading-tight tracking-[-0.015em] px-4 pb-3 pt-5">Result Delivery</h2>
        <p class="text-[#121516] text-base font-normal leading-normal pb-3 pt-1 px-4">Send assessment results directly to the patient electronically, ensuring security.</p>
        <div class="flex items-center gap-4 bg-white px-4 min-h-[72px] py-2 justify-between">
        <div class="flex flex-col justify-center">
            <p class="text-[#121516] text-base font-medium leading-normal line-clamp-1">Send Results to Patient</p>
            <p class="text-[#6a7781] text-sm font-normal leading-normal line-clamp-2">Enable sending results to the patient</p>
        </div>
        <div class="shrink-0">
            <label
            class="relative flex h-[31px] w-[51px] cursor-pointer items-center rounded-full border-none bg-[#f1f2f4] p-0.5 has-[:checked]:justify-end has-[:checked]:bg-[#3f8abf]"
            >
            <div class="h-full w-[27px] rounded-full bg-white" style="box-shadow: rgba(0, 0, 0, 0.15) 0px 3px 8px, rgba(0, 0, 0, 0.06) 0px 3px 1px;"></div>
            <input type="checkbox" class="invisible absolute" />
            </label>
        </div>
        </div>
        <div class="flex max-w-[480px] flex-wrap items-end gap-4 px-4 py-3">
        <label class="flex flex-col min-w-40 flex-1">
            <p class="text-[#121516] text-base font-medium leading-normal pb-2">Email Address</p>
            <input
            class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
            value=""
            placeholder="e.g. your.email@example.com"
            />
        </label>
        </div>
        <h2 class="text-[#121516] text-[22px] font-bold leading-tight tracking-[-0.015em] px-4 pb-3 pt-5">Data Usage for System Improvement</h2>
        <p class="text-[#121516] text-base font-normal leading-normal pb-3 pt-1 px-4">
        Allow the use of anonymized and aggregated patient data to improve the system. Individual data will not be disclosed. See our privacy policy for more details.
        </p>
        <div class="flex items-center gap-4 bg-white px-4 min-h-[72px] py-2 justify-between">
        <div class="flex flex-col justify-center">
            <p class="text-[#121516] text-base font-medium leading-normal line-clamp-1">Use Data for Improvement</p>
            <p class="text-[#6a7781] text-sm font-normal leading-normal line-clamp-2">Allow use of anonymized data for system improvement</p>
        </div>
        <div class="shrink-0">
            <label
            class="relative flex h-[31px] w-[51px] cursor-pointer items-center rounded-full border-none bg-[#f1f2f4] p-0.5 has-[:checked]:justify-end has-[:checked]:bg-[#3f8abf]"
            >
            <div class="h-full w-[27px] rounded-full bg-white" style="box-shadow: rgba(0, 0, 0, 0.15) 0px 3px 8px, rgba(0, 0, 0, 0.06) 0px 3px 1px;"></div>
            <input type="checkbox" class="invisible absolute" />
            </label>
        </div>
        </div>
        <h2 class="text-[#121516] text-[22px] font-bold leading-tight tracking-[-0.015em] px-4 pb-3 pt-5">Consent Summary</h2>
        <p class="text-[#121516] text-base font-normal leading-normal pb-3 pt-1 px-4">
        Selected Preferences: Notifications: Enabled (SMS, 1 day before), Result Delivery: Enabled (sophia.clark@example.com), Data Usage: Allowed
        </p>
        <div class="px-4">
        <label class="flex gap-x-3 py-3 flex-row">
            <input
            type="checkbox"
            class="h-5 w-5 rounded border-[#dde1e3] border-2 bg-transparent text-[#3f8abf] checked:bg-[#3f8abf] checked:border-[#3f8abf] checked:bg-[image:--checkbox-tick-svg] focus:ring-0 focus:ring-offset-0 focus:border-[#dde1e3] focus:outline-none"
            checked=""
            />
            <p class="text-[#121516] text-base font-normal leading-normal">I have discussed these options with the patient and obtained informed consent.</p>
        </label>
        </div>
        <div class="flex justify-stretch">
        <div class="flex flex-1 gap-3 flex-wrap px-4 py-3 justify-end">
            <button
            class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-10 px-4 bg-[#f1f2f4] text-[#121516] text-sm font-bold leading-normal tracking-[0.015em]"
            >
            <span class="truncate">Back to Client Profile</span>
            </button>
            <button
            class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-10 px-4 bg-[#3f8abf] text-white text-sm font-bold leading-normal tracking-[0.015em]"
            >
            <span class="truncate">Save Changes</span>
            </button>
        </div>
        </div>
    </div>
    </div>
    """

    # Combine all parts into the full HTML content
    full_html_content = f"""
    <html>
    <head>
        <link rel="preconnect" href="https://fonts.gstatic.com/" crossorigin="" />
        <link
            rel="stylesheet"
            as="style"
            onload="this.rel='stylesheet'"
            href="https://fonts.googleapis.com/css2?display=swap&family=Noto+Sans%3Awght%40400%3B500%3B700%3B900&family=Public+Sans%3Awght%40400%3B500%3B700%3B900"
        />

        <title>Data Privacy and Consent</title>
        <link rel="icon" type="image/x-icon" href="data:image/x-icon;base64," />

        <script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
        <style>
            /* Ensure the body uses your specified fonts, even within the iframe */
            body {{ font-family: "Public Sans", "Noto Sans", sans-serif; margin: 0; padding: 0; }}
            /* Ensure full height for the layout container within the iframe */
            .relative.flex.size-full.min-h-screen.flex-col {{ min-height: 100vh; }}

            /* Adjustments for width and padding within the embedded HTML */
            .flex.flex-1.justify-center.py-5 {{
                padding-left: 1rem !important; /* Adjust as needed for desired spacing */
                padding-right: 1rem !important; /* Adjust as needed for desired spacing */
            }}

            .layout-content-container.flex.flex-col.flex-1 {{
                max-width: 100% !important; /* Allow it to expand fully */
                width: 100% !important; /* Explicitly set width to 100% */
                padding-left: 1rem !important; /* Add some internal padding */
                padding-right: 1rem !important;
            }}

            /* Adjust header padding to be consistent with content if desired */
            header.px-10.py-3 {{
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }}
            /* Specific styles from your target HTML, particularly for custom SVG backgrounds */
            [style*="--checkbox-tick-svg"] {{
                --checkbox-tick-svg: url('data:image/svg+xml,%3csvg viewBox=%270 0 16 16%27 fill=%27rgb(255,255,255)%27 xmlns=%27http://www.w3.org/2000/svg%27%3e%3cpath d=%27M12.207 4.793a1 1 0 010 1.414l-5 5a1 1 0 01-1.414 0l-2-2a1 1 0 011.414-1.414L6.5 9.086l4.293-4.293a1 1 0 011.414 0z%27/%3e%3c/svg%3e');
            }}
            [style*="--select-button-svg"] {{
                --select-button-svg: url('data:image/svg+xml,%3csvg xmlns=%27http://www.w3.org/2000/svg%27 width=%2724px%27 height=%2724px%27 fill=%27rgb(106,119,129)%27 viewBox=%270 0 256 256%27%3e%3cpath d=%27M181.66,170.34a8,8,0,0,1,0,11.32l-48,48a8,8,0,0,1-11.32,0l-48-48a8,8,0,0,1,11.32-11.32L128,212.69l42.34-42.35A8,8,0,0,1,181.66,170.34Zm-96-84.68L128,43.31l42.34,42.35a8,8,0,0,0,11.32-11.32l-48-48a8,8,0,0,0-11.32,0l-48,48A8,8,0,0,0,85.66,85.66Z%27%3e%3c/path%3e%3c/svg%3e');
            }}
            .form-input.bg-\\[image\\:--select-button-svg\\] {{
                background-position: right 0.75rem center;
                background-repeat: no-repeat;
                background-size: 1.5em 1.5em;
                padding-right: 2.5rem; /* Make space for the icon */
                -webkit-appearance: none; /* Remove default arrow on WebKit browsers */
                -moz-appearance: none;    /* Remove default arrow on Firefox */
                appearance: none;         /* Remove default arrow */
            }}
            .checked\\:bg-\\[image\\:--checkbox-tick-svg\\] {{
                background-position: center;
                background-repeat: no-repeat;
                background-size: 100% 100%;
            }}
        </style>
    </head>
    <body>
        <div
            class="relative flex size-full min-h-screen flex-col bg-white group/design-root overflow-x-hidden"
            style='font-family: "Public Sans", "Noto Sans", sans-serif;'
        >
            <div class="layout-container flex h-full grow flex-col">
                {header_html_content}
                {main_content_html}
                {footer_html_content} </div>
        </div>
    </body>
    </html>
    """

    # Render the HTML content in Streamlit
    # Set scrolling=False to remove the scrollbar.
    # Adjusted height to accommodate all content, including the footer.
    st.components.v1.html(full_html_content, height=1600, scrolling=False)

# Call the function to render the page when this script is run
if __name__ == "__main__":
    st.set_page_config(layout="wide") # Optional: Use wide layout for better display
    data_privacy_and_consent_page()