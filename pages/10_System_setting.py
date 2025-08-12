import streamlit as st

def system_settings_page():
    """
    Renders the System Settings Page by embedding the provided HTML content.
    Adjustments have been made to the HTML's Tailwind classes and internal CSS
    to ensure full-width rendering within the Streamlit application and
    to eliminate unnecessary scrollbars, providing a balanced and professional look.
    """
    # Define the header HTML content with updated links and profile image
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
            <button
                class="flex max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-10 bg-[#f1f2f4] text-[#121516] gap-2 text-sm font-bold leading-normal tracking-[0.015em] min-w-0 px-2.5"
            >
                <div class="text-[#121516]" data-icon="Bell" data-size="20px" data-weight="regular">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" fill="currentColor" viewBox="0 0 256 256">
                        <path
                            d="M221.8,175.94C216.25,166.38,208,139.33,208,104a80,80,0,1,0-160,0c0,35.34-8.26,62.38-13.81,71.94A16,16,0,0,0,48,200H88.81a40,40,0,0,0,78.38,0H208a16,16,0,0,0,13.8-24.06ZM128,216a24,24,0,0,1-22.62-16h45.24A24,24,0,0,1,128,216ZM48,184c7.7-13.24,16-43.92,16-80a64,64,0,1,1,128,0c0,36.05,8.28,66.73,16,80Z"
                        ></path>
                    </svg>
                </div>
            </button>
            <a href="pages/08_Practitioners_Profile.py" class="flex size-10 items-center justify-center rounded-full overflow-hidden">
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

    # Define the footer HTML content with updated links
    footer_html_content = """
    <footer class="flex justify-center">
        <div class="flex flex-1 flex-col">
            <footer class="flex flex-col gap-6 px-5 py-10 text-center @container">
                <div class="flex flex-wrap items-center justify-center gap-6 @[480px]:flex-row @[480px]:justify-around">
                    <a class="text-[#6a7781] text-base font-normal leading-normal min-w-40" href="pages/04_Model_Performance.py">Model Performance</a>
                    <a class="text-[#6a7781] text-base font-normal leading-normal min-w-40" href="pages/10_System_setting.py">System Settings</a>
                    <a class="text-[#6a7781] text-base font-normal leading-normal min-w-40" href="pages/12_Help_and_Support.py">Help & Support</a>
                    <a class="text-[#6a7781] text-base font-normal leading-normal min-w-40" href="pages/13_Logout.py">Log Out</a>
                </div>
                <p class="text-[#6a7781] text-base font-normal leading-normal">© 2025 G4 Pulse. All rights reserved.</p>
            </footer>
        </div>
    </footer>
    """

    full_html_content = f"""
    <html>
    <head>
        <link rel="preconnect" href="https://fonts.gstatic.com/" crossorigin="" />
        <link
            rel="stylesheet"
            as="style"
            onload="this.rel='stylesheet'"
            href="https://fonts.googleapis.com/css2?display=swap&amp;family=Noto+Sans%3Awght%40400%3B500%3B700%3B900&amp;family=Public+Sans%3Awght%40400%3B500%3B700%3B900"
        />

        <title>System Settings</title>
        <link rel="icon" type="image/x-icon" href="data:image/x-icon;base64," />

        <script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
        <style>
            /* Ensure the body uses your specified fonts, even within the iframe */
            body {{ font-family: "Public Sans", "Noto Sans", sans-serif; margin: 0; padding: 0; }}
            /* Removed min-h-screen from the outermost div in HTML to prevent internal scroll */
            .relative.flex.size-full.flex-col {{ min-height: 100vh; }} /* Re-added min-height in CSS for flexibility */

            /* Adjustments for width and padding within the embedded HTML */
            .main-content-area {{
                width: 100%; /* Ensure it takes full width */
                box-sizing: border-box; /* Include padding in width calculation */
                margin-left: auto; /* Center the content */
                margin-right: auto; /* Center the content */
                padding-left: 1rem; /* Default padding for smaller screens */
                padding-right: 1rem; /* Default padding for smaller screens */
            }}

            @media (min-width: 640px) {{ /* Tailwind's sm breakpoint */
                .main-content-area {{
                    padding-left: 2.5rem; /* Equivalent to px-10 */
                    padding-right: 2.5rem;
                }}
            }}

            @media (min-width: 1024px) {{ /* Tailwind's lg breakpoint */
                .main-content-area {{
                    padding-left: 10rem; /* Equivalent to px-40 */
                    padding-right: 10rem;
                }}
            }}

            .layout-content-container {{
                max-width: 960px; /* Constrain content width */
                width: 100%; /* Occupy full width within its parent */
                box-sizing: border-box; /* Include padding in width calculation */
            }}

            /* Adjust header padding to be consistent with content if desired */
            header.px-10.py-3 {{
                padding-left: 2.5rem !important;
                padding-right: 2.5rem !important;
            }}

            /* Ensure select button SVG is correctly rendered */
            [style*='--select-button-svg'] {{
                background-image: var(--select-button-svg);
            }}
        </style>
    </head>
    <body>
        <div class="relative flex size-full flex-col bg-white group/design-root overflow-x-hidden" style='font-family: "Public Sans", "Noto Sans", sans-serif;'>
            <div class="layout-container flex h-full grow flex-col">
                {header_html_content}
                <div class="flex flex-1 justify-center py-5 main-content-area">
                    <div class="layout-content-container flex flex-col flex-1 mx-auto">
                        <div class="flex flex-wrap justify-between gap-3 p-4"> <!-- Kept p-4 as it defines the section padding -->
                            <div class="flex min-w-72 flex-col gap-3">
                                <p class="text-[#121516] tracking-light text-[32px] font-bold leading-tight">System Settings</p>
                            </div>
                        </div>
                        <h2 class="text-[#121516] text-[22px] font-bold leading-tight tracking-[-0.015em] px-4 pb-3 pt-5">Automation Control</h2>
                        <div class="flex items-center gap-4 bg-white px-4 min-h-[72px] py-2 justify-between">
                            <div class="flex flex-col justify-center">
                                <p class="text-[#121516] text-base font-medium leading-normal line-clamp-1">Automated Risk Calculation</p>
                                <p class="text-[#6a7781] text-sm font-normal leading-normal line-clamp-2">Enable or disable automated risk calculation upon patient data entry.</p>
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
                        <div class="flex items-center gap-4 bg-white px-4 min-h-[72px] py-2 justify-between">
                            <div class="flex flex-col justify-center">
                                <p class="text-[#121516] text-base font-medium leading-normal line-clamp-1">High-Risk Alert Threshold</p>
                                <p class="text-[#6a7781] text-sm font-normal leading-normal line-clamp-2">Adjust the risk score threshold that triggers a 'high-risk' alert.</p>
                            </div>
                            <div class="shrink-0"><p class="text-[#121516] text-base font-normal leading-normal">15%</p></div>
                        </div>
                        <h2 class="text-[#121516] text-[22px] font-bold leading-tight tracking-[-0.015em] px-4 pb-3 pt-5">Criteria Customization</h2>
                        <div class="flex items-center gap-4 bg-white px-4 min-h-[72px] py-2 justify-between">
                            <div class="flex flex-col justify-center">
                                <p class="text-[#121516] text-base font-medium leading-normal line-clamp-1">Risk Factor Weighting</p>
                                <p class="text-[#6a7781] text-sm font-normal leading-normal line-clamp-2">Adjust the weighting or importance of certain risk factors.</p>
                            </div>
                            <div class="shrink-0">
                                <button
                                    class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-8 px-4 bg-[#f1f2f4] text-[#121516] text-sm font-medium leading-normal w-fit"
                                >
                                    <span class="truncate">Edit</span>
                                </button>
                            </div>
                        </div>
                        <div class="flex items-center gap-4 bg-white px-4 min-h-[72px] py-2 justify-between">
                            <div class="flex flex-col justify-center">
                                <p class="text-[#121516] text-base font-medium leading-normal line-clamp-1">Risk Factor Inclusion/Exclusion</p>
                                <p class="text-[#6a7781] text-sm font-normal leading-normal line-clamp-2">Include or exclude certain risk factors from the model.</p>
                            </div>
                            <div class="shrink-0">
                                <button
                                    class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-8 px-4 bg-[#f1f2f4] text-[#121516] text-sm font-medium leading-normal w-fit"
                                >
                                    <span class="truncate">Edit</span>
                                </button>
                            </div>
                        </div>
                        <h2 class="text-[#121516] text-[22px] font-bold leading-tight tracking-[-0.015em] px-4 pb-3 pt-5">User Interface Preferences</h2>
                        <div class="flex items-center gap-4 bg-white px-4 min-h-[72px] py-2 justify-between">
                            <div class="flex flex-col justify-center">
                                <p class="text-[#121516] text-base font-medium leading-normal line-clamp-1">Units of Measurement</p>
                                <p class="text-[#6a7781] text-sm font-normal leading-normal line-clamp-2">Select preferred units of measurement for height and weight.</p>
                            </div>
                            <div class="shrink-0"><p class="text-[#121516] text-base font-normal leading-normal">Metric</p></div>
                        </div>
                        <div class="flex items-center gap-4 bg-white px-4 min-h-[72px] py-2 justify-between">
                            <div class="flex flex-col justify-center">
                                <p class="text-[#121516] text-base font-medium leading-normal line-clamp-1">Risk Assessment Results Detail</p>
                                <p class="text-[#6a7781] text-sm font-normal leading-normal line-clamp-2">Adjust the level of detail displayed on the Risk Assessment Results page.</p>
                            </div>
                            <div class="shrink-0"><p class="text-[#121516] text-base font-normal leading-normal">Show All</p></div>
                        </div>
                        <div class="flex items-center gap-4 bg-white px-4 min-h-[72px] py-2 justify-between">
                            <div class="flex flex-col justify-center">
                                <p class="text-[#121516] text-base font-medium leading-normal line-clamp-1">Patient Records Management Display</p>
                                <p class="text-[#6a7781] text-sm font-normal leading-normal line-clamp-2">Set default sorting and filtering options for the Patient Records Management page.</p>
                            </div>
                            <div class="shrink-0">
                                <button
                                    class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-8 px-4 bg-[#f1f2f4] text-[#121516] text-sm font-medium leading-normal w-fit"
                                >
                                    <span class="truncate">Edit</span>
                                </button>
                            </div>
                        </div>
                        <h2 class="text-[#121516] text-[22px] font-bold leading-tight tracking-[-0.015em] px-4 pb-3 pt-5">General Settings</h2>
                        <div class="flex items-center gap-4 bg-white px-4 min-h-14 justify-between">
                            <p class="text-[#121516] text-base font-normal leading-normal flex-1 truncate">Change Password</p>
                            <div class="shrink-0">
                                <button
                                    class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-8 px-4 bg-[#f1f2f4] text-[#121516] text-sm font-medium leading-normal w-fit"
                                >
                                    <span class="truncate">Change</span>
                                </button>
                            </div>
                        </div>
                        <div class="flex items-center gap-4 bg-white px-4 min-h-14 justify-between">
                            <p class="text-[#121516] text-base font-normal leading-normal flex-1 truncate">Update Contact Information</p>
                            <div class="shrink-0">
                                <button
                                    class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-8 px-4 bg-[#f1f2f4] text-[#121516] text-sm font-medium leading-normal w-fit"
                                >
                                    <span class="truncate">Update</span>
                                </button>
                            </div>
                        </div>
                        <div class="flex items-center gap-4 bg-white px-4 min-h-14 justify-between">
                            <p class="text-[#121516] text-base font-normal leading-normal flex-1 truncate">Export Data</p>
                            <div class="shrink-0">
                                <button
                                    class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-8 px-4 bg-[#f1f2f4] text-[#121516] text-sm font-medium leading-normal w-fit"
                                >
                                    <span class="truncate">Export</span>
                                </button>
                            </div>
                        </div>
                        <div class="flex items-center gap-4 bg-white px-4 min-h-14 justify-between">
                            <p class="text-[#121516] text-base font-normal leading-normal flex-1 truncate">Delete Data</p>
                            <div class="shrink-0">
                                <button
                                    class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-8 px-4 bg-[#f1f2f4] text-[#121516] text-sm font-medium leading-normal w-fit"
                                >
                                    <span class="truncate">Delete</span>
                                </button>
                            </div>
                        </div>
                        <div class="flex items-center gap-4 bg-white px-4 min-h-14 justify-between">
                            <p class="text-[#121516] text-base font-normal leading-normal flex-1 truncate">System Updates and New Features</p>
                            <div class="shrink-0">
                                <button
                                    class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-8 px-4 bg-[#f1f2f4] text-[#121516] text-sm font-medium leading-normal w-fit"
                                >
                                    <span class="truncate">View</span>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                {footer_html_content}
            </div>
        </div>
    </body>
    </html>
    """
    st.components.v1.html(full_html_content, height=2500, scrolling=True) # Enabled scrolling to ensure all content is visible

# Call the function to render the page.
system_settings_page()
