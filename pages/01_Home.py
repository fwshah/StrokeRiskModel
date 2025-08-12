import streamlit as st
# Each page will manage its own full HTML content including header and footer.
import streamlit.components.v1 as components

def home_page():
    """
    Renders the Welcome Page (Home Page) of the StrokeRisk AI System
    by embedding the complete original HTML provided by the user using st.components.v1.html.
    This approach ensures exact visual fidelity as per the original design,
    now adjusted for full-width display and to eliminate the inner scrollbar.
    All navigation links within the HTML are updated to point to the correct Streamlit pages.
    """
    
    full_html_content = """
    <html>
    <head>
        <link rel="preconnect" href="https://fonts.gstatic.com/" crossorigin="" />
        <link
            rel="stylesheet"
            as="style"
            onload="this.rel='stylesheet'"
            href="https://fonts.googleapis.com/css2?display=swap&amp;family=Noto+Sans%3Awght%40400%3B500%3B700%3B900&amp;family=Public+Sans%3Awght%40400%3B500%3B700%3B900"
        />

        <title>Welcome to StrokeRisk</title>
        <link rel="icon" type="image/x-icon" href="data:image/x-icon;base64," />

        <script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
        <style>
            /* Ensure the body uses your specified fonts, even within the iframe */
            body { font-family: "Public Sans", "Noto Sans", sans-serif; margin: 0; padding: 0; }
            /* Ensure full height for the layout container within the iframe */
            /* Removed min-h-screen from the outermost div in HTML to prevent internal scroll */
            .relative.flex.size-full.flex-col { min-height: 100vh; } /* Re-added min-height in CSS for flexibility */

            /* --- ADJUSTMENTS FOR WIDTH (Removed conflicting classes from HTML directly) --- */
            /* The following CSS rules are now less critical as the classes are removed from HTML,
               but kept for robustness in case of other implicit styles. */
            .layout-content-container.flex.flex-col.flex-1 { /* Targeting the adjusted class */
                max-width: 100% !important; /* Ensure it takes full width */
                width: 100% !important; /* Explicitly set width to 100% */
                box-sizing: border-box; /* Include padding in width calculation */
            }

            /* Adjust padding for inner sections to maintain some spacing */
            /* For example, the hero section and the "Empowering Healthcare Providers" section */
            .px-4.py-10.\\@container {
                padding-left: 2.5rem !important; /* Equivalent to px-10 (40px) for inner content */
                padding-right: 2.5rem !important;
            }
            .px-10.py-3 { /* Header padding */
                padding-left: 2.5rem !important;
                padding-right: 2.5rem !important;
            }
            /* Adjust the main content wrapper padding to be more flexible */
            .flex.flex-1.py-5 { /* This was previously .px-40... and justify-center */
                padding-left: 2.5rem !important; /* Apply a standard padding */
                padding-right: 2.5rem !important;
            }
            /* Ensure the footer's max-width is also handled */
            footer .flex.flex-1.flex-col { /* Targeting the div that had max-w-[960px] */
                max-width: 100% !important;
                width: 100% !important;
                box-sizing: border-box;
            }
            /* Ensure buttons are responsive and don't overflow */
            .header-button { /* Custom class for header buttons */
                max-width: unset !important; /* Remove fixed max-width */
                flex-shrink: 1; /* Allow buttons to shrink if needed */
                min-width: 0; /* Allow content to shrink if needed */
            }
            .header-nav-links { /* Custom class for navigation links container */
                flex-grow: 1; /* Allow it to grow and take available space */
                flex-shrink: 0; /* Prevent it from shrinking too much */
            }
            .header-action-buttons { /* Custom class for action buttons container */
                flex-shrink: 0; /* Prevent it from shrinking too much */
            }
        </style>
    </head>
    <body>
        <div class="relative flex size-full flex-col bg-white group/design-root overflow-x-hidden" style='font-family: "Public Sans", "Noto Sans", sans-serif;'> <!-- Removed min-h-screen -->
            <div class="layout-container flex h-full grow flex-col">
                <header class="flex items-center justify-between border-b border-solid border-b-[#f1f2f4] px-10 py-3"> <!-- Removed whitespace-nowrap -->
                    <div class="flex items-center gap-4 text-[#121516]">
                        <div class="size-4">
                            <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path fill-rule="evenodd" clip-rule="evenodd" d="M24 4H6V17.3333V30.6667H24V44H42V30.6667V17.3333H24V4Z" fill="currentColor"></path>
                            </svg>
                        </div>
                        <h2 class="text-[#121516] text-lg font-bold leading-tight tracking-[-0.015em]">StrokeRisk</h2>
                    </div>
                    <div class="flex items-center justify-end gap-6 flex-wrap flex-1"> <!-- Consolidated and aligned right -->
                        <!-- Header Navigation Links -->
                        <a class="text-[#121516] text-sm font-medium leading-normal" href="pages/01_Home.py">Home</a>
                        <a class="text-[#121516] text-sm font-medium leading-normal" href="pages/07_About_Us.py">About Us</a>
                        <a class="text-[#121516] text-sm font-medium leading-normal" href="pages/06_Contact_Us.py">Contact Us</a>
                        
                        <!-- Practitioner Profile Icon (linked to 08_Practitioners_Profile.py) -->
                        <a href="pages/08_Practitioners_Profile.py" class="flex size-10 items-center justify-center rounded-full overflow-hidden">
                            <img src="https://lh3.googleusercontent.com/aida-public/AB6AXuBwkktHQpR_bDtX78dae6gQMFY5NXBfpOazltOY9n2KDjANnnbfMSvj1VWr6h9sdxif9ERNDC4qiRzYq3h5DAPM8-R7pLtxTDUn1yabMKiR-M9aKohcwYezZ-0kO1omt02RpBEHBmpq9JeThPsCa8WcsleL1TSS668g3P_dtzNLUOWlc2lRl_IzpdsrsvTBBl7ypv9s3rjUgLBazzzNUdhIp345xspze6IOTt0Cntw09qi6KJ913o7nc9nSNJdeuqd8gyzf5dsfw1k" alt="Practitioner Profile" class="w-full h-full object-cover">
                        </a>

                        <!-- Log Out Button -->
                        <a href="pages/13_Logout.py" class="flex min-w-[84px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-10 px-4 bg-[#3f8abf] text-white text-sm font-bold leading-normal tracking-[0.015em]">
                            <span class="truncate">Log Out</span>
                        </a>
                    </div>
                </header>
                <div class="flex flex-1 py-5"> <!-- Removed px-40 and justify-center -->
                    <div class="layout-content-container flex flex-col flex-1"> <!-- Removed max-w-[960px] -->
                        <div class="@container">
                            <div class="@[480px]:p-4">
                                <div
                                    class="flex min-h-[480px] flex-col gap-6 bg-cover bg-center bg-no-repeat @[480px]:gap-8 @[480px]:rounded-xl items-center justify-center p-4"
                                    style='background-image: linear-gradient(rgba(0, 0, 0, 0.1) 0%, rgba(0, 0, 0, 0.4) 100%), url("https://lh3.googleusercontent.com/aida-public/AB6AXuAN5KrEHd8bfSrxgjbyOWCfioIw_grSARtVuYHBt247tk_TDYty8310mBLzuuGwC6QFhrk9dzBtRVDAWMJifC65a7kfQmv1ok4IubaaKDSOPQIOXBVPZbhrIeY6axGKZqbHOEvKFV2TZ6FXFZ9_Oud_vcOjSmx7KaE7x0Ft6Lvk26kIywW_Reb3KZKbZYWHX_aUMhglJtc7E3wd0KjVJJJk4F7Z1LVvP6ATd6viCAQefB6FIyXgOwbI87VYWwJKZ2x4B5b0vu-gEzw");'
                                >
                                    <div class="flex flex-col gap-2 text-center">
                                        <h1
                                            class="text-white text-4xl font-black leading-tight tracking-[-0.033em] @[480px]:text-5xl @[480px]:font-black @[480px]:leading-tight @[480px]:tracking-[-0.033em]"
                                        >
                                            Predict Stroke Risk for Early Intervention
                                        </h1>
                                        <h2 class="text-white text-sm font-normal leading-normal @[480px]:text-base @[480px]:font-normal @[480px]:leading-normal">
                                            Enhance efficiency, gain data-driven insights, and support equitable care with our advanced AI tool.
                                        </h2>
                                    </div>
                                    <div class="flex-wrap gap-3 flex justify-center">
                                        <a href="pages/02_Patient_Data_Entry.py"
                                            class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-10 px-4 @[480px]:h-12 @[480px]:px-5 bg-[#3f8abf] text-white text-sm font-bold leading-normal tracking-[0.015em]"
                                        >
                                            <span class="truncate">Assess Patient Risk</span>
                                        </a>
                                        <a href="pages/05_Patient_Records.py"
                                            class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-10 px-4 bg-[#f1f2f4] text-[#121516] text-sm font-bold leading-normal tracking-[0.015em]"
                                        >
                                            <span class="truncate">Manage Patient Records</span>
                                        </a>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="flex flex-col gap-10 px-4 py-10 @container">
                            <div class="flex flex-col gap-4">
                                <h1
                                    class="text-[#121516] tracking-light text-[32px] font-bold leading-tight @[480px]:text-4xl @[480px]:font-black @[480px]:leading-tight @[480px]:tracking-[-0.033em] max-w-[720px]"
                                >
                                    Empowering Healthcare Providers
                                </h1>
                                <p class="text-[#121516] text-base font-normal leading-normal max-w-[720px]">
                                    StrokeRisk is designed to seamlessly integrate into your workflow, providing actionable insights to improve patient outcomes.
                                </p>
                            </div>
                            <div class="grid grid-cols-[repeat(auto-fit,minmax(158px,1fr))] gap-3 p-0">
                                <div class="flex flex-1 gap-3 rounded-lg border border-[#dde1e3] bg-white p-4 flex-col">
                                    <div class="text-[#121516]" data-icon="Heart" data-size="24px" data-weight="regular">
                                        <svg xmlns="http://www.w3.org/2000/svg" width="24px" height="24px" fill="currentColor" viewBox="0 0 256 256">
                                            <path
                                                d="M178,32c-20.65,0-38.73,8.88-50,23.89C116.73,40.88,98.65,32,78,32A62.07,62.07,0,0,0,16,94c0,70,103.79,126.66,108.21,129a8,8,0,0,0,7.58,0C136.21,220.66,240,164,240,94A62.07,62.07,0,0,0,178,32ZM128,206.8C109.74,196.16,32,147.69,32,94A46.06,46.06,0,0,1,78,48c19.45,0,35.78,10.36,42.6,27a8,8,0,0,0,14.8,0c6.82-16.67,23.15-27,42.6-27a46.06,46.06,0,0,1,46,46C224,147.61,146.24,196.15,128,206.8Z"
                                            ></path>
                                        </svg>
                                    </div>
                                    <div class="flex flex-col gap-1">
                                        <h2 class="text-[#121516] text-base font-bold leading-tight">Early Detection</h2>
                                        <p class="text-[#6a7781] text-sm font-normal leading-normal">Identify high-risk patients early, enabling timely interventions and potentially preventing strokes.</p>
                                    </div>
                                </div>
                                <div class="flex flex-1 gap-3 rounded-lg border border-[#dde1e3] bg-white p-4 flex-col">
                                    <div class="text-[#121516]" data-icon="ChartLine" data-size="24px" data-weight="regular">
                                        <svg xmlns="http://www.w3.org/2000/svg" width="24px" height="24px" fill="currentColor" viewBox="0 0 256 256">
                                            <path
                                                d="M232,208a8,8,0,0,1-8,8H32a8,8,0,0,1-8-8V48a8,8,0,0,1,16,0v94.37L90.73,98a8,8,0,0,1,10.07-.38l58.81,44.11L218.73,90a8,8,0,1,1,10.54,12l-64,56a8,8,0,0,1-10.07.38L96.39,114.29,40,163.63V200H224A8,8,0,0,1,232,208Z"
                                            ></path>
                                        </svg>
                                    </div>
                                    <div class="flex flex-col gap-1">
                                        <h2 class="text-[#121516] text-base font-bold leading-tight">Accurate Risk Assessment</h2>
                                        <p class="text-[#6a7781] text-sm font-normal leading-normal">
                                            Leverage advanced AI to accurately assess a patient's risk of stroke based on demographic, medical, and lifestyle data.
                                        </p>
                                    </div>
                                </div>
                                <div class="flex flex-1 gap-3 rounded-lg border border-[#dde1e3] bg-white p-4 flex-col">
                                    <div class="text-[#121516]" data-icon="UsersThree" data-size="24px" data-weight="regular">
                                        <svg xmlns="http://www.w3.org/2000/svg" width="24px" height="24px" fill="currentColor" viewBox="0 0 256 256">
                                            <path
                                                d="M244.8,150.4a8,8,0,0,1-11.2-1.6A51.6,51.6,0,0,0,192,128a8,8,0,0,1-7.37-4.89,8,8,0,0,1,0-6.22A8,8,0,0,1,192,112a24,24,0,1,0-23.24-30,8,8,0,1,1-15.5-4A40,40,0,1,1,219,117.51a67.94,67.94,0,0,1,27.43,21.68A8,8,0,0,1,244.8,150.4ZM190.92,212a8,8,0,1,1-13.84,8,57,57,0,0,0-98.16,0,8,8,0,1,1-13.84-8,72.06,72.06,0,0,1,33.74-29.92,48,48,0,1,1,58.36,0A72.06,72.06,0,0,1,190.92,212ZM128,176a32,32,0,1,0-32-32A32,32,0,0,0,128,176ZM72,120a8,8,0,0,0-8-8A24,24,0,1,1,87.24,82a8,8,0,1,0,15.5-4A40,40,0,1,0,37,117.51,67.94,67.94,0,0,0,9.6,139.19a8,8,0,1,0,12.8,9.61A51.6,51.6,0,0,1,64,128,8,8,0,0,0,72,120Z"
                                            ></path>
                                        </svg>
                                    </div>
                                    <div class="flex flex-col gap-1">
                                        <h2 class="text-[#121516] text-base font-bold leading-tight">Personalized Care</h2>
                                        <p class="text-[#6a7781] text-sm font-normal leading-normal">Tailor treatment plans to individual patient needs, optimizing care and improving outcomes.</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <footer class="flex justify-center">
                    <div class="flex flex-1 flex-col"> <!-- Removed max-w-[960px] -->
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
            </div>
        </div>
    </body>
    </html>
    """
    # Use st.components.v1.html to embed the full HTML content
    # st.components.v1.html(full_html_content, height=1350, scrolling=False) # Adjusted height to 1350px
    
    components.html(full_html_content, height=600)

# Call the function to render the home page when this script is run by Streamlit.
home_page()
