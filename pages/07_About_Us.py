import streamlit as st

def about_us_page():
    """
    Renders the About Us page by embedding the complete
    HTML content. The header and footer are made consistent with other pages,
    and all internal links are updated to reflect the correct page paths.
    """
    # Define the header HTML content (consistent with other pages)
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
        <div class="flex items-center justify-end gap-6 flex-wrap flex-1">
            <a class="text-[#121516] text-sm font-medium leading-normal" href="pages/01_Home.py">Home</a>
            <a class="text-[#121516] text-sm font-medium leading-normal" href="pages/06_Contact_Us.py">Contact Us</a>
            <a class="text-[#121516] text-sm font-medium leading-normal" href="pages/05_Patient_Records.py">Manage Patient Records</a>
            
            <a href="pages/08_Practitioners_Profile.py" class="flex size-10 items-center justify-center rounded-full overflow-hidden">
                <img src="https://lh3.googleusercontent.com/aida-public/AB6AXuBwkktHQpR_bDtX78dae6gQMFY5NXBfpOazltOY9n2KDjANnnbfMSvj1VWr6h9sdxif9ERNDC4qiRzYq3h5DAPM8-R7pLtxTDUn1yabMKiR-M9aKohcwYezZ-0kO1omt02RpBEHBmpq9JeThPsCa8WcsleL1TSS668g3P_dtzNLUOWlc2lRl_IzpdsrsvTBBl7ypv9s3rjUgLBazzzNUdhIp345xspze6IOTt0Cntw09qi6KJ913o7nc9nSNJdeuqd8gyzf5dsfw1k" alt="Practitioner Profile" class="w-full h-full object-cover">
            </a>

            <a href="pages/13_Logout.py" class="flex min-w-[84px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-10 px-4 bg-[#3f8abf] text-white text-sm font-bold leading-normal tracking-[0.015em]">
                <span class="truncate">Log Out</span>
            </a>
        </div>
    </header>
    """

    # Define the footer HTML content (consistent with previous pages)
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

        <title>About Us</title>
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

            header.px-10.py-3 {{
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }}
        </style>
    </head>
    <body>
        <div class="relative flex size-full min-h-screen flex-col bg-white group/design-root overflow-x-hidden" style='font-family: "Public Sans", "Noto Sans", sans-serif;'>
            <div class="layout-container flex h-full grow flex-col">
                {header_html_content}
                <div class="flex flex-1 justify-center py-5 px-4">
                    <div class="layout-content-container flex flex-col flex-1">
                        <div class="flex flex-wrap justify-between gap-3 p-4"><p class="text-[#121516] tracking-light text-[32px] font-bold leading-tight min-w-72">About Us</p></div>
                        <p class="text-[#121516] text-base font-normal leading-normal pb-3 pt-1 px-4">
                            StrokeRisk is a clinical decision support tool (CDST) developed by G4 Pulse, a team of four passionate individuals: Fuad, Preston, Marrium, and Femi James. United by
                            a shared vision of leveraging technology to improve healthcare outcomes, we embarked on this project to address the critical need for early and equitable stroke risk
                            assessment.
                        </p>
                        <h3 class="text-[#121516] text-lg font-bold leading-tight tracking-[-0.015em] px-4 pb-2 pt-4">Our Motivation</h3>
                        <p class="text-[#121516] text-base font-normal leading-normal pb-3 pt-1 px-4">
                            Driven by a user-centered design approach, we recognized the potential of AI to empower healthcare providers with actionable insights. StrokeRisk is the culmination
                            of our efforts to create a tool that seamlessly integrates into clinical workflows, providing personalized risk assessments based on demographic, medical, and
                            lifestyle data.
                        </p>
                        <h3 class="text-[#121516] text-lg font-bold leading-tight tracking-[-0.015em] px-4 pb-2 pt-4">Development Process</h3>
                        <ul class="list-disc list-inside text-[#121516] text-base font-normal leading-normal pb-3 pt-1 px-4">
                            <li><strong>Data Analysis:</strong> We meticulously analyzed a comprehensive dataset of patient information to identify key risk factors for stroke.</li>
                            <li><strong>Model Selection:</strong> We carefully evaluated various machine learning models, selecting the one that demonstrated the highest accuracy and reliability.</li>
                            <li><strong>Rigorous Testing:</strong> Our model underwent extensive testing and validation to ensure its performance and robustness across diverse patient populations.</li>
                            <li><strong>Ethical Considerations:</strong> We prioritized fairness and transparency throughout the development process, employing techniques to mitigate bias and ensure responsible AI practices.</li>
                        </ul>
                        <h3 class="text-[#121516] text-lg font-bold leading-tight tracking-[-0.015em] px-4 pb-2 pt-4">Our Commitment</h3>
                        <p class="text-[#121516] text-base font-normal leading-normal pb-3 pt-1 px-4">
                            At G4 Pulse, we are committed to transparency, fairness, and accountability. We have implemented rigorous audits and evaluations to ensure that StrokeRisk operates
                            ethically and responsibly. Our dedication to these principles is reflected in the design and functionality of the tool.
                        </p>
                        <div class="flex justify-stretch">
                            <div class="flex flex-1 gap-3 flex-wrap px-4 py-3 justify-start">
                                <a href="pages/04_Model_Performance.py"
                                    class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-10 px-4 bg-[#3f8abf] text-white text-sm font-bold leading-normal tracking-[0.015em]"
                                >
                                    <span class="truncate">Model Performance/Metrics</span>
                                </a>
                                <a href="pages/06_Contact_Us.py"
                                    class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-10 px-4 bg-[#f1f2f4] text-[#121516] text-sm font-bold leading-normal tracking-[0.015em]"
                                >
                                    <span class="truncate">Contact Us</span>
                                </a>
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
    # Use st.components.v1.html to embed the full HTML content
    # Set height to a reasonable value and enable scrolling.
    st.components.v1.html(full_html_content, height=1200, scrolling=True) # Adjusted height for this page

# Call the function to render the page.
about_us_page()
