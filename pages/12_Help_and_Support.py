import streamlit as st

def help_and_support_page():
    """
    Renders the Help and Support page.
    The header and footer HTML are now embedded directly within this page
    to avoid dependencies on app.py and resolve the reported errors.
    """
    # Define the header HTML content with links and profile image
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

    # Define the footer HTML content with updated links
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

        <title>Help and Support</title>
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
        </style>
    </head>
    <body>
        <div class="relative flex size-full min-h-screen flex-col bg-white group/design-root overflow-x-hidden" style='font-family: "Public Sans", "Noto Sans", sans-serif;'>
            <div class="layout-container flex h-full grow flex-col">
                {header_html_content} <!-- Header is now embedded directly -->
                <div class="flex flex-1 justify-center py-5 px-4">
                    <div class="layout-content-container flex flex-col flex-1">
                        <div class="flex flex-wrap justify-between gap-3 p-4">
                            <p class="text-[#121516] tracking-light text-[32px] font-bold leading-tight min-w-72">Help and Support</p>
                        </div>
                        <p class="text-[#121516] text-base font-normal leading-normal pb-3 pt-1 px-4">
                            Welcome to the StrokeRisk Help and Support page. Here you can find answers to common questions and resources to assist you.
                        </p>
                        <h3 class="text-[#121516] text-lg font-bold leading-tight tracking-[-0.015em] px-4 pb-2 pt-4">Frequently Asked Questions (FAQs)</h3>
                        <div class="px-4 py-3 flex flex-col gap-4">
                            <div class="border rounded-lg p-4">
                                <h4 class="text-[#121516] text-base font-bold leading-tight">How do I assess patient risk?</h4>
                                <p class="text-[#6a7781] text-sm font-normal leading-normal mt-2">
                                    Navigate to the "Patient Data Entry" page from the header or sidebar. Fill in the required patient demographic, medical, and lifestyle information, then click "Submit" to get a risk assessment.
                                </p>
                            </div>
                            <div class="border rounded-lg p-4">
                                <h4 class="text-[#121516] text-base font-bold leading-tight">Where can I view patient records?</h4>
                                <p class="text-[#6a7781] text-sm font-normal leading-normal mt-2">
                                    You can view and manage patient records on the "Manage Patient Records" page, accessible from the header or sidebar.
                                </p>
                            </div>
                            <div class="border rounded-lg p-4">
                                <h4 class="text-[#121516] text-base font-bold leading-tight">How accurate is the stroke risk prediction?</h4>
                                <p class="text-[#6a7781] text-sm font-normal leading-normal mt-2">
                                    Our model's performance metrics, including AUC-ROC and F1-Score, are detailed on the "Model Performance & Fairness" page. We strive for high accuracy and continuously refine the model.
                                </p>
                            </div>
                        </div>
                        <h3 class="text-[#121516] text-lg font-bold leading-tight tracking-[-0.015em] px-4 pb-2 pt-4">Contact Support</h3>
                        <p class="text-[#121516] text-base font-normal leading-normal pb-3 pt-1 px-4">
                            If you cannot find the answer to your question here, please do not hesitate to contact our support team.
                        </p>
                        <div class="flex px-4 py-3 justify-end">
                            <a href="pages/07_Contact_Us.py"
                                class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-10 px-4 bg-[#3f8abf] text-white text-sm font-bold leading-normal tracking-[0.015em]"
                            >
                                <span class="truncate">Contact Us</span>
                            </a>
                        </div>
                    </div>
                </div>
                {footer_html_content} <!-- Footer is now embedded directly -->
            </div>
        </div>
    </body>
    </html>
    """
    st.components.v1.html(full_html_content, height=1200, scrolling=True)

# Call the function to render the page.
help_and_support_page()
