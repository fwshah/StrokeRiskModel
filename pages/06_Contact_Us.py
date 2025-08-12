import streamlit as st

def contact_us_page():
    """
    Renders the Contact Us page by embedding the complete
    original HTML provided by the user using st.components.v1.html.
    Adjustments have been made to the HTML's Tailwind classes and internal CSS
    to ensure full-width rendering within the Streamlit application and
    to eliminate the inner scrollbar.
    The header and footer are made consistent with other pages,
    and the FAQ section answers are populated.
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
            <a class="text-[#121516] text-sm font-medium leading-normal" href="pages/07_About_Us.py">About Us</a>
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

        <title>Contact Us</title>
        <link rel="icon" type="image/x-icon" href="data:image/x-icon;base64," />

        <script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
        <style>
            /* Ensure the body uses your specified fonts, even within the iframe */
            body {{ font-family: "Public Sans", "Noto Sans", sans-serif; margin: 0; padding: 0; }}
            /* Ensure full height for the layout container within the iframe */
            /* Removed min-h-screen from the outermost div in HTML to prevent internal scroll */
            .relative.flex.size-full.flex-col {{ min-height: 100vh; }} /* Re-added min-height to ensure it stretches if content is short */

            /* Adjustments for width and padding within the embedded HTML */
            /* This targets the main content wrapper */
            .main-content-area {{ /* Custom class for clarity and to ensure full width */
                padding-left: 1rem; /* Consistent padding */
                padding-right: 1rem; /* Consistent padding */
                width: 100%; /* Ensure it takes full width */
                box-sizing: border-box; /* Include padding in width calculation */
            }}

            /* This targets the inner content container */
            .layout-content-container {{
                max-width: 100%; /* Allow it to expand fully */
                width: 100%; /* Explicitly set width to 100% */
                box-sizing: border-box; /* Include padding in width calculation */
            }}

            /* Adjust header padding to be consistent with content if desired */
            header.px-10.py-3 {{
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }}

            /* Ensure form fields take full width of their immediate parent */
            .form-field-wrapper {{ /* Custom class for clarity */
                width: 100%; /* Explicitly set width to 100% */
                box-sizing: border-box; /* Include padding in width calculation */
            }}
            label.flex.flex-col.min-w-40.flex-1 {{
                min-width: unset !important; /* Remove min-width constraint if it causes issues on small screens */
            }}
        </style>
    </head>
    <body>
        <div class="relative flex size-full flex-col bg-white group/design-root overflow-x-hidden" style='font-family: "Public Sans", "Noto Sans", sans-serif;'> <!-- Removed min-h-screen -->
            <div class="layout-container flex h-full grow flex-col">
                {header_html_content}
                <div class="flex flex-1 py-5 main-content-area"> <!-- Removed px-40 and justify-center, added main-content-area for consistent padding -->
                    <div class="layout-content-container flex flex-col flex-1"> <!-- Removed max-w-[960px] -->
                        <div class="flex flex-wrap justify-between gap-3 p-4">
                            <div class="flex min-w-72 flex-col gap-3">
                                <p class="text-[#121516] tracking-light text-[32px] font-bold leading-tight">Contact Us</p>
                                <p class="text-[#6a7781] text-sm font-normal leading-normal">We're here to help. Reach out with any questions or feedback.</p>
                            </div>
                        </div>
                        <div class="flex flex-wrap items-end gap-4 py-3 form-field-wrapper"> <!-- Removed max-w-[480px], removed px-4 -->
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">Your Name</p>
                                <input
                                    placeholder="Enter your name"
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                    value=""
                                />
                            </label>
                        </div>
                        <div class="flex flex-wrap items-end gap-4 py-3 form-field-wrapper"> <!-- Removed max-w-[480px], removed px-4 -->
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">Email Address</p>
                                <input
                                    placeholder="Enter your email"
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                    value=""
                                />
                            </label>
                        </div>
                        <div class="flex flex-wrap items-end gap-4 py-3 form-field-wrapper"> <!-- Removed max-w-[480px], removed px-4 -->
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">Subject</p>
                                <input
                                    placeholder="Enter the subject"
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                    value=""
                                />
                            </label>
                        </div>
                        <div class="flex flex-wrap items-end gap-4 py-3 form-field-wrapper"> <!-- Removed max-w-[480px], removed px-4 -->
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">Message</p>
                                <textarea
                                    placeholder="Enter your message"
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] min-h-36 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                ></textarea>
                            </label>
                        </div>
                        <div class="flex px-4 py-3 justify-end">
                            <button
                                class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-10 px-4 bg-[#3f8abf] text-white text-sm font-bold leading-normal tracking-[0.015em]"
                            >
                                <span class="truncate">Submit</span>
                            </button>
                        </div>
                        <h2 class="text-[#121516] text-[22px] font-bold leading-tight tracking-[-0.015em] px-4 pb-3 pt-5">Frequently Asked Questions</h2>
                        <div class="flex flex-col p-4 gap-3">
                            <details class="flex flex-col rounded-xl border border-[#dde1e3] bg-white px-[15px] py-[7px] group">
                                <summary class="flex cursor-pointer items-center justify-between gap-6 py-2">
                                    <p class="text-[#121516] text-sm font-medium leading-normal">How does StrokeRisk work?</p>
                                    <div class="text-[#121516] group-open:rotate-180" data-icon="CaretDown" data-size="20px" data-weight="regular">
                                        <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" fill="currentColor" viewBox="0 0 256 256">
                                            <path d="M213.66,101.66l-80,80a8,8,0,0,1-11.32,0l-80-80A8,8,0,0,1,53.66,90.34L128,164.69l74.34-74.35a8,8,0,0,1,11.32,11.32Z"></path>
                                        </svg>
                                    </div>
                                </summary>
                                <p class="text-[#6a7781] text-sm font-normal leading-normal pb-2">
                                    StrokeRisk is an advanced AI-powered tool that predicts stroke risk based on a patient's demographic, medical history, and lifestyle factors. 
                                    It helps healthcare providers identify high-risk individuals for early intervention and personalized care planning.
                                </p>
                            </details>
                            <details class="flex flex-col rounded-xl border border-[#dde1e3] bg-white px-[15px] py-[7px] group">
                                <summary class="flex cursor-pointer items-center justify-between gap-6 py-2">
                                    <p class="text-[#121516] text-sm font-medium leading-normal">Is my data secure?</p>
                                    <div class="text-[#121516] group-open:rotate-180" data-icon="CaretDown" data-size="20px" data-weight="regular">
                                        <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" fill="currentColor" viewBox="0 0 256 256">
                                            <path d="M213.66,101.66l-80,80a8,8,0,0,1-11.32,0l-80-80A8,8,0,0,1,53.66,90.34L128,164.69l74.34-74.35a8,8,0,0,1,11.32,11.32Z"></path>
                                        </svg>
                                    </div>
                                </summary>
                                <p class="text-[#6a7781] text-sm font-normal leading-normal pb-2">
                                    Yes, data security is our top priority. StrokeRisk employs industry-standard encryption protocols and complies with relevant data privacy regulations 
                                    (e.g., HIPAA, GDPR, PHIPA) to ensure all patient data is protected and handled with the utmost confidentiality. Access is strictly controlled and monitored.
                                </p>
                            </details>
                            <details class="flex flex-col rounded-xl border border-[#dde1e3] bg-white px-[15px] py-[7px] group">
                                <summary class="flex cursor-pointer items-center justify-between gap-6 py-2">
                                    <p class="text-[#121516] text-sm font-medium leading-normal">How accurate is the risk assessment?</p>
                                    <div class="text-[#121516] group-open:rotate-180" data-icon="CaretDown" data-size="20px" data-weight="regular">
                                        <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" fill="currentColor" viewBox="0 0 256 256">
                                            <path d="M213.66,101.66l-80,80a8,8,0,0,1-11.32,0l-80-80A8,8,0,0,1,53.66,90.34L128,164.69l74.34-74.35a8,8,0,0,1,11.32,11.32Z"></path>
                                        </svg>
                                    </div>
                                </summary>
                                <p class="text-[#6a7781] text-sm font-normal leading-normal pb-2">
                                    StrokeRisk utilizes a robust AI model with high predictive accuracy, as demonstrated by its performance metrics like AUC-ROC and F1-Score, detailed on our 
                                    <a href="pages/04_Model_Performance.py" class="text-[#3f8abf] hover:underline">Model Performance & Fairness page</a>. While it's a powerful predictive tool, 
                                    it should always be used in conjunction with a healthcare professional's clinical judgment.
                                </p>
                            </details>
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
    st.components.v1.html(full_html_content, height=2500, scrolling=False) # Adjusted height and scrolling

# Call the function to render the page.
contact_us_page()
