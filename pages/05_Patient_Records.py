import streamlit as st

def patient_records_page():
    """
    Renders the Patient Records Management page by embedding the complete
    original HTML provided by the user using st.components.v1.html.
    Adjustments have been made to the HTML's Tailwind classes and internal CSS
    to ensure full-width rendering within the Streamlit application.
    Includes a dynamic header and footer.
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
            <a class="text-[#121516] text-sm font-medium leading-normal" href="pages/06_Contact_Us.py">Contact Us</a>
            
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

        <title>Patient Records Management</title>
        <link rel="icon" type="image/x-icon" href="data:image/x-icon;base64," />

        <script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
        <style>
            /* Ensure the body uses your specified fonts, even within the iframe */
            body {{ font-family: "Public Sans", "Noto Sans", sans-serif; margin: 0; padding: 0; }}
            /* Ensure full height for the layout container within the iframe */
            .relative.flex.size-full.min-h-screen.flex-col {{ min-height: 100vh; }}

            /* Adjustments for width and padding within the embedded HTML */
            .flex.flex-1.justify-center.py-5 {{
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }}

            .layout-content-container.flex.flex-col.flex-1 {{
                max-width: 100% !important;
                width: 100% !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }}

            header.px-10.py-3 {{
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }}

            .table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-120,
            .table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-240,
            .table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-360,
            .table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-480,
            .table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-600 {{
                width: auto !important;
            }}
            table {{
                width: 100%;
                table-layout: fixed;
            }}
            /* Escaped curly braces for @container rules */
            @container(max-width:120px){{.table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-120{{display: none;}}}}
            @container(max-width:240px){{.table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-240{{display: none;}}}}
            @container(max-width:360px){{.table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-360{{display: none;}}}}
            @container(max-width:480px){{.table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-480{{display: none;}}}}
            @container(max-width:600px){{.table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-600{{display: none;}}}}
        </style>
    </head>
    <body>
        <div class="relative flex size-full min-h-screen flex-col bg-white group/design-root overflow-x-hidden" style='font-family: "Public Sans", "Noto Sans", sans-serif;'>
            <div class="layout-container flex h-full grow flex-col">
                {header_html_content}
                <div class="flex flex-1 justify-center py-5 px-4">
                    <div class="layout-content-container flex flex-col flex-1">
                        <div class="flex flex-wrap justify-between gap-3 p-4">
                            <div class="flex min-w-72 flex-col gap-3">
                                <p class="text-[#121516] tracking-light text-[32px] font-bold leading-tight">Patient Records Management</p>
                                <p class="text-[#6a7781] text-sm font-normal leading-normal">
                                    Efficiently manage and update patient information to ensure accurate risk assessments and personalized care.
                                </p>
                            </div>
                        </div>
                        <div class="px-4 py-3">
                            <label class="flex flex-col min-w-40 h-12 w-full">
                                <div class="flex w-full flex-1 items-stretch rounded-xl h-full">
                                    <div
                                        class="text-[#6a7781] flex border-none bg-[#f1f2f4] items-center justify-center pl-4 rounded-l-xl border-r-0"
                                        data-icon="MagnifyingGlass"
                                        data-size="24px"
                                        data-weight="regular"
                                    >
                                        <svg xmlns="http://www.w3.org/2000/svg" width="24px" height="24px" fill="currentColor" viewBox="0 0 256 256">
                                            <path
                                                d="M229.66,218.34l-50.07-50.06a88.11,88.11,0,1,0-11.31,11.31l50.06,50.07a8,8,0,0,0,11.32-11.32ZM40,112a72,72,0,1,1,72,72A72.08,72.08,0,0,1,40,112Z"
                                            ></path>
                                        </svg>
                                    </div>
                                    <input
                                        placeholder="Search by name, assessment date, or risk level"
                                        class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border-none bg-[#f1f2f4] focus:border-none h-full placeholder:text-[#6a7781] px-4 rounded-l-none border-l-0 pl-2 text-base font-normal leading-normal"
                                        value=""
                                    />
                                </div>
                            </label>
                        </div>
                        <div class="px-4 py-3 @container">
                            <div class="flex overflow-hidden rounded-xl border border-[#dde1e3] bg-white">
                                <table class="flex-1">
                                    <thead>
                                        <tr class="bg-white">
                                            <th class="table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-120 px-4 py-3 text-left text-[#121516] w-[400px] text-sm font-medium leading-normal">
                                                Patient Name
                                            </th>
                                            <th class="table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-240 px-4 py-3 text-left text-[#121516] w-[400px] text-sm font-medium leading-normal">Age</th>
                                            <th class="table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-360 px-4 py-3 text-left text-[#121516] w-[400px] text-sm font-medium leading-normal">
                                                Predicted Risk Score
                                            </th>
                                            <th class="table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-480 px-4 py-3 text-left text-[#121516] w-[400px] text-sm font-medium leading-normal">
                                                Medical History Summary
                                            </th>
                                            <th class="table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-600 px-4 py-3 text-left text-[#121516] w-60 text-[#6a7781] text-sm font-medium leading-normal">
                                                Actions
                                            </th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr class="border-t border-t-[#dde1e3]">
                                            <td class="table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-120 h-[72px] px-4 py-2 w-[400px] text-[#121516] text-sm font-normal leading-normal">
                                                Sophia Carter
                                            </td>
                                            <td class="table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-240 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">62</td>
                                            <td class="table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-360 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">
                                                High (85%)
                                            </td>
                                            <td class="table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-480 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">
                                                Hypertension, Atrial Fibrillation
                                            </td>
                                            <td class="table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-600 h-[72px] px-4 py-2 w-60 text-[#6a7781] text-sm font-bold leading-normal tracking-[0.015em]">
                                                <a href="pages/09_Client_Profile.py" class="text-[#3f8abf] hover:underline">View Full Profile</a>
                                            </td>
                                        </tr>
                                        <tr class="border-t border-t-[#dde1e3]">
                                            <td class="table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-120 h-[72px] px-4 py-2 w-[400px] text-[#121516] text-sm font-normal leading-normal">
                                                Ethan Bennett
                                            </td>
                                            <td class="table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-240 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">55</td>
                                            <td class="table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-360 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">
                                                Moderate (50%)
                                            </td>
                                            <td class="table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-480 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">
                                                Diabetes, High Cholesterol
                                            </td>
                                            <td class="table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-600 h-[72px] px-4 py-2 w-60 text-[#6a7781] text-sm font-bold leading-normal tracking-[0.015em]">
                                                <a href="pages/09_Client_Profile.py" class="text-[#3f8abf] hover:underline">View Full Profile</a>
                                            </td>
                                        </tr>
                                        <tr class="border-t border-t-[#dde1e3]">
                                            <td class="table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-120 h-[72px] px-4 py-2 w-[400px] text-[#121516] text-sm font-normal leading-normal">
                                                Olivia Davis
                                            </td>
                                            <td class="table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-240 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">70</td>
                                            <td class="table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-360 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">
                                                Very High (92%)
                                            </td>
                                            <td class="table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-480 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">
                                                Previous Stroke, Hypertension
                                            </td>
                                            <td class="table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-600 h-[72px] px-4 py-2 w-60 text-[#6a7781] text-sm font-bold leading-normal tracking-[0.015em]">
                                                <a href="pages/09_Client_Profile.py" class="text-[#3f8abf] hover:underline">View Full Profile</a>
                                            </td>
                                        </tr>
                                        <tr class="border-t border-t-[#dde1e3]">
                                            <td class="table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-120 h-[72px] px-4 py-2 w-[400px] text-[#121516] text-sm font-normal leading-normal">
                                                Liam Foster
                                            </td>
                                            <td class="table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-240 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">48</td>
                                            <td class="table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-360 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">
                                                Low (15%)
                                            </td>
                                            <td class="table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-480 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">None</td>
                                            <td class="table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-600 h-[72px] px-4 py-2 w-60 text-[#6a7781] text-sm font-bold leading-normal tracking-[0.015em]">
                                                <a href="pages/09_Client_Profile.py" class="text-[#3f8abf] hover:underline">View Full Profile</a>
                                            </td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                            <style>
                                @container(max-width:120px){{.table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-120{{display: none;}}}}
                                @container(max-width:240px){{.table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-240{{display: none;}}}}
                                @container(max-width:360px){{.table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-360{{display: none;}}}}
                                @container(max-width:480px){{.table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-480{{display: none;}}}}
                                @container(max-width:600px){{.table-8d9121e1-67a3-4ae6-8b81-745fb175b74f-column-600{{display: none;}}}}
                            </style>
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
patient_records_page()
