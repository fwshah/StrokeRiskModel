import streamlit as st

def client_profile_page():
    """
    Renders the Patient Profile Page by embedding the provided HTML content.
    Adjustments have been made to the HTML's Tailwind classes and internal CSS
    to ensure full-width rendering within the Streamlit application and
    to eliminate unnecessary scrollbars, providing a balanced and professional look.
    The header and footer are consistent with other pages, and new sections
    for next appointments and data privacy have been added.
    """
    # Define the header HTML content (consistent with other pages and the trick from patient_records_page)
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
            <a class="text-[#121516] text-sm font-medium leading-normal" href="pages/02_Patient_Data_Entry.py">New Assessment</a>
            <a class="text-[#121516] text-sm font-medium leading-normal" href="pages/05_Patient_Records.py">Manage Patient Records</a>
            
            <a href="pages/08_Practitioners_Profile.py" class="flex size-10 items-center justify-center rounded-full overflow-hidden">
                <div class="bg-center bg-no-repeat aspect-square bg-cover rounded-full size-10"
                    style='background-image: url("https://lh3.googleusercontent.com/aida-public/AB6AXuA9gTOOng2h9CRvl4kmpW1HtpXANHXCK5YLJbnN3_iFsuUrTppj2BtN-JCK9JkOkqffzR3APkZZecXbNG0PsMau39KsmGpjK4dvzc8njIFJG0yklITysB__3-LygBEB99hmdIsUGq2nxFHmTnnCkwVJK4NFFfHg59rS_AbEl2LRWh20IM2V1_zAQ7vI3OszQwORjGRqM8wr5kEll92x8URmGDIz3mS9AHvBxlt2CPj4dKQJcrN84XS1ZOynwreVTAFsw8si4blRjyU");'>
                </div>
            </a>

            <a href="pages/13_Logout.py" class="flex min-w-[84px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-10 px-4 bg-[#3f8abf] text-white text-sm font-bold leading-normal tracking-[0.015em]">
                <span class="truncate">Log Out</span>
            </a>
        </div>
    </header>
    """

    # Define the footer HTML content (consistent with previous pages and the trick from patient_records_page)
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

        <title>Patient Profile</title>
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

            /* Ensure table columns are responsive */
            @container(max-width:120px){{.table-536416ac-2299-4c89-b5ee-2e632543193e-column-120{{display: none;}}}}
            @container(max-width:240px){{.table-536416ac-2299-4c89-b5ee-2e632543193e-column-240{{display: none;}}}}
            @container(max-width:360px){{.table-536416ac-2299-4c89-b5ee-2e632543193e-column-360{{display: none;}}}}
            @container(max-width:480px){{.table-536416ac-2299-4c89-b5ee-2e632543193e-column-480{{display: none;}}}}
            @container(max-width:600px){{.table-536416ac-2299-4c89-b5ee-2e632543193e-column-600{{display: none;}}}}

            /* Ensure select button SVG is correctly rendered */
            [style*='--select-button-svg'] {{
                background-image: var(--select-button-svg);
            }}

            /* Adjust footer position for narrow view */
            @media (max-width: 767px) {{ /* Adjust breakpoint as needed for "narrow view" */
                footer.mt-8 {{
                    margin-top: 2rem; /* Consistent margin */
                    padding-bottom: 1rem; /* Add some padding at the bottom */
                }}
            }}
        </style>
    </head>
    <body>
        <div class="relative flex size-full flex-col bg-white group/design-root overflow-x-hidden" style='font-family: "Public Sans", "Noto Sans", sans-serif;'>
            <div class="layout-container flex h-full grow flex-col">
                {header_html_content}
                <div class="flex flex-1 justify-center py-5 main-content-area">
                    <div class="layout-content-container flex flex-col flex-1 mx-auto">
                        <div class="flex flex-wrap justify-between gap-3 py-4">
                            <div class="flex min-w-72 flex-col gap-3">
                                <p class="text-[#121516] tracking-light text-[32px] font-bold leading-tight">Patient Profile</p>
                                <p class="text-[#6a7781] text-sm font-normal leading-normal">Comprehensive view of patient's medical data</p>
                            </div>
                        </div>
                        <h2 class="text-[#121516] text-[22px] font-bold leading-tight tracking-[-0.015em] pb-3 pt-5">Biographical Information</h2>
                        <div class="flex max-w-[480px] flex-wrap items-end gap-4 py-3">
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">Full Name</p>
                                <input
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                    placeholder="John Doe"
                                />
                            </label>
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">Date of Birth</p>
                                <input
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                    placeholder="1980-05-20"
                                />
                            </label>
                        </div>
                        <div class="flex max-w-[480px] flex-wrap items-end gap-4 py-3">
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">Gender</p>
                                <input
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                    placeholder="Male"
                                />
                            </label>
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">Marital Status</p>
                                <input
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                    placeholder="Married"
                                />
                            </label>
                        </div>
                        <div class="flex max-w-[480px] flex-wrap items-end gap-4 py-3">
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">Contact Information</p>
                                <input
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                    placeholder="john.doe@example.com"
                                />
                            </label>
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">Address</p>
                                <input
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                    placeholder="123 Main St, Anytown, USA"
                                />
                            </label>
                        </div>
                        <div class="flex max-w-[480px] flex-wrap items-end gap-4 py-3">
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">Number of Children</p>
                                <input
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                    placeholder="2"
                                />
                            </label>
                        </div>
                        <h2 class="text-[#121516] text-[22px] font-bold leading-tight tracking-[-0.015em] pb-3 pt-5">Lifestyle Factors</h2>
                        <div class="flex max-w-[480px] flex-wrap items-end gap-4 py-3">
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">Height (cm)</p>
                                <input
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                    placeholder="175"
                                />
                            </label>
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">Weight (kg)</p>
                                <input
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                    placeholder="70"
                                />
                            </label>
                        </div>
                        <div class="flex max-w-[480px] flex-wrap items-end gap-4 py-3">
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">BMI</p>
                                <input
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                    placeholder="22.86"
                                />
                            </label>
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">Smoking Status</p>
                                <input
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                    placeholder="Non-smoker"
                                />
                            </label>
                        </div>
                        <div class="flex max-w-[480px] flex-wrap items-end gap-4 py-3">
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">Alcohol Consumption</p>
                                <input
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                    placeholder="Occasional"
                                />
                            </label>
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">Physical Activity</p>
                                <input
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                    placeholder="Moderate (3-4 times/week)"
                                />
                            </label>
                        </div>
                        <div class="flex max-w-[480px] flex-wrap items-end gap-4 py-3">
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">Dietary Habits</p>
                                <input
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                    placeholder="Balanced"
                                />
                            </label>
                        </div>
                        <h2 class="text-[#121516] text-[22px] font-bold leading-tight tracking-[-0.015em] pb-3 pt-5">Medical History</h2>
                        <div class="flex max-w-[480px] flex-wrap items-end gap-4 py-3">
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">Pre-existing Conditions</p>
                                <input
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                    placeholder="Hypertension, Type 2 Diabetes"
                                />
                            </label>
                        </div>
                        <div class="flex max-w-[480px] flex-wrap items-end gap-4 py-3">
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">Current Medications</p>
                                <input
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                    placeholder="Lisinopril, Metformin"
                                />
                            </label>
                        </div>
                        <div class="flex max-w-[480px] flex-wrap items-end gap-4 py-3">
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">Allergies and Adverse Reactions</p>
                                <input
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                    placeholder="Penicillin"
                                />
                            </label>
                        </div>
                        <div class="flex max-w-[480px] flex-wrap items-end gap-4 py-3">
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">Past Surgeries and Hospitalizations</p>
                                <input
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                    placeholder="Appendectomy (2005)"
                                />
                            </label>
                        </div>
                        <h2 class="text-[#121516] text-[22px] font-bold leading-tight tracking-[-0.015em] pb-3 pt-5">Record of Past Assessments</h2>
                        <div class="py-3 @container">
                            <div class="flex overflow-hidden rounded-xl border border-[#dde1e3] bg-white">
                                <table class="flex-1">
                                    <thead>
                                        <tr class="bg-white">
                                            <th class="px-4 py-3 text-left text-[#121516] w-[15%] text-sm font-medium leading-normal">Date</th>
                                            <th class="px-4 py-3 text-left text-[#121516] w-[10%] text-sm font-medium leading-normal">
                                                Risk Score
                                            </th>
                                            <th class="px-4 py-3 text-left text-[#121516] w-[15%] text-sm font-medium leading-normal">
                                                Risk Level
                                            </th>
                                            <th class="px-4 py-3 text-left text-[#121516] w-[25%] text-sm font-medium leading-normal">
                                                Key Factors
                                            </th>
                                            <th class="px-4 py-3 text-left text-[#121516] w-[15%] text-sm font-medium leading-normal">Last MedicID</th>
                                            <th class="px-4 py-3 text-left text-[#121516] w-[20%] text-sm font-medium leading-normal">
                                                Action
                                            </th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr class="border-t border-t-[#dde1e3]">
                                            <td class="h-[72px] px-4 py-2 text-[#6a7781] text-sm font-normal leading-normal">
                                                2023-08-15
                                            </td>
                                            <td class="h-[72px] px-4 py-2 text-[#6a7781] text-sm font-normal leading-normal">12%</td>
                                            <td class="h-[72px] px-4 py-2 text-sm font-normal leading-normal">
                                                <button
                                                    class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-8 px-4 bg-yellow-500 text-white text-sm font-medium leading-normal w-full"
                                                >
                                                    <span class="truncate">Moderate</span>
                                                </button>
                                            </td>
                                            <td class="h-[72px] px-4 py-2 text-[#6a7781] text-sm font-normal leading-normal">
                                                Age, Hypertension, Family History
                                            </td>
                                            <td class="h-[72px] px-4 py-2 text-[#6a7781] text-sm font-normal leading-normal">AC1234</td>
                                            <td class="h-[72px] px-4 py-2 text-[#3f8abf] text-sm font-bold leading-normal tracking-[0.015em]">
                                                <a href="#" class="hover:underline">View Assessment</a>
                                            </td>
                                        </tr>
                                        <tr class="border-t border-t-[#dde1e3]">
                                            <td class="h-[72px] px-4 py-2 text-[#6a7781] text-sm font-normal leading-normal">
                                                2022-02-20
                                            </td>
                                            <td class="h-[72px] px-4 py-2 text-[#6a7781] text-sm font-normal leading-normal">8%</td>
                                            <td class="h-[72px] px-4 py-2 text-sm font-normal leading-normal">
                                                <button
                                                    class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-8 px-4 bg-green-500 text-white text-sm font-medium leading-normal w-full"
                                                >
                                                    <span class="truncate">Low</span>
                                                </button>
                                            </td>
                                            <td class="h-[72px] px-4 py-2 text-[#6a7781] text-sm font-normal leading-normal">
                                                Age, Family History
                                            </td>
                                            <td class="h-[72px] px-4 py-2 text-[#6a7781] text-sm font-normal leading-normal">SM5678</td>
                                            <td class="h-[72px] px-4 py-2 text-[#3f8abf] text-sm font-bold leading-normal tracking-[0.015em]">
                                                <a href="#" class="hover:underline">View Assessment</a>
                                            </td>
                                        </tr>
                                        <tr class="border-t border-t-[#dde1e3]">
                                            <td class="h-[72px] px-4 py-2 text-[#6a7781] text-sm font-normal leading-normal">
                                                2021-11-10
                                            </td>
                                            <td class="h-[72px] px-4 py-2 text-sm font-normal leading-normal">25%</td>
                                            <td class="h-[72px] px-4 py-2 text-sm font-normal leading-normal">
                                                <button
                                                    class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-8 px-4 bg-red-500 text-white text-sm font-medium leading-normal w-full"
                                                >
                                                    <span class="truncate">High</span>
                                                </button>
                                            </td>
                                            <td class="h-[72px] px-4 py-2 text-[#6a7781] text-sm font-normal leading-normal">
                                                Age, Hypertension, High Cholesterol, Smoking
                                            </td>
                                            <td class="h-[72px] px-4 py-2 text-[#6a7781] text-sm font-normal leading-normal">DR9012</td>
                                            <td class="h-[72px] px-4 py-2 text-[#3f8abf] text-sm font-bold leading-normal tracking-[0.015em]">
                                                <a href="#" class="hover:underline">View Assessment</a>
                                            </td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        <h2 class="text-[#121516] text-[22px] font-bold leading-tight tracking-[-0.015em] pb-3 pt-5">Next Appointments & Referrals</h2>
                        <div class="flex max-w-[480px] flex-wrap items-end gap-4 py-3">
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">Next Appointment Date</p>
                                <input
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                    placeholder="2024-09-01"
                                />
                            </label>
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">Referred to Specialist (MedicID)</p>
                                <input
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                    placeholder="SP9876 (Cardiologist)"
                                />
                            </label>
                        </div>

                        <h2 class="text-[#121516] text-[22px] font-bold leading-tight tracking-[-0.015em] pb-3 pt-5">Data Privacy & Consent</h2>
                        <div class="flex flex-col gap-3 py-4">
                            <div class="flex items-center gap-2">
                                <input type="checkbox" id="dataConsent" class="form-checkbox h-4 w-4 text-[#3f8abf] rounded border-gray-300 focus:ring-[#3f8abf]">
                                <label for="dataConsent" class="text-[#121516] text-sm font-normal leading-normal">
                                    I confirm the patient has given consent for data usage.
                                </label>
                            </div>
                            <p class="text-[#6a7781] text-sm font-normal leading-normal">
                                This patient has provided explicit consent for their data to be used for stroke risk assessment,
                                personalized care planning, and AI model improvement. All data handling complies with
                                strict privacy regulations (e.g., HIPAA, GDPR, PHIPA).
                            </p>
                            <button
                                class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-10 px-4 bg-[#3f8abf] text-white text-sm font-bold leading-normal tracking-[0.015em]"
                                onclick="window.location.href='pages/13_Data_Privacy_Policy.py';"
                            >
                                <span class="truncate">Click here to read more about our Data Privacy Policy</span>
                            </button>
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
    st.components.v1.html(full_html_content, height=2800, scrolling=False) # Increased height slightly for footer

# Call the function to render the page.
client_profile_page()
