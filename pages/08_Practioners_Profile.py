import streamlit as st

def practitioner_profile_page():
    """
    Renders the Practitioner Profile Page by embedding the provided HTML content.
    Adjustments have been made to the HTML's Tailwind classes and internal CSS
    to ensure full-width rendering within the Streamlit application and
    to eliminate unnecessary scrollbars, providing a balanced and professional look.
    The header and footer are consistent with other pages, and the availability
    dropdown and risk level displays are enhanced.
    """
    # Define the header HTML content (consistent with other pages)
    header_html_content = """
    <header class="flex items-center justify-between whitespace-nowrap border-b border-solid border-b-[#f1f2f4] py-3">
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
                <a class="text-[#121516] text-sm font-medium leading-normal" href="pages/02_Patient_Data_Entry.py">New Assessment</a>
                <a class="text-[#121516] text-sm font-medium leading-normal" href="pages/05_Patient_Records.py">Manage Patient Records</a>
            </div>
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

    # Define the footer HTML content (copied from 09_Client_Profile.py for consistency)
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

        <title>Practitioner Profile</title>
        <link rel="icon" type="image/x-icon" href="data:image/x-icon;base64," />

        <script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
        <style>
            /* Ensure the body uses your specified fonts, even within the iframe */
            body {{ font-family: "Public Sans", "Noto Sans", sans-serif; margin: 0; padding: 0; }}
            /* Ensure the outermost div takes full height and does NOT manage its own overflow */
            .relative.flex.size-full.flex-col {{
                min-height: 100vh;
                overflow-x: hidden; /* Prevent horizontal scroll */
            }}

            /* Adjustments for width and padding within the embedded HTML */
            /* This targets the main content wrapper */
            .main-content-area {{
                padding-left: 1rem; /* Consistent padding for smaller screens */
                padding-right: 1rem; /* Consistent padding for smaller screens */
                width: 100%; /* Ensure it takes full width */
                box-sizing: border-box; /* Include padding in width calculation */
                margin-left: auto; /* Center the content */
                margin-right: auto; /* Center the content */
            }}

            /* Responsive padding for main content area */
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

            /* This targets the inner content container */
            .layout-content-container {{
                max-width: 100%; /* Allow it to expand fully */
                width: 100%; /* Explicitly set width to 100% within its parent */
                box-sizing: border-box; /* Include padding in width calculation */
                padding-left: 0; /* Remove inner padding as main-content-area handles it */
                padding-right: 0; /* Remove inner padding as main-content-area handles it */
            }}

            /* Adjust header padding to be consistent with content if desired */
            header.py-3 {{ /* Removed px-10 from HTML, now control via CSS */
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }}
            @media (min-width: 640px) {{
                header.py-3 {{
                    padding-left: 2.5rem !important;
                    padding-right: 2.5rem !important;
                }}
            }}
            @media (min-width: 1024px) {{
                header.py-3 {{
                    padding-left: 10rem !important;
                    padding-right: 10rem !important;
                }}
            }}

            /* Ensure table columns are responsive */
            @container(max-width:120px){{.table-de498274-c850-4dbc-bd57-60039daa2ec4-column-120{{display: none;}}}}
            @container(max-width:240px){{.table-de498274-c850-4dbc-bd57-60039daa2ec4-column-240{{display: none;}}}}
            @container(max-width:360px){{.table-de498274-c850-4dbc-bd57-60039daa2ec4-column-360{{display: none;}}}}
            @container(max-width:480px){{.table-de498274-c850-4dbc-bd57-60039daa2ec4-column-480{{display: none;}}}}

            /* Ensure select button SVG is correctly rendered */
            [style*='--select-button-svg'] {{
                background-image: var(--select-button-svg);
            }}

            /* Footer specific styling for full width (from 09_Client_Profile.py) */
            footer.flex.justify-center {{
                width: 100%;
                max-width: none; /* Override any max-width on the footer itself */
            }}
            footer.flex.justify-center > div.flex.flex-1.flex-col {{
                max-width: 100%; /* Ensure the inner container also expands */
            }}
            /* Adjust footer position for narrow view (from 09_Client_Profile.py) */
            @media (max-width: 767px) {{ /* Adjust breakpoint as needed for "narrow view" */
                footer.mt-8 {{ /* Note: The footer HTML in this file doesn't have mt-8 directly on the outer footer, but the inner one does. */
                    margin-top: 2rem; /* Consistent margin */
                    padding-bottom: 1rem; /* Add some padding at the bottom */
                }}
            }}
        </style>
    </head>
    <body>
        <div
            class="relative flex size-full flex-col bg-white group/design-root"
            style='font-family: "Public Sans", "Noto Sans", sans-serif;'
        >
            <div class="layout-container flex h-full grow flex-col">
                {header_html_content}
                <div class="flex flex-1 justify-center py-0 main-content-area">
                    <div class="layout-content-container flex flex-col flex-1 mx-auto">
                        <div class="flex flex-wrap justify-between gap-3 py-4">
                            <div class="flex min-w-72 flex-col gap-3">
                                <p class="text-[#121516] tracking-light text-[32px] font-bold leading-tight">Practitioner Profile</p>
                                <p class="text-[#6a7781] text-sm font-normal leading-normal">Manage your profile and review your activity within the StrokeRisk system.</p>
                            </div>
                        </div>
                        <div class="flex py-4 @container">
                            <div class="flex w-full flex-col gap-4 @[520px]:flex-row @[520px]:justify-between @[520px]:items-center">
                                <div class="flex gap-4">
                                    <div
                                        class="bg-center bg-no-repeat aspect-square bg-cover rounded-full min-h-32 w-32"
                                        style='background-image: url("https://lh3.googleusercontent.com/aida-public/AB6AXuADSOIVdgeMZfs-OidTeTc-kNcCpIuZ6dsh5LeKp3eT2RvOH-9Ps25LEHCLQIshsX9FRotaAMN6KK-h1FnWlvicJo_yHM1_-JXtyVEH5yXOX57EFnJj-e0N-DyhnaFZnLvErtpLH2JIFKZAZVDO9foXJQiVpP7uuHvzc9r8iKpgoWboWTkmo9SwxcfIlC0CuxSKjNRaXTcoR8EzQ5eUeM157PreQIqRcIk35ob59P9JZnosCXCEEm8XTbPblec8wJRHoChifxukIWQ");'
                                    ></div>
                                    <div class="flex flex-col justify-center">
                                        <p class="text-[#121516] text-[22px] font-bold leading-tight tracking-[-0.015em]">Dr. Amelia Chen</p>
                                        <p class="text-[#6a7781] text-base font-normal leading-normal">Cardiologist</p>
                                        <p class="text-[#6a7781] text-base font-normal leading-normal">Login ID: AC1234</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <h3 class="text-[#121516] text-lg font-bold leading-tight tracking-[-0.015em] pb-2 pt-4">Status</h3>
                        <div class="flex max-w-[480px] flex-wrap items-end gap-4 py-3">
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">Availability</p>
                                <select
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 bg-[image:var(--select-button-svg)] placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                >
                                    <option value="available" selected>Available</option>
                                    <option value="busy">Busy</option>
                                    <option value="on_call">On Call</option>
                                    <option value="other">Other</option>
                                </select>
                            </label>
                        </div>
                        <div class="flex max-w-[480px] flex-wrap items-end gap-4 py-3">
                            <label class="flex flex-col min-w-40 flex-1">
                                <p class="text-[#121516] text-base font-medium leading-normal pb-2">Custom Status Message (Optional)</p>
                                <input
                                    placeholder="e.g., In Consultation, On Leave"
                                    class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border border-[#dde1e3] bg-white focus:border-[#dde1e3] h-14 placeholder:text-[#6a7781] p-[15px] text-base font-normal leading-normal"
                                    value=""
                                />
                            </label>
                        </div>
                        <h3 class="text-[#121516] text-lg font-bold leading-tight tracking-[-0.015em] pb-2 pt-4">Patient Interaction Dashboard</h3>
                        <div class="flex flex-wrap gap-4 py-4">
                            <div class="flex min-w-[158px] flex-1 flex-col gap-2 rounded-xl p-6 border border-[#dde1e3]">
                                <p class="text-[#121516] text-base font-medium leading-normal">Total Assessments</p>
                                <p class="text-[#121516] tracking-light text-2xl font-bold leading-tight">245</p>
                                <p class="text-[#078838] text-base font-medium leading-normal">+10%</p>
                            </div>
                            <div class="flex min-w-[158px] flex-1 flex-col gap-2 rounded-xl p-6 border border-[#dde1e3]">
                                <p class="text-[#121516] text-base font-medium leading-normal">High-Risk Predictions</p>
                                <p class="text-[#121516] tracking-light text-2xl font-bold leading-tight">32</p>
                                <p class="text-[#e73908] text-base font-medium leading-normal">-5%</p>
                            </div>
                            <div class="flex min-w-[158px] flex-1 flex-col gap-2 rounded-xl p-6 border border-[#dde1e3]">
                                <p class="text-[#121516] text-base font-medium leading-normal">Low-Risk Predictions</p>
                                <p class="text-[#121516] tracking-light text-2xl font-bold leading-tight">213</p>
                                <p class="text-[#078838] text-base font-medium leading-normal">+12%</p>
                            </div>
                            <div class="flex min-w-[158px] flex-1 flex-col gap-2 rounded-xl p-6 border border-[#dde1e3]">
                                <p class="text-[#121516] text-base font-medium leading-normal">Average Risk Score</p>
                                <p class="text-[#121516] tracking-light text-2xl font-bold leading-tight">0.65</p>
                                <p class="text-[#e73908] text-base font-medium leading-normal">-2%</p>
                            </div>
                        </div>
                        <h3 class="text-[#121516] text-lg font-bold leading-tight tracking-[-0.015em] pb-2 pt-4">Recent Assessments</h3>
                        <div class="py-3">
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
                                        placeholder="Search by Patient ID"
                                        class="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#121516] focus:outline-0 focus:ring-0 border-none bg-[#f1f2f4] focus:border-none h-full placeholder:text-[#6a7781] px-4 rounded-l-none border-l-0 pl-2 text-base font-normal leading-normal"
                                        value=""
                                    />
                                </div>
                            </label>
                        </div>
                        <div class="py-3 @container">
                            <div class="flex overflow-hidden rounded-xl border border-[#dde1e3] bg-white">
                                <table class="flex-1">
                                    <thead>
                                        <tr class="bg-white">
                                            <th class="table-de498274-c850-4dbc-bd57-60039daa2ec4-column-120 px-4 py-3 text-left text-[#121516] w-[400px] text-sm font-medium leading-normal">
                                                Patient ID
                                            </th>
                                            <th class="table-de498274-c850-4dbc-bd57-60039daa2ec4-column-240 px-4 py-3 text-left text-[#121516] w-[400px] text-sm font-medium leading-normal">
                                                Date/Time
                                            </th>
                                            <th class="table-de498274-c850-4dbc-bd57-60039daa2ec4-column-360 px-4 py-3 text-left text-[#121516] w-60 text-sm font-medium leading-normal">Risk Level</th>
                                            <th class="table-de498274-c850-4dbc-bd57-60039daa2ec4-column-480 px-4 py-3 text-left text-[#121516] w-60 text-[#6a7781] text-sm font-medium leading-normal">
                                                Action
                                            </th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr class="border-t border-t-[#dde1e3]">
                                            <td class="table-de498274-c850-4dbc-bd57-60039daa2ec4-column-120 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">
                                                Patient-123
                                            </td>
                                            <td class="table-de498274-c850-4dbc-bd57-60039daa2ec4-column-240 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">
                                                2024-03-15 10:30 AM
                                            </td>
                                            <td class="table-de498274-c850-4dbc-bd57-60039daa2ec4-column-360 h-[72px] px-4 py-2 w-60 text-sm font-normal leading-normal">
                                                <button
                                                    class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-8 px-4 bg-red-500 text-white text-sm font-medium leading-normal w-full"
                                                >
                                                    <span class="truncate">High</span>
                                                </button>
                                            </td>
                                            <td class="table-de498274-c850-4dbc-bd57-60039daa2ec4-column-480 h-[72px] px-4 py-2 w-60 text-[#6a7781] text-sm font-bold leading-normal tracking-[0.015em]">
                                                <a href="pages/09_Client_Profile.py" class="text-[#3f8abf] hover:underline">View Customer Full Profile</a>
                                            </td>
                                        </tr>
                                        <tr class="border-t border-t-[#dde1e3]">
                                            <td class="table-de498274-c850-4dbc-bd57-60039daa2ec4-column-120 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">
                                                Patient-456
                                            </td>
                                            <td class="table-de498274-c850-4dbc-bd57-60039daa2ec4-column-240 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">
                                                2024-03-14 02:15 PM
                                            </td>
                                            <td class="table-de498274-c850-4dbc-bd57-60039daa2ec4-column-360 h-[72px] px-4 py-2 w-60 text-sm font-normal leading-normal">
                                                <button
                                                    class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-8 px-4 bg-green-500 text-white text-sm font-medium leading-normal w-full"
                                                >
                                                    <span class="truncate">Low</span>
                                                </button>
                                            </td>
                                            <td class="table-de498274-c850-4dbc-bd57-60039daa2ec4-column-480 h-[72px] px-4 py-2 w-60 text-[#6a7781] text-sm font-bold leading-normal tracking-[0.015em]">
                                                <a href="pages/09_Client_Profile.py" class="text-[#3f8abf] hover:underline">View Customer Full Profile</a>
                                            </td>
                                        </tr>
                                        <tr class="border-t border-t-[#dde1e3]">
                                            <td class="table-de498274-c850-4dbc-bd57-60039daa2ec4-column-120 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">
                                                Patient-789
                                            </td>
                                            <td class="table-de498274-c850-4dbc-bd57-60039daa2ec4-column-240 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">
                                                2024-03-13 09:45 AM
                                            </td>
                                            <td class="table-de498274-c850-4dbc-bd57-60039daa2ec4-column-360 h-[72px] px-4 py-2 w-60 text-sm font-normal leading-normal">
                                                <button
                                                    class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-8 px-4 bg-yellow-500 text-white text-sm font-medium leading-normal w-full"
                                                >
                                                    <span class="truncate">Medium</span>
                                                </button>
                                            </td>
                                            <td class="table-de498274-c850-4dbc-bd57-60039daa2ec4-column-480 h-[72px] px-4 py-2 w-60 text-[#6a7781] text-sm font-bold leading-normal tracking-[0.015em]">
                                                <a href="pages/09_Client_Profile.py" class="text-[#3f8abf] hover:underline">View Customer Full Profile</a>
                                            </td>
                                        </tr>
                                        <tr class="border-t border-t-[#dde1e3]">
                                            <td class="table-de498274-c850-4dbc-bd57-60039daa2ec4-column-120 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">
                                                Patient-112
                                            </td>
                                            <td class="table-de498274-c850-4dbc-bd57-60039daa2ec4-column-240 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">
                                                2024-03-11 03:30 PM
                                            </td>
                                            <td class="table-de498274-c850-4dbc-bd57-60039daa2ec4-column-360 h-[72px] px-4 py-2 w-60 text-sm font-normal leading-normal">
                                                <button
                                                    class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-8 px-4 bg-red-500 text-white text-sm font-medium leading-normal w-full"
                                                >
                                                    <span class="truncate">High</span>
                                                </button>
                                            </td>
                                            <td class="table-de498274-c850-4dbc-bd57-60039daa2ec4-column-480 h-[72px] px-4 py-2 w-60 text-[#6a7781] text-sm font-bold leading-normal tracking-[0.015em]">
                                                <a href="pages/09_Client_Profile.py" class="text-[#3f8abf] hover:underline">View Customer Full Profile</a>
                                            </td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        <div class="flex items-center justify-center py-4">
                            <a href="#" class="flex size-10 items-center justify-center">
                                <div class="text-[#121516]" data-icon="CaretLeft" data-size="18px" data-weight="regular">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="18px" height="18px" fill="currentColor" viewBox="0 0 256 256">
                                        <path d="M165.66,202.34a8,8,0,0,1-11.32,11.32l-80-80a8,8,0,0,1,0-11.32l80-80a8,8,0,0,1,11.32,11.32L91.31,128Z"></path>
                                    </svg>
                                </div>
                            </a>
                            <a class="text-sm font-bold leading-normal tracking-[0.015em] flex size-10 items-center justify-center text-[#121516] rounded-full bg-[#f1f2f4]" href="#">1</a>
                            <a class="text-sm font-normal leading-normal flex size-10 items-center justify-center text-[#121516] rounded-full" href="#">2</a>
                            <a class="text-sm font-normal leading-normal flex size-10 items-center justify-center text-[#121516] rounded-full" href="#">3</a>
                            <span class="text-sm font-normal leading-normal flex size-10 items-center justify-center text-[#121516] rounded-full">...</span>
                            <a class="text-sm font-normal leading-normal flex size-10 items-center justify-center text-[#121516] rounded-full" href="#">10</a>
                            <a href="#" class="flex size-10 items-center justify-center">
                                <div class="text-[#121516]" data-icon="CaretRight" data-size="18px" data-weight="regular">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="18px" height="18px" fill="currentColor" viewBox="0 0 256 256">
                                        <path d="M181.66,133.66l-80,80a8,8,0,0,1-11.32-11.32L164.69,128,90.34,53.66a8,8,0,0,1,11.32-11.32l80,80A8,8,0,0,1,181.66,133.66Z"></path>
                                    </svg>
                                </div>
                            </a>
                        </div>
                        <div class="flex py-3 justify-start">
                            <button
                                class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-10 px-4 bg-[#3f8abf] text-white text-sm font-bold leading-normal tracking-[0.015em]"
                            >
                                <span class="truncate">New Assessment</span>
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
    # Set height to a fixed value (increased from 1800 to 2500) and scrolling to False to prevent internal scrollbar.
    st.components.v1.html(full_html_content, height=2500, scrolling=True) # Increased height and enabled scrolling

# Call the function to render the page.
practitioner_profile_page()
