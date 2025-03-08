import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import json
import traceback
import threading
import time
from datetime import datetime
import random
from openai import OpenAI
from PyPDF2 import PdfReader

# ------------------------ GLOBAL VARIABLES ------------------------
selected_api_key = None
selected_api_key_path = None
MODEL_INFO = {}
client = None  # OpenAI client instance
pdf_pages = []  # Cached PDF page texts
tab_data = {}   # Holds data for each page tab
last_api_key_dir = None

# Section info dictionaries (set by section prompt) per page index
section_title = {}
section_number = {}
page_number = {}

# Answer keywords dictionary (set by answer prompt) per page index
answer_array = {}

# Clue types and their descriptions
CLUE_TYPES = [
    "Definition", "Synonym", "Fill-in-the-Blank", "Anagram", "Abbreviation",
    "Charade", "Homophone", "Hidden Word", "Cryptic", "Question",
    "Double Definition", "Acrostic", "Pun", "Cross-Reference",
    "Metaphor/Simile", "Rebus", "Visual"
]

CLUE_TYPE_DESC = {
    "Definition": "Direct definition.",
    "Synonym": "Word(s) with similar meaning.",
    "Fill-in-the-Blank": "Sentence with a blank to be filled.",
    "Anagram": "Clue indicates a scrambled word (e.g., 'scrambled,' 'rearranged').",
    "Abbreviation": "Clue hints at an acronym or abbreviation.",
    "Charade": "Clue formed by combining parts or meanings.",
    "Homophone": "Clue indicates a word sounds like another.",
    "Hidden Word": "Word embedded in the clue sentence (without capitalization hints; avoid all caps).",
    "Cryptic": "Combination of wordplay and definitions.",
    "Question": "Simple question.",
    "Double Definition": "A single clue provides two different definitions for the same word.",
    "Acrostic": "The first letters of each word in the sentence spell the answer.",
    "Pun": "Use puns for humor or trickery.",
    "Cross-Reference": "Refer to another clue or answer in the puzzle.",
    "Metaphor/Simile": "Use figurative language.",
    "Rebus": "Involve letters, numbers, or symbols arranged to depict words or phrases.",
    "Visual": ("Clues rely on the shape or arrangement of symbolic typefaces "
               "(Wingdings, Webdings, Zapf Dingbats, Dingbats, Entypo, FontAwesome, "
               "Material Icons, Octicons, Noto Emoji, EmojiOne (JoyPixels), Twemoji, "
               "Apple Color Emoji, Segoe UI Emoji, Bravura, Petrucci, Opus, Maestro, "
               "Musical Symbols Unicode, Symbol, Cambria Math, STIX Two Math, Asana Math, "
               "TeX Gyre Termes Math, ISO Technical Symbols, MT Extra, Lucida Math, AstroGadget, "
               "AstroSymbols, Alchemy Symbols, Runic Unicode).")
}

# Dictionary to hold per-page randomized clue types
page_clue_types = {}

# Function to preselect and load API key file
def preselect_api_key(file_path):
    global selected_api_key, selected_api_key_path, client, last_api_key_dir
    if os.path.exists(file_path):
        selected_api_key_path = file_path
        last_api_key_dir = os.path.dirname(file_path)
        try:
            with open(file_path, "r") as f:
                raw_key = f.read().strip()
            filename = os.path.basename(file_path)
            print(f"[DEBUG] Preselected API Key: {filename}")
            print(f"[DEBUG] Raw API Key: {raw_key[:5]}...")
            
            if "OpenAI" in filename:
                print("[DEBUG] Detected API.OpenAI.txt, using OpenAI endpoint.")
                selected_api_key = raw_key
                client_temp = OpenAI(api_key=selected_api_key)
            elif "DeepSeek" in filename:
                print("[DEBUG] Detected API.DeepSeek.txt, using DeepSeek endpoint.")
                selected_api_key = raw_key
                client_temp = OpenAI(api_key=selected_api_key, base_url="https://api.deepseek.com")
            else:
                if raw_key.startswith("sk-"):
                    print("[DEBUG] Key starts with 'sk-', treating as OpenAI key.")
                    selected_api_key = raw_key
                    client_temp = OpenAI(api_key=selected_api_key)
                else:
                    print("[DEBUG] Key does not start with 'sk-', treating as DeepSeek key.")
                    selected_api_key = raw_key
                    client_temp = OpenAI(api_key=selected_api_key, base_url="https://api.deepseek.com")
            
            try:
                test_models = client_temp.models.list()
                if hasattr(test_models, "data"):
                    count = len(test_models.data)
                else:
                    count = "unknown"
                print(f"[DEBUG] Key validated successfully. Found {count} models.")
            except Exception as e:
                print("[ERROR] Failed to validate API key.")
                traceback.print_exc()
                messagebox.showerror("API Key Error", f"Failed to validate key: {e}")
                return False
            
            client = client_temp
            return True
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load API key: {e}")
            return False
    else:
        print(f"[ERROR] API key file not found: {file_path}")
        messagebox.showerror("Error", f"API key file not found: {file_path}")
        return False

# ------------------------ API KEY & MODEL FUNCTIONS ------------------------
def browse_api_key():
    global selected_api_key, selected_api_key_path, client, last_api_key_dir
    initial_dir = last_api_key_dir if last_api_key_dir else os.getcwd()
    file_path = filedialog.askopenfilename(
        title="Select API Key File",
        initialdir=initial_dir,
        filetypes=[("Text Files", "*.txt")]
    )
    if file_path:
        last_api_key_dir = os.path.dirname(file_path)
        selected_api_key_path = file_path
        try:
            with open(file_path, "r") as f:
                raw_key = f.read().strip()
            filename = os.path.basename(file_path)
            print(f"[DEBUG] Raw API Key: {raw_key[:20]}...")
            print(f"[DEBUG] File selected: {filename}")
            if "OpenAI" in filename:
                print("[DEBUG] Detected API.OpenAI.txt, using OpenAI endpoint.")
                selected_api_key = raw_key
                client_temp = OpenAI(api_key=selected_api_key)
            elif "DeepSeek" in filename:
                print("[DEBUG] Detected API.DeepSeek.txt, using DeepSeek endpoint.")
                selected_api_key = raw_key
                client_temp = OpenAI(api_key=selected_api_key, base_url="https://api.deepseek.com")
            else:
                if raw_key.startswith("sk-"):
                    print("[DEBUG] Key starts with 'sk-', treating as OpenAI key.")
                    selected_api_key = raw_key
                    client_temp = OpenAI(api_key=selected_api_key)
                else:
                    print("[DEBUG] Key does not start with 'sk-', treating as DeepSeek key.")
                    selected_api_key = raw_key
                    client_temp = OpenAI(api_key=selected_api_key, base_url="https://api.deepseek.com")
            try:
                test_models = client_temp.models.list()
                if hasattr(test_models, "data"):
                    count = len(test_models.data)
                else:
                    count = "unknown"
                print(f"[DEBUG] Key validated successfully. Found {count} models.")
            except Exception as e:
                print("[ERROR] Failed to validate API key.")
                traceback.print_exc()
                messagebox.showerror("API Key Error", f"Failed to validate key: {e}")
                return
            client = client_temp
            api_key_label.config(text=filename)
            retrieve_models()
            load_model_info()
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load API key: {e}")
    else:
        messagebox.showinfo("Info", "No API key file selected.")

def retrieve_models():
    if not client:
        messagebox.showerror("Error", "Please select an API key first.")
        return
    try:
        models = client.models.list()
        available_models = sorted([m.id for m in models])
        if not available_models:
            messagebox.showinfo("Info", "No models found for the selected API key.")
            return
        # Set dropdown values for all three prompt types with defaults:
        section_model_dropdown['values'] = available_models
        section_model_dropdown.set("gpt-3.5-turbo" if "gpt-3.5-turbo" in available_models else available_models[0])
        
        answer_model_dropdown['values'] = available_models
        answer_model_dropdown.set("gpt-3.5-turbo" if "gpt-3.5-turbo" in available_models else available_models[0])
        
        clue_model_dropdown['values'] = available_models
        clue_model_dropdown.set("gpt-3.5-turbo" if "gpt-3.5-turbo" in available_models else available_models[0])
        
        # Update model info labels explicitly on initialization.
        update_section_model_info()
        update_answer_model_info()
        update_clue_model_info()

        # Optionally adjust widths
        max_width = max(len(model) for model in available_models) + 2
        for dd in (section_model_dropdown, answer_model_dropdown, clue_model_dropdown):
            dd.config(width=max(20, max_width))
    except Exception as e:
        traceback.print_exc()
        messagebox.showerror("Error", f"Failed to retrieve models: {e}")

def update_section_model_info(*args):
    selected_model = section_model_dropdown.get().strip()
    print("Section model selected:", selected_model)
    if selected_model in MODEL_INFO:
        details = MODEL_INFO[selected_model]
        input_price = details.get("input_price", "N/A")
        output_price = details.get("output_price", "N/A")
        context_size = details.get("context_size", "N/A")
        is_new = details.get("new_model", False)
        model_info_text = f"Input: ${input_price} | Output: ${output_price} | Context: {context_size}"
        if is_new:
            section_model_info_label.config(text=model_info_text, font=("Helvetica", 10, "bold"), fg="green")
        else:
            section_model_info_label.config(text=model_info_text, font=("Helvetica", 10, "normal"), fg="white")
    else:
        section_model_info_label.config(text="No info available for this model.", font=("Helvetica", 10, "italic"), fg="red")

def update_answer_model_info(*args):
    selected_model = answer_model_dropdown.get().strip()
    print("Answer model selected:", selected_model)
    if selected_model in MODEL_INFO:
        details = MODEL_INFO[selected_model]
        input_price = details.get("input_price", "N/A")
        output_price = details.get("output_price", "N/A")
        context_size = details.get("context_size", "N/A")
        is_new = details.get("new_model", False)
        model_info_text = f"Input: ${input_price} | Output: ${output_price} | Context: {context_size}"
        if is_new:
            answer_model_info_label.config(text=model_info_text, font=("Helvetica", 10, "bold"), fg="green")
        else:
            answer_model_info_label.config(text=model_info_text, font=("Helvetica", 10, "normal"), fg="white")
    else:
        answer_model_info_label.config(text="No info available for this model.", font=("Helvetica", 10, "italic"), fg="red")

def update_clue_model_info(*args):
    selected_model = clue_model_dropdown.get().strip()
    print("Clue model selected:", selected_model)
    if selected_model in MODEL_INFO:
        details = MODEL_INFO[selected_model]
        input_price = details.get("input_price", "N/A")
        output_price = details.get("output_price", "N/A")
        context_size = details.get("context_size", "N/A")
        is_new = details.get("new_model", False)
        model_info_text = f"Input: ${input_price} | Output: ${output_price} | Context: {context_size}"
        if is_new:
            clue_model_info_label.config(text=model_info_text, font=("Helvetica", 10, "bold"), fg="green")
        else:
            clue_model_info_label.config(text=model_info_text, font=("Helvetica", 10, "normal"), fg="white")
    else:
        clue_model_info_label.config(text="No info available for this model.", font=("Helvetica", 10, "italic"), fg="red")

def load_model_info():
    global MODEL_INFO
    json_directory = os.path.dirname(selected_api_key_path) if selected_api_key_path else os.getcwd()
    json_file = os.path.join(json_directory, "model_info.json")
    if not os.path.exists(json_file):
        print(f"Warning: model_info.json not found at {json_file}.")
        MODEL_INFO = {}
        return
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            MODEL_INFO = json.load(f)
        print(f"[DEBUG] Loaded model_info.json from {json_file}.")
    except Exception as e:
        traceback.print_exc()
        MODEL_INFO = {}

# ------------------------ PDF HANDLING ------------------------
def browse_pdf():
    global pdf_pages
    pdf_path = filedialog.askopenfilename(
        title="Select PDF File",
        filetypes=[("PDF Files", "*.pdf")],
        initialdir=os.getcwd()
    )
    if not pdf_path:
        pdf_filename_label.config(text="No PDF selected")
        return
    if not pdf_path.lower().endswith(".pdf"):
        messagebox.showerror("Invalid File", "Please select a valid PDF file.")
        pdf_filename_label.config(text="No PDF selected")
        return
    try:
        reader = PdfReader(pdf_path)
        pdf_pages = [page.extract_text() if page.extract_text() else "Empty Page" for page in reader.pages]
        filename = os.path.basename(pdf_path)
        pdf_filename_label.config(text=filename)
        print(f"[DEBUG] PDF loaded with {len(pdf_pages)} pages.")
    except Exception as e:
        traceback.print_exc()
        messagebox.showerror("Error", f"Failed to process PDF: {e}")
        pdf_filename_label.config(text="No PDF selected")

def preload_pdf(pdf_path):
    """Load a PDF file programmatically without using the file dialog."""
    global pdf_pages
    if not os.path.exists(pdf_path):
        print(f"[ERROR] PDF file not found: {pdf_path}")
        return False
    if not pdf_path.lower().endswith(".pdf"):
        print("[ERROR] Not a valid PDF file.")
        return False
    try:
        reader = PdfReader(pdf_path)
        pdf_pages = [page.extract_text() if page.extract_text() else "Empty Page" for page in reader.pages]
        filename = os.path.basename(pdf_path)
        pdf_filename_label.config(text=filename)
        print(f"[DEBUG] PDF preloaded with {len(pdf_pages)} pages.")
        return True
    except Exception as e:
        traceback.print_exc()
        print(f"[ERROR] Failed to process PDF: {e}")
        return False

# ------------------------ RESPONSE DISPLAY HANDLING ------------------------
def clear_output_display():
    for tab in output_notebook.tabs():
        output_notebook.forget(tab)
    tab_data.clear()
    print("[DEBUG] Response display area cleared.")

def create_tab(page_index, api_response, pdf_text):
    """Creates a new tab to display responses for a given page, with a working Resubmit checkbox."""
    global tab_data
    if page_index in tab_data:
        return  # Tab already exists
    tab_frame = tk.Frame(output_notebook, bg="#444")
    output_notebook.add(tab_frame, text=f"Page {page_index + 1}")
    resubmit_var = tk.IntVar(value=0)
    style = ttk.Style()
    style.configure("Custom.TCheckbutton", background="#2e2e2e", foreground="white")
    resubmit_checkbox = ttk.Checkbutton(
        tab_frame,
        text="Resubmit",
        variable=resubmit_var,
        style="Custom.TCheckbutton"
    )
    resubmit_checkbox.pack(side=tk.TOP, anchor="w")
    columns = ("SectionTitle", "SectionNumber", "PageNumber", "ClueType", "Clue", "Answer")
    tree = ttk.Treeview(tab_frame, columns=columns, show="headings")
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)
    tree.pack(fill=tk.BOTH, expand=True)
    tab_data[page_index] = {
        "frame": tab_frame,
        "treeview": tree,
        "resubmit_var": resubmit_var,
    }
    print(f"[DEBUG] Created tab for page {page_index + 1}.")

# ------------------------ HELPER FUNCTION: VALIDATE ANSWERS ------------------------
def validate_and_replace_answers(page_index, answers):
    """
    Validates each answer so that it is 25 characters or fewer (including spaces).
    If an answer exceeds 25 characters, this function prompts the API (up to 3 attempts)
    to generate a replacement that meets the criteria.
    Returns a list of validated answers.
    """
    valid_answers = []
    selected_model = answer_model_dropdown.get()
    pdf_text = pdf_pages[page_index]

    for ans in answers:
        if len(ans) <= 25:
            valid_answers.append(ans)
        else:
            attempts = 0
            new_answer = None
            while attempts < 3:
                print(f"[DEBUG] Using model '{selected_model}' for Clue Generation.")
                prompt = (
                    f"PDF Content:\n{pdf_text}\n\n"
                    f"The previously generated answer '{ans}' is longer than 25 characters. "
                    "Please generate a new, unique answer keyword/phrase that meets these criteria:\n"
                    "- Contain only UPPERCASE letters (A-Z) and numbers if any.\n"
                    "- May include spaces.\n"
                    "- No special characters, dashes, or punctuation.\n"
                    "- 25 characters or fewer.\n"
                    "- Directly sourced from the document.\n"
                    "Output only the answer."
                )
                try:
                    messages = [{"role": "user", "content": prompt}]
                    params = {"model": selected_model, "messages": messages, "max_completion_tokens": 100, "stream": False}
                    response = client.chat.completions.create(**params)
                    new_answer = response.choices[0].message.content.strip().upper()
                    if len(new_answer) <= 25 and new_answer not in valid_answers:
                        valid_answers.append(new_answer)
                        break
                    else:
                        attempts += 1
                except Exception as e:
                    print(f"[ERROR] Exception during re-generation for an answer on page {page_index+1}: {e}")
                    attempts += 1
            if new_answer is None or len(new_answer) > 25:
                print(f"[DEBUG] Unable to generate a valid answer for '{ans}' on page {page_index+1} after {attempts} attempts.")
    return valid_answers

# ------------------------ SECTION INFO PROMPT (Renamed) ------------------------
def run_section_info_prompt():
    """
    For each page in the PDF, submits its content with instructions to determine SECTIONTITLE, SECTIONNUMBER, and PAGENUMBER.
    The response is expected in a pipe-delimited format and stored in dictionaries keyed by page index.
    Uses the Section Model dropdown.
    """
    global section_title, section_number, page_number
    if not pdf_pages:
        print("[ERROR] No PDF content available. Please load a PDF first.")
        return
    print(f"[DEBUG] Using model '{selected_model}' for Clue Generation.")
    for i, page_text in enumerate(pdf_pages):
        print(f"[DEBUG] Using model '{selected_model}' for Clue Generation.")
        prompt = (
            f"PDF Content:\n{page_text}\n\n"
            "Determine SECTIONTITLE, SECTIONNUMBER, PAGENUMBER. "
            "Output in format: SECTIONTITLE | SECTIONNUMBER | PAGENUMBER."
        )
        print(f"[DEBUG] Sending section info prompt for page {i+1}:\n{prompt}")
        selected_model = section_model_dropdown.get()
        try:
            messages = [{"role": "user", "content": prompt}]
            params = {"model": selected_model, "messages": messages, "max_completion_tokens": 4096, "stream": False}
            response = client.chat.completions.create(**params)
            api_result = response.choices[0].message.content.strip()
            print(f"[DEBUG] API Response for page {i+1}: {api_result}")
            parts = api_result.split('|')
            if len(parts) >= 3:
                section_title[i] = parts[0].strip()
                section_number[i] = parts[1].strip()
                page_number[i] = parts[2].strip()
                print(f"[DEBUG] Parsed Section Info for page {i+1}: SECTIONTITLE='{section_title[i]}', SECTIONNUMBER='{section_number[i]}', PAGENUMBER='{page_number[i]}'")
            else:
                print(f"[ERROR] Unable to parse section info for page {i+1}.")
        except Exception as e:
            print(f"[ERROR] Exception during section info prompt for page {i+1}: {e}")

# ------------------------ ANSWER KEYWORDS PROMPT (Renamed) ------------------------
def run_answer_keywords_prompt():
    """
    For each page in the PDF, submits its content with instructions to determine ANSWER key words/phrases.
    Converts all extracted answers to uppercase and ensures each answer is ≤ 25 characters.
    Uses the Answer Model dropdown.
    """
    global answer_array
    if not pdf_pages:
        print("[ERROR] No PDF content available. Please load a PDF first.")
        return
    print(f"[DEBUG] Using model '{selected_model}' for Clue Generation.")
    for i, page_text in enumerate(pdf_pages):
        print(f"[DEBUG] Using model '{selected_model}' for Clue Generation.")
        prompt = (
            f"PDF Content:\n{page_text}\n\n"
            "Determine the ANSWER key words/phrases from the PDF that meet these criteria:\n"
            "- Contain only UPPERCASE letters (A-Z) and numbers if any.\n"
            "- May include spaces.\n"
            "- No special characters, dashes, or punctuation.\n"
            "- 25 characters or fewer.\n"
            "- All answers must be UNIQUE, concise, and directly sourced from the document.\n"
            "- For phrases with repetitive words, drop the redundant word.\n"
            "- Must not match or be similar to the SECTIONTITLE.\n"
            "Output the answers as a comma-separated list."
        )
        print(f"[DEBUG] Sending answer keywords prompt for page {i+1}:\n{prompt}")
        selected_model = answer_model_dropdown.get()
        try:
            messages = [{"role": "user", "content": prompt}]
            params = {"model": selected_model, "messages": messages, "max_completion_tokens": 4096, "stream": False}
            response = client.chat.completions.create(**params)
            api_result = response.choices[0].message.content.strip()
            print(f"[DEBUG] API Response for ANSWER keywords on page {i+1}: {api_result}")
            raw_answers = [ans.strip().upper() for ans in api_result.split(',') if ans.strip()]
            validated_answers = validate_and_replace_answers(i, raw_answers)
            answer_array[i] = validated_answers
            print(f"[DEBUG] Parsed ANSWER_ARRAY for page {i+1}: {answer_array[i]}")
        except Exception as e:
            print(f"[ERROR] Exception during answer keywords prompt for page {i+1}: {e}")

# ------------------------ REMAINING PROMPTS (Generate Clues) ------------------------
def get_next_clue_type(page_index=0):
    """Returns the next randomized clue type for the given page."""
    if page_index not in page_clue_types or not page_clue_types[page_index]:
        shuffled = CLUE_TYPES.copy()
        random.shuffle(shuffled)
        page_clue_types[page_index] = shuffled
    return page_clue_types[page_index].pop(0)

def process_clue_for_answer(answer, page_index=0):
    """
    Builds the prompt to generate a clue for the given answer using a randomized CLUETYPE
    and its description, then sends it to the API.
    Uses the Clue Model dropdown.
    ClueType 'Acrostic' is only allowed if the answer (excluding spaces) has 10 or fewer letters.
    """
    answer_length = len(answer.replace(" ", ""))
    while True:
        clue_type = get_next_clue_type(page_index)
        if clue_type == "Acrostic" and answer_length > 10:
            continue
        break
    clue_type_desc = CLUE_TYPE_DESC.get(clue_type, "")
    pdf_text = pdf_pages[page_index] if pdf_pages and len(pdf_pages) > page_index else "Cached PDF Context"
    print(f"[DEBUG] Using model '{selected_model}' for Clue Generation.")
    prompt = (
        f"PDF Content:\n{pdf_text}\n\n"
        f"Generate a detailed and engaging clue for the answer '{answer}' using clue type '{clue_type}', '{clue_type_desc}'. "
        "Respond with a single line containing only the clue text. Do not include any labels or extra explanation."
    )
    print(f"[DEBUG] Prompt for answer '{answer}' on page {page_index+1}:\n{prompt}")
    selected_model = clue_model_dropdown.get()
    try:
        messages = [{"role": "user", "content": prompt}]
        params = {"model": selected_model, "messages": messages, "max_completion_tokens": 4096, "stream": False}
        response = client.chat.completions.create(**params)
        api_result = response.choices[0].message.content.strip()
        if api_result.lower().startswith("clue:"):
            api_result = api_result[len("clue:"):].strip()
        api_result = " ".join(api_result.splitlines()).replace("**", "").replace('"', "").strip()
        print(f"[DEBUG] API Response for answer '{answer}' on page {page_index+1}: {api_result}")
        if page_index not in tab_data:
            create_tab(page_index, "", pdf_text)
        tree = tab_data[page_index]["treeview"]
        tree.insert("", "end", values=(section_title.get(page_index, ""), 
                                        section_number.get(page_index, ""), 
                                        page_number.get(page_index, ""), 
                                        clue_type, api_result, answer))
        return api_result
    except Exception as e:
        print(f"[ERROR] Failed to generate clue for answer '{answer}' on page {page_index+1}: {e}")
        return f"Error: {e}"

def run_remaining_prompts():
    """
    For each page that has answers in answer_array, cycles through its ANSWER_ARRAY.
    For each answer, a clue is generated and a row is populated with:
    SECTIONTITLE | SECTIONNUMBER | PAGENUMBER | CLUETYPE | CLUE | ANSWER.
    After processing each page, prints a debug message indicating the last row has been populated.
    """
    if not answer_array:
        print("[ERROR] answer_array is empty. Please run the answer keywords prompt first.")
        return
    for page_index in answer_array.keys():
        for answer in answer_array[page_index]:
            process_clue_for_answer(answer, page_index=page_index)
        print(f"[DEBUG] Last row populated for page {page_index + 1}.")

# ------------------------ COMBINED PROMPTS (All Steps) ------------------------
def run_combined_prompts():
    """
    Processes each page sequentially:
    1. Determines SECTIONTITLE, SECTIONNUMBER, PAGENUMBER using the Section Model.
    2. Determines ANSWER key words/phrases using the Answer Model.
    3. Generates clues for each answer using the Clue Model.
    """
    if not pdf_pages:
        print("[ERROR] No PDF content available. Please load a PDF first.")
        return

    for page_index, page_text in enumerate(pdf_pages):
        # Section Info Prompt
        prompt_section = (
            f"PDF Content:\n{page_text}\n\n"
            "Determine SECTIONTITLE, SECTIONNUMBER, PAGENUMBER. "
            "Output in format: SECTIONTITLE | SECTIONNUMBER | PAGENUMBER."
        )
        print(f"[DEBUG] Sending section info prompt for page {page_index+1}:\n{prompt_section}")
        try:
            messages = [{"role": "user", "content": prompt_section}]
            params = {"model": section_model_dropdown.get(), "messages": messages, "max_completion_tokens": 4096, "stream": False}
            response = client.chat.completions.create(**params)
            api_result = response.choices[0].message.content.strip()
            print(f"[DEBUG] API Response for page {page_index+1}: {api_result}")
            parts = api_result.split('|')
            if len(parts) >= 3:
                section_title[page_index] = parts[0].strip()
                section_number[page_index] = parts[1].strip()
                page_number[page_index] = parts[2].strip()
                print(f"[DEBUG] Parsed Section Info for page {page_index+1}: SECTIONTITLE='{section_title[page_index]}', SECTIONNUMBER='{section_number[page_index]}', PAGENUMBER='{page_number[page_index]}'")
            else:
                print(f"[ERROR] Unable to parse section info for page {page_index+1}.")
        except Exception as e:
            print(f"[ERROR] Exception during section info prompt for page {page_index+1}: {e}")

        # Answer Keywords Prompt
        prompt_answer = (
            f"PDF Content:\n{page_text}\n\n"
            "Determine the ANSWER key words/phrases from the PDF that meet these criteria:\n"
            "- Contain only UPPERCASE letters (A-Z) and numbers if any.\n"
            "- May include spaces.\n"
            "- No special characters, dashes, or punctuation.\n"
            "- 25 characters or fewer.\n"
            "- All answers must be UNIQUE, concise, and directly sourced from the document.\n"
            "- For phrases with repetitive words, drop the redundant word.\n"
            "- Must not match or be similar to the SECTIONTITLE.\n"
            "Output the answers as a comma-separated list."
        )
        print(f"[DEBUG] Sending answer keywords prompt for page {page_index+1}:\n{prompt_answer}")
        try:
            messages = [{"role": "user", "content": prompt_answer}]
            params = {"model": answer_model_dropdown.get(), "messages": messages, "max_completion_tokens": 4096, "stream": False}
            response = client.chat.completions.create(**params)
            api_result = response.choices[0].message.content.strip()
            print(f"[DEBUG] API Response for ANSWER keywords on page {page_index+1}: {api_result}")
            raw_answers = [ans.strip().upper() for ans in api_result.split(',') if ans.strip()]
            validated_answers = validate_and_replace_answers(page_index, raw_answers)
            answer_array[page_index] = validated_answers
            print(f"[DEBUG] Parsed ANSWER_ARRAY for page {page_index+1}: {answer_array[page_index]}")
        except Exception as e:
            print(f"[ERROR] Exception during answer keywords prompt for page {page_index+1}: {e}")

        # Clue Generation Prompt
        if page_index not in tab_data:
            create_tab(page_index, "", page_text)
        for answer in answer_array.get(page_index, []):
            process_clue_for_answer(answer, page_index=page_index)
        print(f"[DEBUG] Finished processing page {page_index + 1}.\n")

# ------------------------ RESUBMIT FUNCTIONALITY ------------------------
def resubmit_tab(page_index):
    """
    If the Resubmit checkbox for the given tab is enabled, prompt the user to either generate new answers
    (with a new, expanded set of answer keywords) or keep the same answers, then re-run process_clue_for_answer
    for each answer accordingly.
    Uses the Answer Model dropdown for new answer generation.
    """
    if page_index not in tab_data:
        print(f"[ERROR] No tab found for page {page_index}.")
        return
    resubmit_var = tab_data[page_index].get("resubmit_var")
    if resubmit_var and resubmit_var.get() == 1:
        choice = messagebox.askyesno(
            "Resubmit Options",
            "Do you want to generate new answers?\n\nYes: Generate new answers (more rows)\nNo: Keep same answers"
        )
        tree = tab_data[page_index]["treeview"]
        tree.delete(*tree.get_children())
        page_clue_types[page_index] = None
        if choice:
            pdf_text = pdf_pages[page_index]
            print(f"[DEBUG] Using model '{selected_model}' for Clue Generation.")
            prompt = (
                f"PDF Content:\n{pdf_text}\n\n"
                "Determine as many unique ANSWER key words/phrases as possible from the PDF that meet these criteria:\n"
                "- Contain only UPPERCASE letters (A-Z) and numbers if any.\n"
                "- May include spaces.\n"
                "- No special characters, dashes, or punctuation.\n"
                "- 25 characters or fewer.\n"
                "- All answers must be UNIQUE, concise, and directly sourced from the document.\n"
                "- For phrases with repetitive words, drop the redundant word.\n"
                "- Must not match or be similar to the SECTIONTITLE.\n"
                "Output the answers as a comma-separated list."
            )
            print(f"[DEBUG] Resubmit: Sending prompt for new answer keywords for page {page_index+1}:\n{prompt}")
            selected_model = answer_model_dropdown.get()
            try:
                messages = [{"role": "user", "content": prompt}]
                params = {"model": selected_model, "messages": messages, "max_completion_tokens": 4096, "stream": False}
                response = client.chat.completions.create(**params)
                api_result = response.choices[0].message.content.strip()
                new_answers = validate_and_replace_answers(
                    page_index,
                    [ans.strip().upper() for ans in api_result.split(',') if ans.strip()]
                )
                answer_array[page_index] = new_answers
                print(f"[DEBUG] Resubmit: Parsed new ANSWER_ARRAY for page {page_index+1}: {new_answers}")
                for answer in new_answers:
                    process_clue_for_answer(answer, page_index=page_index)
            except Exception as e:
                print(f"[ERROR] Exception during resubmit for new answers on page {page_index+1}: {e}")
        else:
            answers = answer_array.get(page_index, [])
            for answer in answers:
                process_clue_for_answer(answer, page_index=page_index)
        print(f"[DEBUG] Tab {page_index + 1} re-populated with clues.")
    else:
        print(f"[DEBUG] Skipping tab {page_index + 1} as Resubmit is not enabled.")

def resubmit_all_tabs():
    for page_index in list(tab_data.keys()):
        resubmit_tab(page_index)

# ------------------------ DROPDOWN ARROW KEY HANDLING ------------------------
def on_arrow_key_dropdown(event, dropdown):
    values = dropdown['values']
    if not values:
        return "break"
    current = dropdown.get()
    try:
        index = values.index(current)
    except ValueError:
        index = 0
    if event.keysym == "Up":
        new_index = (index - 1) % len(values)
    elif event.keysym == "Down":
        new_index = (index + 1) % len(values)
    else:
        return
    dropdown.set(values[new_index])
    # Manually update the corresponding model info label
    if dropdown == section_model_dropdown:
        update_section_model_info()
    elif dropdown == answer_model_dropdown:
        update_answer_model_info()
    elif dropdown == clue_model_dropdown:
        update_clue_model_info()
    return "break"

# ------------------------ BUILD THE GUI ------------------------
root = tk.Tk()
root.title("LLM One-Shot Prompt & Clue Generator")
root.configure(bg="#2e2e2e")
style = ttk.Style(root)
style.theme_use("clam")
style.configure("TButton", background="#444", foreground="white")
style.configure("TLabel", background="#2e2e2e", foreground="white")

# API Key & Main Model Selection Frame
api_frame = tk.Frame(root, bg="#2e2e2e")
api_frame.pack(pady=10, padx=10, fill=tk.X)
# Left side - API Key File
api_key_section = tk.Frame(api_frame, bg="#2e2e2e")
api_key_section.pack(side=tk.LEFT, fill=tk.X)
tk.Label(api_key_section, text="API Key File:", bg="#2e2e2e", fg="white").pack(side=tk.LEFT, padx=5)
tk.Button(api_key_section, text="Browse", command=browse_api_key, bg="#444", fg="white").pack(side=tk.LEFT, padx=5)
api_key_label = tk.Label(api_key_section, text="None selected", bg="#2e2e2e", fg="white")
api_key_label.pack(side=tk.LEFT, padx=5)

# Right side - PDF Browse
pdf_section = tk.Frame(api_frame, bg="#2e2e2e")
pdf_section.pack(side=tk.RIGHT, fill=tk.X)
tk.Button(pdf_section, text="Browse PDF", command=browse_pdf, bg="#444", fg="white").pack(side=tk.LEFT, padx=5)
pdf_filename_label = tk.Label(pdf_section, text="No PDF selected", bg="#2e2e2e", fg="white")
pdf_filename_label.pack(side=tk.LEFT, padx=5)

# Model Selection for Prompts Frame
model_prompt_frame = tk.Frame(root, bg="#2e2e2e")
model_prompt_frame.pack(pady=5, padx=10, fill=tk.X)

# Section Model Dropdown
section_frame = tk.Frame(model_prompt_frame, bg="#2e2e2e")
section_frame.pack(side=tk.LEFT, padx=5)
tk.Label(section_frame, text="Section Model:", bg="#2e2e2e", fg="white").pack(anchor="w")
section_model_var = tk.StringVar()
section_model_dropdown = ttk.Combobox(section_frame, textvariable=section_model_var, state="readonly")
section_model_dropdown.pack(anchor="w")
section_model_dropdown.bind("<Up>", lambda e: on_arrow_key_dropdown(e, section_model_dropdown))
section_model_dropdown.bind("<Down>", lambda e: on_arrow_key_dropdown(e, section_model_dropdown))
section_model_info_label = tk.Label(section_frame, text="Model details will appear here", bg="#2e2e2e", fg="white")
section_model_info_label.pack(anchor="w")
section_model_var.trace_add("write", lambda *args: update_section_model_info())

# Answer Model Dropdown
answer_frame = tk.Frame(model_prompt_frame, bg="#2e2e2e")
answer_frame.pack(side=tk.LEFT, padx=5)
tk.Label(answer_frame, text="Answer Model:", bg="#2e2e2e", fg="white").pack(anchor="w")
answer_model_var = tk.StringVar()
answer_model_dropdown = ttk.Combobox(answer_frame, textvariable=answer_model_var, state="readonly")
answer_model_dropdown.pack(anchor="w")
answer_model_dropdown.bind("<Up>", lambda e: on_arrow_key_dropdown(e, answer_model_dropdown))
answer_model_dropdown.bind("<Down>", lambda e: on_arrow_key_dropdown(e, answer_model_dropdown))
answer_model_info_label = tk.Label(answer_frame, text="Model details will appear here", bg="#2e2e2e", fg="white")
answer_model_info_label.pack(anchor="w")
answer_model_var.trace_add("write", lambda *args: update_answer_model_info())

# Clue Model Dropdown
clue_frame = tk.Frame(model_prompt_frame, bg="#2e2e2e")
clue_frame.pack(side=tk.LEFT, padx=5)
tk.Label(clue_frame, text="Clue Model:", bg="#2e2e2e", fg="white").pack(anchor="w")
clue_model_var = tk.StringVar()
clue_model_dropdown = ttk.Combobox(clue_frame, textvariable=clue_model_var, state="readonly")
clue_model_dropdown.pack(anchor="w")
clue_model_dropdown.bind("<Up>", lambda e: on_arrow_key_dropdown(e, clue_model_dropdown))
clue_model_dropdown.bind("<Down>", lambda e: on_arrow_key_dropdown(e, clue_model_dropdown))
clue_model_info_label = tk.Label(clue_frame, text="Model details will appear here", bg="#2e2e2e", fg="white")
clue_model_info_label.pack(anchor="w")
clue_model_var.trace_add("write", lambda *args: update_clue_model_info())

# Section Info Prompt Button
section_prompt_frame = tk.Frame(root, bg="#2e2e2e")
section_prompt_frame.pack(pady=2, padx=10, fill=tk.X)
tk.Button(section_prompt_frame, text="Determine Section Info", 
          command=lambda: threading.Thread(target=run_section_info_prompt, daemon=True).start(),
          bg="#444", fg="white").pack(pady=1)

# Answer Keywords Prompt Button
answer_prompt_frame = tk.Frame(root, bg="#2e2e2e")
answer_prompt_frame.pack(pady=2, padx=10, fill=tk.X)
tk.Button(answer_prompt_frame, text="Determine Answer Keywords", 
          command=lambda: threading.Thread(target=run_answer_keywords_prompt, daemon=True).start(),
          bg="#444", fg="white").pack(pady=1)

# Clue Generation Prompt Button
clue_prompt_frame = tk.Frame(root, bg="#2e2e2e")
clue_prompt_frame.pack(pady=2, padx=10, fill=tk.X)
tk.Button(clue_prompt_frame, text="Generate Clues for Answers", 
          command=lambda: threading.Thread(target=run_remaining_prompts, daemon=True).start(),
          bg="#444", fg="white").pack(pady=1)

# Resubmit Button (Re-generate clues for tabs with Resubmit enabled)
resubmit_frame = tk.Frame(root, bg="#2e2e2e")
resubmit_frame.pack(pady=2, padx=10, fill=tk.X)
tk.Button(resubmit_frame, text="Resubmit", 
          command=lambda: threading.Thread(target=resubmit_all_tabs, daemon=True).start(),
          bg="#444", fg="white").pack(pady=1)

# Combined Prompts Button
combined_prompt_frame = tk.Frame(root, bg="#2e2e2e")
combined_prompt_frame.pack(pady=2, padx=10, fill=tk.X)
tk.Button(
    combined_prompt_frame,
    text="Process All Prompts for Each Page",
    command=lambda: threading.Thread(target=run_combined_prompts, daemon=True).start(),
    bg="#444",
    fg="white"
).pack(pady=1)

# Output Notebook (For displaying rows for each page)
output_frame = tk.Frame(root, bg="#2e2e2e")
output_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
output_notebook = ttk.Notebook(output_frame)
output_notebook.pack(fill=tk.BOTH, expand=True)

# Function to initialize application with preselected API key
def initialize_app():
    # Preselect API key file
    api_key_path = "../.APIkeys/API.OpenAI.txt"
    if preselect_api_key(api_key_path):
        api_key_label.config(text=os.path.basename(api_key_path))
        # Load models after successful API key selection
        retrieve_models()
        load_model_info()
    
    # Preload default PDF file
    pdf_path = ".\EH200-1.pdf"
    if not preload_pdf(pdf_path):
        print(f"[WARNING] Failed to preload default PDF file: {pdf_path}")

# Call initialization function after a short delay to ensure GUI is fully loaded
root.after(100, initialize_app)

root.mainloop()