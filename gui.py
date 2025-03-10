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
import re

# ------------------------ GLOBAL VARIABLES ------------------------
# API Client (Global)
client = None
selected_api_key = None
selected_api_key_path = None

# Model information dictionary - populated from model_info.json if available
MODEL_INFO = {}

# Global variables to hold PDF content
pdf_pages = []
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
    "Double Definition", "Acrostic", "Pun", "Metaphor/Simile", "Rebus"
]

CLUE_TYPE_DESC = {
    "Definition": "Direct definition.",
    "Synonym": "Word(s) with similar meaning.",
    "Fill-in-the-Blank": "Sentence with a blank to be filled.",
    "Anagram": "Clue signals a word that has been scrambled (e.g., 'rearranged,' 'mixed up') and should be unscrambled to find the answer.",
    "Abbreviation": "Clue suggests a condensed or shortened form, requiring interpretation to find the full meaning.",
    "Charade": "Clue formed by combining parts or meanings.",
    "Homophone": "Clue indicates a word sounds like another.",
    "Hidden Word": "Word embedded in the clue sentence (without capitalization hints; avoid all caps).",
    "Cryptic": "Combination of wordplay and definitions.",
    "Question": "Simple question.",
    "Double Definition": "A single clue provides two different definitions for the same word.",
    "Acrostic": "The first letters of each word in the sentence spell the answer.",
    "Pun": "Use puns for humor or trickery.",
    "Metaphor/Simile": "Use figurative language.",
    "Rebus": "Clue uses numbers, symbols, emojis, or pictograms (Wingdings, Webdings, Zapf Dingbats, Dingbats, Entypo, FontAwesome, "
               "Material Icons, Octicons, Noto Emoji, EmojiOne (JoyPixels), Twemoji, "
               "Apple Color Emoji, Segoe UI Emoji, Bravura, Petrucci, Opus, Maestro, "
               "Musical Symbols Unicode, Symbol, Cambria Math, STIX Two Math, Asana Math, "
               "TeX Gyre Termes Math, ISO Technical Symbols, MT Extra, Lucida Math, AstroGadget, "
               "AstroSymbols, Alchemy Symbols, Runic Unicode) creatively arranged to represent words or phrases without directly spelling them out.",
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
        # Set dropdown values for all prompt types with defaults:
        content_model_dropdown['values'] = available_models
        content_model_dropdown.set("gpt-3.5-turbo" if "gpt-3.5-turbo" in available_models else available_models[0])
        
        clue_model_dropdown['values'] = available_models
        clue_model_dropdown.set("gpt-4o-mini" if "gpt-4o-mini" in available_models else available_models[0])
        
        # Update model info labels explicitly on initialization.
        update_model_info('content')
        update_model_info('clue')

        # Optionally adjust widths
        max_width = max(len(model) for model in available_models) + 2
        for dd in (content_model_dropdown, clue_model_dropdown):
            dd.config(width=max(20, max_width))
        
        # Store model_info
        model_info = {
            'content': content_model_dropdown,
            'clue': clue_model_dropdown
        }
        
        return True
    except Exception as e:
        traceback.print_exc()
        messagebox.showerror("Error", f"Failed to retrieve models: {e}")
        return False

def update_model_info(model_type, *args):
    """
    Updates the model information display for the given model type.
    
    Args:
        model_type (str): The type of model ('content', 'clue')
        *args: Additional arguments passed by trace_add
    """
    dropdown_map = {
        'content': content_model_dropdown,
        'clue': clue_model_dropdown
    }
    
    label_map = {
        'content': content_model_info_label,
        'clue': clue_model_info_label
    }
    
    dropdown = dropdown_map.get(model_type)
    info_label = label_map.get(model_type)
    
    if not dropdown or not info_label:
        print(f"[ERROR] Invalid model type: {model_type}")
        return
        
    selected_model = dropdown.get().strip()
    print(f"{model_type.capitalize()} model selected:", selected_model)
    
    if selected_model in MODEL_INFO:
        details = MODEL_INFO[selected_model]
        input_price = details.get("input_price", "N/A")
        output_price = details.get("output_price", "N/A")
        context_size = details.get("context_size", "N/A")
        model_info_text = f"Input: ${input_price} | Output: ${output_price} | Context: {context_size}"
        
        # Apply standard formatting for all models
        info_label.config(text=model_info_text, font=("Helvetica", 10, "normal"), fg="white")
    else:
        info_label.config(text="No info available for this model.", font=("Helvetica", 10, "italic"), fg="red")

def load_model_info():
    """Load model information after API key is selected."""
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
        
        # Trigger the model info update for all dropdowns
        update_model_info('content')
        update_model_info('clue')
    except Exception as e:
        traceback.print_exc()
        MODEL_INFO = {}

# ------------------------ PDF HANDLING ------------------------
def load_pdf(pdf_path=None, use_dialog=True):
    """
    Load a PDF file either from a provided path or using a file dialog.
    
    Args:
        pdf_path (str, optional): Path to the PDF file to load. If None and use_dialog is True, 
                                  opens a file dialog. Defaults to None.
        use_dialog (bool, optional): Whether to use a file dialog if pdf_path is None. Defaults to True.
    
    Returns:
        bool: True if PDF was successfully loaded, False otherwise.
    """
    global pdf_pages
    
    # If no path provided and dialog requested, use the file dialog
    if pdf_path is None and use_dialog:
        pdf_path = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF Files", "*.pdf")],
            initialdir=os.getcwd()
        )
    
    # If still no path (user cancelled dialog or no path provided), return
    if not pdf_path:
        pdf_filename_label.config(text="No PDF selected")
        return False
    
    # Validate PDF extension
    if not pdf_path.lower().endswith(".pdf"):
        msg = "Please select a valid PDF file."
        if use_dialog:
            messagebox.showerror("Invalid File", msg)
        else:
            print(f"[ERROR] {msg}")
        pdf_filename_label.config(text="No PDF selected")
        return False
    
    # Check if file exists (important for non-dialog modes)
    if not os.path.exists(pdf_path):
        msg = f"PDF file not found: {pdf_path}"
        if use_dialog:
            messagebox.showerror("File Not Found", msg)
        else:
            print(f"[ERROR] {msg}")
        pdf_filename_label.config(text="No PDF selected")
        return False
    
    # Try to process the PDF
    try:
        reader = PdfReader(pdf_path)
        pdf_pages = [page.extract_text() if page.extract_text() else "Empty Page" for page in reader.pages]
        filename = os.path.basename(pdf_path)
        pdf_filename_label.config(text=filename)
        print(f"[DEBUG] PDF loaded with {len(pdf_pages)} pages.")
        return True
    except Exception as e:
        traceback.print_exc()
        msg = f"Failed to process PDF: {e}"
        if use_dialog:
            messagebox.showerror("Error", msg)
        else:
            print(f"[ERROR] {msg}")
        pdf_filename_label.config(text="No PDF selected")
        return False

def browse_pdf():
    """Opens a file dialog to select and load a PDF file."""
    return load_pdf(use_dialog=True)

def preload_pdf(pdf_path):
    """Load a PDF file programmatically without using the file dialog."""
    return load_pdf(pdf_path, use_dialog=False)

# ------------------------ RESPONSE DISPLAY HANDLING ------------------------
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
    Validates answers against criteria and replaces them if needed.
    Criteria: A-Z, 0-9, spaces only, max 25 chars.
    """
    valid_answers = []
    for answer in answers:
        # Keep for answer validation: alphanumeric + spaces only
        prompt = (
            f"Answer: {answer}\n\n"
            "Rules for valid answers:\n"
            "1. Only uppercase letters (A-Z), numbers (0-9), and spaces allowed\n"
            "2. No special characters\n"
            "3. Maximum 25 characters\n"
            "4. Must be a coherent term or phrase\n"
            "5. STRONGLY PREFER single words over phrases when possible\n\n"
            "Provide a valid answer that is as close as possible to the original, "
            "following all rules. If the original requires minimal or no changes, keep it as is.\n"
            "Respond with the valid answer only, no explanation."
        )
        
        api_result = send_prompt(prompt, content_model_dropdown)
        if api_result:
            valid_answers.append(api_result.strip().upper())
        else:
            print(f"[ERROR] Exception during re-generation for an answer on page {page_index+1}")
    return valid_answers

# ------------------------ API PROMPT HANDLING ------------------------
def send_prompt(prompt, model_dropdown):
    """
    Send a prompt to the OpenAI API using the specified model dropdown.
    
    Args:
        prompt (str): The prompt to send to the API
        model_dropdown (ttk.Combobox): The dropdown widget containing the model selection
        
    Returns:
        str: The API response text or None if an error occurred
    """
    selected_model = model_dropdown.get()
    print(f"[DEBUG] Using model '{selected_model}' for API request")
    
    try:
        messages = [{"role": "user", "content": prompt}]
        params = {
            "model": selected_model, 
            "messages": messages,
            "stream": False
        }
        
        # Start API request timer
        start_time = time.time()
        response = client.chat.completions.create(**params)
        elapsed_time = time.time() - start_time
        
        # Log successful request with timing
        print(f"[DEBUG] API request completed in {elapsed_time:.2f} seconds")
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] Exception during API request: {str(e)}")
        # For OpenAI errors, try to get more details
        if hasattr(e, 'response') and hasattr(e.response, 'json'):
            try:
                error_details = e.response.json()
                print(f"[ERROR] API error details: {error_details}")
            except:
                pass
        return None

# ------------------------ PROMPT GENERATION HELPERS ------------------------
def process_page_metadata(page_index, page_text):
    """Process both section info and answer keywords in a single API call."""
    prompt = (
        f"PDF Content:\n{page_text}\n\n"
        "Extract the following information from the PDF content:\n\n"
        "1. SECTIONTITLE: The title of the section\n"
        "2. SECTIONNUMBER: The section number\n"
        "3. PAGENUMBER: The page number\n"
        "4. ANSWERS: A list of all key words/phrases that meet these criteria:\n"
        "   - Uppercase letters (A-Z) and numbers only\n"
        "   - May include spaces\n"
        "   - No special characters\n"
        "   - 25 characters or fewer\n"
        "   - Unique, concise, directly from document\n"
        "   - STRONGLY PREFER single words over phrases when possible\n\n"
        "Format your response exactly as follows:\n"
        "SECTIONTITLE | SECTIONNUMBER | PAGENUMBER\n"
        "ANSWER1, ANSWER2, ANSWER3, ..."
    )
    
    api_result = send_prompt(prompt, content_model_dropdown)
    if api_result:
        lines = api_result.strip().split('\n')
        if len(lines) >= 2:
            # Process section info
            section_parts = lines[0].split('|')
            if len(section_parts) >= 3:
                section_title[page_index] = section_parts[0].strip()
                section_number[page_index] = section_parts[1].strip()
                page_number[page_index] = section_parts[2].strip()
            
            # Process answers
            answer_line = lines[1]
            raw_answers = [ans.strip().upper() for ans in answer_line.split(',') if ans.strip()]
            validated_answers = validate_and_replace_answers(page_index, raw_answers)
            answer_array[page_index] = validated_answers
            
            return True
    return False

def process_clues(page_index):
    """
    Helper function to process clues for all answers on a single page.
    Generates clues for each answer using all available ClueTypes.
    """
    if page_index not in answer_array or not answer_array[page_index]:
        print(f"[ERROR] No answers available for page {page_index+1}.")
        return False
        
    for answer in answer_array[page_index]:
        process_clue_for_answer(answer, page_index=page_index)
    print(f"[DEBUG] Generated clues for all answers on page {page_index + 1}.")
    return True

# ------------------------ COMBINED PROMPTS (All Steps) ------------------------
def run_combined_prompts():
    """Streamlined workflow processing all pages sequentially."""
    if not pdf_pages:
        print("[ERROR] No PDF content available. Please load a PDF first.")
        return

    # Clear existing tabs and data
    for page_idx in list(tab_data.keys()):
        if page_idx in tab_data:
            tree = tab_data[page_idx]["treeview"]
            tree.delete(*tree.get_children())  # Clear existing rows
            page_clue_types[page_idx] = None  # Reset the clue type sequence

    for page_index, page_text in enumerate(pdf_pages):
        # Combined section info and answer extraction
        if process_page_metadata(page_index, page_text):
            # Create tab for displaying results or use existing tab
            if page_index not in tab_data:
                create_tab(page_index, "", page_text)
            
            # Generate clues for each answer
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
    Each answer will have clues for all available ClueTypes.
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
            prompt = (
                f"PDF Content:\n{pdf_text}\n\n"
                "Determine the ANSWER key words/phrases from the PDF that meet these criteria:\n"
                "- Contain only UPPERCASE letters (A-Z) and numbers if any.\n"
                "- May include spaces.\n"
                "- No special characters, dashes, or punctuation.\n"
                "- 25 characters or fewer.\n"
                "- All answers must be UNIQUE, concise, and directly sourced from the document.\n"
                "- For phrases with repetitive words, drop the redundant word.\n"
                "- STRONGLY PREFER single words over phrases when possible.\n"
                "- ANSWER must not match or be similar to the SECTIONTITLE.\n"
                "Output the answers as a comma-separated list."
            )
            print(f"[DEBUG] Resubmit: Sending prompt for new answer keywords for page {page_index+1}:\n{prompt}")
            
            api_result = send_prompt(prompt, content_model_dropdown)
            if api_result:
                new_answers = validate_and_replace_answers(
                    page_index,
                    [ans.strip().upper() for ans in api_result.split(',') if ans.strip()]
                )
                answer_array[page_index] = new_answers
                print(f"[DEBUG] Resubmit: Parsed new ANSWER_ARRAY for page {page_index+1}: {new_answers}")
                for answer in new_answers:
                    process_clue_for_answer(answer, page_index=page_index)
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
    """
    Handles arrow key navigation in dropdowns.
    """
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
    if dropdown == content_model_dropdown:
        update_model_info('content')
    elif dropdown == clue_model_dropdown:
        update_model_info('clue')
    return "break"

# ------------------------ REMAINING PROMPTS (Generate Clues) ------------------------
def get_next_clue_type(page_index=0):
    """Returns the next randomized clue type for the given page."""
    if page_index not in page_clue_types or not page_clue_types[page_index]:
        shuffled = CLUE_TYPES.copy()
        random.shuffle(shuffled)
        page_clue_types[page_index] = shuffled
    return page_clue_types[page_index].pop(0)

def process_clue_for_answer(answer, page_index=0, specified_clue_type=None, max_attempts=3):
    """
    Builds a prompt to generate clues for the given answer using ALL available CLUETYPES
    or a specified ClueType if provided, then sends a single API call to get all responses.
    Uses the Clue Model dropdown.
    ClueType 'Acrostic' is only allowed if the answer (excluding spaces) has 10 or fewer letters.
    Enhanced to prevent answers from appearing in generated clues.
    Now with detailed debugging for batch processing.
    Automatically resubmits clues that contain the answer word.
    """
    answer_length = len(answer.replace(" ", ""))
    pdf_text = pdf_pages[page_index] if pdf_pages and len(pdf_pages) > page_index else "Cached PDF Context"
    
    # Create tab if it doesn't exist
    if page_index not in tab_data:
        create_tab(page_index, "", pdf_text)
    tree = tab_data[page_index]["treeview"]
    
    # For debugging: Get the selected model
    selected_model = clue_model_dropdown.get()
    print(f"[DEBUG] Current selected model for clue generation: '{selected_model}'")
    
    # If a specific clue type is provided, only process that one
    if specified_clue_type:
        clue_types_to_process = [specified_clue_type]
        
        # Skip if Acrostic is specified but answer is too long
        if specified_clue_type == "Acrostic" and answer_length > 10:
            print(f"[WARNING] Acrostic not suitable for answer '{answer}' (too long). Skipping.")
            return False
           
        # Try to generate a valid clue up to max_attempts times 
        for attempt in range(max_attempts):
            # Process single clue type with existing method
            clue_type_desc = CLUE_TYPE_DESC.get(specified_clue_type, "")
            prompt = (
                f"PDF Content:\n{pdf_text}\n\n"
                f"Generate a detailed and engaging clue for the answer '{answer}' using clue type '{specified_clue_type}', '{clue_type_desc}'. "
                "IMPORTANT: Your clue MUST NOT contain the answer itself or directly reveal it. "
                f"The clue should challenge the user to figure out the answer without explicitly mentioning it or any part of '{answer}'. "
                "Respond with a single line containing only the clue text. Do not include any labels or explanation."
            )
            
            # Add special instructions for problematic clue types
            if specified_clue_type == "Rebus":
                prompt += " For this Rebus clue, do not include '= [ANSWER]' or any equation that shows the answer. Only provide the symbols/emojis."

            if attempt > 0:
                print(f"[DEBUG] Retry attempt #{attempt+1} for answer '{answer}' with clue type '{specified_clue_type}' on page {page_index+1}")
            else:
                print(f"[DEBUG] Prompt for answer '{answer}' with clue type '{specified_clue_type}' on page {page_index+1}")
            
            api_result = send_prompt(prompt, clue_model_dropdown)
            if api_result:
                # Clean up the response
                if api_result.lower().startswith("clue:"):
                    api_result = api_result[len("clue:"):].strip()
                api_result = " ".join(api_result.splitlines()).replace("**", "").replace('"', "").strip()
                
                # Check if answer is in clue
                if answer.lower() in api_result.lower():
                    if attempt < max_attempts - 1:
                        print(f"[DEBUG] Answer found in clue, retrying (attempt {attempt+1}/{max_attempts})")
                        continue
                    else:
                        print(f"[WARNING] Failed to generate valid clue after {max_attempts} attempts, using best effort")
                
                # Insert into tree
                tree.insert("", "end", values=(section_title.get(page_index, ""), 
                                            section_number.get(page_index, ""), 
                                            page_number.get(page_index, ""), 
                                            specified_clue_type, api_result, answer))
                return True
            
            # If API call failed, try again
            if attempt < max_attempts - 1:
                print(f"[WARNING] API call failed, retrying (attempt {attempt+1}/{max_attempts})")
            else:
                print(f"[ERROR] Failed to generate clue after {max_attempts} attempts")
                return False
                
        # If we get here, all attempts failed
        return False
    
    # For all clue types, make a single batched API call
    # Filter out Acrostic if answer is too long
    clue_types_to_process = [ct for ct in CLUE_TYPES if not (ct == "Acrostic" and answer_length > 10)]
    print(f"[DEBUG] Processing these clue types for '{answer}': {clue_types_to_process}")
    
    # Build a comprehensive prompt requesting all clue types at once
    clue_type_requests = []
    for clue_type in clue_types_to_process:
        clue_type_desc = CLUE_TYPE_DESC.get(clue_type, "")
        clue_type_requests.append(f"Clue Type: {clue_type}\nDescription: {clue_type_desc}")
    
    all_clue_types_text = "\n\n".join(clue_type_requests)
    
    # Track which clue types need to be resubmitted
    clue_types_needing_resubmission = set()
    
    # Make up to max_attempts batch requests
    for batch_attempt in range(max_attempts):
        if batch_attempt > 0:
            # If this is a retry, only request the problematic clue types
            if not clue_types_needing_resubmission:
                break
                
            clue_type_requests = []
            for clue_type in clue_types_needing_resubmission:
                clue_type_desc = CLUE_TYPE_DESC.get(clue_type, "")
                clue_type_requests.append(f"Clue Type: {clue_type}\nDescription: {clue_type_desc}")
            
            all_clue_types_text = "\n\n".join(clue_type_requests)
            print(f"[DEBUG] Retry batch attempt #{batch_attempt+1} for {len(clue_types_needing_resubmission)} clue types")
        
        # Create a simpler prompt for better batch processing
        prompt = (
            f"ANSWER: '{answer}'\n\n"
            f"TASK: Generate one unique clue for each of the following clue types:\n\n"
            f"{all_clue_types_text}\n\n"
            "FORMAT REQUIREMENTS:\n"
            "1. Format each clue as: '[ClueType] Clue text'\n"
            "2. Respond with a numbered list, one clue per line\n"
            "3. Be concise - clues should be under 15 words each\n"
            f"4. IMPORTANT: Do NOT include the answer word '{answer}' or any variation of it in any clue\n\n"
            "EXAMPLE RESPONSE FORMAT:\n"
            "1. [Definition] Brief definition clue here\n"
            "2. [Synonym] Brief synonym clue here\n"
            "And so on for each clue type."
        )
        
        # Add truncated PDF context at the end to avoid overweighting it
        simplified_pdf = pdf_text[:300] + "..." if len(pdf_text) > 300 else pdf_text
        prompt += f"\n\nCONTEXT (for reference):\n{simplified_pdf}"
        
        if batch_attempt == 0:
            print(f"[DEBUG] Making batch request for all clue types for answer '{answer}' on page {page_index+1}")
        else:
            print(f"[DEBUG] Making retry batch request for {len(clue_types_needing_resubmission)} clue types for answer '{answer}'")
        print(f"[DEBUG] Prompt length: {len(prompt)} characters")
        
        # Make a single API call for all clue types without token limitation
        api_result = send_prompt(prompt, clue_model_dropdown)
        
        # Reset the resubmission set for this attempt
        clue_types_needing_resubmission = set()
        
        # Check if we got a valid response
        if api_result and len(api_result.strip()) > 0:
            print(f"[DEBUG] Received batch response of {len(api_result)} characters")
            # Parse the results to extract individual clues
            # Expected format: "[ClueType] Clue text" or "1. [ClueType] Clue text"
            clue_lines = api_result.strip().split('\n')
            print(f"[DEBUG] Response contains {len(clue_lines)} lines")
            
            clues_processed = 0
            for line in clue_lines:
                # Skip empty lines
                if not line.strip():
                    continue
                    
                print(f"[DEBUG] Processing line: {line[:50]}..." if len(line) > 50 else f"[DEBUG] Processing line: {line}")
                    
                # Remove numbering if present (e.g., "1. [ClueType] Clue text")
                if re.match(r'^\d+\.', line):
                    line = re.sub(r'^\d+\.\s*', '', line)
                
                # Extract clue type and text using regex
                match = re.match(r'\[([^\]]+)\](.*)', line)
                if match:
                    clue_type = match.group(1).strip()
                    clue_text = match.group(2).strip()
                    
                    print(f"[DEBUG] Extracted clue type: '{clue_type}', clue text: '{clue_text[:30]}...'")
                    
                    # Verify clue type is valid
                    if clue_type in CLUE_TYPES:
                        # Check if answer is in clue
                        if answer.lower() in clue_text.lower():
                            # If this is not the final attempt, add to resubmission list
                            if batch_attempt < max_attempts - 1:
                                print(f"[DEBUG] Answer found in '{clue_type}' clue, will retry")
                                clue_types_needing_resubmission.add(clue_type)
                                continue
                            else:
                                print(f"[WARNING] Failed to generate valid clue after {max_attempts} attempts, using best effort")
                        
                        clues_processed += 1
                        # Insert into tree
                        tree.insert("", "end", values=(section_title.get(page_index, ""), 
                                                    section_number.get(page_index, ""), 
                                                    page_number.get(page_index, ""), 
                                                    clue_type, clue_text, answer))
                    else:
                        print(f"[WARNING] Invalid clue type '{clue_type}' in response for answer '{answer}'")
                else:
                    print(f"[WARNING] Could not parse line: '{line}'")
            
            # If we have some clues needing resubmission but also processed some valid ones
            if clues_processed > 0:
                if not clue_types_needing_resubmission or batch_attempt == max_attempts - 1:
                    print(f"[DEBUG] Successfully processed {clues_processed} clues for answer '{answer}' in batch mode")
                    return True
            else:
                print(f"[ERROR] Batch API call succeeded but no valid clues found. Response format may be incorrect.")
                # Print the first part of the API response for debugging
                print(f"[DEBUG] First 300 chars of response: {api_result[:300]}...")
                return False
        else:
            # If batch API call failed completely and this is the last attempt
            if batch_attempt == max_attempts - 1:
                print(f"[ERROR] Batch API call failed for answer '{answer}' after {max_attempts} attempts. No response received.")
                return False
    
    # If we get here and still have clue types needing resubmission
    if clue_types_needing_resubmission:
        print(f"[WARNING] Could not generate valid clues for {len(clue_types_needing_resubmission)} clue types after {max_attempts} attempts")
    
    return True

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

# Content Model Dropdown (replaces section and answer model dropdowns)
content_frame = tk.Frame(model_prompt_frame, bg="#2e2e2e")
content_frame.pack(side=tk.LEFT, padx=5)
tk.Label(content_frame, text="Content Analysis Model:", bg="#2e2e2e", fg="white").pack(anchor="w")
content_model_var = tk.StringVar()
content_model_dropdown = ttk.Combobox(content_frame, textvariable=content_model_var, state="readonly")
content_model_dropdown.pack(anchor="w")
content_model_dropdown.bind("<Up>", lambda e: on_arrow_key_dropdown(e, content_model_dropdown))
content_model_dropdown.bind("<Down>", lambda e: on_arrow_key_dropdown(e, content_model_dropdown))
content_model_info_label = tk.Label(content_frame, text="Model details will appear here", bg="#2e2e2e", fg="white")
content_model_info_label.pack(anchor="w")
content_model_var.trace_add("write", lambda *args: update_model_info('content'))

# Clue Model Dropdown (keep this as is)
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
clue_model_var.trace_add("write", lambda *args: update_model_info('clue'))

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
    pdf_path = "./EH200-1.pdf"
    if not preload_pdf(pdf_path):
        print(f"[WARNING] Failed to preload default PDF file: {pdf_path}")

# Call initialization function after a short delay to ensure GUI is fully loaded
root.after(100, initialize_app)

root.mainloop()
