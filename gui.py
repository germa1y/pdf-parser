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
        # Set dropdown values for all three prompt types with defaults:
        content_model_dropdown['values'] = available_models
        content_model_dropdown.set("gpt-4o-mini" if "gpt-4o-mini" in available_models else available_models[0])
        
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
        is_new = details.get("new_model", False)
        model_info_text = f"Input: ${input_price} | Output: ${output_price} | Context: {context_size}"
        
        if is_new:
            info_label.config(text=model_info_text, font=("Helvetica", 10, "bold"), fg="green")
        else:
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
        
        api_result = send_prompt(prompt, content_model_dropdown, max_tokens=100)
        if api_result:
            valid_answers.append(api_result.strip().upper())
        else:
            print(f"[ERROR] Exception during re-generation for an answer on page {page_index+1}")
    return valid_answers

# ------------------------ API PROMPT HANDLING ------------------------
def send_prompt(prompt, model_dropdown, max_tokens=4096):
    """
    Send a prompt to the OpenAI API using the specified model dropdown.
    
    Args:
        prompt (str): The prompt to send to the API
        model_dropdown (ttk.Combobox): The dropdown widget containing the model selection
        max_tokens (int): Maximum tokens for completion
        
    Returns:
        str: The API response text or None if an error occurred
    """
    selected_model = model_dropdown.get()
    print(f"[DEBUG] Using model '{selected_model}' for API request.")
    
    try:
        messages = [{"role": "user", "content": prompt}]
        params = {
            "model": selected_model, 
            "messages": messages, 
            "max_completion_tokens": max_tokens, 
            "stream": False
        }
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] Exception during API request: {e}")
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
        "4. ANSWERS: A list of 5-10 key words/phrases that meet these criteria:\n"
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
    """Helper function to process clues for all answers on a single page."""
    if page_index not in answer_array or not answer_array[page_index]:
        print(f"[ERROR] No answers available for page {page_index+1}.")
        return False
        
    for answer in answer_array[page_index]:
        process_clue_for_answer(answer, page_index=page_index)
    print(f"[DEBUG] Last row populated for page {page_index + 1}.")
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

def process_clue_for_answer(answer, page_index=0):
    """
    Builds the prompt to generate a clue for the given answer using a randomized CLUETYPE
    and its description, then sends it to the API.
    Uses the Clue Model dropdown.
    ClueType 'Acrostic' is only allowed if the answer (excluding spaces) has 10 or fewer letters.
    Enhanced to prevent answers from appearing in generated clues.
    """
    answer_length = len(answer.replace(" ", ""))
    while True:
        clue_type = get_next_clue_type(page_index)
        if clue_type == "Acrostic" and answer_length > 10:
            continue
        break
    clue_type_desc = CLUE_TYPE_DESC.get(clue_type, "")
    pdf_text = pdf_pages[page_index] if pdf_pages and len(pdf_pages) > page_index else "Cached PDF Context"
    
    # Enhanced prompt with explicit instructions not to include the answer
    prompt = (
        f"PDF Content:\n{pdf_text}\n\n"
        f"Generate a detailed and engaging clue for the answer '{answer}' using clue type '{clue_type}', '{clue_type_desc}'. "
        "IMPORTANT: Your clue MUST NOT contain the answer itself or directly reveal it. "
        "The clue should challenge the user to figure out the answer without explicitly mentioning it. "
        "Respond with a single line containing only the clue text. Do not include any labels or extra explanation."
    )
    
    # Add special instructions for problematic clue types
    if clue_type == "Rebus":
        prompt += " For this Rebus clue, do not include '= [ANSWER]' or any equation that shows the answer. Only provide the symbols/emojis."

    print(f"[DEBUG] Prompt for answer '{answer}' on page {page_index+1}:\n{prompt}")
    
    # Try up to 3 times to get a valid clue without the answer
    max_attempts = 3
    attempt = 0
    valid_clue = False
    api_result = None
    
    while attempt < max_attempts and not valid_clue:
        api_result = send_prompt(prompt, clue_model_dropdown)
        if not api_result:
            error_msg = "Error generating clue"
            print(f"[ERROR] Failed to generate clue for answer '{answer}' on page {page_index+1}")
            return error_msg
            
        # Clean up the response
        if api_result.lower().startswith("clue:"):
            api_result = api_result[len("clue:"):].strip()
        api_result = " ".join(api_result.splitlines()).replace("**", "").replace('"', "").strip()
        
        # Check if the answer is directly in the clue (case insensitive)
        if answer.lower() not in api_result.lower():
            valid_clue = True
        else:
            attempt += 1
            print(f"[WARNING] Answer '{answer}' found in clue. Retrying with stronger instructions (Attempt {attempt}/{max_attempts})")
            
            # Create a stronger prompt for the retry
            prompt = (
                f"Generate a challenging clue for '{answer}' using clue type '{clue_type}'. "
                f"CRITICAL REQUIREMENT: The clue MUST NOT contain '{answer}' or directly reveal it. "
                f"Previous attempt failed because it revealed the answer. "
                f"Respond with only the clue text without labels or explanation."
            )
            
            # Add even more specific instructions for problematic clue types
            if clue_type == "Rebus":
                prompt += f" Use only symbols and emojis that suggest the concept without spelling out '{answer}' or including '= {answer}'."
    
    # If we still have the answer in the clue after all attempts, add a warning note
    if not valid_clue and api_result:
        api_result = f"[NEEDS REVIEW - Answer may be revealed] {api_result}"
    
    print(f"[DEBUG] API Response for answer '{answer}' on page {page_index+1}: {api_result}")
    
    # Update the UI with the result
    if page_index not in tab_data:
        create_tab(page_index, "", pdf_text)
    tree = tab_data[page_index]["treeview"]
    tree.insert("", "end", values=(section_title.get(page_index, ""), 
                                   section_number.get(page_index, ""), 
                                   page_number.get(page_index, ""), 
                                   clue_type, api_result, answer))
    return api_result

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
    pdf_path = ".\EH200-1.pdf"
    if not preload_pdf(pdf_path):
        print(f"[WARNING] Failed to preload default PDF file: {pdf_path}")

# Call initialization function after a short delay to ensure GUI is fully loaded
root.after(100, initialize_app)

root.mainloop()
