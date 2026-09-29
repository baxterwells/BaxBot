import base64
import os
import glob
import shutil
import io
import csv
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from datetime import datetime
from PIL import Image
import pillow_heif
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

# Register HEIF opener with Pillow
pillow_heif.register_heif_opener()

# --- CONFIGURATION ---
MODEL_NAME = "llava"
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif")
MAX_WORKERS = 4 

def get_user_inputs():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    source_folder = None
    output_name = "Found Photos"
    target_subjects_input = "" # Initialize empty

    try:
        # 1. Get Folder
        source_folder = filedialog.askdirectory(title="Select the folder containing images")
        
        if source_folder:
            # 2. Get Output Folder Name
            name_input = simpledialog.askstring(
                "Folder Name", 
                "What should the main results folder be named?", 
                initialvalue=output_name
            )
            if name_input:
                output_name = name_input
            
            # 3. NEW: Get Search Keywords via Popup
            keywords_input = simpledialog.askstring(
                "Search Criteria", 
                "What are you looking for? (e.g., 'people, pets, cars')",
                initialvalue=""
            )
            if keywords_input:
                target_subjects_input = keywords_input
                
    finally:
        root.quit()
        root.destroy()

    # Return three values now instead of two
    return source_folder, output_name, target_subjects_input


def create_cancel_window():
    cancel_state = {"requested": False}
    root = tk.Tk()
    root.title("Photo Search")
    root.attributes("-topmost", True)
    root.resizable(False, False)

    def request_cancel(event=None):
        cancel_state["requested"] = True

    tk.Label(root, text="Searching photos...").pack(padx=20, pady=(14, 4))
    tk.Button(root, text="Cancel (Command+X)", command=request_cancel).pack(
        padx=20, pady=(4, 14)
    )
    root.bind("<Command-x>", request_cancel)
    root.bind("<Control-x>", request_cancel)
    root.protocol("WM_DELETE_WINDOW", request_cancel)
    return root, cancel_state


def encode_image_to_base64(image_path):
    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")
        buffer = io.BytesIO()
        rgb_img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def get_images_from_folder(folder_path):
    if not os.path.exists(folder_path): return []
    image_paths = []
    for ext in SUPPORTED_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        image_paths.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
    return image_paths

def worker_task(img_path, prompt, target_subjects_list):
    filename = os.path.basename(img_path)
    try:
        llm = ChatOllama(model=MODEL_NAME)
        base64_image = encode_image_to_base64(img_path)
        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
        ])
        response = llm.invoke([message])
        ai_res_text = response.content.strip().lower()
        
        return {
            "status": "SUCCESS",
            "filename": filename,
            "img_path": img_path,
            "ai_response": ai_res_text,
            "error": None
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "filename": filename,
            "img_path": img_path,
            "ai_response": str(e),
            "error": str(e)
        }

def main():
    # UPDATED: Unpack three values from the input function
    knowledge_folder, saved_folder_name, raw_input_string = get_user_inputs()
    
    if not knowledge_folder:
        print("No folder selected. Exiting.")
        return

    # Process the string from the popup into a list
    target_subjects = [s.strip() for s in raw_input_string.lower().split(',') if s.strip()]
    
    if not target_subjects:
        print("No subjects provided. Exiting.")
        return

    # Create the base output folder
    base_saved_folder = os.path.join(os.getcwd(), saved_folder_name)
    if not os.path.exists(base_saved_folder):
        os.makedirs(base_saved_folder)

    print("\n==================================================")
    print("PARALLEL MULTI-SUBJECT SEARCH AGENT")
    print(f"Concurrency Level: {MAX_WORKERS} workers")
    print("==================================================")
    
    image_files = get_images_from_folder(knowledge_folder)
    if not image_files:
        print(f"No images found.")
        return

    print(f"\n[!] TARGETS: {', '.join(target_subjects).upper()}")
    print(f"[!] PROCESSING {len(image_files)} IMAGES...\n")

    stats = {"found": 0, "skipped": 0, "errors": 0}
    log_file_path = os.path.join(base_saved_folder, "detection_report.csv")
    
    subjects_str = ", ".join(target_subjects)
    combined_prompt = (
        f"Analyze this image. I am looking for these specific categories: [{subjects_str}]. "
        f"For every category in the list above that you see in the image, write the category name exactly as it appears in my list. "
        f"If you see something that belongs to a category (like a 'dog' belongs to 'pets'), write the category name: '{subjects_str}'. "
        f"Respond with the category names separated by commas. If nothing matches, say 'none'."
    )

    with open(log_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        log_writer = csv.writer(csvfile)
        log_writer.writerow(["Timestamp", "Filename", "Found_Subjects", "AI_Response"])

        executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
        cancel_window, cancel_state = create_cancel_window()
        cancelled = False
        
        try:
            futures = {executor.submit(worker_task, img, combined_prompt, target_subjects): img for img in image_files}
            pending = set(futures)
            completed_count = 0
            
            while pending:
                cancel_window.update_idletasks()
                cancel_window.update()
                if cancel_state["requested"]:
                    cancelled = True
                    for future in pending:
                        future.cancel()
                    break

                done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                for future in done:
                    completed_count += 1
                    result = future.result()
                    filename = result['filename']
                    img_path = result['img_path']
                    ai_res = result['ai_response']

                    print(f"[{completed_count}/{len(image_files)}] Finished: {filename}", end="\r")

                    if result['status'] == "SUCCESS":
                        matches = []
                        for subject in target_subjects:
                            if subject in ai_res:
                                matches.append(subject)
                        
                        if not matches:
                            for subject in target_subjects:
                                if subject.lower() in ai_res:
                                    matches.append(subject)

                        if matches:
                            stats["found"] += 1
                            unique_matches = list(set(matches))
                            for match in unique_matches:
                                subject_folder = os.path.join(base_saved_folder, match)
                                if not os.path.exists(subject_folder):
                                    os.makedirs(subject_folder)
                                shutil.copy2(img_path, os.path.join(subject_folder, filename))
                            
                            log_writer.writerow([datetime.now(), filename, ", ".join(unique_matches), ai_res])
                        else:
                            stats["skipped"] += 1
                            log_writer.writerow([datetime.now(), filename, "NONE", ai_res])
                    else:
                        print(f"\n[!] Error on {filename}: {result['error']}")
                        log_writer.writerow([datetime.now(), filename, "ERROR", result['error']])
                        stats["errors"] += 1

        except KeyboardInterrupt:
            cancelled = True
        finally:
            cancel_window.destroy()
            executor.shutdown(wait=not cancelled, cancel_futures=True)

    print("\n==================================================")
    print("SEARCH CANCELLED" if cancelled else "MISSION COMPLETE")
    print(f"Found: {stats['found']} | Skipped: {stats['skipped']} | Errors: {stats['errors']}")
    print(f"Results saved in: {base_saved_folder}")
    print("==================================================\n")

if __name__ == "__main__":
    main()
