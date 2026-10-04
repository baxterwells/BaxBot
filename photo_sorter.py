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
import multiprocessing
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# Register HEIF opener with Pillow
pillow_heif.register_heif_opener()

# --- CONFIGURATION ---
VISUAL_MODEL_NAME = "llava"
JUDGE_MODEL_NAME = "mistral"
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif")
MAX_WORKERS = 4 

def get_user_inputs():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    source_folder = None
    output_name = "Sorted Photos"
    target_subjects_input = ""
    try:
        source_folder = filedialog.askdirectory(title="Select the folder containing images you want to sort")
        if source_folder:
            name_input = simpledialog.askstring("Folder Name", "What should the master results folder be named?", initialvalue=output_name)
            if name_input: output_name = name_input
            keywords_input = simpledialog.askstring("Search Criteria", "What are you looking for? (e.g., 'brown dog, blue car')", initialvalue="")
            if keywords_input: target_subjects_input = keywords_input
    finally:
        root.quit()
        root.destroy()
    return source_folder, output_name, target_subjects_input

def create_cancel_window():
    cancel_state = {"requested": False}
    root = tk.Tk()
    root.title("Photo Search")
    root.attributes("-topmost", True)
    root.resizable(False, False)
    def request_cancel(event=None): cancel_state["requested"] = True
    tk.Label(root, text="Searching photos...").pack(padx=20, pady=(14, 4))
    tk.Button(root, text="Cancel (Command+X)", command=request_cancel).pack(padx=20, pady=(4, 14))
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

def worker_task(img_path, visual_prompt, target_subjects_list):
    filename = os.path.basename(img_path)
    try:
        # --- STAGE 1: VISION ---
        vision_llm = ChatOllama(model=VISUAL_MODEL_NAME)
        base64_image = encode_image_to_base64(img_path)
        message = HumanMessage(content=[
            {"type": "text", "text": visual_prompt},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
        ])
        description = vision_llm.invoke([message]).content.strip().lower()

        # --- STAGE 2: JUDGE ---
        judge_llm = ChatOllama(model=JUDGE_MODEL_NAME)
        judge_prompt = (
            f"You are a precise classification tool. Identify if any of these exact categories are in the description.\n\n"
            f"RULES:\n"
            f"1. Return ONLY the exact keywords from the list below if they are found.\n"
            f"2. If NO match is found, you MUST respond with exactly one word: 'none'.\n"
            f"3. DO NOT provide explanations, parentheses, or any extra text.\n\n"
            f"CATEGORIES: {', '.join(target_subjects_list)}\n"
            f"DESCRIPTION: {description}"
        )
        
        judgment_raw = judge_llm.invoke([judge_prompt]).content.strip().lower()
        
        # --- STAGE 3: STRICT MEMBERSHIP PARSING ---
        # This is the fix. We only accept words that are EXACTLY in your target list.
        matches = []
        # First, ignore the 'none' signal
        if 'none' not in judgment_raw:
            # Split by comma and clean up
            potential_matches = [m.strip() for m in judgment_raw.split(',')]
            # Only keep the match if it is an EXACT match to one of your input words
            for p in potential_matches:
                if p in target_subjects_list:
                    matches.append(p)

        return {
            "status": "SUCCESS",
            "filename": filename,
            "img_path": img_path,
            "ai_description": description,
            "matches": matches,
            "error": None
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "filename": filename,
            "img_path": img_path,
            "ai_description": str(e),
            "error": str(e)
        }

def main():
    knowledge_folder, saved_folder_name, raw_input_string = get_user_inputs()
    if not knowledge_folder: return

    target_subjects = [s.strip().lower() for s in raw_input_string.split(',') if s.strip()]
    if not target_subjects: return

    base_saved_folder = os.path.join(os.getcwd(), saved_folder_name)
    if not os.path.exists(base_saved_folder): os.makedirs(base_saved_folder)

    print("\n==================================================")
    print("AI VISION-JUDGE PIPELINE (STRICT MODE)")
    print(f"Vision: {VISUAL_MODEL_NAME} | Judge: {JUDGE_MODEL_NAME}")
    print("==================================================")
    
    image_files = get_images_from_folder(knowledge_folder)
    if not image_files: return

    print(f"\n[!] SEARCHING FOR: {', '.join(target_subjects)}")
    print(f"[!] PROCESSING {len(image_files)} IMAGES...\n")

    stats = {"found": 0, "skipped": 0, "errors": 0}
    log_file_path = os.path.join(base_saved_folder, "detection_report.csv")
    
    vision_prompt = (
        "Provide a detailed description of this image. "
        "Include colors, specific objects, and their characteristics."
    )

    with open(log_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        log_writer = csv.writer(csvfile)
        log_writer.writerow(["Timestamp", "Filename", "Found_Subjects", "AI_Description"])

        pool = multiprocessing.Pool(processes=MAX_WORKERS)
        cancel_window, cancel_state = create_cancel_window()
        cancelled = False
        
        try:
            async_results = []
            for img in image_files:
                res = pool.apply_async(worker_task, (img, vision_prompt, target_subjects))
                async_results.append(res)

            completed_count = 0
            
            while async_results:
                cancel_window.update_idletasks()
                cancel_window.update()
                
                if cancel_state["requested"]:
                    print("\n[!] Cancellation Requested. Terminating processes...")
                    cancelled = True
                    pool.terminate() 
                    break

                for i in range(len(async_results) - 1, -1, -1):
                    result_obj = async_results[i]
                    
                    if result_obj.ready():
                        result = result_obj.get()
                        async_results.pop(i) 
                        completed_count += 1
                        
                        filename = result['filename']
                        img_path = result['img_path']
                        ai_desc = result['ai_description']

                        print(f"[{completed_count}/{len(image_files)}] Finished: {filename}", end="\r")

                        if result['status'] == "SUCCESS":
                            matches = result['matches']
                            if matches:
                                stats["found"] += 1
                                unique_matches = list(set(matches))
                                for match in unique_matches:
                                    subject_folder = os.path.join(base_saved_folder, match)
                                    if not os.path.exists(subject_folder): os.makedirs(subject_folder)
                                    shutil.copy2(img_path, os.path.join(subject_folder, f"{filename}.jpg"))
                                log_writer.writerow([datetime.now(), filename, ", ".join(unique_matches), ai_desc])
                            else:
                                stats["skipped"] += 1
                                log_writer.writerow([datetime.now(), filename, "NONE", ai_desc])
                        else:
                            print(f"\n[!] Error on {filename}: {result['error']}")
                            log_writer.writerow([datetime.now(), filename, "ERROR", result['error']])
                            stats["errors"] += 1

                import time
                time.sleep(0.1)

        except KeyboardInterrupt:
            cancelled = True
            pool.terminate()
        finally:
            cancel_window.destroy()
            pool.close()
            pool.join()

    print("\n==================================================")
    print("SEARCH CANCELLED" if cancelled else "MISSION COMPLETE")
    print(f"Found: {stats['found']} | Skipped: {stats['skipped']} | Errors: {stats['errors']}")
    print(f"Results saved in: {base_saved_folder}")
    print("==================================================\n")

if __name__ == "__main__":
    main()
