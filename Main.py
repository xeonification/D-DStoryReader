import os
import re
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import soundfile as sf

# Initialize placeholder for model global variable
model = None

# --- INITIALIZE VISUAL GRAPHICAL WINDOW FIRST ---
root = tk.Tk()
root.title("Kokoro Voice Studio")
root.geometry("550x540")
root.resizable(False, False)

# Main App Status Banner
status_label = tk.Label(root, text="⚙️ Initializing local text frameworks...", fg="blue", font=("Arial", 10, "italic"))
status_label.pack(pady=5)

def async_model_load():
    global model
    try:
        from mlx_audio.tts.utils import load_model
        # Patch pointers
        try:
            from phonemizer.backend.espeak.wrapper import EspeakWrapper
            if not hasattr(EspeakWrapper, "set_data_path"):
                EspeakWrapper.set_data_path = lambda path: setattr(EspeakWrapper, "data_path", path)
        except Exception:
            pass
            
        status_label.config(text="📥 Fetching core model matrices from hub...", fg="orange")
        root.update()
        
        # Pull model weight files safely
        model = load_model("mlx-community/Kokoro-82M-bf16")
        
        status_label.config(text="✅ Apple Silicon Model Fully Loaded & Ready", fg="green")
        play_button.config(state=tk.NORMAL)
    except Exception as e:
        status_label.config(text="❌ Failed to initialize weights matrix", fg="red")
        messagebox.showerror("Error", f"Initialization crash: {e}")

def run_tts():
    if model is None:
        messagebox.showerror("Error", "Model is still caching files. Please wait.")
        return

    text_input = text_box.get("1.0", tk.END).strip()
    selected_voice = voice_dropdown.get()
    filename_input = filename_entry.get().strip()

    if not text_input:
        messagebox.showwarning("Warning", "Please enter text first!")
        return
        
    if not filename_input:
        filename_input = "pyl"
    else:
        filename_input = re.sub(r'[\/*?:\'\"<>|]', '', filename_input)

    lang_code = "en-gb" if selected_voice.startswith("b") else "en-us"
    play_button.config(text="Processing...", state=tk.DISABLED)
    status_label.config(text="🔊 Synthesizing audio stream channels...", fg="purple")
    root.update()

    # --- ULTRACLEAN CONTRACTION SANITIZATION ENGINE ---
    # Strip layout double quotes cleanly
    cleaned_text = text_input.replace('"', " ").replace("`", " ")
    
    # Standardise any variant of split contractions into fluent, clean words
    cleaned_text = re.sub(r"\b(don\s*t|don\'t|dont)\b", "do not", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\b(can\s*t|can\'t|cant)\b", "cannot", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\b(won\s*t|won\'t|wont)\b", "will not", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\b(isn\s*t|isn\'t|isnt)\b", "is not", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\b(aren\s*t|aren\'t|arent)\b", "are not", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\b(wasn\s*t|wasn\'t|wasnt)\b", "was not", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\b(weren\s*t|weren\'t|werent)\b", "were not", cleaned_text, flags=re.IGNORECASE)
    
    # Strip any remaining single hanging quotes or standalone 't letters
    cleaned_text = cleaned_text.replace("'", " ")
    cleaned_text = re.sub(r"\s+t\b", "", cleaned_text) # Wipes out any accidentally repeated trailing 't' markers
    
    # Standardise long dashes, hyphens, and brackets into clean pauses
    cleaned_text = cleaned_text.replace("—", ", ").replace("–", ", ").replace("-", " ")
    cleaned_text = cleaned_text.replace("(", ", ").replace(")", ", ")
    
    # Standard conversion parameters
    cleaned_text = cleaned_text.replace("%", " percent ").replace("&", " and ").replace("$", " dollars ")
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    initial_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned_text) if s.strip()]
    final_chunks = []
    for sentence in initial_sentences:
        if len(sentence.split()) > 12:
            final_chunks.extend([p.strip() for p in re.split(r"(?<=[,;:])\s+", sentence) if p.strip()])
        else:
            final_chunks.append(sentence)

    audio_chunks = []
    
    for idx, phrase in enumerate(final_chunks):
        # Trim remaining trailing symbols cleanly
        phrase = phrase.strip(" ,-–—")
        
        # Safety Check: Ignore lines missing alphanumerics
        if not re.search(r"[a-zA-Z0-9]", phrase) or len(phrase.strip()) < 2:
            print(f" -> Skipping fragment {idx+1} containing no speakable alphanumeric tokens.")
            continue
            
        print(f"Generating segment {idx+1}/{len(final_chunks)}: \"{phrase}\"")
        
        base_phrase = phrase.rstrip(".,;:!? ")
        last_char = base_phrase[-1] if base_phrase else "a"
        
        # Dynamic Middle-Splitter Calculation for the fallback strategy
        words = phrase.split()
        if len(words) > 2:
            midpoint = len(words) // 2
            mid_split_phrase = " ".join(words[:midpoint]) + "... " + " ".join(words[midpoint:])
        else:
            mid_split_phrase = phrase + " ."

        # Silent variations to shift internal matrix lengths
        text_variations = [
            phrase,                                      # 1. Base Text
            base_phrase + ", ",                          # 2. Soft trailing breath comma
            base_phrase + "...",                         # 3. Trailing ellipsis padding
            " " + phrase,                                # 4. Single leading alignment space
            "  " + phrase,                               # 5. Double leading alignment space
            "; " + phrase,                               # 6. Silent leading semicolon pause marker
            mid_split_phrase                             # 7. Safe mid-sentence breath break
        ]
        
        success = False
        for attempt_idx, text_attempt in enumerate(text_variations):
            try:
                audio_generator = model.generate(text_attempt, voice=selected_voice, lang_code=lang_code)
                segment_chunks = []
                for chunk in audio_generator:
                    if hasattr(chunk, "audio"):
                        segment_chunks.append(np.array(chunk.audio, dtype=np.float32).flatten())
                    elif isinstance(chunk, dict) and 'audio' in chunk:
                        segment_chunks.append(np.array(chunk["audio"], dtype=np.float32).flatten())
                    else:
                        segment_chunks.append(np.array(chunk, dtype=np.float32).flatten())
                
                if segment_chunks:
                    audio_chunks.extend(segment_chunks)
                    success = True
                    break 
            except ValueError as ve:
                if "broadcast_shapes" in str(ve):
                    print(f" -> Tensor shape error on attempt {attempt_idx+1}. Mutating array frames...")
                    continue
                else:
                    print(f"Skipped segment error: {ve}")
                    break
            except Exception as e:
                print(f"Skipped segment error: {e}")
                break
                
        if not success:
            print(f" ❌ Skipping phrase permanently: \"{phrase}\"")

    if audio_chunks:
        final_audio = np.concatenate(audio_chunks)
        output_path = os.path.expanduser(f"~/Desktop/{filename_input}.wav")
        sf.write(output_path, final_audio, 24000)
        status_label.config(text="✅ Apple Silicon Model Fully Loaded & Ready", fg="green")
        messagebox.showinfo("🎉 Success", f"Audio generated successfully!\nSaved to Desktop as {filename_input}.wav")
    else:
        messagebox.showerror("Error", "Failed to compile audio stream data.")

    play_button.config(text="Generate & Play", state=tk.NORMAL)

# --- USER INTERFACE DESIGN ELEMENTS ---
tk.Label(root, text="Enter your audiobook text payload below:", font=("Arial", 11)).pack(pady=5)

text_box = tk.Text(root, height=12, width=60, font=("Arial", 10))
text_box.pack(pady=5)
text_box.insert("1.0", "Testing the new desktop dashboard pipeline. Select a voice profile below and generate the wave files.")

voice_frame = tk.Frame(root)
voice_frame.pack(pady=10)

tk.Label(voice_frame, text="Select Voice Profile: ", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)

available_voices = ["bm_fable", "bm_george", "bm_lewis", "bf_emma", "bf_isabella", "am_michael", "am_adam", "af_heart", "af_bella", "af_nicole"]
voice_dropdown = ttk.Combobox(voice_frame, values=available_voices, state="readonly", width=15)
voice_dropdown.set("bm_fable")
voice_dropdown.pack(side=tk.LEFT, padx=5)

filename_frame = tk.Frame(root)
filename_frame.pack(pady=10)

tk.Label(filename_frame, text="Output File Name: ", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)

filename_entry = tk.Entry(filename_frame, width=25, font=("Arial", 10))
filename_entry.insert(0, "my_narration")
filename_entry.pack(side=tk.LEFT, padx=5)

tk.Label(filename_frame, text=".wav", font=("Arial", 10, "bold"), fg="gray").pack(side=tk.LEFT)

play_button = tk.Button(root, text="Generate & Play", font=("Arial", 11, "bold"), width=20, height=2, state=tk.DISABLED, command=run_tts)
play_button.pack(pady=10)

root.after(100, async_model_load)
root.mainloop()
