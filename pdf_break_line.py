import tkinter as tk
from tkinter import ttk
import pyperclip

def convert_text():
    input_text = input_box.get("1.0", "end-1c")  # get the input text
    output_text = input_text.replace("\n", " ")  # replace line breaks with spaces
    output_box.delete("1.0", "end")  # clear the output box
    output_box.insert("end", output_text)  # display the converted text in the output box
    copy_button.config(state="normal")  # enable the copy button

def clear_text():
    input_box.delete("1.0", "end")  # clear the input box
    output_box.delete("1.0", "end")  # clear the output box
    copy_button.config(state="disabled")  # disable the copy button

def copy_text():
    output_text = output_box.get("1.0", "end-1c")  # get the output text
    pyperclip.copy(output_text)  # copy the text to the clipboard
    copy_button.config(bg="#007acc", text="Copied!")  # change the color and text of the copy button
    root.after(1500, reset_copy_button)  # reset the copy button after 1.5 seconds

def reset_copy_button():
    copy_button.config(bg="#f0f0f0", text="Copy")  # reset the color and text of the copy button

# create the main window
root = tk.Tk()
root.title("Text Converter")

# create the input box and label
input_label = tk.Label(root, text="Input Text:")
input_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)

input_scrollbar = ttk.Scrollbar(root, orient="vertical")
input_scrollbar.grid(row=1, column=1, sticky="ns")

input_box = tk.Text(root, height=10, width=40, wrap="word", yscrollcommand=input_scrollbar.set)
input_box.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
input_scrollbar.config(command=input_box.yview)

# create the output box and label
output_label = tk.Label(root, text="Output Text:")
output_label.grid(row=0, column=2, sticky="w", padx=5, pady=5)

output_scrollbar = ttk.Scrollbar(root, orient="vertical")
output_scrollbar.grid(row=1, column=3, sticky="ns")

output_box = tk.Text(root, height=10, width=40, wrap="word", yscrollcommand=output_scrollbar.set)
output_box.grid(row=1, column=2, padx=5, pady=5, sticky="nsew")
output_scrollbar.config(command=output_box.yview)

# create the convert button
convert_button = tk.Button(root, text="Convert", command=convert_text, width=10, bg="#007acc", fg="white", font=("Arial", 10, "bold"))
convert_button.grid(row=2, column=0, padx=5, pady=5, sticky="e")

# create the clear button
clear_button = tk.Button(root, text="Clear", command=clear_text, width=10, bg="#007acc", fg="white", font=("Arial", 10, "bold"))
clear_button.grid(row=2, column=2, padx=5, pady=5, sticky="w")

# create the copy button
copy_button = tk.Button(root, text="Copy", command=copy_text, width=10, bg="#f0f0f0", fg="black", font=("Arial", 10, "bold"), state="disabled")
copy_button.grid(row=2, column=2, padx=5, pady=5, sticky="e")

# configure the grid
root.columnconfigure(0, weight=1)
root.columnconfigure(2, weight=1)
root.rowconfigure(1, weight=1)

# start the main event loop
root.mainloop()

