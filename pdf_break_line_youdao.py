import tkinter as tk
from tkinter import ttk
import pyperclip
import requests
import hashlib
import json


def convert_text():
    input_text = input_box.get("1.0", "end-1c")  # get the input text
    output_text = input_text.replace("\n", " ")  # replace line breaks with spaces
    output_box.delete("1.0", "end")  # clear the output box
    output_box.insert("end", output_text)  # display the converted text in the output box
    copy_button.config(state="normal")  # enable the copy button
    translate_text()  # call the function to translate the text


def clear_text():
    input_box.delete("1.0", "end")  # clear the input box
    output_box.delete("1.0", "end")  # clear the output box
    translated_box.delete("1.0", "end")  # clear the translated output box
    copy_button.config(state="disabled")  # disable the copy button


def copy_text():
    output_text = output_box.get("1.0", "end-1c")  # get the output text
    pyperclip.copy(output_text)  # copy the text to the clipboard
    copy_button.config(bg="#007acc", text="Copied!")  # change the color and text of the copy button
    root.after(1500, reset_copy_button)  # reset the copy button after 1.5 seconds


def reset_copy_button():
    copy_button.config(bg="#f0f0f0", text="Copy")  # reset the color and text of the copy button


def translate_text():
    input_text = output_box.get("1.0", "end-1c")  # get the converted input text, that is the output text
    youdao_url = 'https://openapi.youdao.com/api'  # Youdao address
    app_key = "ymVGHyqr16JAJp2QfLXfAwtvoqg4DxYL"  # insert your Youdao Cloud app key here
    app_id = "0b5be8a47efabb9d"  # insert your Youdao Cloud app ID here
    salt = str(hash(input_text))
    sign = app_id + input_text + salt + app_key
    sign = hashlib.md5(sign.encode("utf-8")).hexdigest()
    payload = {
        "q": input_text,
        "from": "auto",
        "to": "auto",
        "appKey": app_id,
        "salt": salt,
        "sign": sign
    }
    response = requests.get(youdao_url, params=payload)
    translation = json.loads(response.content.decode("utf-8"))
    translated_box.delete("1.0", "end")  # clear the translated output box
    translated_box.insert("end", translation["translation"][0])  # display the translated text in the output box


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
output_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)

output_scrollbar = ttk.Scrollbar(root, orient="vertical")
output_scrollbar.grid(row=3, column=1, sticky="ns")

output_box = tk.Text(root, height=10, width=40, wrap="word", yscrollcommand=output_scrollbar.set)
output_box.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")
output_scrollbar.config(command=output_box.yview)

# create the translated output box and label
translated_label = tk.Label(root, text="Youdao Translated Text:")
translated_label.grid(row=5, column=0, sticky="w", padx=5, pady=5)

translated_scrollbar = ttk.Scrollbar(root, orient="vertical")
translated_scrollbar.grid(row=6, column=1, sticky="ns")

translated_box = tk.Text(root, height=10, width=40, wrap="word", yscrollcommand=translated_scrollbar.set)
translated_box.grid(row=6, column=0, padx=5, pady=5, sticky="nsew")
translated_scrollbar.config(command=translated_box.yview)

# create the clear button
clear_button = tk.Button(root, text="Clear", command=clear_text, width=5, bg="#007acc", fg="white", font=("Arial", 10, "bold"))
clear_button.grid(row=2, column=0, padx=2, pady=2, sticky="e")

# create the convert button
convert_button = tk.Button(root, text="Convert", command=convert_text, width=5, bg="#007acc", fg="white", font=("Arial", 10, "bold"))
convert_button.grid(row=4, column=0, padx=2, pady=2, sticky="e")

# create the copy button
copy_button = tk.Button(root, text="Copy", state="disabled", command=copy_text, width=5, bg="#f0f0f0", fg="black", font=("Arial", 10, "bold"))
copy_button.grid(row=3, column=0, padx=2, pady=2, sticky="es")

# set the focus on the input box
input_box.focus()

# configure the grid
root.columnconfigure(0, weight=1)
root.columnconfigure(2, weight=1)
root.rowconfigure(1, weight=1)

# run the main event loop
root.mainloop()
