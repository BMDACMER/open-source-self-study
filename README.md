### Fluent PDF Text

This is a small software application or script designed to improve your reading speed. When you need to copy several paragraphs of text in PDF format, there are often gaps between them, which can affect the accuracy of translation software. This software is designed to help you remove the spaces and join the paragraphs together to form a single block of text, which can be conveniently used with translation software to achieve more accurate translations.



We have developed two versions of the software: one simply converts broken sentences into fluent ones (offline, `pdf_break_line.py`), and another translates the output text using an online translation software (online, 'pdf_break_line_youdao.py'). The layout and functions of the software are both very simple, so feel free to submit issues or pull requests.



### Quick start

First, you are expected to make sure the correct development environment.  

```python
python>=3.8
tkinter
```

After that, just run this script `pdf_break_line.py`. If you want to release it as a software that can run on Windows platform,  just follow the steps below.

```python
pip install pyinstaller
pyinstaller --onefile pdf_break_line.py
```

Finally, you can find the newly released exe in the dist folder.



**Note**: If you want to integrate with an online translation software, first ensure that the software has an open developer API, then register on their official website and apply for an `APP_KEY` and `APP_ID`. For specific development processes, please refer to the official documentation.



### Layout

**offline**

![image-20230414214444391](.\img\image-20230414214444391.png)



**online**

![image-20230414214952807](.\img\image-20230414214952807.png)