import sys
from pathlib import Path

import win32com.client as win32

src = Path(sys.argv[1]).resolve()
dst = src.with_suffix(".pdf")

word = win32.DispatchEx("Word.Application")
word.Visible = False
doc = word.Documents.Open(str(src), ReadOnly=False)
try:
    doc.SaveAs(str(dst), FileFormat=17)
finally:
    doc.Close(False)
    word.Quit()
print("wrote", dst)
