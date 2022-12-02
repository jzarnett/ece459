# Author: Chris Gravel < cgravel@edu.uwaterloo.ca >

# Purpose of this file is to concatenate all lecture summaries to make a notebook with TOC for use in exam

# What this file does:
# 1. Prepares a sorted list of all lectures (7 letter long strings like Lxx.tex)
# 2. Make minor changes to Latex (described below) so files can display nicely

# Latex changes:
# 1. Remove the \include{header} line and insert the contents of header.tex at the top
# 2. Change document class to report and change config so TOC will have trailling dots to page nums
# 3. Add table of contents at the beginning using \tableofcontents
# 4. Change \lecture{} to \chapter*{} To display more nicely
# 5. Mark location of each chapter in the TOC using \addcontentsline
# 6. Remove \begin{document}, \end{document} and \input{bibliography.tex}
# 7. Add \end{document} and \input{bibliography.tex} after the last iteration

import os
import re

CUTOFF = float('inf')

files = os.listdir()
tex_lectures = sorted(list(filter(lambda x: '.tex' in x and len(x) == 7, files)))

buf = ''

with open('../common/header.tex', 'r') as f:
    buf = f.read()

# Replace header
buf = buf.replace('''\\documentclass[letterpaper,10pt]{article}''',
'''\\documentclass[a4paper]{report}
\\usepackage{tocloft}
\\renewcommand{\\cftchapleader}{\\cftdotfill{\\cftdotsep}} ''')

for (index, file) in enumerate(tex_lectures):
    if (index < CUTOFF):
        print('reading: ', file)
        with open(file, 'r') as f:
            s = f.read()
            # Remove include header
            s = s.replace('\\input{../common/header}', '')

            # Add table of contents if first notes read
            if (index == 0):
                s = s.replace('\\begin{document}', '\\begin{document}\n\n\\tableofcontents\n')
            else:
                s = s.replace('\\begin{document}', '') # Remove begin document

            # Remove end of document tag and bibliography tag
            s = s.replace('\\input{bibliography.tex}', '')
            s = s.replace('\\end{document}', '')

            # Get chapter title
            title = re.findall(r'\\lecture{(.+?)}', s)[0].strip()

            # Find and replace lecture line
            le = re.findall(r'^\\lecture{.+$', s, re.MULTILINE)[0]

            # replace with chapter and add line for putting in table of contents
            s = s.replace(le, "\\chapter*{{{0}}}\n\n\\addcontentsline{{toc}}{{chapter}}{{{0}}}".format(title)) # Double curly brackets used to escape
            
            # Use original \lecture{} command. Makes a bug with the TOC links - correct page numbers in TOC, breaks doc link
            # s = s.replace(le, "\\clearpage\n\n{0}\n\n\\addcontentsline{{toc}}{{chapter}}{{{1}}}".format(le, title)) # Double curly brackets used to escape

            buf += '\n' + s

buf += '\\input{bibliography.tex}\n\n\\end{document}'

with open('notebook.tex', 'w') as f:
    f.write(buf)

print("Finished")
