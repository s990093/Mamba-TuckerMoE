import os
import re
import subprocess

readme_path = "README.md"
with open(readme_path, "r") as f:
    content = f.read()

latex_template_math = r"""\documentclass[preview,border=10pt]{standalone}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage{graphicx}
\begin{document}
\scalebox{2.0}{%
$\displaystyle %%MATH%% $%
}
\end{document}
"""

latex_template_table = r"""\documentclass[preview,border=10pt]{standalone}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage{xeCJK}
\setCJKmainfont{PingFang SC}
\usepackage{booktabs}
\usepackage{graphicx}
\begin{document}
\scalebox{1.5}{%
%%TABLE%%
}
\end{document}
"""

# Extract Math $$ ... $$
math_blocks = re.findall(r"\$\$(.*?)\$\$", content, flags=re.DOTALL)
table_blocks = re.findall(r"```latex(.*?)```", content, flags=re.DOTALL)

replacements = {}
counter = 1

XELATEX_CMD = "/usr/local/texlive/2025/bin/universal-darwin/xelatex"
ASSET_DIR = os.path.join("assets", "latex")
BUILD_DIR = os.path.join("archive", "latex-build")
SOURCE_DIR = os.path.join("archive", "latex-sources")
os.makedirs(BUILD_DIR, exist_ok=True)
os.makedirs(SOURCE_DIR, exist_ok=True)
os.makedirs(ASSET_DIR, exist_ok=True)

for idx, math in enumerate(math_blocks):
    tex_file = os.path.join(SOURCE_DIR, f"tex_math_{counter}.tex")
    pdf_file = os.path.join(BUILD_DIR, f"tex_math_{counter}.pdf")
    png_file = os.path.join(ASSET_DIR, f"tex_math_{counter}.png")
    
    with open(tex_file, "w") as f:
        f.write(latex_template_math.replace("%%MATH%%", math.strip()))
    
    # Compile
    subprocess.run(
        [XELATEX_CMD, "-interaction=nonstopmode", "-output-directory", BUILD_DIR, tex_file],
        stdout=subprocess.DEVNULL,
    )
    if os.path.exists(pdf_file):
        subprocess.run(["sips", "-s", "format", "png", pdf_file, "--out", png_file], stdout=subprocess.DEVNULL)
        replacements[f"$${math}$$"] = f"![Math Equation {counter}]({png_file})"
        print(f"Compiled {png_file}")
    
    counter += 1

for idx, table in enumerate(table_blocks):
    # Remove \begin{table} and \end{table} to avoid float errors in standalone
    clean_table = re.sub(r"\\begin\{table\}.*?\\centering", "", table, flags=re.DOTALL)
    clean_table = re.sub(r"\\caption\{.*?\}", "", clean_table, flags=re.DOTALL)
    clean_table = re.sub(r"\\end\{table\}", "", clean_table, flags=re.DOTALL)
    
    tex_file = os.path.join(SOURCE_DIR, f"tex_table_{counter}.tex")
    pdf_file = os.path.join(BUILD_DIR, f"tex_table_{counter}.pdf")
    png_file = os.path.join(ASSET_DIR, f"tex_table_{counter}.png")
    
    with open(tex_file, "w") as f:
        f.write(latex_template_table.replace("%%TABLE%%", clean_table.strip()))
        
    # Compile
    subprocess.run(
        [XELATEX_CMD, "-interaction=nonstopmode", "-output-directory", BUILD_DIR, tex_file],
        stdout=subprocess.DEVNULL,
    )
    if os.path.exists(pdf_file):
        subprocess.run(["sips", "-s", "format", "png", pdf_file, "--out", png_file], stdout=subprocess.DEVNULL)
        replacements[f"```latex{table}```"] = f"![Table {counter}]({png_file})"
        print(f"Compiled {png_file}")
        
    counter += 1

# Replace
new_content = content
for k, v in replacements.items():
    new_content = new_content.replace(k, v)

with open(readme_path, "w") as f:
    f.write(new_content)

print("Done substituting LaTeX with PNG images.")
