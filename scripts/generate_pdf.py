from markdown_pdf import MarkdownPdf, Section
import fitz
import os

def build_pdf():
    # Change working directory so local image paths resolve correctly
    original_cwd = os.getcwd()
    os.chdir("docs")
    
    # 1. Generate standard PDF from markdown
    temp_pdf = "final_report_raw.pdf"
    pdf = MarkdownPdf(toc_level=1)
    
    with open("final_report.md", "r", encoding="utf-8") as f:
        md_text = f.read()
        
    parts = md_text.split("<!-- pagebreak -->")
    
    for i, part in enumerate(parts):
        pdf.add_section(Section(part.strip(), toc=False))
        
    pdf.meta["title"] = "Final Report"
    pdf.save(temp_pdf)
    
    # 2. Compress the PDF using PyMuPDF (fitz)
    doc = fitz.open(temp_pdf)
    doc.save("final_report.pdf", garbage=4, deflate=True, clean=True)
    doc.close()
    
    # 3. Clean up raw temp file
    if os.path.exists(temp_pdf):
        os.remove(temp_pdf)
        
    orig_size = os.path.getsize("final_report.pdf")
    print(f"Compressed PDF generated successfully: docs/final_report.pdf ({orig_size / 1024 / 1024:.2f} MB)")
    
    # Restore working directory
    os.chdir(original_cwd)

if __name__ == "__main__":
    build_pdf()
