# %%
#!/usr/bin/env python3
"""
Script to create a PDF from PNG files in the directory.
Each PNG is placed on a separate page with its batch name and animal number as title.
"""

import os
import re
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph, Frame
from reportlab.lib.styles import getSampleStyleSheet


def extract_batch_animal(filename):
    """
    Extract batch name and animal number from filename.
    Expected format: psycho_corner_3-params-T_0_fixed_from_vanilla_{batch_name}_{animal_num}.png
    """
    pattern = r'psycho_corner_3-params-T_0_fixed_from_vanilla_(.+)_(\d+)\.png'
    match = re.match(pattern, filename)
    if match:
        batch_name = match.group(1)
        animal_num = match.group(2)
        return batch_name, animal_num
    return None, None


def create_pdf_from_pngs(directory_path, output_filename="combined_plots_T_0_fixed.pdf", page_size=A4):
    """
    Create a PDF with each PNG file on a separate page with title.
    """
    # Get all PNG files matching the pattern
    png_files = [f for f in os.listdir(directory_path) if f.endswith('.png') and f.startswith('psycho_corner_3-params-T_0_fixed_from_vanilla_')]
    
    # Sort files for consistent ordering
    png_files.sort()
    
    # Create PDF
    c = canvas.Canvas(os.path.join(directory_path, output_filename), pagesize=page_size)
    width, height = page_size
    
    # Styles for title
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    
    for png_file in png_files:
        # Extract batch name and animal number
        batch_name, animal_num = extract_batch_animal(png_file)
        
        if batch_name and animal_num:
            title_text = f"{batch_name} - {animal_num}"
        else:
            # Fallback if pattern doesn't match
            title_text = png_file
        
        # Add title to page
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(width/2, height - 40, title_text)
        
        # Add image to page
        img_path = os.path.join(directory_path, png_file)
        try:
            # Calculate image size to fit on page with some margins
            img_width = width - 100  # 50px margin on each side
            img_height = height - 120  # Account for title and bottom margin
            
            # Draw image centered
            c.drawInlineImage(img_path, 
                             x=(width - img_width) / 2,
                             y=(height - img_height) / 2 - 20,  # Adjust for title
                             width=img_width,
                             height=img_height)
        except Exception as e:
            print(f"Error adding image {png_file}: {e}")
        
        # Create new page for next image
        c.showPage()
    
    # Save PDF
    c.save()
    print(f"PDF created: {os.path.join(directory_path, output_filename)} with {len(png_files)} pages.")


# if __name__ == "__main__":
# %%
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create the PDF
create_pdf_from_pngs(script_dir)
