import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/create_submission_zip.py <firstname> <lastname>")
        sys.exit(1)
        
    first = sys.argv[1].strip()
    last = sys.argv[2].strip()
    
    zip_name = f"{first.lower()}_{last.lower()}.zip"
    print(f"Creating submission zip: {zip_name} ...")
    
    # Verify that the Technical Report PDF and test_output.csv exist
    pdf_path = ROOT / "Technical_Report.pdf"
    csv_path = ROOT / "test_output.csv"
    
    if not pdf_path.exists():
        print(f"Error: Technical_Report.pdf not found at {pdf_path}. Run scripts/generate_report.py first.")
        sys.exit(1)
        
    if not csv_path.exists():
        print(f"Error: test_output.csv not found at {csv_path}. Run scripts/align_pairs.py and verify test_output.csv was generated.")
        sys.exit(1)
        
    # 1. Create source_code.zip
    src_zip_path = ROOT / "source_code.zip"
    print("Packaging source code...")
    with zipfile.ZipFile(src_zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        # Package src, scripts, configs, models, and outputs
        for folder in ["src", "scripts", "configs", "models", "output"]:
            folder_path = ROOT / folder
            if folder_path.exists():
                for root, dirs, files in os.walk(folder_path):
                    # Exclude python cache folders
                    dirs[:] = [d for d in dirs if d != "__pycache__"]
                    for file in files:
                        file_path = Path(root) / file
                        # Don't include temporary zips or generated root reports in source_code.zip
                        if file_path.name in ["source_code.zip", zip_name, "Technical_Report.pdf", "test_output.csv"]:
                            continue
                        arcname = file_path.relative_to(ROOT)
                        z.write(file_path, arcname)
                        
        # Package main root config and setup files
        for file in ["README.md", "requirements.txt", "setup_real_data.py", ".gitignore"]:
            file_path = ROOT / file
            if file_path.exists():
                z.write(file_path, file)
                
    # 2. Package the main submission zip
    main_zip_path = ROOT / zip_name
    print(f"Assembling main submission package: {zip_name} ...")
    with zipfile.ZipFile(main_zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(src_zip_path, "source_code.zip")
        z.write(csv_path, "test_output.csv")
        z.write(pdf_path, "Technical_Report.pdf")
        
    # Clean up intermediate zip
    if src_zip_path.exists():
        src_zip_path.unlink()
        
    print(f"Successfully created submission package at {main_zip_path}!")
    print(f"Size: {main_zip_path.stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()
