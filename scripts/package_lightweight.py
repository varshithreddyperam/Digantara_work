"""Creates a lightweight deliverables ZIP (<50MB) for portals with upload size limits."""

import os
import zipfile

def package_lightweight(zip_path="submission/Digantara_Report_and_Deliverables_Lightweight.zip"):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write("docs/ASSESSMENT_REPORT.pdf", "ASSESSMENT_REPORT.pdf")
        z.write("docs/ASSESSMENT_REPORT.md", "ASSESSMENT_REPORT.md")
        z.write("docs/SUBMISSION_README.md", "SUBMISSION_README.md")
        z.write("data/processed/dataset/dataset.yaml", "dataset.yaml")
        z.write("docs/validation_report.json", "validation/validation_report.json")
        z.write("docs/pipeline_summary.json", "validation/pipeline_summary.json")
        z.write("docs/dataset_inspection.json", "validation/dataset_inspection.json")
        
        for mf in sorted(os.listdir("data/processed/manifests")):
            z.write(os.path.join("data/processed/manifests", mf), f"manifests/{mf}")
            
        for root, _, files in os.walk("data/processed/review_crops"):
            for f in files:
                full_p = os.path.join(root, f)
                rel_p = os.path.relpath(full_p, "data/processed")
                z.write(full_p, rel_p)
                
        for split in ["train", "val"]:
            lbl_dir = f"data/processed/dataset/labels/{split}"
            for f in os.listdir(lbl_dir):
                z.write(os.path.join(lbl_dir, f), f"dataset/labels/{split}/{f}")
                
        lbl_dir = "data/processed/reconstructed/labels"
        for f in os.listdir(lbl_dir):
            z.write(os.path.join(lbl_dir, f), f"reconstructed/labels/{f}")

    sz_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Created lightweight package: {zip_path} ({sz_mb:.2f} MB)")

if __name__ == "__main__":
    package_lightweight()
