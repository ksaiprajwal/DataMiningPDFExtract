import os
import pdfplumber
import pandas as pd
import json

def extract_tables_from_pdf(pdf_path, output_dir='data/tables'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    tables_found = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            tables = page.extract_tables()
            if not tables:
                continue
            for idx, table in enumerate(tables, 1):
                df = pd.DataFrame(table)
                base_name = f"page{page_num}_table{idx}"
                csv_path = os.path.join(output_dir, f"{base_name}.csv")
                json_path = os.path.join(output_dir, f"{base_name}.json")
                df.to_csv(csv_path, index=False, header=False)
                df.to_json(json_path, orient="split")
                tables_found += 1
                print(f"Saved table from page {page_num} as {csv_path} and {json_path}")
    print(f"Total tables found: {tables_found}")

if __name__ == "__main__":
    pdf_path = os.path.join('data', 'sample.pdf')
    extract_tables_from_pdf(pdf_path)
