import pymupdf 

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF document.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A string containing the extracted text.
    """

    extracted_text = ""

    try:
        # Open the PDF document
        document = pymupdf.open(pdf_path)

        # Read each page
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text()

            if page_text.strip():
                extracted_text += f"\n--- Page {page_number} ---\n"
                extracted_text += page_text

        document.close()

        if not extracted_text.strip():
            raise ValueError("No readable text found in this PDF.")

        return extracted_text

    except Exception as error:
        raise RuntimeError(f"Failed to extract PDF text: {error}") from error
