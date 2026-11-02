
import pymupdf
import pytest

from src.pdf_loader import extract_text_from_pdf


def test_extract_text_from_pdf(tmp_path):
    # Create a temporary PDF for testing
    pdf_path = tmp_path / "sample.pdf"

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Hello AI Document Assistant!")
    document.save(pdf_path)
    document.close()

    # Extract text
    result = extract_text_from_pdf(str(pdf_path))

    assert "Hello AI Document Assistant!" in result
    assert "Page 1" in result


def test_invalid_pdf():
    with pytest.raises(RuntimeError):
        extract_text_from_pdf("non_existing_file.pdf")
