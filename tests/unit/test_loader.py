"""
Tests for ingestion/loader.py
"""

import pytest

from ingestion.loader import (
    Document,
    _strip_html_tags,
    load_file,
    load_from_bytes,
)


class TestDocument:
    def test_filename_property_returns_filename(self):
        doc = Document(content="hello", metadata={"filename": "test.pdf"})
        assert doc.filename == "test.pdf"

    def test_filename_property_defaults_to_unknown(self):
        doc = Document(content="hello", metadata={})
        assert doc.filename == "unknown"


class TestStripHtmlTags:
    def test_strips_simple_tags(self):
        html = "<p>Hello <strong>world</strong></p>"
        assert _strip_html_tags(html) == "Hello world"

    def test_removes_script_blocks(self):
        html = "<p>Hello</p><script>alert('xss')</script><p>World</p>"
        assert "alert" not in _strip_html_tags(html)
        assert "Hello" in _strip_html_tags(html)
        assert "World" in _strip_html_tags(html)

    def test_removes_style_blocks(self):
        html = "<style>body { color: red; }</style><p>Content</p>"
        assert _strip_html_tags(html) == "Content"

    def test_collapse_whitespace(self):
        html = "<p>Hello</p>\n\n\n   <p>World</p>"
        result = _strip_html_tags(html)
        assert "\n" not in result
        assert "  " not in result

    def test_empty_string(self):
        assert _strip_html_tags("") == ""
        assert _strip_html_tags("   ") == ""

    def test_no_tags(self):
        text = "Plain text without any tags."
        assert _strip_html_tags(text) == text


class TestLoadFile:
    def test_loads_markdown_file(self, temp_dir):
        md_file = temp_dir / "test.md"
        md_file.write_text("# Hello\n\nThis is content.", encoding="utf-8")

        doc = load_file(md_file)

        assert isinstance(doc, Document)
        assert "Hello" in doc.content
        assert doc.metadata["filename"] == "test.md"
        assert doc.metadata["file_type"] == ".md"
        assert doc.metadata["file_type_label"] == "markdown"

    def test_loads_plaintext_file(self, temp_dir):
        txt_file = temp_dir / "notes.txt"
        txt_file.write_text("Line one.\nLine two.", encoding="utf-8")

        doc = load_file(txt_file)

        assert isinstance(doc, Document)
        assert "Line one" in doc.content
        assert doc.metadata["file_type"] == ".txt"
        assert doc.metadata["file_type_label"] == "plaintext"

    def test_loads_html_file(self, temp_dir):
        html_file = temp_dir / "page.html"
        html_file.write_text(
            "<html><body><h1>Title</h1><p>Content here.</p></body></html>",
            encoding="utf-8",
        )

        doc = load_file(html_file)

        assert isinstance(doc, Document)
        assert "Title" in doc.content
        assert "Content here" in doc.content
        assert doc.metadata["file_type"] == ".html"

    def test_raises_on_nonexistent_file(self, temp_dir):
        with pytest.raises(FileNotFoundError):
            load_file(temp_dir / "does_not_exist.pdf")

    def test_raises_on_unsupported_file_type(self, temp_dir):
        bad_file = temp_dir / "document.docx"
        bad_file.write_bytes(b"PK\x03\x04")  # fake zip header

        with pytest.raises(ValueError, match="Unsupported file type"):
            load_file(bad_file)

    def test_loads_with_source_name_override(self, temp_dir):
        md_file = temp_dir / "uuid123.md"
        md_file.write_text("Content here.", encoding="utf-8")

        doc = load_file(md_file, source_name="report-final.md")

        assert doc.metadata["filename"] == "report-final.md"


class TestLoadFromBytes:
    def test_loads_pdf_bytes(self, sample_pdf_bytes):
        doc = load_from_bytes(sample_pdf_bytes, "test.pdf", ".pdf")

        assert isinstance(doc, Document)
        assert len(doc.content) > 0
        assert doc.metadata["filename"] == "test.pdf"
        assert doc.metadata["file_type"] == ".pdf"
        assert "page_count" in doc.metadata

    def test_loads_markdown_bytes(self):
        content = b"# Heading\n\nSome **bold** text."
        doc = load_from_bytes(content, "notes.md", ".md")

        assert isinstance(doc, Document)
        assert "Heading" in doc.content
        assert doc.metadata["filename"] == "notes.md"
        assert doc.metadata["file_type"] == ".md"

    def test_loads_text_bytes(self):
        content = "Hello world.\nSecond line.".encode("utf-8")
        doc = load_from_bytes(content, "notes.txt", ".txt")

        assert isinstance(doc, Document)
        assert "Hello world" in doc.content
        assert doc.metadata["file_type"] == ".txt"

    def test_loads_html_bytes_strips_tags(self):
        content = b"<html><body><p>Hello</p></body></html>"
        doc = load_from_bytes(content, "page.html", ".html")

        assert isinstance(doc, Document)
        assert "Hello" in doc.content
        # HTML tags should be stripped
        assert "<p>" not in doc.content

    def test_handles_invalid_utf8_gracefully(self):
        # Invalid UTF-8 sequence: continuation byte without start byte
        content = b"\x80\x90\xff"
        doc = load_from_bytes(content, "binary.bin", ".txt")

        assert isinstance(doc, Document)
        # Should not raise, should decode with replacement
        assert isinstance(doc.content, str)
