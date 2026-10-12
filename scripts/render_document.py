"""Use Word for PDF export and the documents skill renderer for page images.

The Windows artifact runtime has no bundled LibreOffice. This adapter replaces
only the DOCX-to-PDF conversion step, retaining the packaged rasterisation.
Run with the managed document Python runtime, not the project environment.
"""
import argparse
import importlib.util
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--renderer", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("skill_renderer", args.renderer)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)

    def word_export(doc_path, user_profile, convert_tmp_dir, stem, verbose=False):
        output = Path(convert_tmp_dir) / f"{stem}.pdf"
        process = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(Path(__file__).with_name("render_word.ps1")), "-InputPath", doc_path,
            "-OutputPath", str(output)], capture_output=True, text=True, check=True)
        return str(output), process.stdout + process.stderr

    renderer.convert_to_pdf = word_export
    pages = renderer.rasterize(str(args.input.resolve()), str(args.output.resolve()),
                              120, verbose=False, emit_pdf=True)
    print(f"Rendered {len(pages)} pages through Microsoft Word and Poppler.")


if __name__ == "__main__":
    main()

