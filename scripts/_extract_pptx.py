"""Extract text from a .pptx by unzipping and parsing slide XML (no python-pptx needed)."""
import re
import sys
import zipfile
from pathlib import Path


def slide_text(xml: bytes) -> str:
    text = xml.decode("utf-8", errors="replace")
    # pull every <a:t>...</a:t> run
    runs = re.findall(r"<a:t>(.*?)</a:t>", text, re.S)
    return " ".join(runs)


def main(path: str, out: str):
    with zipfile.ZipFile(path) as z:
        names = sorted((n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                       key=lambda n: int(re.search(r"\d+", n).group()))
        found_notes = [n for n in z.namelist() if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", n)]
        lines = [f"slides: {len(names)}  |  notes slides: {len(found_notes)}", ""]
        for i, n in enumerate(names, 1):
            txt = slide_text(z.read(n))
            lines.append(f"--- Slide {i} ---")
            lines.append(txt.strip())
            lines.append("")
    Path(out).write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out, "slides:", len(names))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])