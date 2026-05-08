import fitz
import os

doc = fitz.open("Cáncer apoptosis.pdf")
out_dir = "pdf_images"
os.makedirs(out_dir, exist_ok=True)

for page_num in range(len(doc)):
    page = doc[page_num]
    images = page.get_images(full=True)
    for img_idx, img in enumerate(images):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        ext = base_image["ext"]
        w = base_image["width"]
        h = base_image["height"]
        fname = os.path.join(out_dir, f"page{page_num+1}_img{img_idx+1}.{ext}")
        with open(fname, "wb") as f:
            f.write(image_bytes)
        print(f"Saved: {fname} ({w}x{h}, {len(image_bytes)} bytes)")

doc.close()
print("Done!")
