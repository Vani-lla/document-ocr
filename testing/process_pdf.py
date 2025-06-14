from pdf2image import convert_from_path

for i in range(1, 6):
    images = convert_from_path(f"pdfs/f{i}.pdf")
    images[0].save(f"pdfs/images/i{i}.jpg", "JPEG")