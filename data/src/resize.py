from PIL import Image

#im = Image.open("12NWc.png")

im = Image.open("15TANZANIAc.jpg")
#rgba_image.load()

#im = Image.new("RGB", rgba_image.size, (255, 255, 255))
#im.paste(rgba_image, mask = rgba_image.split()[3])

im = im.resize((15*256, 15*256))
im.save("15TANZANIAc.jpg")