
from PIL import Image, ImageOps

#xmin, ymin, xmax, ymax = path.split(',')[:4]
#xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)

#width = float(xmax) - float(xmin)
#height = float(ymax) - float(ymin)

stitch = Image.new('RGB', (256*63, 256*63))

for y in range(0,256 * 63,256):
    for x in range(0,256 * 63,256):
        tile = Image.open(f'PyT_SAT_MARS3/{int(x / 256)},{int(y / 256)}.jpg')
        stitch.paste(tile, (y,x))

stitch.save('stitched_mars3.jpg')
