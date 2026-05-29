from PIL import Image, ImageDraw, ImageFont
import math

width = 2048
height = 1024
img = Image.new('RGB', (width, height), color=(30, 30, 30))
draw = ImageDraw.Draw(img)

# Draw Grid
for x in range(0, width, 64):
    color = (255, 0, 0) if x == width//2 else (100, 100, 100)
    draw.line((x, 0, x, height), fill=color, width=2)
for y in range(0, height, 64):
    color = (0, 0, 255) if y == height//2 else (100, 100, 100)
    draw.line((0, y, width, y), fill=color, width=2)

# Try loading a default font, otherwise standard
try:
    font = ImageFont.truetype("arial.ttf", 60)
except IOError:
    font = ImageFont.load_default()

def draw_text(text, x, y):
    # Depending on Pillow version, text bbox or size might be used. 
    # Just draw roughly centered
    draw.text((x-50, y-30), text, fill=(255, 255, 0), font=font)

# Front (Center)
draw_text("Center (Front)", width//2, height//2)
# Left
draw_text("Left", width//4, height//2)
# Right
draw_text("Right", (width*3)//4, height//2)
# Back (Edges)
draw_text("Back", 100, height//2)
draw_text("Back", width-100, height//2)

# Top
draw_text("Top (Zenith)", width//2, 100)
# Bottom
draw_text("Bottom (Nadir)", width//2, height-100)

img.save('sample_equirectangular.jpg', quality=95)
print("Created sample_equirectangular.jpg")
