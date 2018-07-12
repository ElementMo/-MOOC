from PIL import Image

img = Image.open('pic/0.png')
img = img.resize((28,28), Image.ANTIALIAS)
img.show()