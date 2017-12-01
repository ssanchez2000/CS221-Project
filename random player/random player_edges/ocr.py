from PIL import Image
import pytesseract

img=Image.open("test.png")
#img=img.convert('1')

reward=pytesseract.image_to_string(img,config='-psm 100')
print(reward)
