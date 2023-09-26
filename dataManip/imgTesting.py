imgPath = "MMBCReco/dataManip/imgs/track2.png"

from PIL import Image, ImageFilter 
from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
)
image = Image.open(imgPath)
#image = image.filter(EMBOSS)
#image = image.filter(CONTOUR)
image = image.filter(EDGE_ENHANCE_MORE)
image.show()