from PIL import Image
import numpy
values={}
for i in range(10):
    A=Image.open(str(i)+".png")
    A= numpy.array(A)
    values[(A[2,2,0],A[2,5,0],A[4,1,0],A[5,1,0])]=str(i)

print(values)
