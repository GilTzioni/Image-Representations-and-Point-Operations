# Image-Representations-and-Point-Operations:camera: :zap:
Image Processing and Computer Vision Course Assignment 1 <br />
**Written in Python on PyCharm workspace** ![22](https://user-images.githubusercontent.com/73976733/104122372-55ae3700-534d-11eb-8de4-492973fca972.jpeg)
                                               
In ex1.utils.py you will find an image processing like:

**Reading an image into a given representation** ->repressentation of RGB or Gray Scale image.<br />
the mathods imReadAndConvert() and imDisplay() using Opencv2

**Transforming an RGB image to YIQ color space**-> <br />
Transform an RGB image into the YIQ color space and vice versa. Given the red (R), green (G), and blue (B) pixel components of an RGB color image, the corresponding luminance (Y), and the chromaticity components (I and Q) in the YIQ color space are linearly related as follows:

<img width="498" alt="Screenshot 2023-04-19 at 17 12 50" src="https://user-images.githubusercontent.com/58401645/233102591-7e16af26-c744-43e2-a33c-56ecebd772f5.png">

<img width="575" alt="Screenshot 2023-04-19 at 16 49 51" src="https://user-images.githubusercontent.com/58401645/233099251-64a26bba-1153-46a1-b828-0ebfd46a4ac0.png">

**Histogram equalization**->Equalizes the histogram of an image. <br />
the method hsitogramEqualize() - algorithm that calculate the cumsum of the image and change it Linear as much as possible for Equale histogram



**Optimal image quantization**-> Quantized an image in to **nQuant** colors. <br />
the method quantizeImage() - create a new image with input number of colors  
 from this image: <br />
 <img width="573" alt="Screenshot 2023-04-19 at 16 50 55" src="https://user-images.githubusercontent.com/58401645/233112582-9758293b-2fde-40cf-8ca4-b19bec8439fd.png">

to this : <br />
<img width="551" alt="Screenshot 2023-04-19 at 16 51 54" src="https://user-images.githubusercontent.com/58401645/233112821-f772e5e5-1cd2-4f3d-bc76-31880b003c25.png">


In gamma.py you will find :
 
**Gamma Correction**->GUI for gamma correction with trackBar from(• OpenCV trackbar example and • Gamma Correction Wikipedia)
<br />
<img width="595" alt="Screenshot 2023-04-19 at 16 57 32" src="https://user-images.githubusercontent.com/58401645/233114216-ed5811ca-6383-468d-8929-a14f6da67bdd.png">

