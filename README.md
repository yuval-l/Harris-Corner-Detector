Harris Corner Detector

--------------------------------------------------------------------------------------------------

HIT | Algorithms in Multimedia and machine learning in Python enviorment

Yuval Levi & Ortal Michael

May 2021

--------------------------------------------------------------------------------------------------

Algorithm steps:

1. Compute image gradients: Gx, Gy

2. Compute products: Gx*Gx, Gy*Gy, Gx*Gy

3. Filter products with Gaussian window

4. For each pixel (i,j) define the matrix 
  
        M = [ m11(i,j)   m12(i,j)
  
        m21(i,j)   m22(i,j) ]
        
5. For each pixel compute the score R
   
        R = det(M) - a[tr(M)]^2
   
        a~0.06
   
6. Threshold R and perform NMS (Non-maxima suppression)

--------------------------------------------------------------------------------------------------

The algorithm is wrapped in PySimpleGUI User Interface:

1. Load image from OS

2. Choose sensetivity (low, medium, high) of the algorithm

3. Click "show corners" to present the algorithm output on the selected image

--------------------------------------------------------------------------------------------------



