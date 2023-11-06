# LL_detection

With the rapid development of society, automobiles have become one of the transportation tools for people. As the number of vehicles are increasing day by day, the number of car accidents are increasing every year. Lane discipline is crucial to road safety for both drivers and pedestrians Following lane lines is one of the most important traffic rules, so detecting them is a significant task while building models for autonomous (self-driving)vehicles.

![image](https://github.com/Vijaya0869/LL_detection/assets/109131720/db457069-e633-4ee0-8d16-3e5d4846d7ce)

The main objective of this project is to detect lane lines on the roads using the Convolutional Neural Network. In this proposed work, we will detect the lane through which the vehicle is moving. This system is aimed to operate in a real time environment for enhanced safety by faster acquisition and detecting the lanes to assist the driver in controlling and performing scrutinized operations.

The annual increase in vehicle ownerships has caused traffic safety to become an important factor affecting the development of a society. The frequent occurrence of traffic accidents are caused because of human errors .To eliminate all these to a certain extent Smart vehicles (self-driving) were invented.
Lane lines are used to describe the path for self-driving vehicles so, it's necessary to detect lane line on the road surface. Based on the driving lane, determining an effective driving direction for the smart car and providing the accurate position of the vehicle in the lane are possible.

![image](https://github.com/Vijaya0869/LL_detection/assets/109131720/84ffbc7f-02cc-42aa-ac2a-4c2315ae808d)


# Mathematical Background of the project

# RGB to GreyScale Conversion :

![image](https://github.com/Vijaya0869/LL_detection/assets/109131720/05241413-2688-473b-80bf-1794afef97ae)

where  gray(x,y) is the output greyscale image
              x,y are the image coordinates
              f(x,y,R) represents red channel pixel values in specific (x,y) coordinates
              f(x,y,G) represents green channel pixel values in specific (x,y) coordinates
              f(x,y,B) represents blue channel pixel values in specific (x,y) coordinates

# Medain Filter

Median filter is most popular filter for noise-removal and less blurring of an image.
It also remove salt-and pepper noise in an image while preserving useful features and image edges.

![image](https://github.com/Vijaya0869/LL_detection/assets/109131720/cf63f93a-f50f-43a9-91ca-109f7c910d59)

where f(x , y) is the output image obtained after applying median filter 
      x , y are the image coordinates 
      Sxy - set of coordinates in a sub-image window(kernel) of size centered at point (x, y).
	    g(s , t)  represents the computation elements for median calculation within the window size.

# Sobel Filter

Edges are the Sudden &Significant changes in the intensity of an image. Sobel operator  is used to detect both horizontal and vertical edges.

![image](https://github.com/Vijaya0869/LL_detection/assets/109131720/d1ed14b7-346f-4ba1-966e-9a9659c90c0d)

where, Gx = Sobel filter (horizontal)  
       Gy = Sobel filter (vertical) 
       G = √((𝐺𝑥^2+𝐺𝑦^2)
       

# Convolutional Filter:

𝑆[𝑡]=(𝑥 ∗𝑤)[𝑡]  
![image](https://github.com/Vijaya0869/LL_detection/assets/109131720/39345527-04a2-42fd-98df-b5931e074c00)

# Pooling Layer

![image](https://github.com/Vijaya0869/LL_detection/assets/109131720/c4d4d053-3705-4836-b55a-463529991c1e)

# CNN Predictor

![image](https://github.com/Vijaya0869/LL_detection/assets/109131720/d4f624db-559e-40ce-a5c9-dc0c7a5983a0)


# Result Analysis & Comparison :

Evaluated a number of studies which uses minimum distance classifier(MDC), support vector machines(SVM) and Hough transform algorithms.
The following table shows the comparison of above mentioned algorithms with proposed model based on some performance metrics such as accuracy, precision and recall score.

![image](https://github.com/Vijaya0869/LL_detection/assets/109131720/02f79021-52e9-4762-b8f7-70da4d9f5ca8)









