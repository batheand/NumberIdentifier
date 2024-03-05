# NumberIdentifier
A series of AI models that takes square image that has a handwritten one digit number as input and processes and finds the number written on the image. Using Python and TensorFlow.

There are 4 Models already implemented, all you need to do to switch between them is change the line "model_name = "tf_docs"" in the main function.
The models are: "tf_docs", "small", "medium" and "large"

There is a pipeline to predict your own image (handwritten, one-digit, preferrably a square)
For that al you need to do is add the path to the line "image_path = """ in the main function.

If you get the error "'Adam' object has no attribute 'build'", The problem is likely on the TensorFlow version, You can test it on a cloud page such as https://colab.research.google.com.
