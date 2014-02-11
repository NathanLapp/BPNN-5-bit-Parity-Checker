User Manual
By: Nathan Lapp

Installation:
Download Python version 3.3.0 for Windows.
Here is the link to the official download page:
http://www.python.org/download/

When installing Python, make sure to add Python to your path so that you can use Python in any folder on your system.

Execution:
Open a command prompt window and navigate to the folder that you have the program files in. Verify that the ‘weights_nvl002_CSC475.txt’ file is in the same folder as the ‘BPNN_nvl002_CSC475.py’ program. Now if you have added Python to your path, you can simply type 

‘python BPNN_nvl002_CSC475.py’

into the command prompt and then hit the Enter key. It should output a lengthy result containing the percent error from the training and the final value for each of the training sets.

About the code:
The program starts at line 151 of ‘BPNN_nvl002_CSC475.py’. I run two different experiments on each execution of the program. I create 2 neural networks (located on lines 215 and 225, respectively) I train the first one on pattern 1 then test it on pattern 2. I train the second one on pattern 2 then test it on pattern 2.

If you are interested in changing a few things about the program, here is what to do for a few options.

Changing the weights: open up the ’weights_nvl002_CSC475.txt’ file and change the weights. Make sure not to change the number of weights. The program reads each weight line by line; so do not change the input format.

Changing the learning rate: the learning rate is located on line 137 and is represented with the variable ‘N’.

Changing the number of iterations: the number of iterations is located on line 137 and is represented with the variable ‘iterations’.
