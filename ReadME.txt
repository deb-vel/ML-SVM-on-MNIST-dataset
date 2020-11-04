This folder contains the Documentation inside which the plagiarism declaration form and statement of completion can be found.

The folder ML_Assignment contains the code.
it has two python files:
•	svm_all_kernels.py : this python file runs all the parameter variations discussed in this report by the use of nested loops.
•	svm_one_kernel.py : contains the same SVM implementation of the above file, but lets the user choose his own desired kernel, gamma and C value.

It also has two .cmd files to run the above.  Double click them to execute code:
•	run_svm_all_kernels.cmd : this runs svm_all_kernels.py
•	run_svm_one_kernel.cmd : this runs svm_one_kernel.py


For the programs to run successfully the following libraries must be already installed on the device: numpy, timeit, pandas, matplotlib, seaborn, and sklearn.  
Furthermore, when an image is successfully display in its window, the window should be closed for the program to continue running.

The datasets are csv files:
trainingSet.csv : contains 12,100 training examples
mnist_train.csv : contains 60,000 training examples
mnist_test.csv : contains the test set