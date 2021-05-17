# Seq2Seq-picture


This is an example of the seq2seq addition task on the LSTM model, implemented on images

![alt text](https://raw.githubusercontent.com/manishemirani/Seq2Seq-picture/main/images/output.JPG)

We've got label 7 or 8(in other examples) of the plus sign, because I didn't train plus on the model, but I fixed it with a simple math trick

## Run

To run the code you should do:
          
         python main.py -i [IMAGE PATH] (in this case is ./images/[IMAGE NAME.JPG])

Result will be:

    [INFO] 4 - 91.32%
    [INFO] 8 - 96.93%
    [INFO] 3 - 99.95%
    [INFO] 7 - 92.68%
    [INFO] 2 - 89.00%
    [INFO] 2 - 82.98%
    Your operation is: 483+22
    Result: 483+22 = 505
