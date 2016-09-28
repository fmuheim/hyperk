To create the ssh tunnel, first log into the gateway:

$ ssh -L 38080:localhost:38080 ph-ppe

Then from the gateway log into the remote machine:

$ ssh -L 38080:localhost:38080 abaddon

In another terminal start the notebook:

jupyter notebook --no-browser --port=38080

