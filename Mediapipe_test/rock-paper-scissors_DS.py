

from roboflow import Roboflow

#Get the data set
rf = Roboflow(api_key="Xy9TSDYjSvrRt3AWLG80")
project = rf.workspace("object-detection-quyvj").project("rock-paper-scissors-sxsw-wrkkl")
version = project.version(1)
dataset = version.download("yolov11")


#Train

#Test