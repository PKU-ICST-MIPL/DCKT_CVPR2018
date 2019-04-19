# Content:

1) XMediaNet: Source model on XMediaNet dataset (for pre-training)
2) Wiki: Target model on Wikipedia dataset (for pre-training), which also has model.prototxt, solver.prototxt, and test.prototxt
3) DCKT: codes of training and testing for DCKT
4) VGG19: Pre-train model of VGG19, please download http://59.108.48.34/tiki/tiki-download_file.php?fileId=10051
5) Evaluate: Test codes for generating MAP scores


Step #1: Setup transfer-caffe
 Setup transfer-caffe from the following URL:https://github.com/zhuhan1236/transfer-caffe

Step #2: Pre-training for Source and Target model
 1) Training of Source model in folder "XMediaNet", as SourceModel.caffemodel. According to model.prototxt, you need:
	1-1) Extracting the pool5 feature maps of XMediaNet dataset, as .LMDB format, using vgg19_cvgj_iter_300000.caffemodel and test.prototxt in folder "VGG19". 
	     You need images' folder, and list in .txt format (including label). Remember to set your paths in test.prototxt.  
		 Each line of List is in the format as "filepath label" like "n04347754_15004.JPEG 833"
	1-2) Extracting the text features, in .LMDB format, and .TXT format, respetively. In our paper, each text is represented as a 300-d Word CNN feature. 
	     For .LMDB format, each entry of lmdb includes this vector and its label.
		 for .TXT format, each line is the vector, but no label.
	1-3) Training source model as SourceModel.caffemodel. Use solver.prototxt and model.prototxt, with pre-train model vgg19_cvgj_iter_300000_TripleNet.caffemodel. 
	     Remember to set your paths.
 2) Training of Target model in folder "Wiki", as TargetModel.caffemodel. Similar to the Source model.
	XMediaNet dataset can be download via http://www.icst.pku.edu.cn/mipl/XmediaNet
	Wikipeia dataset can be download via: http://www.svcl.ucsd.edu/projects/crossmodal/

Step #2: Progressive Transfer
 1) Prepare the setting in folder "DCKT", including:
    1-1) Set your path for files in folder "Proto". Note that the path prefix like "TrainData" needs to be preserved.
	     That is to say, you need to keep a folder for each GPU number to save the training data for each Iter.
	1-2) Put the list for training and testing set of Wikipedia dataset in folder "TrainData", as imageTrainList.txt and textTrainList.txt
		 Each line of List is in the format as "filepath label" like "n04347754_15004.JPEG 833"
	1-3) Set your path in CurIdxExtractor_Para.m
		 A .MAT file containing the training labels of Wikipedia dataset
		 A .TXT file containing text features of Wikipedia dataset
	1-3) Set your Wikipedia dataset path in test_cur.prototxt, test_cur_Target.prototxt and vgg.prototxt.
	1-4) Set your Caffe path, TargetModel path, and SourceModel path in runall.sh, and TrainerGenerator_Para.m.
 2) Run runall.sh with 4 parameters:
    2-1) gpu=$1  GPU number you are using
    2-2) alpha=$2 parameter \alpha
    2-3) maxIter=$3 the max Iter number
    2-4) CuIter=$4 the current Iter number, for convenience of resuming from break point.
 Note: The most common problem may come from the path settings. I suggest first dealing with TargetModel and SourceModel respectively, and then only one 1 Iter of progressive transfer.

Step #4: Evaluation
 1) If runall.sh is succesfully run, the common representations can be found in folder "ProtoX"
 2) Compute MAP scores with extracted representations with Evaluate/evaluate_wiki.m. Note: We set an exapmle Label.mat file in this folder. You must create yourselves to match the labels of your test data.
