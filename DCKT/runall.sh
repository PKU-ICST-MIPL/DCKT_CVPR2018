#Iter 1

gpu=$1
alpha=$2
maxIter=$3
CuIter=$4

iter="1"

rm Wiki/Cur${gpu}*
rm -rf ./TrainData${gpu}
rm -rf ./Proto${gpu} 


mkdir ./TrainData${gpu}
mkdir ./Proto${gpu}
cp -r ./TrainData/* ./TrainData${gpu}/
cp -r ./Proto/* ./Proto${gpu}

#Extract feature with source pre-trained model(XMediaNet)
exe="transfer-caffe-2/install/bin/extract_features" #Remmber to set your caffe path
model="SourceModel.caffemodel"
prototxt="./test_cur.prototxt"
feature="./TrainData${gpu}"
batch="2173"
GPU="GPU ${gpu}"
$exe ${model} $prototxt xmedia_img_prob ${feature}/wiki_img_prob_S $batch leveldb $GPU
$exe ${model} $prototxt xmedia_txt_prob ${feature}/wiki_txt_prob_S $batch leveldb $GPU

#Extract feature with source pre-trained model(Target Domain)
exe="transfer-caffe-2/install/bin/extract_features"
model="TargetModel.caffemodel"
prototxt="./test_cur_Target.prototxt"
feature="./TrainData${gpu}"
batch="2173"
GPU="GPU ${gpu}"
$exe ${model} $prototxt img_prob ${feature}/wiki_img_prob $batch leveldb $GPU
$exe ${model} $prototxt txt_prob ${feature}/wiki_txt_prob $batch leveldb $GPU

sudo matlab -nodesktop -nosplash -nojvm -r "CurIdxExtractor_Para(${iter},${gpu},${alpha});quit;"

mkdir TrainData${gpu}/Pool5_Cur

#Generating training data (img pool5 and txt wcnn)
exe="transfer-caffe-2/install/bin/extract_features" #Remmber to set your caffe path
model="vgg19_cvgj_iter_300000.caffemodel"
cp ./vgg.prototxt ./Proto${gpu}/vgg.prototxt
sudo sed -i "s/TrainData/TrainData${gpu}/g" ./Proto${gpu}/vgg.prototxt
prototxt="./Proto${gpu}/vgg.prototxt"
feature="./TrainData${gpu}/Pool5_Cur"
batch="2173"
GPU="GPU ${gpu}"
$exe ${model} $prototxt pool5 ${feature}/wiki_train_pool5 $batch lmdb $GPU

exe="convert_float2lmdb" #You can use the "convert_float2lmdb.cpp" codes in folder "DCKT"
feature="./TrainData${gpu}/train_feature_Curr.txt"
list="./TrainData${gpu}/textTrainList_Curr.txt"
dim="300"
lmdb="./TrainData${gpu}/WCNN_Cur"
$exe $feature $list $dim $lmdb


sudo matlab -nodesktop -nosplash -nojvm -r "TrainerGenerator_Para(${iter},${gpu},${maxIter});quit;" 
sed -i "s/TrainData/TrainData${gpu}/g" ./Proto${gpu}/model_Cur.prototxt

rm -rf ./Proto${gpu}/Feature/Iter${iter}
sh ./Proto${gpu}/train.sh
sh ./Proto${gpu}/extract_tr.sh
sh ./Proto${gpu}/extract_te.sh
temp=$(($((maxIter))*$((iter))))
cp ./Proto${gpu}/Feature/Iter${iter}/${temp}/wiki_txt_prob_tr/* ./TrainData${gpu}/wiki_img_prob
cp ./Proto${gpu}/Feature/Iter${iter}/${temp}/wiki_img_prob_tr/* ./TrainData${gpu}/wiki_txt_prob
cp ./Proto${gpu}/Feature/Iter${iter}/${temp}/wiki_txt_prob_tr_S/* ./TrainData${gpu}/wiki_img_prob_S
cp ./Proto${gpu}/Feature/Iter${iter}/${temp}/wiki_img_prob_tr_S/* ./TrainData${gpu}/wiki_txt_prob_S


#Inter 2

for iter in `seq  2  1 ${CuIter}`
do

temp=$(($((maxIter))*$((iter))))

sudo matlab -nodesktop -nosplash -nojvm -r "CurIdxExtractor_Para(${iter},${gpu},${alpha});quit;"


 #Generating training data (img pool5 and txt wcnn)
 rm -rf ./TrainData${gpu}/Pool5_Cur/wiki_train_pool5
 rm -rf ./TrainData${gpu}/WCNN_Cur
 
 exe="transfer-caffe-2/install/bin/extract_features"
 model="vgg19_cvgj_iter_300000.caffemodel"
 prototxt="./Proto${gpu}/vgg.prototxt"
 feature="./TrainData${gpu}/Pool5_Cur"
 batch="2173"
 GPU="GPU ${gpu}"
 $exe ${model} $prototxt pool5 ${feature}/wiki_train_pool5 $batch lmdb $GPU
 exe="convert_float2lmdb"
 feature="./TrainData${gpu}/train_feature_Curr.txt"
 list="./TrainData${gpu}/textTrainList_Curr.txt"
 dim="300"
 lmdb="./TrainData${gpu}/WCNN_Cur"
 $exe $feature $list $dim $lmdb
 
 #Generate train.sh extract_tr.sh, extract_te.sh and solver.prototxt
 sudo matlab -nodesktop -nosplash -nojvm -r "TrainerGenerator_Para(${iter},${gpu},${maxIter});quit;"  
 rm -rf ./Proto${gpu}/Feature/Iter${iter}
 sh ./Proto${gpu}/train.sh
 sh ./Proto${gpu}/extract_tr.sh
 sh ./Proto${gpu}/extract_te.sh
 cp ./Proto${gpu}/Feature/Iter${iter}/${temp}/wiki_txt_prob_tr/* ./TrainData${gpu}/wiki_img_prob
 cp ./Proto${gpu}/Feature/Iter${iter}/${temp}/wiki_img_prob_tr/* ./TrainData${gpu}/wiki_txt_prob
 cp ./Proto${gpu}/Feature/Iter${iter}/${temp}/wiki_txt_prob_tr_S/* ./TrainData${gpu}/wiki_img_prob_S
 cp ./Proto${gpu}/Feature/Iter${iter}/${temp}/wiki_img_prob_tr_S/* ./TrainData${gpu}/wiki_txt_prob_S
 
 rm /media/junchao/yuanmingkuan/huangxin/modelSaver/CVPR2018/Wiki/Cur${gpu}_Iter$(($((iter))-1))_*

done



