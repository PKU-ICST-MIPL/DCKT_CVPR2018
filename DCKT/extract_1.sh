exe="/home/yuanmingkuan/transfer-caffe-2/install/bin/extract_features"
model="/media/junchao/yuanmingkuan/model/ysm/VGG19_xmedia/_iter_5000.caffemodel"
prototxt="/home/yuanmingkuan/huangxin/CVPR2018/Cur/test_1.prototxt"
feature="/home/yuanmingkuan/huangxin/CVPR2018/Cur/TrainData_Wiki"
batch="2173"
GPU="GPU 3"
$exe ${model} $prototxt xmedia_img_prob ${feature}/wiki_img_prob $batch leveldb $GPU
$exe ${model} $prototxt xmedia_txt_prob ${feature}/wiki_txt_prob $batch leveldb $GPU
