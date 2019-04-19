function X = TrainerGenerator_Para(Iter,gpu,maxIter)
X = 1;

%Generate Train_Curr.sh
trainer = fopen(['Proto' num2str(gpu) '/train.sh'],'w');
if Iter == 1
  fprintf(trainer, '%s\n','exe="/home/yuanmingkuan/transfer-caffe-master/build/tools/caffe"');
  fprintf(trainer, '%s\n',['solver="Proto' num2str(gpu) '/solver_Cur.prototxt"']);
  fprintf(trainer, '%s\n','model1="TargetModel.caffemodel"'); %Set your path
  fprintf(trainer, '%s\n','model2=SourceModel.caffemodel"'); %Set your path
  fprintf(trainer, '%s\n',['gpu="' num2str(gpu) '"']);
  fprintf(trainer, '%s\n','$exe train --solver=$solver --weights=$model2,$model1 --gpu=$gpu');
else
  fprintf(trainer,'%s\n','exe="transfer-caffe-2/build/tools/caffe"');
  fprintf(trainer,'%s\n',['solver="Proto' num2str(gpu) '/solver_Cur.prototxt"']);
  %****************************************************
  %Set pre-train model: train from last Iter
  fprintf(trainer, '%s%d%s\n',['snap="Wiki/Cur' num2str(gpu) '_Iter'],Iter-1,['_iter_'  num2str(maxIter*(Iter-1)) '.solverstate"']);
  %****************************************************
  fprintf(trainer, '%s\n',['gpu="' num2str(gpu) '"']);
  fprintf(trainer, '%s\n','$exe train --solver=$solver --snapshot=$snap --gpu=$gpu');
end
fclose(trainer);

%****************************************************
%Generate solver_Curr.sh
solver = fopen(['Proto' num2str(gpu) '/solver_Cur.prototxt'],'w');
fprintf(solver, 'test_iter: 7\n');
fprintf(solver, ['test_interval: ' num2str(maxIter) '\n']);
fprintf(solver, 'base_lr: 0.01\n');
fprintf(solver, 'display: 100\n');
fprintf(solver, ['max_iter: ' num2str(maxIter*Iter) '\n']);
fprintf(solver, 'lr_policy: "inv"\n');
fprintf(solver, 'power: 0.75\n');
fprintf(solver, 'gamma: 0.0001\n');
fprintf(solver, 'momentum: 0.9\n');
fprintf(solver, 'weight_decay: 0.0005\n');
fprintf(solver, 'snapshot: 5000\n');
fprintf(solver, '%s%d"\n', ['snapshot_prefix: "Wiki/Cur' num2str(gpu) '_Iter'],Iter);
fprintf(solver, 'random_seed: 0\n');
fprintf(solver, ['net: "Proto' num2str(gpu) '/model_Cur.prototxt"\n']);
fprintf(solver, 'test_initialization: false\n');
fprintf(solver, 'iter_size: 10\n'); 
fclose(solver);

Ex_tr = fopen(['Proto' num2str(gpu) '/extract_tr.sh'],'w');
fprintf(Ex_tr, 'exe="transfer-caffe-2/install/bin/extract_features"\n'); %Set your path
fprintf(Ex_tr, '%s%d"\n', ['model="Wiki/Cur' num2str(gpu) '_Iter'],Iter);
fprintf(Ex_tr, ['prototxt="Proto' num2str(gpu) '/test_tr_Cur.prototxt"\n']);
fprintf(Ex_tr, '%s%d"\n', ['feature="Proto' num2str(gpu) '/Feature/Iter'],Iter);
fprintf(Ex_tr, 'batch="2173"\n');
fprintf(Ex_tr, ['GPU="GPU ' num2str(gpu) '"\n']);
fprintf(Ex_tr, 'mkdir ${feature}\n');
fprintf(Ex_tr, ['iter="' num2str(maxIter*Iter) '"\n']);
fprintf(Ex_tr, ' mkdir ${feature}/${iter}\n');
fprintf(Ex_tr, ' $exe ${model}_iter_${iter}.caffemodel $prototxt wiki_img_prob ${feature}/${iter}/wiki_img_prob_tr $batch leveldb $GPU\n');
fprintf(Ex_tr, ' $exe ${model}_iter_${iter}.caffemodel $prototxt wiki_txt_prob ${feature}/${iter}/wiki_txt_prob_tr $batch leveldb $GPU\n');
fprintf(Ex_tr, ' $exe ${model}_iter_${iter}.caffemodel $prototxt xmedia_img_prob ${feature}/${iter}/wiki_img_prob_tr_S $batch leveldb $GPU\n');
fprintf(Ex_tr, ' $exe ${model}_iter_${iter}.caffemodel $prototxt xmedia_txt_prob ${feature}/${iter}/wiki_txt_prob_tr_S $batch leveldb $GPU\n');
%fprintf(Ex_tr, 'done\n'); 
fclose(Ex_tr);

Ex_te = fopen(['Proto' num2str(gpu) '/extract_te.sh'],'w');
fprintf(Ex_te, 'exe="transfer-caffe-2/install/bin/extract_features"\n');
fprintf(Ex_tr, '%s%d"\n', ['model="Wiki/Cur' num2str(gpu) '_Iter'],Iter);
fprintf(Ex_tr, ['prototxt="Proto' num2str(gpu) '/test_te_Cur.prototxt"\n']);
fprintf(Ex_tr, '%s%d"\n', ['feature="Proto' num2str(gpu) '/Feature/Iter'],Iter);
fprintf(Ex_te, 'batch="693"\n');
fprintf(Ex_tr, ['GPU="GPU ' num2str(gpu) '"\n']);
fprintf(Ex_te, 'mkdir ${feature}\n');
fprintf(Ex_tr, ['iter="' num2str(maxIter*Iter) '"\n']);
fprintf(Ex_te, ' mkdir ${feature}/${iter}\n');
fprintf(Ex_te, ' $exe ${model}_iter_${iter}.caffemodel $prototxt wiki_img_prob ${feature}/${iter}/wiki_img_prob_te $batch leveldb $GPU\n');
fprintf(Ex_te, ' $exe ${model}_iter_${iter}.caffemodel $prototxt wiki_txt_prob ${feature}/${iter}/wiki_txt_prob_te $batch leveldb $GPU\n');
%fprintf(Ex_te, 'done\n'); 
fclose(Ex_te);

end
