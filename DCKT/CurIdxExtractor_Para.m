
function CurIdx = CurIdxExtractor_Para_Dual(Iter,gpu,alpha)

load('TrainLabel.mat'); %Training labels of Wikipedia dataset
tr_n_I = size(trCatAll,1);
tr_n_T = size(trCatAll,1);


%% -------------------Search Task Definition(Wikipedia)-----------------------
I_tr = importdata(['TrainData' num2str(gpu) '/wiki_img_prob/feature.txt']);
T_tr = importdata(['TrainData' num2str(gpu) '/wiki_txt_prob/feature.txt']);
I_tr_S = importdata(['TrainData' num2str(gpu) '/wiki_img_prob_S/feature.txt']);
T_tr_S = importdata(['TrainData' num2str(gpu) '/wiki_txt_prob_S/feature.txt']);
[ImgName ImgLabel] = textread(['TrainData' num2str(gpu) '/imageTrainList.txt'],'%s%d');
[TxtName TxtLabel] = textread(['TrainData' num2str(gpu) '/textTrainList.txt'],'%s%d');
TxtFeature = load('train_txt.txt'); %Text feature in .txt file of Wikipedia dataset.


tr_n_I = size(trCatAll,1);
tr_n_T = size(trCatAll,1);
I_tr = I_tr(1:tr_n_I,:);
T_tr = T_tr(1:tr_n_T,:);
I_tr_S = I_tr_S(1:tr_n_I,:);
T_tr_S = T_tr_S(1:tr_n_T,:);


D = pdist2(I_tr,T_tr, 'cosine');
DS = pdist2(I_tr_S,T_tr_S, 'cosine');
WS = 1-DS;
DT = pdist2(I_tr,I_tr, 'cosine'); 
DI = pdist2(T_tr,T_tr, 'cosine');
W = 1-D;
WI = 1-DI;
WT = 1-DT;

Sim_I = QryonTestBi_Train(WS, trCatAll, trCatAll);
Sim_T = QryonTestBi_Train(WS', trCatAll, trCatAll);
Sim = Sim_I+Sim_T;

save(['TrainData' num2str(gpu) '/SimSaver_' num2str(Iter)],'Sim');

if Iter>20
Iter = 20;
end

Sim = mapminmax(Sim);
Sim = max(Sim)-Sim;
Sim = Sim./Iter;
Prob = 1-log(Sim+1);
CurIdx = [];
for i =1:size(Sim,1)
    if rand<=(Prob(i)*alpha)
        CurIdx = [CurIdx; i];
    end
end
disp (size(CurIdx,1));
ImgName = ImgName(CurIdx,:);
ImgLabel = ImgLabel(CurIdx,:);
TxtName = TxtName(CurIdx,:);
TxtLabel = TxtLabel(CurIdx,:);
TxtFeature = TxtFeature(CurIdx,:);

%Generate TrainImgList
I_List = fopen(['TrainData' num2str(gpu) '/imageTrainList_Curr.txt'],'w');
for i = 1:size(CurIdx,1)
    fprintf(I_List,'%s ',ImgName{i});
    fprintf(I_List,'%d\n',ImgLabel(i));
end
for i = size(CurIdx,1)+1:tr_n_I
    if mod(i,size(CurIdx,1)) ==0
      fprintf(I_List,'%s ',ImgName{size(CurIdx,1)});
      fprintf(I_List,'%d\n',ImgLabel(size(CurIdx,1)));
    else
      fprintf(I_List,'%s ',ImgName{mod(i,size(CurIdx,1))});
      fprintf(I_List,'%d\n',ImgLabel(mod(i,size(CurIdx,1))));
    end
end
fclose(I_List);

%Generate TrainTxtList
T_List = fopen(['TrainData' num2str(gpu) '/textTrainList_Curr.txt'],'w');
for i = 1:size(CurIdx,1)
    fprintf(T_List,'%s ',TxtName{i});
    fprintf(T_List,'%d\n',TxtLabel(i));
end
for i = size(CurIdx,1)+1:tr_n_I
    if mod(i,size(CurIdx,1)) ==0
      fprintf(T_List,'%s ',TxtName{size(CurIdx,1)});
      fprintf(T_List,'%d\n',TxtLabel(size(CurIdx,1)));
    else
    fprintf(T_List,'%s ',TxtName{mod(i,size(CurIdx,1))});
    fprintf(T_List,'%d\n',TxtLabel(mod(i,size(CurIdx,1))));
    end
end
fclose(T_List);

%Generate TrainTxtFeature
T_Fea = fopen(['TrainData' num2str(gpu) '/train_feature_Curr.txt'],'w');
for i = 1:size(CurIdx,1)
	for j = 1:size(TxtFeature,2)-1
		fprintf(T_Fea,'%d ',TxtFeature(i,j));
	end
	 fprintf(T_Fea,'%d\n',TxtFeature(i,j+1));
end
for i = size(CurIdx,1)+1:tr_n_I
    if mod(i,size(CurIdx,1)) ==0
		for j = 1:size(TxtFeature,2)-1
			fprintf(T_Fea,'%d ',TxtFeature(size(CurIdx,1),j));
		end
		fprintf(T_Fea,'%d\n',TxtFeature(size(CurIdx,1),j+1));
    else
		for j = 1:size(TxtFeature,2)-1
			fprintf(T_Fea,'%d ',TxtFeature(mod(i,size(CurIdx,1)),j));
		end
		fprintf(T_Fea,'%d\n',TxtFeature(mod(i,size(CurIdx,1)),j+1));
    end
end
fclose(T_Fea);



end
