%% ----------Initialization------------
function MAP = Evaluate_Wiki_Cur_Para(Iter,gpu,alpha,maxIter)

%% -------------- setup Wikipedia dataset -------------------------
load('HDFeature.mat');
load('index.mat');
index = index + 1;
teCatAll = teCatAll(index(232:693), :);

te_n_I = size(teCatAll,1);
te_n_T = size(teCatAll,1);
teImgCat = teCatAll;
teTxtCat = teCatAll;

disp(['Iter :' num2str(Iter)]);

%% -------------------Search Task Definition(Wikipedia)-----------------------
I_te = importdata(['Proto' num2str(gpu) '/Feature/Iter' num2str(Iter) '/' num2str(maxIter*Iter) '/wiki_img_prob_te/feature.txt' ]);
T_te = importdata(['Proto' num2str(gpu) '/Feature/Iter' num2str(Iter) '/' num2str(maxIter*Iter) '/wiki_txt_prob_te/feature.txt' ]);
I_tr = importdata(['Proto' num2str(gpu) '/Feature/Iter' num2str(Iter) '/' num2str(maxIter*Iter) '/wiki_img_prob_tr/feature.txt' ]);
T_tr = importdata(['Proto' num2str(gpu) '/Feature/Iter' num2str(Iter) '/' num2str(maxIter*Iter) '/wiki_txt_prob_tr/feature.txt' ]);
I_tr_S = importdata(['Proto' num2str(gpu) '/Feature/Iter' num2str(Iter) '/' num2str(maxIter*Iter) '/wiki_img_prob_tr_S/feature.txt' ]);
T_tr_S = importdata(['Proto' num2str(gpu) '/Feature/Iter' num2str(Iter) '/' num2str(maxIter*Iter) '/wiki_txt_prob_tr_S/feature.txt' ]);

I_tr = I_tr(1:2173, :);
T_tr = T_tr(1:2173, :);
I_tr_S = I_tr_S(1:2173, :);
T_tr_S = T_tr_S(1:2173, :);

I_te = I_te(index(232:693), :);
T_te = T_te(index(232:693), :);

%I_te = mnorm(I_te);
%T_te = mnorm(T_te);

%I_te = sqrtNorm(I_te);
% T_te = sqrtNorm(T_te);

%I_te = pre_cnn(I_te);
%T_te = pre_cnn(T_te);

output = fopen(['Log/Log' num2str(gpu) '-' num2str(maxIter) '-' num2str(alpha) '.txt'], 'a');

fprintf(output,['Iter :', num2str(Iter) '-Train\n']);
disp('Train');
D = pdist2(I_tr, T_tr, 'cosine');
Z = - D;
W = Z;

DS = pdist2(I_tr_S, T_tr_S, 'cosine');
ZS = - DS;
WS = ZS;


 %Image->Text
MAP = QryonTestBi(W, trCatAll, trCatAll);
fprintf(output,['Image->Text: ',num2str(MAP) '\n']);
% %Text->Image
MAP = QryonTestBi(W',trCatAll, trCatAll);
fprintf(output,['Text->Image: ',num2str(MAP) '\n']);
MAP = QryonTestBi(WS, trCatAll, trCatAll);
fprintf(output,['Image->Text: ',num2str(MAP) '\n']);
% %Text->Image
MAP = QryonTestBi(WS',trCatAll, trCatAll);
fprintf(output,['Text->Image: ',num2str(MAP) '\n']);

disp('Test');
D = pdist([I_te; T_te], 'cosine');
Z = - squareform(D);
W = Z;
W_II = W(1:te_n_I, 1:te_n_I);
W_TT = W(te_n_I+1:te_n_I+te_n_T, te_n_I+1:te_n_I+te_n_T);
W_IT = W(1:te_n_I, te_n_I+1:te_n_I+te_n_T);

W = [W_II,W_IT;...
    W_IT',W_TT];


for i = 1:length(W)
    W(i,i) = 9999;
end

WIA = W(1:te_n_I,:);
WTA = W(te_n_I+1:end,:);
WII = W(1:te_n_I,1:te_n_I);
WTT = W(te_n_I+1:end,te_n_I+1:end);
WIT = W(1:te_n_I,te_n_I+1:end);
% WTI = W(te_n_I+1:te_n_I+te_n_T,1:te_n_I);
WTI = W(te_n_I+1:end,1:te_n_I);

fprintf(output,['Iter :', num2str(Iter) '-Test\n']);
% %Image->All
MAP = QryonTestBi(WIA, teImgCat, [teImgCat;teTxtCat]);
fprintf(output,['Image->All: ',num2str(MAP) '\n']);
%Text->All
MAP = QryonTestBi(WTA, teTxtCat, [teImgCat;teTxtCat]);
fprintf(output,['Text->All: ',num2str(MAP) '\n']);

%Image->Image
MAP = QryonTestBi(WII, teImgCat, teImgCat);
fprintf(output,['Image->Image: ',num2str(MAP) '\n']);
%Text->Text
MAP = QryonTestBi(WTT, teTxtCat, teTxtCat);
fprintf(output,['Text->Text: ',num2str(MAP) '\n']);

% %Image->Text
MAP = QryonTestBi(WIT, teImgCat, teTxtCat);
fprintf(output,['Image->Text: ',num2str(MAP) '\n']);
% %Text->Image
MAP = QryonTestBi(WTI, teTxtCat, teImgCat);
fprintf(output,['Text->Image: ',num2str(MAP) '\n\n\n']);
fclose(output);
MAP = 0;
end
