clc;
clear;
close all;

fprintf('🚀 [FINAL INTEGRATED] MLP 모델 훈련 및 심층 분석 시작...\n\n');

%% Data Loading & Preprocessing
base_data_directory = "C:\Users\Desktop\연주\code";

data_paths = {
    fullfile(base_data_directory, "1-1-1. data100_cutoff0.05"), ...
    fullfile(base_data_directory, "1-1-2. data100_cutoff0.1"), ...
    fullfile(base_data_directory, "1-1-3. data100_cutoff0.2")
};
  
cutoff_values = [0.05, 0.1, 0.2];
strains = linspace(0, 0.8, 9); 
BoxSize = 10; 
current_box_vol = BoxSize^3;

ml_table = table();

fprintf('️데이터셋 통합 중...\n');

for i = 1:length(data_paths)
    cutoff_val = cutoff_values(i);
    data_path  = data_paths{i};
    
    if abs(cutoff_val - 0.05) < 1e-8
        file_name_str = '0.05';
    else
        file_name_str = sprintf('%.1f', cutoff_val);
    end
    
    file_name = sprintf('Full_Simulation_Results_100Reps_%s_Avg.mat', file_name_str);
    master_file = fullfile(data_path, file_name);
    
    if ~isfile(master_file)
        if isfile([master_file, '.mat'])
            master_file = [master_file, '.mat'];
        else
            error('파일을 찾을 수 없습니다.\n   경로: %s', master_file);
        end
    end
    
    fprintf('   > 로드 중 (\\delta=%.2f): %s\n', cutoff_val, file_name);
    load(master_file, 'results');
    
    for j = 1:length(results)
        if results(j).ParticleCount < 2
            continue; 
        end
        
        num_steps = numel(strains);
        
        VolFrac = repmat(results(j).TotalVolume / current_box_vol, num_steps, 1);
        VolFrac(VolFrac > 1) = 1; 
        
        ParticleSize = repmat(results(j).ParticleSize, num_steps, 1);
        PoissonRatio = repmat(results(j).PoissonRatio, num_steps, 1);
        Strain = strains';
        TunnelingCutoff = repmat(cutoff_val, num_steps, 1);
        
        MeanInfiniteClusterRatio = results(j).MeanInfiniteParticleCounts / results(j).ParticleCount;
        
        step_table = table(VolFrac, ParticleSize, PoissonRatio, Strain, TunnelingCutoff, MeanInfiniteClusterRatio);
        ml_table = [ml_table; step_table]; %#ok<AGROW>
    end
end

fprintf('총 %d개의 데이터 포인트 준비 완료.\n\n', height(ml_table));

%% ========================================================================
%% PART 2. 데이터 분할 및 모델 훈련 (Splitting & Training)
%% ========================================================================
fprintf('데이터 분할 및 모델 훈련...\n');

X = ml_table(:, {'VolFrac','ParticleSize','PoissonRatio','Strain','TunnelingCutoff'});
Y = ml_table.MeanInfiniteClusterRatio;

rng(42); 
cv = cvpartition(height(ml_table), 'HoldOut', 0.2);
idxTrain = training(cv);
idxTest  = test(cv);

XTrain = X(idxTrain,:);  YTrain = Y(idxTrain);
XTest  = X(idxTest,:);   YTest  = Y(idxTest);

model = fitrnet(XTrain, YTrain, ...
    'LayerSizes', [50 30 15], ...
    'Activations', 'relu', ...
    'Standardize', true, ...
    'Lambda', 0.001, ...
    'ValidationData', {XTest, YTest}, ...
    'Verbose', 1);

trainInfo = model.TrainingHistory; 
fprintf('모델 훈련 완료.\n\n');

%% Evaluation & Saving

fprintf('모델 성능 평가 및 저장...\n');

YPred = predict(model, XTest);

SS_res = sum((YTest - YPred).^2);
SS_tot = sum((YTest - mean(YTest)).^2);
R2   = 1 - (SS_res / SS_tot);
RMSE = sqrt(mean((YTest - YPred).^2));

fprintf(' R²: %.2f%% | RMSE: %.5f\n', R2*100, RMSE);

model_save_dir = fullfile(base_data_directory, "4-1. MLP figure");
if ~isfolder(model_save_dir), mkdir(model_save_dir); end


%% [추가 검증] 5-Fold 

K = 5; 
cv_kfold = cvpartition(height(ml_table), 'KFold', K);
R2_scores = zeros(K, 1);
RMSE_scores = zeros(K, 1);

for k = 1:K
    fprintf('   ▶ Fold %d / %d 훈련 중... ', k, K);
    idxTr = training(cv_kfold, k); idxTe = test(cv_kfold, k);
    XTr = X(idxTr,:); YTr = Y(idxTr); XTe = X(idxTe,:); YTe = Y(idxTe);
    
    % fitrnet 호출
    model_cv = fitrnet(XTr, YTr, 'LayerSizes', [50 30 15], 'Activations', 'relu', ...
        'Standardize', true, 'Lambda', 0.001, 'Verbose', 0); 
    
    YPred_cv = predict(model_cv, XTe);
    SS_res_cv = sum((YTe - YPred_cv).^2);
    SS_tot_cv = sum((YTe - mean(YTe)).^2);
    R2_scores(k) = 1 - (SS_res_cv / SS_tot_cv);
    RMSE_scores(k) = sqrt(mean((YTe - YPred_cv).^2));
    
    fprintf('완료. (R²: %.2f%%)\n', R2_scores(k)*100);
end

mean_R2 = mean(R2_scores) * 100;
std_R2  = std(R2_scores) * 100;

fprintf('\n [5-Fold 교차 검증 최종 결과]\n');
fprintf('  평균 정확도 (Mean R²): %.2f%% (표준편차 ±%.3f%%)\n', mean_R2, std_R2);


if mean(R2_scores) > 0.95
    fprintf('\n 결론: 이 모델은 데이터 분할(Seed)에 상관없이 매우 안정적이고 견고합니다.\n');
end
