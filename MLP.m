clc;
clear;
close all;

fprintf('🚀 [FINAL INTEGRATED] MLP 모델 훈련 및 심층 분석 시작...\n\n');

%% ========================================================================
%% PART 1. 데이터 로드 및 전처리 (Data Loading & Preprocessing)
%% ========================================================================
base_data_directory = "C:\Users\건희\Desktop\연주\251113\matlab";

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

fprintf('1️⃣ 데이터셋 통합 중...\n');

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
            error('❌ 파일을 찾을 수 없습니다.\n   경로: %s', master_file);
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

fprintf('✅ 총 %d개의 데이터 포인트 준비 완료.\n\n', height(ml_table));

%% ========================================================================
%% PART 2. 데이터 분할 및 모델 훈련 (Splitting & Training)
%% ========================================================================
fprintf('2️⃣ 데이터 분할 및 모델 훈련...\n');

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
fprintf('✅ 모델 훈련 완료.\n\n');

%% ========================================================================
%% PART 3. 성능 평가 및 저장 (Evaluation & Saving)
%% ========================================================================
fprintf('3️⃣ 모델 성능 평가 및 저장...\n');

YPred = predict(model, XTest);

SS_res = sum((YTest - YPred).^2);
SS_tot = sum((YTest - mean(YTest)).^2);
R2   = 1 - (SS_res / SS_tot);
RMSE = sqrt(mean((YTest - YPred).^2));

fprintf('   📊 R²: %.2f%% | RMSE: %.5f\n', R2*100, RMSE);

model_save_dir = fullfile(base_data_directory, "4-1. MLP figure");
if ~isfolder(model_save_dir), mkdir(model_save_dir); end

% ---------------- [Fig 4] Model Performance ----------------
fig_perf = figure('Name','Model Performance','Position',[100 100 1000 500]);

subplot(1,2,1);
scatter(YTest, YPred, 25, 'b', 'filled', 'MarkerFaceAlpha', 0.35); hold on;
plot([0 1],[0 1],'r--','LineWidth',2);

p = polyfit(YTest, YPred, 1);
plot(YTest, polyval(p, YTest), 'g-','LineWidth',1.5);

legend('Predictions','Ideal (y=x)', sprintf('Trend (Slope=%.2f)', p(1)), 'Location','southeast');
xlabel('Actual Value'); ylabel('Predicted Value');
title(sprintf('Predicted vs Actual (R²=%.2f%%)', R2*100)); grid on; axis square;

subplot(1,2,2);
histogram(YTest - YPred, 30, 'FaceColor', [0.2 0.6 0.8]);
xlabel('Prediction Error'); ylabel('Frequency');
title(sprintf('Error Distribution (RMSE=%.4f)', RMSE)); grid on; axis square;

saveas(fig_perf, fullfile(model_save_dir, 'Fig4_Model_Performance.png'));

% ---------------- Feature Importance ----------------
featureNames = X.Properties.VariableNames;
baseLoss = loss(model, XTest, YTest);
importanceScores = zeros(1, length(featureNames));

for i = 1:length(featureNames)
    Xshuffled = XTest;
    Xshuffled.(featureNames{i}) = Xshuffled.(featureNames{i})(randperm(height(Xshuffled)));
    importanceScores(i) = loss(model, Xshuffled, YTest) - baseLoss;
end

importanceScores = max(importanceScores, 0); 
importanceScores = importanceScores / sum(importanceScores) * 100;

% Feature 이름을 학술적 기호로 변경 (Size -> Vp, VolFrac -> Vf 등)
prettyNames = featureNames;
prettyNames = strrep(prettyNames, 'VolFrac', 'V_f');
prettyNames = strrep(prettyNames, 'ParticleSize', 'V_p'); 
prettyNames = strrep(prettyNames, 'PoissonRatio', '\nu'); 
prettyNames = strrep(prettyNames, 'Strain', '\epsilon');
prettyNames = strrep(prettyNames, 'TunnelingCutoff', '\delta'); 

% ---------------- [Fig 5] Feature Importance (분홍색) ----------------
fig_imp = figure('Name','Feature Importance','Position',[200 120 800 600]);
bar(importanceScores, 'FaceColor', [0.9 0.5 0.6]); % Pink
xticks(1:length(featureNames)); 
xticklabels(prettyNames); 
ylabel('Relative Importance (%)');
title('Feature Importance Analysis'); grid on;

text(1:length(importanceScores), importanceScores, num2str(importanceScores', '%.1f%%'), ...
    'VerticalAlignment','bottom','HorizontalAlignment','center','FontSize',11,'FontWeight','bold');

saveas(fig_imp, fullfile(model_save_dir, 'Fig5_Feature_Importance.png'));

% ---------------- 모델 저장 ----------------
metadata = struct('Features', featureNames, 'Created', datetime('now'), 'Layers', [50 30 15]);
save(fullfile(model_save_dir, 'Final_MLP_Model_FullEnhanced.mat'), ...
     'model', 'trainInfo', 'R2', 'RMSE', 'importanceScores', 'metadata');

writetable(table(prettyNames', importanceScores', 'VariableNames', {'Feature_Symbol','ImportancePercent'}), ...
           fullfile(model_save_dir, 'Feature_Importance_Table.csv'));

fprintf('✅ 기본 평가 완료 및 모델 저장됨.\n\n');

%% ========================================================================
%% PART 4. 심층 시각화 (3D/2D/Loss)
%% ========================================================================
fprintf('4️⃣ 심층 시각화 그래프 생성 중...\n');

vf_range = linspace(0.05, 0.8, 50); 
strain_range = linspace(0, 0.8, 50);
[X_mesh, Y_mesh] = meshgrid(vf_range, strain_range);

fix_PS = 1.0; fix_PR = 0.3; fix_Cutoff = 0.1;
num_points = numel(X_mesh);

T_grid = table(X_mesh(:), repmat(fix_PS, num_points, 1), repmat(fix_PR, num_points, 1), ...
               Y_mesh(:), repmat(fix_Cutoff, num_points, 1), ...
               'VariableNames', {'VolFrac', 'ParticleSize', 'PoissonRatio', 'Strain', 'TunnelingCutoff'});

Z_pred = predict(model, T_grid);
Z_mesh = reshape(Z_pred, size(X_mesh));
Z_mesh(Z_mesh < 0) = 0; Z_mesh(Z_mesh > 1) = 1;

% ---------------- [Fig 6] 3D Response Surface ----------------
fig3d = figure('Name', '3D Response Surface', 'Position', [100, 100, 1000, 800]);
surf(X_mesh, Y_mesh, Z_mesh, 'EdgeColor', 'none', 'FaceAlpha', 0.9); hold on;
contour(X_mesh, Y_mesh, Z_mesh, 15, 'LineWidth', 1.2, 'LineColor', 'k'); 
colormap(jet); c = colorbar; 
c.Label.String = 'Infinite Cluster Ratio'; % Mean 제거

% [수정] 라벨 위치 자동 정렬로 복귀 (왜곡 방지) + 전체 명칭 사용
xlabel('Volume Fraction (V_f)'); 
ylabel('Compressive Strain (\epsilon)'); 
zlabel('Infinite Cluster Ratio'); 

title({'3D Response Surface Analysis'; sprintf('(Fixed: V_p=%.1f, \\nu=%.1f, \\delta=%.2f)', fix_PS, fix_PR, fix_Cutoff)}, 'FontSize', 14);

view(135, 30); grid on; axis square; light('Position', [1 0 1]); lighting gouraud;
xlim([0.05 0.8]); ylim([0 0.8]); zlim([0 1.0]);

saveas(fig3d, fullfile(model_save_dir, 'Fig6_3D_Response_Surface.png'));

% ---------------- [Fig 7] 2D Contour Map ----------------
fig2d = figure('Name', '2D Response Contour', 'Position', [150, 150, 800, 600]);
contourf(X_mesh, Y_mesh, Z_mesh, 20, 'LineColor', 'none');
colormap(jet); c2 = colorbar; 
c2.Label.String = 'Infinite Cluster Ratio'; % Mean 제거

xlabel('Volume Fraction (V_f)'); 
ylabel('Compressive Strain (\epsilon)'); % 전체 명칭 사용

title({'2D Contour Map'; sprintf('(Fixed: V_p=%.1f, \\nu=%.1f, \\delta=%.2f)', fix_PS, fix_PR, fix_Cutoff)}, 'FontSize', 14);

axis tight; grid on;
saveas(fig2d, fullfile(model_save_dir, 'Fig7_2D_Response_Contour.png'));

% ---------------- [Fig 8] Training Loss Curve ----------------
figLoss = figure('Name', 'Training History', 'Position', [200, 200, 800, 600]);
if ismember('TrainingLoss', trainInfo.Properties.VariableNames)
    plot(trainInfo.TrainingLoss, 'LineWidth', 1.5, 'Color', 'b'); hold on;
    legend_str = {'Training Loss'};
    
    if ismember('ValidationLoss', trainInfo.Properties.VariableNames)
        valLoss = trainInfo.ValidationLoss;
        valid_idx = ~isnan(valLoss);
        if any(valid_idx)
            plot(find(valid_idx), valLoss(valid_idx), 'LineWidth', 1.5, 'Color', 'r', 'LineStyle', '--');
            legend_str{end+1} = 'Validation Loss';
        end
    end
    
    legend(legend_str, 'Location', 'northeast');
    xlabel('Iteration'); ylabel('Loss (MSE)');
    title('Training and Validation Loss Curves', 'FontSize', 14); grid on;
    
    saveas(figLoss, fullfile(model_save_dir, 'Fig8_Training_Loss.png'));
end

fprintf('\n🎉 기본 분석 완료! 결과 파일이 저장되었습니다: %s\n', model_save_dir);

%% ========================================================================
%% [추가 검증] 5-Fold 교차 검증 및 시각화 (Bar Chart: 하늘색)
%% ========================================================================
fprintf('\n🔄 [추가 검증] 5-Fold 교차 검증을 수행합니다 (데이터 편향 확인용)...\n');

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

fprintf('\n📊 [5-Fold 교차 검증 최종 결과]\n');
fprintf('   ⭐ 평균 정확도 (Mean R²): %.2f%% (표준편차 ±%.3f%%)\n', mean_R2, std_R2);

% ---------------- [Fig CV] Cross Validation Results ----------------
fig_cv = figure('Name', '5-Fold Cross Validation Results', 'Position', [300, 300, 800, 600]);
b = bar(1:K, R2_scores * 100, 'FaceColor', [0.3 0.7 0.95]); % Sky Blue
hold on; grid on;

yline(mean_R2, 'r--', 'LineWidth', 2, 'Label', sprintf('Mean: %.2f%%', mean_R2), ...
      'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'bottom', 'FontSize', 11, 'FontWeight', 'bold');

xtips = b.XEndPoints; ytips = b.YEndPoints;
labels = string(round(ytips, 2)) + '%';
text(xtips, ytips, labels, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');

xlabel('Fold Number', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Accuracy (R^2 Score, %)', 'FontSize', 12, 'FontWeight', 'bold');
title({'5-Fold Cross Validation Results'; '(Check for Data Bias)'}, 'FontSize', 14, 'FontWeight', 'bold');

y_low = max(80, min(R2_scores)*100 - 5); 
ylim([y_low, 100]); 
xticks(1:K);

saveas(fig_cv, fullfile(model_save_dir, 'Fig_Cross_Validation_Results.png'));
fprintf('🎉 교차 검증 그래프 저장 완료.\n');

if mean(R2_scores) > 0.95
    fprintf('\n✅ 결론: 이 모델은 데이터 분할(Seed)에 상관없이 매우 안정적이고 견고합니다.\n');
end