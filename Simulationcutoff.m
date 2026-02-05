clc;
clear;
close all;
rng('shuffle'); % 실행할 때마다 다른 랜덤 시드를 사용

%% -------------------- A. 시뮬레이션 변수 정의 (Parameter Sweep) --------------------
num_reps = 100; % [신규] 각 조합당 반복할 횟수 (몬테카를로)

% [수정 1] 푸아송 비 (일반적인 고분자 범위로 한정)
poisson_ratios = [0.0, 0.3, 0.5]; 
% [수정 2] 부피 분율 (5% ~ 80%, 임계영역 포함)
total_volumes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]; 
% [수정 3] 입자 크기 (박스 크기(1000) 대비 현실적인 값으로 한정)
particle_sizes = [0.1, 0.2, 0.5, 1, 2, 5];

% --- [신규] 결과 저장을 위한 설정 ---
% [수정 4] 저장 경로를 1회차 및 100회차로 명확히 분리
save_directory = "C:\Users\user\OneDrive\바탕 화면\251113\Simulation data\1-1-1. data100_cutoff0.05";

% 저장 폴더가 없으면 생성
if ~isfolder(save_directory)
    mkdir(save_directory);
    fprintf('저장 폴더를 생성했습니다: %s\n', save_directory);
end

num_combinations = length(poisson_ratios) * length(total_volumes) * length(particle_sizes);
fprintf('총 %d개의 조합으로 시뮬레이션을 시작합니다.\n', num_combinations);
fprintf('각 조합당 %d회 반복 실행합니다.\n', num_reps);

% [수정] 결과 저장을 위한 구조체 배열 초기화 (Std Dev 필드 추가)
results = struct('Combination', {}, 'PoissonRatio', {}, 'TotalVolume', {}, ...
                 'ParticleSize', {}, 'ParticleCount', {}, ...
                 'MeanClusterCounts', {}, 'MeanInfiniteClusterCounts', {}, 'MeanInfiniteParticleCounts', {}, ...
                 'StdInfiniteParticleCounts', {}); % [신규] 표준편차 저장

combination_counter = 0; % 현재 몇 번째 조합인지 카운트

%% -------------------- B. 파라미터 스윕 루프 시작 --------------------
% [루프 1] 포아송 비
for i_pr = 1:length(poisson_ratios)
    current_poisson_ratio = poisson_ratios(i_pr);
    
    % [루프 2] 총 입자 부피
    for j_vol = 1:length(total_volumes)
        current_volume = total_volumes(j_vol);
        
        % [루프 3] 입자 크기
        for k_size = 1:length(particle_sizes)
            current_size = particle_sizes(k_size);
            
            % --- 루프에 종속된 변수 계산 ---
            current_count = round(current_volume / current_size); % 정수로 변환
            
            % (물리적) 구형 반지름
            particle_radius = (current_size * 3 / (4 * pi))^(1/3);
            n = 10; % Grid size
            
            % [수정 5] 논리적 오류 방지 (입자 수가 적거나, 입자가 너무 클 때)
            if current_count < 2
                fprintf('--- 조합 스킵 (입자 수 %d개) ---\n', current_count);
                continue; % 다음 루프로
            end
            if particle_radius >= (n / 2)
                 warning('--- 조합 경고: 입자 반지름(%.2f)이 박스 절반(%.2f)보다 큽니다. (Finite size effect) ---', particle_radius, n/2);
            end
            
            combination_counter = combination_counter + 1;
            fprintf('\n--- 조합 %d / %d 시작 --- \n', combination_counter, num_combinations);
            fprintf('Poisson: %.2f, TotalVol: %d, P_Size: %.2f, P_Count: %d\n', ...
                    current_poisson_ratio, current_volume, current_size, current_count);
            fprintf('  > %d회 반복 시뮬레이션을 시작합니다...\n', num_reps);
            
            % =================================================================
            %             ↓ (여기부터 Soft-Shell 100회 반복) ↓
            % =================================================================
            
            time_steps = 9; % Total time steps
            total_strain = 0.8;
            n_Particles = current_count;
            poisson_ratio = current_poisson_ratio;
            
            % 터널링 거리 (0.05로 고정)
            tunneling_cutoff = 0.05; 
            % 유효 접촉 거리 (미리 계산)
            effective_contact_distance = (particle_radius * 2) + tunneling_cutoff;
            
            % [신규] 100회 반복 결과를 저장할 누적 배열 (9 steps x 100 reps)
            all_rep_cluster_counts = zeros(time_steps, num_reps);
            all_rep_inf_cluster_counts = zeros(time_steps, num_reps);
            all_rep_inf_particle_counts = zeros(time_steps, num_reps);
            
            % [신규] 100회 반복(Repetition) 루프
            for rep = 1:num_reps
                
                %% --- 1. 파라미터 (반복마다 초기화) ---
                Particles0 = rand(n_Particles, 3) * n; % [중요] 반복마다 새 위치 생성
                
                % 1회 반복(9 스텝) 결과를 저장할 벡터
                cluster_counts_single_rep = zeros(time_steps, 1);
                Infinite_cluster_counts_single_rep = zeros(time_steps, 1);
                Infinite_cluster_particle_counts_single_rep = zeros(time_steps, 1);
            
                %% --- 2. 3D 애니메이션 루프 수정 ---
                for t = 1:time_steps
                    
                    % 현재 스텝의 변형률 및 Z축 크기 계산
                    current_strain = (t - 1) * (total_strain / (time_steps-1)); 
                    if t == 1, current_strain = 0; end
                    current_z_size = n * (1 - current_strain);
                    
                    if current_z_size <= 0
                        if t > 1
                            cluster_counts_single_rep(t:end) = cluster_counts_single_rep(t-1);
                            Infinite_cluster_counts_single_rep(t:end) = Infinite_cluster_counts_single_rep(t-1);
                            Infinite_cluster_particle_counts_single_rep(t:end) = Infinite_cluster_particle_counts_single_rep(t-1);
                            break; 
                        else
                            error('Z-axis size is zero or below at t=1.');
                        end
                    end
                    
                    % 변형 계산
                    scale_xy = 1 + poisson_ratio * current_strain;
                    scale_z = 1 - current_strain;
                    Current_Particles = Particles0;
                    Current_Particles(:, 1:2) = Particles0(:, 1:2) * scale_xy;
                    Current_Particles(:, 3) = Particles0(:, 3) * scale_z;
                    
                    % 경계 판별
                    boundary_z_min_idx = find(Current_Particles(:, 3) <= particle_radius);
                    boundary_z_max_idx = find(Current_Particles(:, 3) >= current_z_size - particle_radius);
                    
                    %% --- 3. [성능 최적화] 연결 로직 (벡터화) ---
                    distances = pdist(Current_Particles);
                    connected_pairs = distances <= effective_contact_distance;
                    AdjMatrix = squareform(connected_pairs);
                    G = graph(AdjMatrix);
                    bins = conncomp(G);
                    
                    %% --- 4. [성능 최적화] 클러스터 분석 (벡터화) ---
                    cluster_count = max(bins);
                    total_Infinite_particles = 0;
                    infinite_cluster_IDs = [];
                    
                    if cluster_count > 0 
                        for c_id = 1:cluster_count
                            particles_in_cluster_idx = find(bins == c_id);
                            is_touch_min = any(ismember(particles_in_cluster_idx, boundary_z_min_idx));
                            is_touch_max = any(ismember(particles_in_cluster_idx, boundary_z_max_idx));
                            
                            if is_touch_min && is_touch_max
                                infinite_cluster_IDs = [infinite_cluster_IDs; c_id];
                                total_Infinite_particles = total_Infinite_particles + length(particles_in_cluster_idx);
                            end
                        end
                    end
                    
                    % 1회 시뮬레이션의 t번째 스텝 결과 저장
                    cluster_counts_single_rep(t) = cluster_count;
                    Infinite_cluster_counts_single_rep(t) = numel(infinite_cluster_IDs);
                    Infinite_cluster_particle_counts_single_rep(t) = total_Infinite_particles;
                    
                end % (end of time_step 't' loop)
                
                % [신규] 100회 누적 배열에 1회차(9스텝) 결과 저장
                all_rep_cluster_counts(:, rep) = cluster_counts_single_rep;
                all_rep_inf_cluster_counts(:, rep) = Infinite_cluster_counts_single_rep;
                all_rep_inf_particle_counts(:, rep) = Infinite_cluster_particle_counts_single_rep;
                
                if mod(rep, 20) == 0 % 20회마다 진행 상황 표시
                    fprintf('    > Repetition %d/%d 완료.\n', rep, num_reps);
                end
                
            end % (end of repetition 'rep' loop)
            
            % =================================================================
            %             ↑ (여기까지 Soft-Shell 100회 반복) ↑
            % =================================================================
            
            %% --- 7. [수정] 100회 결과 통계 계산 및 저장 ---
            
            % [신규] 100회 반복 결과의 평균 및 표준편차 계산 (time_steps별 9x1 벡터)
            mean_cluster_counts = mean(all_rep_cluster_counts, 2);
            mean_inf_cluster_counts = mean(all_rep_inf_cluster_counts, 2);
            mean_inf_particle_counts = mean(all_rep_inf_particle_counts, 2);
            
            std_inf_particle_counts = std(all_rep_inf_particle_counts, 0, 2);
            
            % [1. 수정] 전체 results 구조체에 (평균) 결과 집계
            results(combination_counter).Combination = combination_counter;
            results(combination_counter).PoissonRatio = current_poisson_ratio;
            results(combination_counter).TotalVolume = current_volume;
            results(combination_counter).ParticleSize = current_size;
            results(combination_counter).ParticleCount = current_count;
            results(combination_counter).MeanClusterCounts = mean_cluster_counts; % 평균값 저장
            results(combination_counter).MeanInfiniteClusterCounts = mean_inf_cluster_counts; % 평균값 저장
            results(combination_counter).MeanInfiniteParticleCounts = mean_inf_particle_counts; % 평균값 저장
            results(combination_counter).StdInfiniteParticleCounts = std_inf_particle_counts; % [신규] 표준편차 저장
            
            % [2. 수정] 엑셀 저장을 위한 (평균) 요약 데이터 생성
            % 마지막 time_step (t=9)의 평균/표준편차 결과 추출
            final_mean_inf_particles = mean_inf_particle_counts(time_steps);
            final_std_inf_particles = std_inf_particle_counts(time_steps);
            final_mean_clusters = mean_cluster_counts(time_steps);
            
            if n_Particles > 0
                mean_ratio = final_mean_inf_particles / n_Particles; 
                std_ratio = final_std_inf_particles / n_Particles;
            else
                mean_ratio = 0;
                std_ratio = 0;
            end
            
            % 엑셀용 셀 배열 생성 (통계 값 포함)
            output_data_for_ai = {
                'Parameter', 'Value', 'Unit';
                'Total_Volume', current_volume, '';
                'n_Particles', current_count, '';
                'Individual_Volume', current_size, '';
                'Poisson_Ratio', current_poisson_ratio, '';
                'Mean Infinite Cluster Ratio', mean_ratio, ''; 
                'StdDev of Ratio', std_ratio, ''; % [신규] 표준편차 비율
                'Mean Cluster Size (final step)', final_mean_clusters, ''
            };
            
            % [3. 신규] 저장할 파일 이름 생성 (이전과 동일)
            poisson_str = strrep(sprintf('%.2f', current_poisson_ratio), '.', 'p');
            base_filename = sprintf('Results_PR%s_TV%d_PS%.1f', ...
                                    poisson_str, current_volume, current_size);
            base_filename = strrep(base_filename, '.', 'p');
            
            mat_filename = fullfile(save_directory, [base_filename, '.mat']);
            excel_filename = fullfile(save_directory, [base_filename, '.xlsx']);
            
            % [4. 수정] .mat 파일 저장 (100회 원본 + 통계)
            current_result_data_summary = results(combination_counter);
            save(mat_filename, ...
                 'current_result_data_summary', ... % 평균/표준편차 요약 (results 구조체)
                 'all_rep_inf_particle_counts'); % 100회 원본 데이터 (9x100 배열)
            
            % [5. 수정] .xlsx 파일 저장 (요약 통계)
            writecell(output_data_for_ai, excel_filename, 'Sheet', 'Summary');
            
            fprintf('--- 조합 %d 완료. 100회 평균/표준편차 집계 및 파일 저장 완료. ---\n', combination_counter);
            
        end % [루프 3] particle_sizes
    end % [루프 2] total_volumes
end % [루프 1] poisson_ratios

%% -------------------- C. 시뮬레이션 종료 및 저장 --------------------
% [수정] 모든 조합의 (평균/표준편차) 결과가 담긴 마스터 .mat 파일 저장
save(fullfile(save_directory, 'Full_Simulation_Results_100Reps_0.05_Avg.mat'), 'results');
fprintf('\n모든 시뮬레이션 완료. 개별 파일 및 마스터 요약 파일이 아래 경로에 저장되었습니다:\n');
fprintf('%s\n', save_directory);