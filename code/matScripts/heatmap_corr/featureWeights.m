clear;
clear all;

mem_MSC01=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/featureWeights/train_MSC01_test_MSC02_mem.mat'];
mem_MSC02=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/featureWeights/train_MSC02_test_MSC03_mem.mat'];

mem_MSC01_file=load(mem_MSC01);
mem_MSC01_fW=mem_MSC01_file.fW_mat;
fig_mem_MSC01=figure_corrmat_network_generic(mem_MSC01_fW, atlas_parameters('Parcels','~/Box/Quest_Backup/Atlases/Evan_parcellation/'));
%colormapeditor
saveas(fig_mem_MSC01, '~/Desktop/MSC_Alexis/analysis/output/images/between_sub_test/featureWeights/train_MSC01_test_MSC02_mem_fW.png');

mem_MSC02_file=load(mem_MSC02);
mem_MSC02_fW=mem_MSC02_file.fW_mat;
fig_mem_MSC02=figure_corrmat_network_generic(mem_MSC02_fW, atlas_parameters('Parcels','~/Box/Quest_Backup/Atlases/Evan_parcellation/'));
%colormapeditor
saveas(fig_mem_MSC02, '~/Desktop/MSC_Alexis/analysis/output/images/between_sub_test/featureWeights/train_MSC02_test_MSC03_mem_fW.png');

motor_MSC01=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/featureWeights/train_MSC01_test_MSC02_motor.mat'];
motor_MSC02=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/featureWeights/train_MSC02_test_MSC03_motor.mat'];

motor_MSC01_file=load(motor_MSC01);
motor_MSC01_fW=motor_MSC01_file.fW_mat;
fig_motor_MSC01=figure_corrmat_network_generic(motor_MSC01_fW, atlas_parameters('Parcels','~/Box/Quest_Backup/Atlases/Evan_parcellation/'));
%colormapeditor
saveas(fig_motor_MSC01, '~/Desktop/MSC_Alexis/analysis/output/images/between_sub_test/featureWeights/train_MSC01_test_MSC02_motor_fW.png');

motor_MSC02_file=load(motor_MSC02);
motor_MSC02_fW=motor_MSC02_file.fW_mat;
fig_motor_MSC02=figure_corrmat_network_generic(motor_MSC02_fW, atlas_parameters('Parcels','~/Box/Quest_Backup/Atlases/Evan_parcellation/'));
colormapeditor
saveas(fig_motor_MSC02, '~/Desktop/MSC_Alexis/analysis/output/images/between_sub_test/featureWeights/train_MSC02_test_MSC03_motor_fW.png');


mix_MSC01=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/featureWeights/train_MSC01_test_MSC02_mixed.mat'];
mix_MSC02=['~/Desktop/MSC_Alexis/analysis/output/results/MVPA_mat/between_sub_test/featureWeights/train_MSC02_test_MSC03_mixed.mat'];

mix_MSC01_file=load(mix_MSC01);
mix_MSC01_fW=mix_MSC01_file.fW_mat;
fig_mix_MSC01=figure_corrmat_network_generic(mix_MSC01_fW, atlas_parameters('Parcels','~/Box/Quest_Backup/Atlases/Evan_parcellation/'));
colormapeditor
saveas(fig_mix_MSC01, '~/Desktop/MSC_Alexis/analysis/output/images/between_sub_test/featureWeights/train_MSC01_test_MSC02_mix_fW.png');

mix_MSC02_file=load(mix_MSC02);
mix_MSC02_fW=mix_MSC02_file.fW_mat;
fig_mix_MSC02=figure_corrmat_network_generic(mix_MSC02_fW, atlas_parameters('Parcels','~/Box/Quest_Backup/Atlases/Evan_parcellation/'));
colormapeditor
saveas(fig_mix_MSC02, '~/Desktop/MSC_Alexis/analysis/output/images/between_sub_test/featureWeights/train_MSC02_test_MSC03_mix_fW.png');

