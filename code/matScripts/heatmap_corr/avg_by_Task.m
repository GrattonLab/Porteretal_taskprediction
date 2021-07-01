clear;
clear all;

load('/Users/Alexis/Box/Quest_Backup/MSC/TaskFC/FC_Parcels/mem/allsubs_mem_corrmats_orig.mat', 'avg_task_corrmat')
mem=avg_task_corrmat.AllMem
mem(isinf(mem)|isnan(mem)) = 0; % Replace NaNs and infinite values with zeros

mem_avg=mean(mem,3)
mem_avg=real(mem_avg)%convert to real numbers

f=figure_corrmat_network_generic(mem_avg, atlas_parameters('Parcels','~/Box/Quest_Backup/Atlases/Evan_parcellation/'))
colormapeditor
saveas(f, '~/Box/My Box Notes/MSC_Alexis/analysis/output/images/heatmap_corr/mem_avg_all.png')

clear;
clear all;
load('~/Box/Quest_Backup/MSC/TaskFC/FC_Parcels/motor/allsubs_motor_corrmats_orig.mat', 'avg_task_corrmat')
motor=avg_task_corrmat.AllMotor
motor(isinf(motor)|isnan(motor)) = 0; % Replace NaNs and infinite values with zeros

motor_avg=mean(motor,3)
motor_avg=real(motor_avg)%convert to real numbers

f=figure_corrmat_network_generic(motor_avg, atlas_parameters('Parcels','~/Box/Quest_Backup/Atlases/Evan_parcellation/'))
colormapeditor
saveas(f, '~/Box/My Box Notes/MSC_Alexis/analysis/output/images/heatmap_corr/motor_avg_all.png')

clear;
clear all;
load('~/Box/Quest_Backup/MSC/TaskFC/FC_Parcels/mixed/allsubs_mixed_corrmats_orig.mat', 'avg_task_corrmat')
mixed_glass=avg_task_corrmat.AllGlass
mixed_glass(isinf(mixed_glass)|isnan(mixed_glass)) = 0; % Replace NaNs and infinite values with zeros

mixed_glass_avg=mean(mixed_glass,3)
mixed_glass_avg=real(mixed_glass_avg)%convert to real numbers

f=figure_corrmat_network_generic(mixed_glass_avg, atlas_parameters('Parcels','~/Box/Quest_Backup/Atlases/Evan_parcellation/'))
colormapeditor
saveas(f, '~/Box/My Box Notes/MSC_Alexis/analysis/output/images/heatmap_corr/mixed_glass_avg_all.png')



clear;
clear all;
load('~/Box/Quest_Backup/MSC/TaskFC/FC_Parcels/rest/allsubs_rest_corrmats_orig.mat', 'avg_rest_corrmat')
avg_rest_corrmat(isinf(avg_rest_corrmat)|isnan(avg_rest_corrmat)) = 0; % Replace NaNs and infinite values with zeros

rest_avg=mean(avg_rest_corrmat,3)
rest_avg=real(rest_avg)%convert to real numbers

f=figure_corrmat_network_generic(rest_avg, atlas_parameters('Parcels','~/Box/Quest_Backup/Atlases/Evan_parcellation/'))
colormapeditor
saveas(f, '~/Box/My Box Notes/MSC_Alexis/analysis/output/images/heatmap_corr/rest_avg_all.png')



memMrest=mem_avg-rest_avg
motorMrest=motor_avg-rest_avg
mixMrest=mixed_glass_avg-rest_avg

f=figure_corrmat_network_generic(memMrest, atlas_parameters('Parcels','~/Box/Quest_Backup/Atlases/Evan_parcellation/'))
colormapeditor
saveas(f, '~/Box/My Box Notes/MSC_Alexis/analysis/output/images/heatmap_corr/memMrest_all.png')

f=figure_corrmat_network_generic(mixMrest, atlas_parameters('Parcels','~/Box/Quest_Backup/Atlases/Evan_parcellation/'))
colormapeditor
saveas(f, '~/Box/My Box Notes/MSC_Alexis/analysis/output/images/heatmap_corr/mixMrest_all.png')

f=figure_corrmat_network_generic(motorMrest, atlas_parameters('Parcels','~/Box/Quest_Backup/Atlases/Evan_parcellation/'))
colormapeditor
saveas(f, '~/Box/My Box Notes/MSC_Alexis/analysis/output/images/heatmap_corr/motorMrest_all.png')



