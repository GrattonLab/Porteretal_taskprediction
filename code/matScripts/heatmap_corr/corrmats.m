function corrmats(sub)
    %load all your data
    
    mem_file=['/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/mvpa_data/mem/' sub '_parcel_corrmat.mat'];
    motor_file=['/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/mvpa_data/motor/' sub '_parcel_corrmat.mat'];
    mixed_file=['/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/mvpa_data/mixed/' sub '_parcel_corrmat.mat'];
    rest_file=['/Users/aporter1350/Desktop/MSC_Alexis/analysis/data/mvpa_data/rest/' sub '_parcel_corrmat.mat'];
    
    mem=load(mem_file).parcel_corrmat;
    motor=load(motor_file).parcel_corrmat;
    mixed=load(mixed_file).parcel_corrmat;
    rest=load(rest_file).parcel_corrmat;
    %loop through each day 
    for i=1:size(mem,3)
       day=mem(:,:, i);
       fig=figure_corrmat_network_generic(day, atlas_parameters('Parcels','~/Box/Quest_Backup/Atlases/Evan_parcellation/'));
       caxis([-.4,1]);
       saveas(fig, sprintf('~/Desktop/MSC_Alexis/analysis/output/images/heatmap_corr/%s/day%d_mem.png', sub, i));
    end 
    for i=1:size(motor,3)
       day=motor(:,:, i);
       fig=figure_corrmat_network_generic(day, atlas_parameters('Parcels','~/Box/Quest_Backup/Atlases/Evan_parcellation/'));
       caxis([-.4,1]);
       saveas(fig, sprintf('~/Desktop/MSC_Alexis/analysis/output/images/heatmap_corr/%s/day%d_motor.png', sub, i));
    end 
    for i=1:size(mixed,3)
       day=mixed(:,:, i);
       fig=figure_corrmat_network_generic(day, atlas_parameters('Parcels','~/Box/Quest_Backup/Atlases/Evan_parcellation/'));
       caxis([-.4,1]);
       saveas(fig, sprintf('~/Desktop/MSC_Alexis/analysis/output/images/heatmap_corr/%s/day%d_mixed.png', sub, i));
    end 
    for i=1:size(rest,3)
       day=rest(:,:, i);
       fig=figure_corrmat_network_generic(day, atlas_parameters('Parcels','~/Box/Quest_Backup/Atlases/Evan_parcellation/'));
       caxis([-.4,1]);
       saveas(fig, sprintf('~/Desktop/MSC_Alexis/analysis/output/images/heatmap_corr/%s/day%d_rest.png', sub, i));
    end 
end 

