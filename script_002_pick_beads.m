
close all;
clear;

loadpick=1;

for kk=1:1
    
    for ll=1:10
        
        [kk ll]
        
        load(['widefield_',num2str(kk),'_',num2str(ll),'.mat']);
        
        if loadpick==0
            figure;
            imagesc(tmp);
            axis equal;
            axis off;
            title('left click to pick / right click to quit');
            [XX,YY]=getpts; % pick the points manually
            XX(end)=[];
            YY(end)=[];
            save(['picked_',num2str(kk),'_',num2str(ll),'.mat'],'XX','YY');
        else
            load(['picked_',num2str(kk),'_',num2str(ll),'.mat']);
        end
        
        rebuild=zeros(size(tmp));
        for pp=1:length(XX)
            rebuild(floor(YY(pp)),floor(XX(pp)))=1;
        end
        x=-4:4;
        y=-4:4;
        [xx,yy]=meshgrid(x,y);
        RR=sqrt(xx.*xx+yy.*yy);
        psf=zeros(size(RR));
        psf(RR<=3)=1;
        rebuild=conv2(rebuild,psf,'same');
        save(['rebuild_',num2str(kk),'_',num2str(ll),'.mat'],'rebuild');
  
    end 
    
end

