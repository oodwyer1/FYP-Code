
close all;
clear;

encodeorpick=1; % 1 for training encoder and 2 for creating picks of particles (used to train the classifier)
whichone=1; % 1 to split the first image in the folder, and 2 to split the other one, can also set at 1:2 etc to process both images etc

myfolder='C:\Users\christophe.silien\Documents\Christophe\Matlab\IMSfluoML\v003_CNN_class\Data';

if encodeorpick==1
    npix=32;
    gap=6;
elseif encodeorpick==2
    npix=200;
    gap=199;
end

%%

list=dir(myfolder);
for kk=1:length(list)-2 % get the overall content of the teams folder (linked to my OneDrive via sharepoint first)
    temp{kk,1}=list(kk+2).name;
end
counter=0;
for kk=1:length(temp) % keep only the names of the folder where the data will be found
    if endsWith(squeeze(temp{kk,1}),'.bmp')
        counter=counter+1;
        filenames{counter,1}=squeeze(temp{kk,1});
    end
end
nfile=length(filenames);

kcount=0;
for kk=whichone %(whichone=1 to split the first image in the folder, and 2 to split the other one)
    clear temp;
    temp=imread([myfolder,'\',filenames{kk}]);
    img=double(temp(:,:,1))+double(temp(:,:,2))+double(temp(:,:,3));
    [nx,ny]=size(img);
    xx=1:gap:nx-1*npix;
    yy=1:gap:ny-1*npix;  
    xstart=[];
    ystart=[];
    for ll=1:length(xx)
        xstart=[xstart repmat(xx(ll),1,length(yy))];
        ystart=[ystart yy];
    end
    tmpimgframe=zeros(npix,npix,1,length(xstart));
    for ll=1:length(xstart)
        tmpimgframe(:,:,1,ll)=squeeze(img(xstart(ll):xstart(ll)+npix-1,ystart(ll):ystart(ll)+npix-1));
        if encodeorpick==2
            tmp=squeeze(tmpimgframe(:,:,1,ll));
            save(['widefield_',num2str(kk),'_',num2str(ll),'.mat'],'tmp');
        end
    end 
    if kcount==0
        imgframe=tmpimgframe;
        kcount=1;
    else
        imgframe=cat(4,imgframe,tmpimgframe);
    end

end

if encodeorpick==1
    save('widefield_frames.mat','imgframe');
end

%%

ndispl=100;
[nx,ny,~,nf]=size(imgframe);
idx=randi(nf,1,ndispl);
nh=round(sqrt(ndispl));
if nh*nh<ndispl
    nh=nh+1;
end
nv=round(ndispl/nh);
if nv*nh<ndispl
    nv=nv+1;
end
figure;
for kk=1:ndispl
    subplot(nh,nv,kk);
    imagesc(squeeze(imgframe(:,:,1,idx(kk))));
    axis off;
    axis equal;
end