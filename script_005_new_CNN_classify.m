
close all;
clear;

npix=32;
gap=1;
loadold=1;
load('Encoder_norm.mat');

%% create training set

counter=1;
for kk=1:1
    
    for mm=1:9 
        
        counter
        
        load(['widefield_',num2str(kk),'_',num2str(mm),'.mat']);
        img=tmp;
        load(['rebuild_',num2str(kk),'_',num2str(mm),'.mat']);
        target=rebuild;
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
            hw=floor(npix/2);
            tempclass=rebuild(xstart(ll)+hw,ystart(ll)+hw);
            xpos(kk,mm,ll)=xstart(ll)+hw;
            ypox(kk,mm,ll)=ystart(ll)+hw;
            if tempclass==1
                tmptargetclass{ll}='YES';
                tmpclass(ll,1)=1;
                tmpclass(ll,2)=0;
            else
                tmptargetclass{ll}='NO';
                tmpclass(ll,1)=0;
                tmpclass(ll,2)=1;
            end
        end 
        if counter==1
            inputval=tmpimgframe;
            targetclass=tmptargetclass;
            numclass=tmpclass;
        else
            inputval=cat(4,inputval,tmpimgframe);
            targetclass=cat(2,targetclass,tmptargetclass);
            numclass=cat(1,numclass,tmpclass);
        end
        counter=counter+1;
        
    end
end
inputval=inputval/wide_norm;
targetclass=categorical(squeeze(targetclass'));

% Keep only "twice" as many NO than YES to balance the training set
idxNO=find(targetclass=='NO');
nNO=length(idxNO);
idxYES=find(targetclass=='YES');
nYES=length(idxYES);
idxpick=randi(nNO,nYES*4+1,1);
idxkeep=sort([idxNO(idxpick);idxYES]);
targetclass_cropped=targetclass(idxkeep);
inputval_cropped=inputval(:,:,:,idxkeep);
numclass_cropped=numclass(idxkeep,:);

%%
%%%% create encoder network

numLatentInputs = 48; % number of targeted variables after encoding

% % layersEncoder = [
% %     imageInputLayer([32 32 1],"Name","in","Normalization","none")
% %     dropoutLayer(0.01,"Name","dpL1")
% %     convolution2dLayer([4 4],12,"Name","conv1","Padding","same","Stride",[2 2])
% %     leakyReluLayer(0.2,"Name","lrelu1")
% %     convolution2dLayer([3 3],24,"Name","conv2","Padding","same","Stride",[2 2])
% %     batchNormalizationLayer("Name","bn2")
% %     leakyReluLayer(0.2,"Name","lrelu2")
% %     convolution2dLayer([4 4],48,"Name","conv3","Padding","same","Stride",[2 2])
% %     batchNormalizationLayer("Name","bn3")
% %     leakyReluLayer(0.2,"Name","lrelu3")
% %     convolution2dLayer([4 4],numLatentInputs,"Name","conv5")];
% % 
% % % analyzeNetwork(layersEncoder);
% % 
% % lgraphEncoder = layerGraph(layersEncoder);
% % dlnetEncoder = dlnetwork(lgraphEncoder);

%% create classifier network

classes = ["YES" "NO"];
numClasses = numel(classes);

layersDecoder = [
    featureInputLayer(numLatentInputs,"Name","in")
    fullyConnectedLayer(100,"Name","fc21")
    leakyReluLayer(0.2,"Name","lrelu21")
    fullyConnectedLayer(numClasses,"Name","fc22")
    softmaxLayer("Name","Soft21")];

% analyzeNetwork(layersDecoder);

lgraphDecoder = layerGraph(layersDecoder);
dlnetDecoder = dlnetwork(lgraphDecoder);

%%

if loadold==1
    load('Classifier.mat');
end

load('Encoder.mat');


%%

numEpochs = 5; % for a given loaded data set, this is the number times that the optimization is cycled
miniBatchSize = 2000; % size of a minibatch (how many data points, here how many CCD frames are examined at a same time)

learnRate = 0.001;
gradientDecayFactor = 0.5;
squaredGradientDecayFactor = 0.999;

validationFrequency = 5; % number of iteration before updating graphs during optimization, and saving the networks
numValidationImages = 64; % number of images displayed 

trailingAvgEncoder = [];
trailingAvgSqEncoder = [];
trailingAvgDecoder = [];
trailingAvgSqDecoder = [];

f = figure;
f.Position(3) = 2*f.Position(3);
imageAxes = subplot(1,3,1);
imageAxes2 = subplot(1,3,2);
scoreAxes = subplot(1,3,3);
lineScore = animatedline(scoreAxes,'Color',[0 0.447 0.741]);
legend('Encoder-Decoder');
xlabel("Iteration")
ylabel("log10 loss")
grid on

iteration = 0;
start = tic;

%% Loop over epochs.

[~,~,~,nspl]=size(inputval_cropped);
allspl=1:nspl;
temp = randperm(length(allspl),numValidationImages);
idspl_validation=allspl(temp); % pick a series of CCD images index to always display same during training

niter=floor(nspl/miniBatchSize);

for epoch = 1:numEpochs
    
    clear idx idspl;
    allspl=1:nspl;
    for kk=1:niter % create random draws without repeat for each iterations within a period (ie., those are the minibatches)
        temp = randperm(length(allspl),miniBatchSize);
        idspl(kk,:)=allspl(temp); % (generator)
        allspl(temp)=[];
    end
    
    for kk=1:niter
        iteration = iteration + 1;
        % 1 iteration of update the Encoder-Decoder using the myCCD function    
        Zin=inputval_cropped(:,:,:,idspl(kk,:));
        Zout=numclass_cropped(idspl(kk,:),:)';
        [dlnetEncoder,trailingAvgEncoder,trailingAvgSqEncoder,stateEncoder,...
            dlnetDecoder,trailingAvgDecoder,trailingAvgSqDecoder,stateDecoder,...
            loss]=dlfeval(@myCCD,Zin,Zout,iteration,learnRate,...
            gradientDecayFactor, squaredGradientDecayFactor,...
            dlnetEncoder,trailingAvgEncoder, trailingAvgSqEncoder,...
            dlnetDecoder,trailingAvgDecoder, trailingAvgSqDecoder);
        dlnetEncoder.State = stateEncoder;
        dlnetDecoder.State = stateDecoder;    
        
        % After "validationFrequency" iterations, update the display of images      
        if mod(iteration,validationFrequency) == 0 || iteration == 1 
            
        Zin=inputval_cropped(:,:,:,idspl_validation);
        Zout=numclass_cropped(idspl_validation,:)';

            input=dlarray(Zin,'SSCB');
            target=dlarray(Zout,'CB');
            [outEncoder,stateEncoder]=forward(dlnetEncoder,input);
            outEncoder=dlarray(squeeze(outEncoder),'CB');
            [outDecoder,stateDecoder]=forward(dlnetDecoder,outEncoder);
            subplot(1,3,1);  
            histogram(extractdata(squeeze(outDecoder(1,:))));
            subplot(1,3,2);  
            histogram(extractdata(squeeze(outDecoder(2,:))));
                                                 
% %             save('Encoder.mat','dlnetEncoder');
            save('Classifier.mat','dlnetDecoder');
            
        end
        
        % Update the scores plot.
        subplot(1,3,3)
        addpoints(lineScore,iteration,...
            log10(double(gather(extractdata(loss)))));       
        
        % Update the title with training progress information.
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        title(...
            "Epoch: " + (epoch) + ", " + ...
            "Iteration: " + iteration + ", " + ...
            "Elapsed: " + string(D))
        
        drawnow
   
    end

end


%% create test set

counter=1;
for kk=1:1
    
    for mm=10 
        
        counter
        
        load(['widefield_',num2str(kk),'_',num2str(mm),'.mat']);
        img=tmp;
        load(['rebuild_',num2str(kk),'_',num2str(mm),'.mat']);
        target=rebuild;
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
            hw=floor(npix/2);
            tempclass=rebuild(xstart(ll)+hw,ystart(ll)+hw);
            xpos(kk,mm,ll)=xstart(ll)+hw;
            ypos(kk,mm,ll)=ystart(ll)+hw;
            if tempclass==1
                tmptargetclass{ll}='YES';
                tmpclass(ll,1)=1;
                tmpclass(ll,2)=0;
            else
                tmptargetclass{ll}='NO';
                tmpclass(ll,1)=0;
                tmpclass(ll,2)=1;
            end
        end 
        if counter==1
            inputval=tmpimgframe;
            targetclass=tmptargetclass;
            numclass=tmpclass;
        else
            inputval=cat(4,inputval,tmpimgframe);
            targetclass=cat(2,targetclass,tmptargetclass);
            numclass=cat(1,numclass,tmpclass);
        end
        counter=counter+1;
        
    end
end
inputval_test=inputval/wide_norm;
numclass_test=numclass;
targetclass_test=categorical(squeeze(targetclass'));

%% testing testing

Zin=inputval_test;
Zout=numclass_test';

input=dlarray(Zin,'SSCB');
target=dlarray(Zout,'CB');
[outEncoder,stateEncoder]=forward(dlnetEncoder,input);
outEncoder=dlarray(squeeze(outEncoder),'CB');
[outDecoder,stateDecoder]=forward(dlnetDecoder,outEncoder);

idx=find(extractdata(squeeze(target(1,:)))==1);
YESs=extractdata(outDecoder(1,:));
NOs=extractdata(outDecoder(2,:));

XX=squeeze(xpos(1,10,:));
YY=squeeze(ypos(1,10,:));
idx_guess=find(YESs>0.5);
XX=XX(idx_guess);
YY=YY(idx_guess);
recon=zeros(200,200);
for kk=1:length(XX)
    recon(XX(kk),YY(kk))=1; 
end


x=16+1:200-16;
figure;
subplot(2,3,1);
imagesc(img(x,x));
axis off;
axis equal;
subplot(2,3,2);
imagesc(rebuild(x,x));
axis off;
axis equal;
subplot(2,3,3);
imagesc(recon(x,x));
axis off;
axis equal;

    
 %%

function [dlnetEncoder,trailingAvgEncoder,trailingAvgSqEncoder,stateEncoder,...
            dlnetDecoder,trailingAvgDecoder,trailingAvgSqDecoder,stateDecoder,...
            loss] = myCCD(Zin,Zout,iteration,learnRate,...
            gradientDecayFactor, squaredGradientDecayFactor,...
            dlnetEncoder,trailingAvgEncoder, trailingAvgSqEncoder,...
            dlnetDecoder,trailingAvgDecoder, trailingAvgSqDecoder)

    input=dlarray(Zin,'SSCB');
    target=dlarray(Zout,'CB');
    [outEncoder,stateEncoder]=forward(dlnetEncoder,input);
    outEncoder=dlarray(squeeze(outEncoder),'CB');
    [outDecoder,stateDecoder]=forward(dlnetDecoder,outEncoder);

    loss=crossentropy(outDecoder,target);
       
%     gradientsEncoder = dlgradient(loss, dlnetEncoder.Learnables,'RetainData',true);
%     [dlnetEncoder,trailingAvgEncoder,trailingAvgSqEncoder] = ...
%         adamupdate(dlnetEncoder, gradientsEncoder, ...
%         trailingAvgEncoder, trailingAvgSqEncoder, iteration, ...
%         learnRate, gradientDecayFactor, squaredGradientDecayFactor);

    gradientsDecoder = dlgradient(loss, dlnetDecoder.Learnables,'RetainData',true);
    [dlnetDecoder,trailingAvgDecoder,trailingAvgSqDecoder] = ...
        adamupdate(dlnetDecoder, gradientsDecoder, ...
        trailingAvgDecoder, trailingAvgSqDecoder, iteration, ...
        learnRate, gradientDecayFactor, squaredGradientDecayFactor);

end







