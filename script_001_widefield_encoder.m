
close all;
clear;

%%

loadold=1;

myfolder='C:\Users\christophe.silien\Documents\Christophe\Matlab\IMSfluoML\v003_CNN_class';
filename='widefield_frames.mat';

load([myfolder,'\',filename]);
wide_norm=max(max(max(max(imgframe))));


%% create encoder network

numLatentInputs = 48; % number of targeted variables after encoding

layersEncoder = [
    imageInputLayer([32 32 1],"Name","in","Normalization","none")
    dropoutLayer(0.01,"Name","dpL1")
    convolution2dLayer([4 4],12,"Name","conv1","Padding","same","Stride",[2 2])
    leakyReluLayer(0.2,"Name","lrelu1")
    convolution2dLayer([3 3],24,"Name","conv2","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","bn2")
    leakyReluLayer(0.2,"Name","lrelu2")
    convolution2dLayer([4 4],48,"Name","conv3","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","bn3")
    leakyReluLayer(0.2,"Name","lrelu3")
    convolution2dLayer([4 4],numLatentInputs,"Name","conv5")];

% analyzeNetwork(layersEncoder);

lgraphEncoder = layerGraph(layersEncoder);
dlnetEncoder = dlnetwork(lgraphEncoder);

%% create decoder network

layersDecoder = [
    featureInputLayer(numLatentInputs,"Name","in")
    projectAndReshapeLayer([4 4 48],numLatentInputs,'Name','proj');
    dropoutLayer(0.01,"Name","dpL1")
    transposedConv2dLayer([4 4],24,"Name","tconv1","Cropping","same","Stride",[2 2])
    batchNormalizationLayer("Name","bnorm1")
    reluLayer("Name","relu1")
    transposedConv2dLayer([4 4],12,"Name","tconv2","Cropping","same","Stride",[2 2])
    batchNormalizationLayer("Name","bnorm2")
    reluLayer("Name","relu2")
    transposedConv2dLayer([4 4],1,"Name","tconv3","Cropping","same","Stride",[2 2])
    tanhLayer("Name","tanh")];

% analyzeNetwork(layersDecoder);

lgraphDecoder = layerGraph(layersDecoder);
dlnetDecoder = dlnetwork(lgraphDecoder);

%%

if loadold==1
    load('Encoder.mat');
    load('Decoder.mat');
    load('Encoder_norm.mat');
end

imgframe=imgframe/wide_norm;

%%

numEpochs = 20; % for a given loaded data set, this is the number times that the optimization is cycled
miniBatchSize = 1000; % size of a minibatch (how many data points, here how many CCD frames are examined at a same time)

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

croppedImg=imgframe;

[~,~,~,nspl]=size(croppedImg);
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
        Zin=croppedImg(:,:,:,idspl(kk,:));
        Zout=croppedImg(:,:,:,idspl(kk,:));
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
            
            Zin=croppedImg(:,:,:,idspl_validation);
            Zout=croppedImg(:,:,:,idspl_validation);

            input=dlarray(Zin,'SSCB');
            target=dlarray(Zout,'SSCB');
            [outEncoder,stateEncoder]=forward(dlnetEncoder,input);
            outEncoder=dlarray(squeeze(outEncoder),'CB');
            [outDecoder,stateDecoder]=forward(dlnetDecoder,outEncoder);
                                      
            II = imtile(extractdata(outDecoder));
            IItg = imtile(extractdata(target));           
            % Display the images.
            subplot(1,3,1);            
            imagesc(II);
            axis equal;
            axis off;
            colorbar;
            xticklabels([]);
            yticklabels([]);
            caxis([0 1]);
            title("reconstructions");
            subplot(1,3,2);
            imagesc(IItg);
            axis equal;
            axis off;
            colorbar;
            xticklabels([]);
            yticklabels([]);
            caxis([0 1]);
            title("targets");
            
            save('Encoder.mat','dlnetEncoder');
            save('Decoder.mat','dlnetDecoder');
            save('Encoder_norm.mat','wide_norm');
            
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

    
 %%

function [dlnetEncoder,trailingAvgEncoder,trailingAvgSqEncoder,stateEncoder,...
            dlnetDecoder,trailingAvgDecoder,trailingAvgSqDecoder,stateDecoder,...
            loss] = myCCD(Zin,Zout,iteration,learnRate,...
            gradientDecayFactor, squaredGradientDecayFactor,...
            dlnetEncoder,trailingAvgEncoder, trailingAvgSqEncoder,...
            dlnetDecoder,trailingAvgDecoder, trailingAvgSqDecoder)

    input=dlarray(Zin,'SSCB');
    target=dlarray(Zout,'SSCB');
    [outEncoder,stateEncoder]=forward(dlnetEncoder,input);
    outEncoder=dlarray(squeeze(outEncoder),'CB');
    [outDecoder,stateDecoder]=forward(dlnetDecoder,outEncoder);

    loss=mse(outDecoder,target);
       
    gradientsEncoder = dlgradient(loss, dlnetEncoder.Learnables,'RetainData',true);
    [dlnetEncoder,trailingAvgEncoder,trailingAvgSqEncoder] = ...
        adamupdate(dlnetEncoder, gradientsEncoder, ...
        trailingAvgEncoder, trailingAvgSqEncoder, iteration, ...
        learnRate, gradientDecayFactor, squaredGradientDecayFactor);

    gradientsDecoder = dlgradient(loss, dlnetDecoder.Learnables,'RetainData',true);
    [dlnetDecoder,trailingAvgDecoder,trailingAvgSqDecoder] = ...
        adamupdate(dlnetDecoder, gradientsDecoder, ...
        trailingAvgDecoder, trailingAvgSqDecoder, iteration, ...
        learnRate, gradientDecayFactor, squaredGradientDecayFactor);

end

        