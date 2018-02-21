%% Script for learning FCSS descriptor network
%% Written by Seungryong Kim, Yonsei University, Seoul, Korea
function main_FCSS_train_Tatsunori(varargin)
    run('vlfeat-0.9.20/toolbox/vl_setup');
    run('matconvnet-1.0-beta23/matlab/vl_setupnn.m');
    addpath('SIFTflow');
    addpath('flow-code-matlab');
    addpath('init_model');
    addpath('model');
    addpath('function');

    %load('data/imdb_correspondence_Tatsunori.mat');
    load('data/imdb_correspondence_Lake.mat');

    init_model = true; % Using pretrained model as an initial parameter (or not)
    net = init_FCSS(init_model);

    trainOpts.batchSize = 850;
    trainOpts.numEpochs = 1600;
    trainOpts.continue = true;
    trainOpts.gpus = 1;
    trainOpts.learningRate = 5e-2;
    trainOpts.derOutputs = {'objective', 1};
    trainOpts.expDir = 'data/fcss_Lake1e-1';
    trainOpts.expFrequency = 200;
    trainOpts.fileLog = 'logb5e-2.out';
    trainOpts = vl_argparse(trainOpts, varargin);

    cnn_train_dag_pairwise_learning(net, imdb, getBatch, trainOpts);
end

function inputs = getBatch()
    inputs = @(imdb,batch) getBatch_Tatsunori(imdb,batch);
end
