run('vlfeat-0.9.20/toolbox/vl_setup');
run('matconvnet-1.0-beta23/matlab/vl_setupnn.m');
addpath('SIFTflow');
addpath('PFflow');
addpath('NNoptimization');
addpath('flow-code-matlab');
addpath('init_model');
addpath('model');
addpath('function');

load('data/fcss/net-epoch.mat');

net = dagnn.DagNN.loadobj(net);

same_count = 0;
min_d_x = 1000;
min_d_y = 1000;
min_d = 1000;
sum_d = 0;

for l = 1:3
    for i = 1:64
        sf_s_dx = sprintf('sf_s_dx%.2d_level%.1d', i, l);
        sf_s_dy = sprintf('sf_s_dy%.2d_level%.1d', i, l);
        sf_t_dx = sprintf('sf_t_dx%.2d_level%.1d', i, l);
        sf_t_dy = sprintf('sf_t_dy%.2d_level%.1d', i, l);
        
        sdx = net.getParam(sf_s_dx).value;
        sdy = net.getParam(sf_s_dy).value;
        tdx = net.getParam(sf_t_dx).value;
        tdy = net.getParam(sf_t_dy).value;
        
        if abs(sdx - tdx) < 2 && abs(sdy - tdy) < 2
            same_count = same_count + 1;
        end
        if abs(sdx - tdx) < 1
            d = abs(sdy - tdy);
            if d < min_d_y
                min_d_y = d;
            end
        elseif abs(sdy - tdy) < 1
            d = abs(sdx - tdx);
            if d < min_d_x
                min_d_x = d;
            end
        end
        
        d = sqrt((sdx-tdx)^2 + (sdy-tdy)^2);
        if d < min_d
            min_d = d;
        end
        sum_d = sum_d + d;
    end
end

average_d = sum_d / (3*64);
    