require 'torch'  
require 'nn'  
require 'nngraph'  
require 'cunn'  
require 'cudnn'  
require 'optim'  
require 'pl'  
require 'paths'  
require 'image'  
require 'utils'
require 'src/InstanceNormalization'
require 'stn'
require 'src/descriptor_net'

TF = require 'SpynetLossNetwork/transforms'
flowX = require 'SpynetLossNetwork/flowExtensions'

local spynet = paths.dofile('SpynetLossNetwork/models/fullModel2.lua')
spynet = spynet:cuda()
 
----------------------------------------------------------------------  
opt = lapp[[
  --learningRate     (default 0.0001)             learning rate
  --beta             (default 0.9)               momentum term for adam
  --batchSize        (default 4)               batch size  
  --save_root        (default 'logs/')           base directory to save logs  
  --dataRoot         (default '/path/to/data/')  data root directory
  --optimizer        (default 'adam')            optimizer to train with
  --nEpochs          (default 5000)               max training epochs  
  --seed             (default 1)                 random seed  
  --epochSize        (default 1000)             number of samples per epoch  
  --imageSize        (default 128)                size of image
  --dataset          (default DTexture)      dataset
  --movingDigits     (default 1)                 if moving mnist dataset, how many digits to use
  --cropSize         (default 227)               size of crop (for kitti only)
  --maxStep          (default 3)                max future time from which to sample future frame from
  --nShare           (default 1)                 number of frame to use for content encoding
  --loss_model        (default 'models/perceptual/vgg_normalised.t7')
  --perceptualWeight  (default 1.0)
  --content_layers    (default '1,4,11,18,31')
]]  

opt.save = ('%s/%s/%s'):format(opt.save_root, opt.dataset, 'flow2rgb')
os.execute('mkdir -p ' .. opt.save .. '/gen/')

assert(optim[opt.optimizer] ~= nil, 'unknown optimizer: ' .. opt.optimizer)
opt.optimizer = optim[opt.optimizer]

torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)
math.randomseed(opt.seed)

local nc = 3
local nf = 2
opt.geometry = {nc, opt.imageSize, opt.imageSize}
opt.geometry_flow = {nf, opt.imageSize, opt.imageSize}


if paths.filep(opt.save .. '/model.t7') then
  checkpoint = torch.load(opt.save .. '/model.t7') 
end


if checkpoint then
  netC = checkpoint.netC
  netF = checkpoint.netF
  netD = checkpoint.netD
  print('Loaded models from file')
else
  netC = require('models/content_conv4_1.lua')
  netF = require('models/flow_conv4_1.lua')
  netD = require('models/invert_conv4_1_concat.lua')
  print('Initialized models from scratch')
end

optimStateC = {learningRate = opt.learningRate, beta=opt.beta}
optimStateF = {learningRate = opt.learningRate, beta=opt.beta}
optimStateD = {learningRate = opt.learningRate, beta=opt.beta}

netC:cuda()
netF:cuda()
netD:cuda()

params_C, grads_C = netC:getParameters()
params_F, grads_F = netF:getParameters()
params_D, grads_D = netD:getParameters()

rec_criterion = nn.MSECriterion()
rec_criterion:cuda()

local x = {}
for i=1,opt.maxStep do
  x[i] = torch.CudaTensor(opt.batchSize, unpack(opt.geometry))
end
local x1, x2 = {}, {}
for i=1,opt.nShare do
  x1[i] = torch.CudaTensor(opt.batchSize, unpack(opt.geometry))
  x2[i] = torch.CudaTensor(opt.batchSize, unpack(opt.geometry))
end

function flow_warping_model()
    local imgData = nn.Identity()() 
    local floData = nn.Identity()()
    
    local img2 = nn.Transpose({2,3},{3,4})(imgData) -- Warping on the second image
    local floOut = nn.Transpose({2,3},{3,4})(floData)
    local output = nn.Transpose({3,4},{2,3})(nn.BilinearSamplerBHWD()({img2, floOut}))
    
    local net = nn.gModule({imgData,floData},{output})
    
    return net 
end

WarpNet = flow_warping_model():cuda()

function plot_pred(plot_x, fname)
  
  for i=1,opt.maxStep do
    x[i]:copy(plot_x[i])
  end
  for i=1,opt.nShare do
    x1[i]:copy(x[i])
  end
  
  local offset = math.random(opt.nShare+1, opt.maxStep-opt.nShare)
  
  for i=1,opt.nShare do
    x2[i]:copy(x[i+offset])
  end

  local pair_gt = torch.cat(x2[1], x1[1], 2)
  pair_gt = TF.normalize(pair_gt)
  local flow_gt = spynet:forward(pair_gt):clone()
  local hp2 = netF:forward(flow_gt)

  local x1_warp = WarpNet:forward({x1[1], flow_gt})

  local hc1 = netC:forward(x1_warp)
  local pred = netD:forward({hc1, hp2})
  
  
  local pair_gen = torch.cat(pred, x1[1], 2)
  pair_gen = TF.normalize(pair_gen)
  local flow_gen = spynet:forward(pair_gen):clone()
  
  local N = math.min(20, opt.batchSize)

  local to_plot = {}
  for i=1,N do
    for ii=1,opt.nShare do
      table.insert(to_plot, x1[ii][i]:float())
      table.insert(to_plot, x2[ii][i]:float())
    end

    table.insert(to_plot, pred[i]:float())
    local flow_rgb_gt = flowX.xy2rgb(flow_gt[i][1]:float(), flow_gt[i][2]:float())
    table.insert(to_plot, flow_rgb_gt:float())
    local flow_rgb_gen = flowX.xy2rgb(flow_gen[i][1]:float(), flow_gen[i][2]:float())
    table.insert(to_plot, flow_rgb_gen:float())
  end

  image.save(('%s/gen/%s_%d.png'):format(opt.save, fname, epoch), image.toDisplayTensor{input=to_plot, scaleeach=false, nrow=10})
end



function train(x_cpu)
    
  for i=1,opt.maxStep do
    x[i]:copy(x_cpu[i])
  end
  for i=1,opt.nShare do
    x1[i]:copy(x[i])
  end
  
  local offset = math.random(opt.nShare+1, opt.maxStep-opt.nShare)
  
  for i=1,opt.nShare do
    x2[i]:copy(x[i+offset])
  end

  grads_C:zero()
  grads_F:zero()
  grads_D:zero()
  

  local pair_gt = torch.cat(x2[1], x1[1], 2)
  pair_gt = TF.normalize(pair_gt)
  local flow_gt = spynet:forward(pair_gt):clone()
  local hp2 = netF:forward(flow_gt)

  local x1_warp = WarpNet:forward({x1[1], flow_gt})
  local hc1 = netC:forward(x1_warp)

  local pred = netD:forward({hc1, hp2})  
  -- minimize ||P(hc1, hp2), x2||
  local pred_mse = rec_criterion:forward(pred, x2[1])
  local dpred = rec_criterion:backward(pred, x2[1])
  
  
  local dper, loss_content = 0, 0
  if opt.perceptualWeight > 0 then
    local descriptor_net, content_losses = create_descriptor_net(x2[1])
    descriptor_net:forward(pred)
    dper = descriptor_net:backward(pred, nil)
    
    for _, mod in ipairs(content_losses) do
      loss_content = loss_content + mod.loss
    end

  end
    
  local dhc1, dhp2 = unpack(netD:backward({hc1, hp2}, dper))

  netC:backward(x1_warp, dhc1) 
  netF:backward(flow_gt, dhp2) 

  opt.optimizer(function() return 0, grads_C end, params_C, optimStateC)
  opt.optimizer(function() return 0, grads_F end, params_F, optimStateF)
  opt.optimizer(function() return 0, grads_D end, params_D, optimStateD)

  return loss_content
end


require(('data.%s'):format(opt.dataset))

plot_x_train = trainLoader:getBatch(opt.batchSize, opt.maxStep)
plot_x_val = valLoader:getBatch(opt.batchSize, opt.maxStep)

if checkpoint then
  best = checkpoint.best
  start_epoch = checkpoint.epoch+1
  total_iter = checkpoint.total_iter
  print('Starting training at epoch ' .. start_epoch)
else
  best = 1e10
  start_epoch = 0 
  total_iter = 0
end
epoch = start_epoch
while true do
  collectgarbage()
  collectgarbage()

  -- train
  print('\n<trainer> Epoch ' .. epoch )
  netC:training()
  netF:training()
  netD:training()
  local iter, pred_mse, gram = 0, 0, 0
  
  local nTrain = opt.epochSize
  for i=1,nTrain,opt.batchSize do
    xlua.progress(i, nTrain)
    local batch = trainLoader:getBatch(opt.batchSize, opt.maxStep)
    local p_mse = train(batch)
    pred_mse = pred_mse + p_mse
    iter=iter+1
    total_iter = total_iter + 1
  end
  print(('\n(%d)\tperceptual loss = %.4f'):format(total_iter, pred_mse/iter))

  if pred_mse/iter < best then
    best = pred_mse / iter
    print(('Saving best model so far (pred mse = %.4f) %s/model_best.t7'):format(pred_mse/iter, opt.save))
    torch.save(('%s/model_best.t7'):format(opt.save), {netC=netC:clearState(), netF=netF:clearState(), netD=netD:clearState(), opt=opt, epoch=epoch, best=best, total_iter=total_iter})
  end
 
  -- plot 
  netC:evaluate()
  netF:evaluate()
  netD:evaluate()
  plot_pred(plot_x_train, 'train')
  plot_pred(plot_x_val, 'val')

  --back to training
  netC:training()
  netF:training()
  netD:training()
  
  if epoch % 1 == 0 then
    print(('Saving model %s/model.t7'):format(opt.save))
    torch.save(('%s/model.t7'):format(opt.save), {netC=netC:clearState(), netF=netF:clearState(), netD=netD:clearState(), opt=opt, epoch=epoch, best=best, total_iter=total_iter})
  end
  epoch = epoch+1
  if epoch > opt.nEpochs then break end
end
