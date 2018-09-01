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
require 'src/Gaussian'
TF = require 'SpynetLossNetwork/transforms'
flowX = require 'SpynetLossNetwork/flowExtensions'
require 'src/descriptor_net'

local VAE = require 'src/VAE'
local spynet = paths.dofile('SpynetLossNetwork/models/fullModel2.lua')
spynet = spynet:cuda()
  
----------------------------------------------------------------------  
opt = lapp[[  
  --learningRate     (default 0.00002)             learning rate  
  --beta1            (default 0.5)               momentum term for adam
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
  --maxStep          (default 17)                max future time from which to sample future frame from
  --nShare           (default 1)                 number of frame to use for content encoding
  --advWeight        (default 0)                 weight on adversarial scene discriminator loss 
  --KLWeight         (default 0.1)  
]]  


opt.save = ('%s/%s/%s'):format(opt.save_root, opt.dataset, 'flow_prediction')
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

opt.geometry_flowIm = {nf+nc, opt.imageSize, opt.imageSize}

local z_dim = 2000
local ngf = 64
local naf = 64

if paths.filep(opt.save .. '/model.t7') then
  checkpoint = torch.load(opt.save .. '/model.t7') 
end
if checkpoint then
  encoder = checkpoint.netG:get(1)
  sampler = VAE.Sampler()
  netI = checkpoint.netI
  netD = checkpoint.netD
  print('Loaded models from file')
else
  print('Initialized models from scratch')
  encoder = VAE.VolEncoder(nf+nc, naf, z_dim)
  sampler = VAE.Sampler()
  netD = VAE.VolDecoder(nf, ngf, z_dim)
  netI = VAE.ImEncoder(nc, naf, z_dim)
end

netG = nn.Sequential()
netG:add(encoder)
netG:add(sampler)

netG:cuda()
netI:cuda()
netD:cuda()

optimStateG = {learningRate = opt.learningRate, beta=opt.beta1}
params_G, grads_G = netG:getParameters()

optimStateI = {learningRate = opt.learningRate, beta=opt.beta1}
params_I, grads_I = netI:getParameters()

optimStateD = {learningRate = opt.learningRate, beta=opt.beta1}
params_D, grads_D = netD:getParameters()

m_criterion = nn.AbsCriterion()
m_criterion:cuda()
-------------------------------------------------------------------------------------------
local x = {}
local x_flow = {}
local x_flowIm = {}
local flow_test = {}
local flow_gt_test = {}

local y = torch.CudaTensor(opt.maxStep-1, opt.batchSize, unpack(opt.geometry_flow))
local y_flowIm = torch.CudaTensor(opt.maxStep-1, opt.batchSize, unpack(opt.geometry_flowIm))

for i=1,opt.maxStep do
  x[i] = torch.CudaTensor(opt.batchSize, unpack(opt.geometry))
end


for i=1,opt.maxStep-1 do
  x_flowIm[i] = torch.CudaTensor(opt.batchSize, unpack(opt.geometry_flowIm))
  x_flow[i] = torch.CudaTensor(opt.batchSize, unpack(opt.geometry_flow))
  flow_test[i] = torch.CudaTensor(opt.batchSize, unpack(opt.geometry_flow))
  flow_gt_test[i] = torch.CudaTensor(opt.batchSize, unpack(opt.geometry_flow))
end

local x1 = {}
for i=1,opt.nShare do
  x1[i] = torch.CudaTensor(opt.batchSize, unpack(opt.geometry))
end

function plot_pred(plot_x, fname)

  for i=1,opt.maxStep do
    x[i]:copy(plot_x[i])
  end
  
  for i=1,opt.nShare do
    x1[i]:copy(x[i])
  end
  
  for i=1,opt.maxStep-1 do
    local pair_gt = torch.cat(x[i+1], x[i], 2)
    pair_gt = TF.normalize(pair_gt)
    x_flow[i] = spynet:forward(pair_gt):clone()
    y[{{i},{},{},{},{}}] = x_flow[i]

    x_flowIm[i] = torch.cat(x[i]*5, x_flow[i], 2)
    y_flowIm[{{i},{},{},{},{}}] = x_flowIm[i]
  end 
  local flow_gt = y:transpose(1,2):transpose(2,3)
  local input_flowIm = y_flowIm:transpose(1,2):transpose(2,3)
  
  
  local flow_embedding = netG:forward(input_flowIm)
  local im_embedding = netI:forward(x1[1])
  local pred_flow = netD:forward({im_embedding, flow_embedding})
  
  
  for i=1,opt.maxStep-1 do
    local temp_flow = pred_flow[{{},{},{i},{},{}}] 
    flow_test[i] = torch.squeeze(temp_flow)
    
    local temp_gt_flow = flow_gt[{{},{},{i},{},{}}]
    flow_gt_test[i] = torch.squeeze(temp_gt_flow)
  end

  local N = math.min(5, opt.batchSize)

  local to_plot = {}
  for i=1,N do
      for j=1,opt.maxStep-1  do       
        local flow_rgb_gen = flowX.xy2rgb(flow_test[j][i][1]:float(), flow_test[j][i][2]:float())
        table.insert(to_plot, flow_rgb_gen:float())
      end
      for j=1,opt.maxStep-1  do       
        local flow_rgb_gt = flowX.xy2rgb(flow_gt_test[j][i][1]:float(), flow_gt_test[j][i][2]:float())
        table.insert(to_plot, flow_rgb_gt:float())
      end
      
   end
   image.save(('%s/gen/%s_%d.png'):format(opt.save, fname, epoch), image.toDisplayTensor{input=to_plot, scaleeach=false, nrow=opt.maxStep-1})
end


function train(x_cpu)
  for i=1,opt.maxStep do
    x[i]:copy(x_cpu[i])
  end

  for i=1,opt.maxStep-1 do
    local pair_gt = torch.cat(x[i+1], x[i], 2)
    pair_gt = TF.normalize(pair_gt)
    x_flow[i] = spynet:forward(pair_gt):clone()
    y[{{i},{},{},{},{}}] = x_flow[i]
    x_flowIm[i] = torch.cat(x[1]*5, x_flow[i], 2)
    y_flowIm[{{i},{},{},{},{}}] = x_flowIm[i]
  end
   
  local flow_gt = y:transpose(1,2):transpose(2,3)
  local input_flowIm = y_flowIm:transpose(1,2):transpose(2,3)
  
  for i=1,opt.nShare do
    x1[i]:copy(x[i])
  end
   
  grads_G:zero()
  grads_D:zero()
  grads_I:zero()

  local pred_mse = 0

  local flow_embedding = netG:forward(input_flowIm)
  local im_embedding = netI:forward(x1[1])
  local pred_flow = netD:forward({im_embedding, flow_embedding})

  local errA = m_criterion:forward(pred_flow, flow_gt)
  local df_do = m_criterion:backward(pred_flow, flow_gt)
  

  local dI, df = unpack(netD:backward({im_embedding, flow_embedding}, df_do))
  netI:backward(x1[1], dI)
  netG:backward(input_flowIm, df)
  
  local KLLoss = 0
  mean, log_var = table.unpack(encoder.output)
  var = torch.exp(log_var)
  KLLoss = -0.5 * torch.sum(1 + log_var - torch.pow(mean, 2) - var)
  gradKLLoss = {opt.KLWeight*mean, opt.KLWeight*0.5*(var - 1)}
  encoder:backward(input_flowIm, gradKLLoss)
	
  opt.optimizer(function() return 0, grads_D end, params_D, optimStateD)
  opt.optimizer(function() return 0, grads_G end, params_G, optimStateG)
  opt.optimizer(function() return 0, grads_I end, params_I, optimStateI)
	
  return errA, KLLoss
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
  netG:training()
  netI:training()
  netD:training()
  local iter, pred_mse, flow_epe = 0, 0, 0
  local nTrain = opt.epochSize
  for i=1,nTrain,opt.batchSize do
    xlua.progress(i, nTrain)
    local batch= trainLoader:getBatch(opt.batchSize, opt.maxStep)
    local p_mse, f_epe = train(batch)
    pred_mse = pred_mse + p_mse
    flow_epe = flow_epe  + f_epe
    iter=iter+1
    total_iter = total_iter + 1
  end
  print(('\n(%d)\tprediction mse = %.4f, KL = %.6f'):format(total_iter, pred_mse/iter, flow_epe/iter))

  if pred_mse/iter < best then
    best = pred_mse / iter
    print(('Saving best model so far (pred mse = %.4f) %s/model_best.t7'):format(pred_mse/iter, opt.save))
    torch.save(('%s/model_best.t7'):format(opt.save), {netG=netG:clearState(), netI=netI:clearState(), netD=netD:clearState(), opt=opt, epoch=epoch, best=best, total_iter=total_iter})
  end

  -- test
  netG:evaluate()
  netI:evaluate()
  netD:evaluate()
	 
  -- plot 
  plot_pred(plot_x_train, 'train')
  plot_pred(plot_x_train, 'val')

  -- back to training
  netG:training()
  netI:training()
  netD:training()
  if epoch % 1 == 0 then
    print(('Saving model %s/model.t7'):format(opt.save))
    torch.save(('%s/model.t7'):format(opt.save), {netG=netG:clearState(), netI=netI:clearState(), netD=netD:clearState(), opt=opt, epoch=epoch, best=best, total_iter=total_iter})
  end
  epoch = epoch+1
  if epoch > opt.nEpochs then break end
end