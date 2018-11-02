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
  

opt = lapp[[  
  --batchSize        (default 4)               batch size  
  --save_root        (default 'logs/')           base directory to save logs  
  --dataRoot         (default 'datasets/DTexture')  data root directory
  --seed             (default 1)                 random seed  
  --imageSize        (default 128)                size of image
  --dataset          (default DTexture)      dataset
  --maxStep          (default 17)                max future time from which to sample future frame from
  --nShare           (default 1)                 number of frame to use for content encoding
  --startingFrame    (default 'c1.png')
]]  

opt.save = ('%s/%s/%s'):format(opt.save_root, opt.dataset, 'flow_prediction')
os.execute('mkdir -p ' .. opt.save .. '/TestResults/')

torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')


local nc = 3
local nf = 2

opt.geometry = {nc, opt.imageSize, opt.imageSize}

opt.geometry_flow = {nf, opt.imageSize, opt.imageSize}

--flow prediction model
if paths.filep(opt.save .. '/model_flag_eccv.t7') then
  checkpoint1 = torch.load(opt.save .. '/model_flag_eccv.t7') 
end
if checkpoint1 then
  netG = checkpoint1.netG
  netI = checkpoint1.netI
  netD1 = checkpoint1.netD
  netG:cuda()
  netI:cuda()
  netD1:cuda()
end
netG:evaluate()
netI:evaluate()
netD1:evaluate()

--flow2rgb model
opt.save2 = ('%s/%s/%s'):format(opt.save_root, opt.dataset, 'flow2rgb')
if paths.filep(opt.save2 .. '/model_flag_eccv.t7') then
  checkpoint2 = torch.load(opt.save2 .. '/model_flag_eccv.t7') 
end

if checkpoint2 then
  netC = checkpoint2.netC
  netF = checkpoint2.netF
  netD2 = checkpoint2.netD
  netC:cuda()
  netF:cuda()
  netD2:cuda()
end
netC:evaluate()
netF:evaluate()
netD2:evaluate()

--------------------------------------------------------------------
local x = {}
local x_flow = {}
local flow_test = {}
local flow_gt_test = {}

local y = torch.CudaTensor(opt.maxStep-1, opt.batchSize, unpack(opt.geometry_flow))
local dpred_flow = torch.CudaTensor(opt.maxStep-1, opt.batchSize, unpack(opt.geometry_flow))

for i=1,opt.maxStep do
  x[i] = torch.CudaTensor(opt.batchSize, unpack(opt.geometry))
end

for i=1,opt.maxStep-1 do
  x_flow[i] = torch.CudaTensor(opt.batchSize, unpack(opt.geometry_flow))
  flow_test[i] = torch.CudaTensor(opt.batchSize, unpack(opt.geometry_flow))
  flow_gt_test[i] = torch.CudaTensor(opt.batchSize, unpack(opt.geometry_flow))
end

local x1 = {}
for i=1,opt.nShare do
  x1[i] = torch.CudaTensor(opt.batchSize, unpack(opt.geometry))
end

local z_dim = 2000
noise_x = torch.Tensor(opt.batchSize, z_dim, 1, 1, 1)
noise_x:normal(0, 1)
noise_x = noise_x:cuda()

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

  local id = 1
  local im = image.load(opt.startingFrame)
  im = image.scale(im, opt.imageSize, opt.imageSize)
  x1[1][id]:copy(im)
  
  for i=1,opt.maxStep-1 do
    local pair_gt = torch.cat(x[i+1], x[i], 2)
    pair_gt = TF.normalize(pair_gt)
    x_flow[i] = spynet:forward(pair_gt):clone()
    y[{{i},{},{},{},{}}] = x_flow[i]
  end 
  local flow_gt = y:transpose(1,2):transpose(2,3)
  
  local im_embedding = netI:forward(x1[1])
  local pred_flow = netD1:forward({im_embedding, noise_x})
  
  
  for i=1,opt.maxStep-1 do
    local temp_flow = pred_flow[{{},{},{i},{},{}}] 
    flow_test[i] = torch.squeeze(temp_flow)
    local temp_gt_flow = flow_gt[{{},{},{i},{},{}}]
    flow_gt_test[i] = torch.squeeze(temp_gt_flow)
  end


  local N = math.min(id, opt.batchSize)

  local to_plot = {}
  for i=id, id do
      for j=1,opt.maxStep-1  do       
        local flow_rgb_gen = flowX.xy2rgb(flow_test[j][i][1]:float(), flow_test[j][i][2]:float())
        table.insert(to_plot, flow_rgb_gen:float())
        image.save(('%s/TestResults/%03d_%02d_predflow.png'):format(opt.save, i, j), flow_rgb_gen:float())
      end
      for j=1,opt.maxStep-1  do       
        local flow_rgb_gt = flowX.xy2rgb(flow_gt_test[j][i][1]:float(), flow_gt_test[j][i][2]:float())
      end

      local pred_pre = x1[1]
      image.save(('%s/TestResults/%03d_00_pred.png'):format(opt.save, i), x1[1][id]:float())
      for j=1,opt.maxStep-1 do

        local hp1 = netF:forward(flow_test[j])
        local x1_warp = WarpNet:forward({pred_pre, flow_test[j]})
        local hc1 = netC:forward(x1_warp)
        local pred  = netD2:forward({hc1, hp1})

        table.insert(to_plot, pred[i]:float())
        image.save(('%s/TestResults/%03d_%02d_pred.png'):format(opt.save, i, j), pred[i]:float())
        pred_pre:copy(pred)
      end
      
   end
   image.save(('%s/TestResults/test_%s.png'):format(opt.save, fname), image.toDisplayTensor{input=to_plot, scaleeach=false, nrow=opt.maxStep-1})
end

--main
require(('data.%s'):format(opt.dataset))

plot_pred(valLoader:getBatch(opt.batchSize, opt.maxStep), 'val')
collectgarbage()
