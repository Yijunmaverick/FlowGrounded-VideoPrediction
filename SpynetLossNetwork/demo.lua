require 'nn'
require 'cunn'
require 'cudnn'
require 'image'
require 'stn'
require 'nngraph'
local TF = require 'transforms'
local flowX = require 'flowExtensions'

function flow_warping_model()
    
    local imgData = nn.Identity()() 
    local floData = nn.Identity()()
    local img2 = nn.Transpose({2,3},{3,4})(nn.Narrow(2,4,3)(imgData)) -- Warping on the second image
    local floOut = nn.Transpose({2,3},{3,4})(floData)
    local output = nn.Transpose({3,4},{2,3})(nn.BilinearSamplerBHWD()({img2, floOut}))
    net = nn.gModule({imgData,floData},{output})
    
    return net 
end


local model = paths.dofile('models/fullModel2_1.lua')
model = model:cuda()

local im1 = image.load('image-075_128x128.png' )
local im2 = image.load('image-076_128x128.png' )
local im = torch.cat(im1, im2, 1)

im = im:resize(1, im:size(1), im:size(2), im:size(3)):cuda()

im = TF.normalize(im)

local flow = model:forward(im)

local flow_rgb = flowX.xy2rgb(flow[1][1]:double(), flow[1][2]:double())
image.save('flow.jpg',flow_rgb)