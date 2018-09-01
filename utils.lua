function strsplit(inputstr, sep)
  if sep == nil then
    sep = "%s"
  end
  local t={} ; i=1
  for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
    t[i] = str
    i = i + 1
  end
  return t
end

function center_crop(x, crop)
  local crop = math.min(crop, math.min(x:size(2), x:size(3)))
  local sx = math.floor((x:size(2) - crop)/2)
  local sy = math.floor((x:size(3) - crop)/2)
  return image.crop(x, sy, sx, sy+crop, sx+crop)
end

function random_crop(x, crop, sx, sy)
  assert(x:dim() == 3)
  local crop = math.min(crop, math.min(x:size(2), x:size(3)))
  local sx = sx or math.random(0, x:size(2) - crop)
  local sy = sy or math.random(0, x:size(3) - crop)
  return image.crop(x, sy, sx, sy+crop, sx+crop), sx, sy
end

function adjust_meanstd(x, mean, std)
  for c = 1,3 do
    x[c]:add(-mean[c]):div(std[c])
  end
  return x
end

function normalize(x, min, max)
  local new_min = min or -1
  local new_max = max or 1
  local old_min, old_max = x:min(), x:max()
  local eps = 1e-7
  x:add(-old_min)
  x:mul(new_max - new_min)
  x:div(old_max - old_min + eps)
  x:add(new_min)
  return x
end

-- based on https://github.com/wojzaremba/lstm/blob/master/base.lua
function clone_many(net, T)
  local clones = {}
  local params, grads = net:parameters()
  local mem = torch.MemoryFile('w'):binary()
  mem:writeObject(net)
  for t = 1,T do
    local reader = torch.MemoryFile(mem:storage(), 'r'):binary()
    local clone = reader:readObject()
    reader:close()
    local cloneParams, cloneGrads = clone:parameters()
    for i = 1, #params do
      cloneParams[i]:set(params[i])
      cloneGrads[i]:set(grads[i])
    end 
    clones[t] = clone
    collectgarbage() 
  end 
  mem:close()
  return clones
end

function updateConfusion(confusion, output, targets)
  local correct = 0
  for i = 1,targets:nElement() do
    if targets[i] ~= -1 then
      local _, ind = output[i]:max(1)
      confusion:add(ind[1], targets[i])
      if ind[1] == targets[i] then
        correct = correct+1
      end
    end
  end
  return correct
end


local function weights_init(m)
  local name = torch.type(m)
  if name:find('Convolution') or name:find('Linear') then
    m.weight:normal(0.0, 0.02)
    m.bias:fill(0)
  elseif name:find('BatchNormalization') then
    if m.weight then m.weight:normal(1.0, 0.02) end
    if m.bias then m.bias:fill(0) end
  end
end


function initModel(model)
  for _, m in pairs(model:listModules()) do
    weights_init(m)
  end
end

function isNan(x)
  return x:ne(x):sum() > 0 
end

function sampleNoise(z)
  if opt.noise == 'uniform' then
    z:uniform(-1, 1)
  else
    z:normal()
  end
end

function clone_table(t)
  local tt = {}
  for i=1,#t do
    tt[i] = t[i]:clone()
  end
end

function zero_table(t)
  for k, v in pairs(t) do
    t[k]:zero()
  end
end

function replace_table(t1, t2)
  for i=1,#t1 do
    t1[i]:copy(t2[i])
  end
end

function write_opt(opt)
  local opt_file = io.open(('%s/opt.log'):format(opt.save), 'w')
  for k, v in pairs(opt) do
    opt_file:write(('%s = %s\n'):format(k, v))
  end
  opt_file:close()
end


function borderPlot(to_plot, k)
  local k = k or 1
  local sx = to_plot[1]:size(2)
  local sy = to_plot[1]:size(3)
  for i=1,#to_plot do
    to_plot[i] = to_plot[i]:clone()
    to_plot[i][{ {}, {}, {1,k}}]:fill(1)
    to_plot[i][{ {}, {}, {sy-k+1,sy}}]:fill(1)
    to_plot[i][{ {}, {1,k}, {}}]:fill(1)
    to_plot[i][{ {}, {sx-k+1,sx}, {}}]:fill(1)
  end
end

function borderPlotRGB(to_plot, rgb)
  local nc = to_plot[1]:size(1)
  local sx = to_plot[1]:size(2)
  local sy = to_plot[1]:size(3)
  for i=1,#to_plot do
    local im 
    if nc == 1 then
      im = torch.expand(to_plot[i], 3, sx, sy):clone()
    else 
      im = to_plot[i]
    end
    to_plot[i] = im
    for c=1,3 do 
      to_plot[i][{ c, {}, 1}]:fill(rgb[c])
      to_plot[i][{ c, {}, sy}]:fill(rgb[c])
      to_plot[i][{ c, 1, {}}]:fill(rgb[c])
      to_plot[i][{ c, sx, {}}]:fill(rgb[c])
    end
  end
end

function borderPlotTensorRGB(x, rgb)
  local nc = x:size(1)
  local sx = x:size(2)
  local sy = x:size(3)
  local im 
  if nc == 1 then
    im = torch.expand(x, 3, sx, sy):clone()
  else 
    im = x
  end
  for c=1,3 do 
    im[{ c, {}, 1}]:fill(rgb[c])
    im[{ c, {}, sy}]:fill(rgb[c])
    im[{ c, 1, {}}]:fill(rgb[c])
    im[{ c, sx, {}}]:fill(rgb[c])
  end
  return im
end

function slice_table(input, start, end_)
  local result = {}

  local index = 1

  for i=start, end_ do
    result[index] = input[i]
    index = index + 1
  end

  return result
end


function extend_table(input, tail)
  for i=1, #tail do
    table.insert(input, tail[i])
  end
end

function find_index(t, e)
  for k, v in pairs(t) do
    if v == e then return k end
  end
end



---------------------------------------------------------
-- DummyGradOutput
---------------------------------------------------------

-- Simpulates Identity operation with 0 gradOutput

local DummyGradOutput, parent = torch.class('nn.DummyGradOutput', 'nn.Module')

function DummyGradOutput:__init()
  parent.__init(self)
  self.gradInput = nil
end


function DummyGradOutput:updateOutput(input)
  self.output = input
  return self.output
end

function DummyGradOutput:updateGradInput(input, gradOutput)
  self.gradInput = self.gradInput or input.new():resizeAs(input):fill(0)
  if not input:isSameSizeAs(self.gradInput) then
    self.gradInput = self.gradInput:resizeAs(input):fill(0)
  end  
  return self.gradInput 
end


----------------------
-- adds first dummy dimension
function torch.add_dummy(self)
  local sz = self:size()
  local new_sz = torch.Tensor(sz:size()+1)
  new_sz[1] = 1
  new_sz:narrow(1,2,sz:size()):copy(torch.Tensor{sz:totable()})

  if self:isContiguous() then
    return self:view(new_sz:long():storage())
  else
    return self:reshape(new_sz:long():storage())
  end
end

function torch.FloatTensor:add_dummy()
  return torch.add_dummy(self)
end
function torch.DoubleTensor:add_dummy()
  return torch.add_dummy(self)
end

function torch.CudaTensor:add_dummy()
  return torch.add_dummy(self)
end
