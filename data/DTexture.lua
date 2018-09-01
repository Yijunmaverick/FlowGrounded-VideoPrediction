require 'torch'
require 'paths'
require 'image'
require 'utils'


local DTextureDataset = torch.class('DTextureLoader')

if torch.getmetatable('dataLoader') == nil then
   torch.class('dataLoader')
end

function DTextureDataset:__init(opt, data_type)
  self.data_type = data_type
  self.opt = opt or {}
  self.path = self.opt.dataRoot 
  self.data = torch.load(('%s/processed/meta/WavingFlag_%s_meta.t7'):format(self.path, data_type))
  self.classes = {'WavingFlag'}
end


function DTextureDataset:getBatch(n, T, delta)
  local xx = torch.Tensor(T, unpack(self.opt.geometry))
  local x = {}
  for t=1,T do
    x[t] = torch.Tensor(n, unpack(self.opt.geometry))
  end
  for i = 1,n do
    while not self:getSequence(xx, delta) do
    end
    for t=1,T do
      x[t][i]:copy(xx[t])
    end
  end 
  return x
end


function DTextureDataset:getSequence(x, delta)
  local delta = math.random(1, delta or self.opt.delta or 1) 
  local c = self.classes[math.random(#self.classes)]
  local vid = self.data[c][math.random(#self.data[c])]
  local seq_length = vid.n
  local basename = ('%s/processed/%s/%s'):format(self.path, c, vid.vid) 
  --print('basename = ', basename)

  local T = x:size(1)
  while T*delta > seq_length do
    delta = delta-1
    if delta < 1 then return false end
  end

  local offset = math.random(seq_length-T*delta)
  local start = vid.startframe

  for t = 1,T do
    local tt = start + offset+(t-1)*delta - 1
    local img = image.load(('%s/image-%03d_%dx%d.png'):format(basename, tt, self.opt.imageSize, self.opt.imageSize))
    x[t]:copy(img)
  end
  return true
end

function DTextureDataset:plotSeq(fname)
  print('plotting sequence: ' .. fname)
  local to_plot = {}
  local t = 10 --30 
  local n = 50
  for i = 1,n do
    local x = self:getBatch(1, t)
    for j = 1,t do
      table.insert(to_plot, x[j][1])
    end
  end 
  image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=false, nrow=t})
end

function DTextureDataset:plot()
  local savedir = self.opt.save  .. '/data/'
  os.execute('mkdir -p ' .. savedir)
  self:plotSeq(savedir .. '/' .. self.data_type .. '_seq.png')
end

trainLoader = DTextureLoader(opt_t or opt, 'train')
valLoader = DTextureLoader(opt_t or opt, 'test')
