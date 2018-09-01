require 'nn'
require 'dpnn'

local VAE = {}


function VAE.VolEncoder(channels, naf, z_dim)
    local encoder = nn.Sequential()
    encoder:add(nn.MulConstant(0.2))
    encoder:add(nn.VolumetricConvolution(channels, naf, 3, 3, 3, 1, 1, 1, 1, 1, 1))
    encoder:add(nn.VolumetricBatchNormalization(naf))
    encoder:add(nn.ReLU(true))
    encoder:add(nn.VolumetricMaxPooling(1, 2, 2))
    encoder:add(nn.VolumetricConvolution(naf, naf, 3, 3, 3, 1, 1, 1, 1, 1, 1))
    encoder:add(nn.VolumetricBatchNormalization(naf))
    encoder:add(nn.ReLU(true))
    encoder:add(nn.VolumetricMaxPooling(1, 2, 2))
    encoder:add(nn.VolumetricConvolution(naf, naf*2, 3, 3, 3, 1, 1, 1, 1, 1, 1))
    encoder:add(nn.VolumetricBatchNormalization(naf*2))
    encoder:add(nn.ReLU(true))             
    encoder:add(nn.VolumetricMaxPooling(2, 2, 2))                                                            
    encoder:add(nn.VolumetricConvolution(naf*2, naf*4, 3, 3, 3, 1, 1, 1, 1, 1, 1))
    encoder:add(nn.VolumetricBatchNormalization(naf*4))
    encoder:add(nn.ReLU(true))
    encoder:add(nn.VolumetricMaxPooling(2, 2, 2))
    encoder:add(nn.VolumetricConvolution(naf*4, naf*8, 3, 3, 3, 1, 1, 1, 1, 1, 1))
    encoder:add(nn.VolumetricBatchNormalization(naf*8))
    encoder:add(nn.ReLU(true))
    encoder:add(nn.VolumetricMaxPooling(2, 2, 2))
        
    zLayer = nn.ConcatTable()
    z1 = nn.Sequential()
    z1:add(nn.VolumetricConvolution(naf * 8, z_dim, 2, 4, 4))
    z2 = nn.Sequential()
    z2:add(nn.VolumetricConvolution(naf * 8, z_dim, 2, 4, 4))
    zLayer:add(z1)
    zLayer:add(z2)
    encoder:add(zLayer)
    
    return encoder
end


function VAE.Sampler()
    local noiseModule = nn.Sequential()
    --
    local stdModule = nn.Sequential()
    stdModule:add(nn.MulConstant(0.5)) -- Compute 1/2 log σ^2 = log σ
    stdModule:add(nn.Exp()) -- Compute σ
    --
    local noiseModuleInternal = nn.ConcatTable()
    noiseModuleInternal:add(stdModule) -- Standard deviation σ
    noiseModuleInternal:add(nn.Gaussian(0, 1)) -- Sample noise z
    noiseModule:add(noiseModuleInternal) 
    noiseModule:add(nn.CMulTable()) --z*σ
    --
    local sampler = nn.Sequential()
    samplerInternal = nn.ParallelTable()
    samplerInternal:add(nn.Identity()) --u
    samplerInternal:add(noiseModule)  --z*σ
    sampler:add(samplerInternal)
    sampler:add(nn.CAddTable())  --u + z*σ
    
    return sampler
end


function VAE.ImEncoder(channels, naf, z_dim)
    local encoder = nn.Sequential()
    encoder:add(nn.SpatialConvolution(channels, naf, 4, 4, 2, 2, 1, 1))
    encoder:add(nn.ReLU())
    encoder:add(nn.SpatialConvolution(naf, naf, 4, 4, 2, 2, 1, 1))
    encoder:add(nn.ReLU())
    encoder:add(nn.SpatialConvolution(naf, naf * 2, 4, 4, 2, 2, 1, 1))
    encoder:add(nn.ReLU())
    encoder:add(nn.SpatialConvolution(naf * 2, naf * 4, 4, 4, 2, 2, 1, 1))
    encoder:add(nn.ReLU())
    encoder:add(nn.SpatialConvolution(naf * 4, naf * 8, 4, 4, 2, 2, 1, 1))
    encoder:add(nn.ReLU())
    encoder:add(nn.SpatialConvolution(naf * 8, z_dim, 4, 4))
    encoder:add(nn.View(z_dim, 1, 1, 1))
    return encoder
end


function VAE.VolDecoder(channels, ngf, z_dim)
    local im_embedding = nn.Identity()() 
    local flow_embedding = nn.Identity()()

    local r000 = nn.CMulTable(2)({im_embedding, flow_embedding})
    local r001 = nn.CAddTable(2)({im_embedding, r000})

    local r002 = nn.VolumetricFullConvolution(z_dim, ngf * 8, 2, 4, 4)(r001)
    local r003 = nn.VolumetricBatchNormalization(ngf * 8)(r002)
    local r004 = nn.ReLU(true)(r003)

    local r005 = nn.VolumetricFullConvolution(ngf * 8, ngf * 4, 4,4,4, 2,2,2, 1,1,1)(r004)
    local r006 = nn.VolumetricBatchNormalization(ngf * 4)(r005)
    local r007 = nn.ReLU(true)(r006)

    local r008 = nn.VolumetricFullConvolution(ngf * 4, ngf * 2, 4,4,4, 2,2,2, 1,1,1)(r007)
    local r009 = nn.VolumetricBatchNormalization(ngf * 2)(r008)
    local r010 = nn.ReLU(true)(r009)

    local r011 = nn.VolumetricFullConvolution(ngf * 2, ngf, 4,4,4, 2,2,2, 1,1,1)(r010)
    local r012 = nn.VolumetricBatchNormalization(ngf)(r011)
    local r013 = nn.ReLU(true)(r012)

    local r014 = nn.VolumetricFullConvolution(ngf, ngf, 3,4,4, 1,2,2, 1,1,1)(r013)
    local r015 = nn.VolumetricBatchNormalization(ngf)(r014)
    local r016 = nn.ReLU(true)(r015)

    local r017 = nn.VolumetricFullConvolution(ngf, channels, 3,4,4, 1,2,2, 1,1,1)(r016)
    local r018 = nn.Tanh()(r017)
    local output = nn.MulConstant(5)(r018)

    local decoder = nn.gModule({im_embedding, flow_embedding},{output})
    
    return decoder
end


return VAE
