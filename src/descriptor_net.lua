require 'src/content_loss'

require 'loadcaffe'

function nop()
  -- nop.  not needed by our net
end

function create_descriptor_net(content_image)
    
  local cnn = torch.load(opt.loss_model):cuda()
  local content_layers = opt.content_layers:split(",") 

  -- Set up the network, inserting texture and content loss modules
  local content_losses = {}
  local next_content_idx = 1
  local net = nn.Sequential()

  for i = 1, #cnn do
    if next_content_idx <= #content_layers then
      local layer = cnn:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
      
      if layer_type == 'nn.SpatialConvolution' or layer_type == 'nn.SpatialConvolutionMM' or layer_type == 'cudnn.SpatialConvolution' then
        layer.accGradParameters = nop
      end

      net:add(layer)
   
      ---------------------------------
      -- Content loss
      ---------------------------------
      if opt.content_weight ~= 0 and i == tonumber(content_layers[next_content_idx]) then

        local target = net:forward(content_image):clone()

        local norm = false
        local loss_module = nn.ContentLoss(opt.perceptualWeight , target, norm):cuda()

        net:add(loss_module)
        table.insert(content_losses, loss_module)
        next_content_idx = next_content_idx + 1
      end
    end
  end

  net:add(nn.DummyGradOutput())

  -- We don't need the base CNN anymore, so clean it up to save memory.
  cnn = nil
  for i=1,#net.modules do
    local module = net.modules[i]
    if torch.type(module) == 'nn.SpatialConvolutionMM' or torch.type(module) == 'nn.SpatialConvolution' or torch.type(module) == 'cudnn.SpatialConvolution' then
        module.gradWeight = nil
        module.gradBias = nil
    end
  end
  collectgarbage()
      
  return net, content_losses
end


