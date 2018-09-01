local conv_num = 256

net = nn.Sequential()

--------------------------
net:add(nn.SpatialReflectionPadding(1,1,1,1))
net:add(nn.SpatialConvolution(3, conv_num/4, 3, 3))
net:add(nn.ReLU())

net:add(nn.SpatialReflectionPadding(1,1,1,1))
net:add(nn.SpatialConvolution(conv_num/4, conv_num/4,3,3))
net:add(nn.ReLU())

net:add(nn.SpatialMaxPooling(2,2))

net:add(nn.SpatialReflectionPadding(1,1,1,1))
net:add(nn.SpatialConvolution(conv_num/4, conv_num/2,3,3))
net:add(nn.ReLU())

net:add(nn.SpatialReflectionPadding(1,1,1,1))
net:add(nn.SpatialConvolution(conv_num/2, conv_num/2,3,3))
net:add(nn.ReLU())

net:add(nn.SpatialMaxPooling(2,2))

net:add(nn.SpatialReflectionPadding(1,1,1,1))
net:add(nn.SpatialConvolution(conv_num/2, conv_num,3,3))
net:add(nn.ReLU())

net:add(nn.SpatialReflectionPadding(1,1,1,1))
net:add(nn.SpatialConvolution(conv_num, conv_num,3,3))
net:add(nn.ReLU())

net:add(nn.SpatialReflectionPadding(1,1,1,1))
net:add(nn.SpatialConvolution(conv_num, conv_num,3,3))
net:add(nn.ReLU())

net:add(nn.SpatialReflectionPadding(1,1,1,1))
net:add(nn.SpatialConvolution(conv_num, conv_num,3,3))
net:add(nn.ReLU())

net:add(nn.SpatialMaxPooling(2,2))

net:add(nn.SpatialReflectionPadding(1,1,1,1))
net:add(nn.SpatialConvolution(conv_num, conv_num*2,3,3))
net:add(nn.ReLU())


return net
