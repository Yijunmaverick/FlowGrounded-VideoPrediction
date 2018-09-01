require 'paths'

opt = lapp[[  
  --imageSize        (default 128)                size of image
  --dataRoot         (default 'DTexture/')  data root directory
  --classes          (default 'yourVideo')          
]]
image_size = opt.imageSize 
data_root = opt.dataRoot
if not paths.dir(data_root) then
  error(('Error with data directory: %s'):format(data_root))
end

classes = {opt.classes}

frame_rate = 24

for _, class in pairs(classes) do
  print(' ---- ')
  print(class)

  for vid in paths.iterfiles(data_root .. 'raw/' .. class) do
    print(vid)
    local fname = vid:sub(1,-5)
    os.execute(('mkdir -p %s/processed/%s/%s'):format(data_root, class, fname))
    os.execute(('ffmpeg -i %s/raw/%s/%s -r %d -f image2 -s %dx%d  %s/processed/%s/%s/image-%%03d_%dx%d.png'):format(data_root, class, vid, frame_rate, image_size, image_size, data_root, class, fname, image_size, image_size))
  end
end
 
