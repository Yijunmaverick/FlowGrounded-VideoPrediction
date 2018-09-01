opt = lapp[[
  --classes          (default 'yourVideo')          
]]

file = "./DTexture/processed/" .. opt.classes .. "ID.txt"
f = assert(io.open(file, "r"))

ids = {}
starts = {}
endings = {}
num_lines = 0 
for line in io.lines(file) do 
	local id, start, ending = unpack(line:split(" "))
    ids[#ids + 1] = id
    starts[#starts + 1] = tonumber(start)
    endings[#endings + 1] = tonumber(ending)
    num_lines = num_lines + 1
end

vid = {}
data={}

for i=1,num_lines do
   local cubic = {startframe = starts[i], endframe = endings[i], vid = ids[i], n = endings[i]-starts[i]+1}
   vid[#vid + 1] = cubic
end

class = opt.classes
data[class] =vid
print(data)

torch.save("./DTexture/processed/meta/" .. opt.classes .. "_train_meta.t7", data)