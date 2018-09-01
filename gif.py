from os import system

savedir_pred = "./logs/DTexture/flow_prediction/TestResults"

cmd1 = ("ffmpeg -y -f image2 -framerate 8 -i "+savedir_pred+"/001_%02d_pred.png "+savedir_pred+"/pred.gif")
system(cmd1)
print "Done."

cmd2 = ("ffmpeg -y -f image2 -framerate 8 -i "+savedir_pred+"/001_%02d_pred.png "+"-vcodec libx264 -pix_fmt yuv420p " +savedir_pred+"/pred.mp4")
system(cmd2)
