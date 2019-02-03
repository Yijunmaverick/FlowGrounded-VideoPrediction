cd models/perceptual/

wget -c http://vllab1.ucmerced.edu/~yli62/VideoPrediction/vgg/vgg_normalised.t7

cd ../..

cd logs/DTexture/flow_prediction/
wget -c http://vllab1.ucmerced.edu/~yli62/VideoPrediction/flag/flow_prediction/model_flag_eccv.t7
wget -c http://vllab1.ucmerced.edu/~yli62/VideoPrediction/kth/flow_prediction/model_kth_eccv.t7

cd ..
cd flow2rgb/
wget -c http://vllab1.ucmerced.edu/~yli62/VideoPrediction/flag/flow2rgb/model_flag_eccv.t7
wget -c http://vllab1.ucmerced.edu/~yli62/VideoPrediction/kth/flow2rgb/model_kth_eccv.t7

cd ../../../..
