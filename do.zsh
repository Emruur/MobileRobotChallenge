# 1) Clone the TF models repo and enter the OD API directory
git clone https://github.com/tensorflow/models.git
cd models/research/object_detection

# 2) Download the Quantized SSD-MobileNet V2 COCO checkpoint (includes .pb and .ckpt)
wget -c http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz
tar xvf ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz
# └─ creates folder ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/  
#    containing model.ckpt.* and pipeline.config :contentReference[oaicite:0]{index=0}

# 3) Export the TFLite-compatible graph
python3 export_tflite_ssd_graph.py \
  --pipeline_config_path=samples/configs/ssd_mobilenet_v2_quantized_300x300_coco.config \
  --trained_checkpoint_prefix=ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/model.ckpt \
  --output_directory=exported_tflite_ssd \
  --add_postprocessing_op=true
# └─ writes tflite_graph.pb and a 300×300 input signature :contentReference[oaicite:1]{index=1}

# 4) Convert to a standalone .tflite
cd exported_tflite_ssd
tflite_convert \
  --output_file=detect.tflite \
  --graph_def_file=tflite_graph.pb \
  --input_shapes=1,300,300,3 \
  --input_arrays=normalized_input_image_tensor \
  --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
  --mean_values=128 \
  --std_dev_values=128
# └─ produces detect.tflite that you can copy back to your tracker folder :contentReference[oaicite:2]{index=2}

# 5) Move it into your project and re-run your script:
mv detect.tflite /path/to/your/Challenge/
