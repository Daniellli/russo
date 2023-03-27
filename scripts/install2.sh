

#* refer data 
mkdir datasets
cd datasets
ln -s  ~/exp/data/butd_data/scanrefer scanrefer
ln -s  ~/exp/data/butd_data/refer_it_3d refer_it_3d
mv  ~/exp/data/butd_data/gf_detector_l6o256.pth  ./
mv  ~/exp/data/butd_data/group_free_pred_bboxes_* ./
mv  ~/exp/data/butd_data/pretrained ../
ln -s  ~/exp/data/scannet/scans scans
ln -s  ~/exp/data/scannet/scans_test scans_test


#* cache 
mv ~/exp/data/butd_data/roberta   ~/.cache/huggingface/transformers/ 






#* can not run 
# python scripts/download_scannet_files.py  > logs/install.log 



#* 
# mkdir logs
# python prepare_data.py --data_root datasets  2>&1 | tee -a logs/install.log





#* hhelo 