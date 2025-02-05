export http_proxy=http://127.0.0.1:7890;export https_proxy=http://127.0.0.1:7890
python eval_AET.py --config ./configs/Retrieval_coco.yaml \
	--cuda_id 1 \
	--model_list ALBEF TCL CLIP_ViT CLIP_CNN \
	--source_model ALBEF \
	--albef_ckpt ./checkpoints/albef_coco.pth \
	--tcl_ckpt ./checkpoints/tcl_coco.pth \
	--original_rank_index_path ./std_eval_idx/coco/\
	# --batch_size 1

# python eval_AET.py --config ./configs/Retrieval_coco.yaml \
# 	--cuda_id 3 \
# 	--model_list ALBEF TCL CLIP_ViT CLIP_CNN \
# 	--source_model CLIP_ViT \
# 	--albef_ckpt ./checkpoints/albef_coco.pth \
# 	--tcl_ckpt ./checkpoints/tcl_coco.pth \
# 	--original_rank_index_path ./std_eval_idx/coco/\