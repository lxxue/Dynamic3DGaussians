seq=mocap_240724_Take12
seq=mocap_240724_Take10
seq=mocap_240906_Take3  
seq=mocap_240906_Take8
python create_metadata.py --seq $seq
python create_init_pcd.py --seq $seq
