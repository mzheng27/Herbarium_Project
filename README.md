# Label Segmentation 

This code is mainly based on CRAFT algorithm for text detection: https://github.com/clovaai/CRAFT-pytorch

After downloading, replace the file paths (image path, model path, results path) in test.py. Put images in t01, then run test.py. 

Output a cropped image of the largest joinned (based on [Jason's script](https://github.com/mzheng27/Herbarium_Project/blob/main/craft_res_proc.ipynb) , expand -> intersection -> union) bounding box. 
