# UNet: semantic segmentation with PyTorch

## this branch is for training with superpoint features on megadepth and nyu images

The indicated dataset is of such format:
```
training_dataset/
		├── anns/
			└── train.txt
			└── test.txt
			└── val.txt		
				
		├── megadepth/
			└── 0037/
				└── dense0/
					└── images/
						└── 66590517_dbeb3b7fbc_o.jpg
						└── ......
					└── depth/
						└── 66590517_dbeb3b7fbc_o.jpg.npz
						└── ......
			└── ......
		├── nyu/
			└── home_office_0002/
				└── images/
					└──r-1315165279.359014-2909829786.jpg
					└── ......
				└── depth/
					└── r-1315165279.359014-2909829786.npz
					└── ......
				└── ......
			├── superpoint/
				└──megadepth_5k^_^5013^_^dense0^_^images^_^00310.jpg.npz
				└── ......
```
 
---

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox: [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)
