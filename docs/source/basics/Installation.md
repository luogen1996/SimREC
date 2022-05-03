## Installation
Here we provide the basic installation guide for SimREC, including setups of the environments, datasets and the pretrained weights.

### Environment Setup
- Clone this repo
```bash
git clone https://github.com/luogen1996/SimREC.git
cd SimREC
```
- Create a conda virtual environment and activate it
```bash
conda create -n simrec python=3.7 -y
conda activate simrec
```
- Install Pytorch following the [official installation instructions](https://pytorch.org/get-started/locally/)
- Install mmcv following the [installation guide](https://github.com/open-mmlab/mmcv#installation)
- Install [Spacy](https://spacy.io/) and initialize the [GloVe](https://github-releases.githubusercontent.com/84940268/9f4d5680-4fed-11e9-9dd2-988cce16be55?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210815%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210815T072922Z&X-Amz-Expires=300&X-Amz-Signature=1bd1bd4fc52057d8ac9eec7720e3dd333e63c234abead471c2df720fb8f04597&X-Amz-SignedHeaders=host&actor_id=48727989&key_id=0&repo_id=84940268&response-content-disposition=attachment%3B%20filename%3Den_vectors_web_lg-2.1.0.tar.gz&response-content-type=application%2Foctet-stream) and install other requirements as follows:
```
pip install -r requirements.txt
wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
pip install en_vectors_web_lg-2.1.0.tar.gz
```


### Dataset Setup
Prepare the datasets before running experiments.

The project directory is ``$ROOT``，and current directory is located at ``$ROOT/data``  to generate annotations.

1. Download the cleaned referring expressions datasets and extract them into `$ROOT/data` folder:

<!-- | Dataset | Download URL |
|:--------|:-------------|
| RefCOCO  | http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip  |
| RefCOCO+ | http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip |
| RefCOCOg | http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip |
| RefClef  | https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip | -->

<table class="docutils">
  <tbody>
    <tr>
      <th width="80" align="left"> Dataset </th>
      <th valign="bottom" align="center" width="120">Download URL</th>
    </tr>
    <tr>
      <td align="left"> RefCOCO </td>
      <td align="left"> http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip </td>
    </tr>
    <tr>
      <td align="left"> RefCOCO+ </td>
      <td align="left"> http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip </td>
    </tr>
    <tr>
      <td align="left"> RefCOCOg </td>
      <td align="left"> http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip </td>
    </tr>
    <tr>
      <td align="left"> RefClef </td>
      <td align="left"> https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip </td>
    </tr>
    </tr>
  </tbody>
</table>

1. Prepare [mscoco train2014 images](https://pjreddie.com/projects/coco-mirror),  [original Flickr30K images](http://shannon.cs.illinois.edu/DenotationGraph/), [ReferItGame images ](https://drive.google.com/file/d/1R6Tm7tQTHCil6A_eOhjudK3rgaBxkD2t/view?usp=sharing)and [Visual Genome images](http://visualgenome.org/api/v0/api_home.html), and unzip the annotations. Then the file structure should look like:
```
$ROOT/data
|-- refcoco
    |-- instances.json
    |-- refs(google).p
    |-- refs(unc).p
|-- refcoco+
    |-- instances.json
    |-- refs(unc).p
|-- refcocog
    |-- instances.json
    |-- refs(google).p
    |-- refs(umd).p
|-- refclef
    |-- instances.json
    |-- refs(berkeley).p
    |-- refs(unc).p
|-- images
    |-- train2014
    |-- refclef
    |-- flickr
    |-- VG   
```

3. Run [data_process.py](./data/data_process.py) to generate the annotations. For example, running the following code to generate the annotations for **RefCOCO**:

```
cd $ROOT/data
python data_process.py --data_root $ROOT/data --output_dir $ROOT/data --dataset refcoco --split unc --generate_mask
```
- `--dataset={'refcoco', 'refcoco+', 'refcocog', 'refclef'}` to set the dataset to be processd.

​For **Flickr** and **merged pre-training data**, we provide the pre-processed json files: [flickr.json](https://1drv.ms/u/s!AmrFUyZ_lDVGim3OYlbaTGP7hzZV?e=rhFf29), [merge.json](https://1drv.ms/u/s!AmrFUyZ_lDVGim7ufJ41Z0anf0A4?e=vraV1O).

**Note:** The merged pre-training data contains the training data from RefCOCO *train*,  RefCOCO+ *train*, RefCOCOg  *train*, Referit *train*, Flickr *train* and VG. We also remove the images appearing the validation and testing set of RefCOCO, RefCOCO+ and RefCOCOg.

4. At this point the directory  `$ROOT/data` should look like: 
```
$ROOT/data
|-- refcoco
    |-- instances.json
    |-- refs(google).p
    |-- refs(unc).p
|-- refcoco+
    |-- instances.json
    |-- refs(unc).p
|-- refcocog
    |-- instances.json
    |-- refs(google).p
    |-- refs(umd).p
|-- anns
    |-- refcoco
        |-- refcoco.json
    |-- refcoco+
        |-- refcoco+.json
    |-- refcocog
        |-- refcocog.json
    |-- refclef
        |-- refclef.json
    |-- flickr
        |-- flickr.json
    |-- merge
        |-- merge.json
|-- masks
    |-- refcoco
    |-- refcoco+
    |-- refcocog
    |-- refclef
|-- images
    |-- train2014
    |-- refclef
    |-- flickr
    |-- VG       
|-- weights
    |-- pretrained_weights
```

### Pretrained Weight Setup

We provide the pretrained weights of visual backbones on MS-COCO. We remove all images appearing in the *val+test* splits of RefCOCO, RefCOCO+ and RefCOCOg. Please download the following weights into `$ROOT/data/weights`.

<!-- | Pretrained Weights of Backbone       |                             Link                             |
| ------------------------------------ | :----------------------------------------------------------: |
| DarkNet53-coco                       | [OneDrive](https://1drv.ms/u/s!AmrFUyZ_lDVGinNMjv1ST758T4lj?e=UqumPe) , Baidu Cloud |
| CSPDarkNet-coco                      | [OneDrive](https://1drv.ms/u/s!AmrFUyZ_lDVGinF-8LK_9tzqArs9?e=vvADN9) , Baidu Cloud |
| Vgg16-coco                           | [OneDrive](https://1drv.ms/u/s!AmrFUyZ_lDVGinDBG42mcf3E5Rhg?e=T4qVqu) , Baidu Cloud |
| DResNet101-voc                       | [OneDrive](https://1drv.ms/u/s!AmrFUyZ_lDVGinK9ZJI1D-kvUWh8?e=0B2F5t) , Baidu Cloud | -->

<table class="docutils">
  <tbody>
    <tr>
      <th align="left"> Pretrained Weights of Backbone </th>
      <th align="center" width="120">Download URL</th>
    </tr>
    <tr>
      <td align="left"> DarkNet53-coco </td>
      <td align="left"> <a href="https://1drv.ms/u/s!AmrFUyZ_lDVGinNMjv1ST758T4lj?e=UqumPe">OneDrive</a> | <a herf="">Baidu Cloud (coming soon)</a> </td>
    </tr>
    <tr>
      <td align="left"> CSPDarkNet-coco </td>
      <td align="left"> <a href="https://1drv.ms/u/s!AmrFUyZ_lDVGinF-8LK_9tzqArs9?e=vvADN9">OneDrive</a> | <a herf="">Baidu Cloud (coming soon)</a> </td>
    </tr>
    <tr>
      <td align="left"> VGG16-coco </td>
      <td align="left"> <a href="https://1drv.ms/u/s!AmrFUyZ_lDVGinDBG42mcf3E5Rhg?e=T4qVqu">OneDrive</a> | <a herf="">Baidu Cloud (coming soon)</a> </td>
    </tr>
    <tr>
      <td align="left"> DResNet101-coco </td>
      <td align="left"> <a href="https://1drv.ms/u/s!AmrFUyZ_lDVGinK9ZJI1D-kvUWh8?e=0B2F5t">OneDrive</a> | <a herf="">Baidu Cloud (coming soon)</a> </td>
    </tr>
    </tr>
  </tbody>
</table>

We also provide the weights of **SimREC** that are pretrained on 0.2M images.

<!-- | **Pretrained Weights of REC Models** |                           **Link**                           |
| ------------------------------------ | :----------------------------------------------------------: |
| SimREC (merge)                       | [OneDrive](https://1drv.ms/u/s!AmrFUyZ_lDVGinTJlKFvD_Alg-r8?e=5woU7Y) , Baidu Cloud | -->

<table class="docutils">
  <tbody>
    <tr>
      <th align="left"> Pretrained Weights of REC Models </th>
      <th align="center" width="120">Download URL</th>
    </tr>
    <tr>
      <td align="left"> SimREC(merge) </td>
      <td align="left"> <a href="https://1drv.ms/u/s!AmrFUyZ_lDVGinTJlKFvD_Alg-r8?e=5woU7Y">OneDrive</a> | <a herf="">Baidu Cloud (coming soon)</a> </td>
    </tr>
    </tr>
  </tbody>
</table>