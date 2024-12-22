# Turkish-Sentiment-Classification



### <span style="color:#3D9F03"> Create Enviroment

Type the conda command below for creating enviroment.

```bash
conda create --name turkish-sentiment python==3.8
```
After that press 'y' for continue creating enviroment.
When finished download some python packages type the command below for activate enviroment.

```bash
conda activate turkish-sentiment
```
Upgrade setuptools and pip.

```bash
pip install --upgrade pip setuptools
```
### <span style="color:#3D9F03"> Install Requirements

```bash
pip install -r requirements.txt
```

## <span style="color:#A5EF00">  Usage  </span>
### <span style="color:#3D9F03"> Run for training</span>
```bash
python app.py --env local --train
```

### <span style="color:#3D9F03"> Run for inference</span>
```bash
python app.py --env local --infer
```

#### <span style="color:#3D9F03"> Train Classification</span>
```bash
python app.py --env local --train --classification
```

#### <span style="color:#3D9F03"> Inference YOLO</span>
```bash
python app.py --env local --infer --yolo
```
#### <span style="color:#3D9F03"> Inference Classification</span>
```bash
python app.py --env local --infer --classification
```

#### <span style="color:#3D9F03"> Model Architecture YOLO in Detail</span>
```bash
python app.py --env local --yolo --summary
```
#### <span style="color:#3D9F03"> Model Architecture Classification in Detail</span>
```bash
python app.py --env local --classification --summary
```

#### <span style="color:#3D9F03"> Board To Visualize </span>
```bash
tensorboard --logdir=runs --load_fast=false
```

#### <span style="color:#3D9F03"> Docker-Compose (not active)</span>
##### <span style="color:#B8DD21"> Debug </span>
```bash
docker-compose up --build
```
##### <span style="color:#B8DD21"> Prod </span>
```bash
docker-compose up --build --detach
```
