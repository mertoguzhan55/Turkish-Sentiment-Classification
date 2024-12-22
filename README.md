# Turkish-Sentiment-Classification

This repo is created as a base repo for NLP. Many downstream tasks will be added but now, with this repo you can classify sentences(angry, sad, happy, surprised) with the dataset you provide.

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

#### <span style="color:#3D9F03"> Docker-Compose (will be added)</span>
##### <span style="color:#B8DD21"> Debug (will be added)</span>

