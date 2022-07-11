### Setup
* Install python3 requirements: `pip3 install -r requirements.txt`
* Initialize GloVe as follows:
```bash
$ wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.3.0/en_vectors_web_lg-2.3.0.tar.gz -O en_vectors_web_lg-2.3.0.tar.gz
$ pip install en_vectors_web_lg-2.3.0.tar.gz
```

For overfitting a single batch, set the `--DEBUG` flag to `True`

Other important parameters:
1. `--RESUME`, start training with saved checkpoint. You should assign checkpoint version `--VERSION int` and resumed epoch `--CKPT_E int`.
2. `--VERSION`, to assign seed and version of the model.
3. `--CPU`, to force training and/or inference using CPU.


### Running the trained model

To run the trained model (version 1) run the following command

```bash
$ python run.py --RUN_MODE test --VERSION 1 --CKPT_E 30
```