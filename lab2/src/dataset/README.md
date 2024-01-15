unzip the source_code file and move it under this directory.


##### download source code
```
git clone https://github.com/TheAlgorithms/Python
git clone https://github.com/TheAlgorithms/Java
git clone https://github.com/TheAlgorithms/C-Plus-Plus
```

##### Move the dowloaded two folders into here this `dataset/` directory and then run

```
python convert.py --segment_len 256 --stride 10 --dev_size 0.1
```

You will find a train set named train.jsonl and dev set named dev.jsonl under `source_code/json/`.

