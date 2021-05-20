# wai-annotations-tf
wai.annotations module for managing TFRecords datasets.

## Plugins
### FROM-TF-OD
Reads image object-detection annotations in the TFRecords binary format

#### Domain(s):
- **Image Object-Detection Domain**

#### Options:
```
usage: from-tf-od [-I FILENAME] [-i FILENAME] [-N FILENAME] [-n FILENAME] [--seed SEED] [--mask-threshold THRESHOLD] [--sample-stride STRIDE]

optional arguments:
  -I FILENAME, --inputs-file FILENAME
                        Files containing lists of input files (can use glob syntax)
  -i FILENAME, --input FILENAME
                        Input files (can use glob syntax)
  -N FILENAME, --negatives-file FILENAME
                        Files containing lists of negative files (can use glob syntax)
  -n FILENAME, --negative FILENAME
                        Files that have no annotations (can use glob syntax)
  --seed SEED           the seed to use for randomisation
  --mask-threshold THRESHOLD
                        the threshold to use when calculating polygons from masks
  --sample-stride STRIDE
                        the stride to use when calculating polygons from masks
```

### TO-TF-OD
Writes image object-detection annotations in the TFRecords binary format

#### Domain(s):
- **Image Object-Detection Domain**

#### Options:
```
usage: to-tf-od [--dense] -o PATH [-p FILENAME] [-s FILENAME [FILENAME ...]] [--split-names SPLIT NAME [SPLIT NAME ...]] [--split-ratios RATIO [RATIO ...]]

optional arguments:
  --dense               outputs masks in the dense numerical format instead of PNG-encoded
  -o PATH, --output PATH
                        name of output file for TFRecords
  -p FILENAME, --protobuf FILENAME
                        for storing the label strings and IDs
  -s FILENAME [FILENAME ...], --shards FILENAME [FILENAME ...]
                        additional shards to write to
  --split-names SPLIT NAME [SPLIT NAME ...]
                        the names to use for the splits
  --split-ratios RATIO [RATIO ...]
                        the ratios to use for the splits
```
