import glob
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

#%% Retrieve credentials following https://github.com/Kaggle/kaggle-api and upload the kaggle.json file and add them in the right directory
kaggle_config_dir = Path.home() / ".kaggle"
kaggle_config_dir.mkdir(exist_ok=True)
Path("kaggle.json").replace(kaggle_config_dir / "kaggle.json")

#%% Some setups for the notebook
np.random.seed(42)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.width", 500)
pd.set_option("display.max_rows", 100)

#%% Download data
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
api.competition_download_files("dog-breed-identification", "data")

with zipfile.ZipFile("data/dog-breed-identification.zip", "r") as zip_file:
    zip_file.extractall("data")

Path("data/dog-breed-identification.zip").unlink()

#%% Read labels
categories = [
    "beagle",
    "chihuahua",
    "doberman",
    "french_bulldog",
    "golden_retriever",
    "malamute",
    "pug",
    "saint_bernard",
    "scottish_deerhound",
    "tibetan_mastiff",
]
dataframe = (
    pd.read_csv("data/labels.csv")
    .loc[lambda df: df.breed.isin(categories)]
    .assign(
        filename=lambda df: "data/train/" + df.id + ".jpg",
        is_file=lambda df: df.filename.map(lambda filename: Path(filename).is_file()),
        label=lambda df: pd.Categorical(df.breed, categories=categories).codes,
        dataset=lambda df: np.random.choice(
            ["train", "val"], size=len(df), p=[0.8, 0.2]
        ),
    )
)

#%% Safety check on the data set
missing_files = dataframe.query("is_file == False")
if not missing_files.empty:
    raise ValueError(f"Some files are missing:\n{missing_files}")

#%% Identifying how many images are there for each breed
dataframe.breed.value_counts().plot.bar()
plt.show()

#%% Define model
model = tf.keras.models.Sequential(
    [
        tf.keras.applications.ResNet50(
            include_top=False, input_shape=(224, 224, 3), pooling="avg"
        ),
        tf.keras.layers.Dense(
            units=len(dataframe.label.unique()), activation="softmax"
        ),
    ]
)

model.summary()


#%% Define preprocessing
@tf.function
def preprocessing(filename):
    output_tensor = tf.io.decode_jpeg(tf.io.read_file(filename))
    output_tensor = tf.image.convert_image_dtype(output_tensor, tf.float32)
    output_tensor = tf.image.resize(output_tensor, size=model.input_shape[1:3])
    return output_tensor


#%% Define datasets
train_dataset, val_dataset = [
    (
        tf.data.Dataset.from_tensor_slices(
            dataframe.query(f"dataset == '{dataset}'").to_dict("list")
        )
        .map(
            lambda x: (preprocessing(x["filename"]), x["label"]),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .cache()
    )
    for dataset in ["train", "val"]
]

#%% Train the model
batch_size = 64
model.layers[0].trainable = False
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(lr=1e-5),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3),
    ],
)
model.fit(
    x=train_dataset.batch(batch_size),
    epochs=10,
    validation_data=val_dataset.batch(batch_size),
    callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir="logs"),
        tf.keras.callbacks.ReduceLROnPlateau(),
        tf.keras.callbacks.EarlyStopping(),
        tf.keras.callbacks.ModelCheckpoint(
            "logs/best_weights.hdf5", save_best_only=True
        ),
    ],
)

#%% Evaluate
model = tf.keras.models.load_model("logs/best_weights.hdf5")
test_set = pd.Index(glob.glob("data/test/*.jpg"), name="id")
test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_set.values)
    .map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
)
predictions = model.predict(test_dataset.batch(64), verbose=1)

#%% Export results
(
    pd.DataFrame(predictions, columns=categories, index=test_set)
    .reset_index()
    .assign(id=lambda df: df.id.str.extract(r"data/test/(?P<id>.*)\.jpg").id)
    .to_csv("logs/submission.csv", index=False)
)
