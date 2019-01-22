
# 

# <img src="https://raw.githubusercontent.com/GokuMohandas/practicalAI/master/images/cnn_cv.png" width=650>

# # Configuration
config = {
  "seed": 1234,
  "cuda": True,
  "data_url": "https://raw.githubusercontent.com/GokuMohandas/practicalAI/master/data/surnames.csv",
  "data_dir": "cifar10",
  "shuffle": True,
  "train_size": 0.7,
  "val_size": 0.15,
  "test_size": 0.15,
  "vectorizer_file": "vectorizer.json",
  "model_file": "model.pth",
  "save_dir": "experiments",
  "num_epochs": 5,
  "early_stopping_criteria": 5,
  "learning_rate": 1e-3,
  "batch_size": 128,
  "fc": {
    "hidden_dim": 100,
    "dropout_p": 0.1
  }
}

# # Set up

# Load PyTorch library

import os
import json
import numpy as np
import time
import torch
import uuid

# ### Components
def set_seeds(seed, cuda):
    """ Set Numpy and PyTorch seeds.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    print ("==> 🌱 Set NumPy and PyTorch seeds.")
def generate_unique_id():
    """Generate a unique uuid
    preceded by a epochtime.
    """
    timestamp = int(time.time())
    unique_id = "{}_{}".format(timestamp, uuid.uuid1())
    print ("==> 🔑 Generated unique id: {0}".format(unique_id))
    return unique_id
def create_dirs(dirpath):
    """Creating directories.
    """
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        print ("==> 📂 Created {0}".format(dirpath))
def check_cuda(cuda):
    """Check to see if GPU is available.
    """
    if not torch.cuda.is_available():
        cuda = False
    device = torch.device("cuda" if cuda else "cpu")
    print ("==> 💻 Device: {0}".format(device))
    return device

# ### Operations

# Set seeds for reproducability
set_seeds(seed=config["seed"], cuda=config["cuda"])

# Generate unique experiment ID
config["experiment_id"] = generate_unique_id()

# Create experiment directory
config["save_dir"] = os.path.join(config["save_dir"], config["experiment_id"])
create_dirs(dirpath=config["save_dir"])

# Expand file paths to store components later
config["vectorizer_file"] = os.path.join(config["save_dir"], config["vectorizer_file"])
config["model_file"] = os.path.join(config["save_dir"], config["model_file"])
print ("Expanded filepaths: ")
print ("{}".format(config["vectorizer_file"]))
print ("{}".format(config["model_file"]))

# Save config
config_fp = os.path.join(config["save_dir"], "config.json")
with open(config_fp, "w") as fp:
    json.dump(config, fp)

# Check CUDA
config["device"] = check_cuda(cuda=config["cuda"])

# # Load data

# We are going to get CIFAR10 data which contains images from ten unique classes. Each image has length 32, width 32 and three color channels (RGB). We are going to save these images in a directory. Each image will have its own directory (name will be the class).
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import tensorflow as tf

# ### Components
def get_data():
    """Get CIFAR10 data.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X = np.vstack([x_train, x_test])
    y = np.vstack([y_train, y_test]).squeeze(1)
    print ("==> 🌊 Downloading Cifar10 data using TensorFlow.")
    return X, y
def create_class_dirs(data_dir, classes):
    """Create class directories.
    """
    create_dirs(dirpath=data_dir)
    for _class in classes.values():
        classpath = os.path.join(data_dir, _class)
        create_dirs(dirpath=classpath)
def visualize_samples(data_dir, classes):
    """Visualize sample images for
    each class.
    """
    # Visualize some samples
    num_samples = len(classes)
    for i, _class in enumerate(classes.values()):  
        for file in os.listdir(os.path.join(data_dir, _class)):
            if file.endswith((".png", ".jpg", ".jpeg")):
                plt.subplot(1, num_samples, i+1)
                plt.title("{0}".format(_class))
                img = Image.open(os.path.join(data_dir, _class, file))
                plt.imshow(img)
                plt.axis("off")
                break
def img_to_array(fp):
    """Conver image file to NumPy array.
    """
    img = Image.open(fp)
    array = np.asarray(img, dtype="float32")
    return array
def load_data(data_dir, classes):
    """Load data into Pandas DataFrame.
    """
    # Load data from files
    data = []
    for i, _class in enumerate(classes.values()):  
        for file in os.listdir(os.path.join(data_dir, _class)):
            if file.endswith((".png", ".jpg", ".jpeg")):
                full_filepath = os.path.join(data_dir, _class, file)
                data.append({"image": img_to_array(full_filepath), "category": _class})
                
    # Load to Pandas DataFrame
    df = pd.DataFrame(data)
    print ("==> 🖼️ Image dimensions: {0}".format(df.image[0].shape))
    print ("==> 🍣 Raw data:")
    print (df.head())
    return df

# ### Operations

# Get CIFAR10 data
X, y = get_data()
print ("X:", X.shape)
print ("y:", y.shape)

# Classes
classes = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 
           6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

# Create image directories
create_class_dirs(data_dir=config["data_dir"], classes=classes)

# Save images for each class
for i, (image, label) in enumerate(zip(X, y)):
    _class = classes[label]
    im = Image.fromarray(image)
    im.save(os.path.join(config["data_dir"], _class, "{0:02d}.png".format(i)))

# Visualize each class
visualize_samples(data_dir=config["data_dir"], classes=classes)

# Load data into DataFrame
df = load_data(data_dir=config["data_dir"], classes=classes)

# # Split data

# Split the data into train, validation and test sets where each split has similar class distributions.
import collections

# ### Components
def split_data(df, shuffle, train_size, val_size, test_size):
    """Split the data into train/val/test splits.
    """
    # Split by category
    by_category = collections.defaultdict(list)
    for _, row in df.iterrows():
        by_category[row.category].append(row.to_dict())
    print ("\n==> 🛍️ Categories:")
    for category in by_category:
        print ("{0}: {1}".format(category, len(by_category[category])))
    # Create split data
    final_list = []
    for _, item_list in sorted(by_category.items()):
        if shuffle:
            np.random.shuffle(item_list)
        n = len(item_list)
        n_train = int(train_size*n)
        n_val = int(val_size*n)
        n_test = int(test_size*n)
      # Give data point a split attribute
        for item in item_list[:n_train]:
            item['split'] = 'train'
        for item in item_list[n_train:n_train+n_val]:
            item['split'] = 'val'
        for item in item_list[n_train+n_val:]:
            item['split'] = 'test'
        # Add to final list
        final_list.extend(item_list)
    # df with split datasets
    split_df = pd.DataFrame(final_list)
    print ("\n==> 🖖 Splits:")
    print (split_df["split"].value_counts())
    return split_df

# ### Operations

# Split data
split_df = split_data(
    df=df, shuffle=config["shuffle"],
    train_size=config["train_size"],
    val_size=config["val_size"],
    test_size=config["test_size"])

# # Vocabulary

# Create vocabularies for the image classes.

# ### Components
class Vocabulary(object):
    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        # Token to index
        if token_to_idx is None:
            token_to_idx = {}
        self.token_to_idx = token_to_idx
        # Index to token
        self.idx_to_token = {idx: token                              for token, idx in self.token_to_idx.items()}
        
        # Add unknown token
        self.add_unk = add_unk
        self.unk_token = unk_token
        if self.add_unk:
            self.unk_index = self.add_token(self.unk_token)
    def to_serializable(self):
        return {'token_to_idx': self.token_to_idx,
                'add_unk': self.add_unk, 'unk_token': self.unk_token}
    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)
    def add_token(self, token):
        if token in self.token_to_idx:
            index = self.token_to_idx[token]
        else:
            index = len(self.token_to_idx)
            self.token_to_idx[token] = index
            self.idx_to_token[index] = token
        return index
    def add_tokens(self, tokens):
        return [self.add_token[token] for token in tokens]
    def lookup_token(self, token):
        if self.add_unk:
            index = self.token_to_idx.get(token, self.unk_index)
        else:
            index =  self.token_to_idx[token]
        return index
    def lookup_index(self, index):
        if index not in self.idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self.idx_to_token[index]
    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)
    def __len__(self):
        return len(self.token_to_idx)

# ### Operations

# Vocabulary instance
category_vocab = Vocabulary(add_unk=False)
for index, row in df.iterrows():
    category_vocab.add_token(row.category)
print (category_vocab) # __str__
print (len(category_vocab)) # __len__
index = category_vocab.lookup_token("bird")
print (index)
print (category_vocab.lookup_index(index))

# # Sequence vocbulary

# We will also create a vocabulary object for the actual images. It will store the mean and standard deviations for eahc image channel (RGB) which we will use later on for normalizing our images with the Vectorizer.
from collections import Counter
import string

# ### Components
class SequenceVocabulary(Vocabulary):
    def __init__(self, train_means, train_stds):
        
        self.train_means = train_means
        self.train_stds = train_stds
        
    def to_serializable(self):
        contents = {'train_means': self.train_means,
                    'train_stds': self.train_stds}
        return contents
    
    @classmethod
    def from_dataframe(cls, df):
        train_data = df[df.split == "train"]
        means = {0:[], 1:[], 2:[]}
        stds = {0:[], 1:[], 2:[]}
        for image in train_data.image:
            for dim in range(3):
                means[dim].append(np.mean(image[:, :, dim]))
                stds[dim].append(np.std(image[:, :, dim]))
        train_means = np.array((np.mean(means[0]), np.mean(means[1]), 
                                np.mean(means[2])), dtype="float64").tolist()
        train_stds = np.array((np.mean(stds[0]), np.mean(stds[1]), 
                               np.mean(stds[2])), dtype="float64").tolist()
            
        return cls(train_means, train_stds)
        
    def __str__(self):
        return "<SequenceVocabulary(train_means: {0}, train_stds: {1}>".format(
            self.train_means, self.train_stds)

# ### Operations

# Create SequenceVocabulary instance
image_vocab = SequenceVocabulary.from_dataframe(split_df)
print (image_vocab) # __str__

# # Vectorizer

# The vectorizer will normalize our images using the vocabulary.

# ### Components
class ImageVectorizer(object):
    def __init__(self, image_vocab, category_vocab):
        self.image_vocab = image_vocab
        self.category_vocab = category_vocab
    def vectorize(self, image):
        # Avoid modifying the actual df
        image = np.copy(image)
        
        # Normalize
        for dim in range(3):
            mean = self.image_vocab.train_means[dim]
            std = self.image_vocab.train_stds[dim]
            image[:, :, dim] = ((image[:, :, dim] - mean) / std)
            
        # Reshape from (32, 32, 3) to (3, 32, 32)
        image = np.swapaxes(image, 0, 2)
        image = np.swapaxes(image, 1, 2)
                
        return image
    
    @classmethod
    def from_dataframe(cls, df):
        # Create vocabularies
        image_vocab = SequenceVocabulary.from_dataframe(df)
        category_vocab = Vocabulary(add_unk=False)   
        for category in sorted(set(df.category)):
            category_vocab.add_token(category)
        return cls(image_vocab, category_vocab)
    @classmethod
    def from_serializable(cls, contents):
        image_vocab = SequenceVocabulary.from_serializable(contents['image_vocab'])
        category_vocab = Vocabulary.from_serializable(contents['category_vocab'])
        return cls(image_vocab=image_vocab, 
                   category_vocab=category_vocab)
    
    def to_serializable(self):
        return {'image_vocab': self.image_vocab.to_serializable(),
                'category_vocab': self.category_vocab.to_serializable()}

# ### Operations

# Vectorizer instance
vectorizer = ImageVectorizer.from_dataframe(split_df)
print (vectorizer.image_vocab)
print (vectorizer.category_vocab)
print (vectorizer.category_vocab.token_to_idx)
image_vector = vectorizer.vectorize(split_df.iloc[0].image)
print (image_vector.shape)

# # Dataset

# The Dataset will create vectorized data from the data.
import random
from torch.utils.data import Dataset, DataLoader

# ### Components
class ImageDataset(Dataset):
    def __init__(self, df, vectorizer, infer=False):
        self.df = df
        self.vectorizer = vectorizer
        
        # Data splits
        if not infer:
            self.train_df = self.df[self.df.split=='train']
            self.train_size = len(self.train_df)
            self.val_df = self.df[self.df.split=='val']
            self.val_size = len(self.val_df)
            self.test_df = self.df[self.df.split=='test']
            self.test_size = len(self.test_df)
            self.lookup_dict = {'train': (self.train_df, self.train_size), 
                                'val': (self.val_df, self.val_size),
                                'test': (self.test_df, self.test_size)}
            self.set_split('train')
            # Class weights (for imbalances)
            class_counts = df.category.value_counts().to_dict()
            def sort_key(item):
                return self.vectorizer.category_vocab.lookup_token(item[0])
            sorted_counts = sorted(class_counts.items(), key=sort_key)
            frequencies = [count for _, count in sorted_counts]
            self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)
        elif infer:
            self.infer_df = self.df[self.df.split=="infer"]
            self.infer_size = len(self.infer_df)
            self.lookup_dict = {'infer': (self.infer_df, self.infer_size)}
            self.set_split('infer')
    @classmethod
    def load_dataset_and_make_vectorizer(cls, df):
        train_df = df[df.split=='train']
        return cls(df, ImageVectorizer.from_dataframe(train_df))
    @classmethod
    def load_dataset_and_load_vectorizer(cls, df, vectorizer_filepath):
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(df, vectorizer)
    def load_vectorizer_only(vectorizer_filepath):
        with open(vectorizer_filepath) as fp:
            return ImageVectorizer.from_serializable(json.load(fp))
    def save_vectorizer(self, vectorizer_filepath):
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self.vectorizer.to_serializable(), fp)
    def set_split(self, split="train"):
        self.target_split = split
        self.target_df, self.target_size = self.lookup_dict[split]
    def __str__(self):
        return "<Dataset(split={0}, size={1})".format(
            self.target_split, self.target_size)
    def __len__(self):
        return self.target_size
    def __getitem__(self, index):
        row = self.target_df.iloc[index]
        image_vector = self.vectorizer.vectorize(row.image)
        category_index = self.vectorizer.category_vocab.lookup_token(row.category)
        return {'image': image_vector, 
                'category': category_index}
    def get_num_batches(self, batch_size):
        return len(self) // batch_size
    def generate_batches(self, batch_size, shuffle=True, drop_last=True, device="cpu"):
        dataloader = DataLoader(dataset=self, batch_size=batch_size, 
                                shuffle=shuffle, drop_last=drop_last)
        for data_dict in dataloader:
            out_data_dict = {}
            for name, tensor in data_dict.items():
                out_data_dict[name] = data_dict[name].to(device)
            yield out_data_dict
def sample(dataset):
    """Some sanity checks on the dataset.
    """
    sample_idx = random.randint(0,len(dataset))
    sample = dataset[sample_idx]
    print ("\n==> 🔢 Dataset:")
    print ("Random sample: {0}".format(sample))
    print ("Unvectorized category: {0}".format(
        dataset.vectorizer.category_vocab.lookup_index(sample['category'])))

# ### Operations

# Load dataset and vectorizer
dataset = ImageDataset.load_dataset_and_make_vectorizer(split_df)
dataset.save_vectorizer(config["vectorizer_file"])
vectorizer = dataset.vectorizer
print (dataset.class_weights)

# Sample checks
sample(dataset=dataset)

# # Model

# Basic CNN architecture for image classification.
import torch.nn as nn
import torch.nn.functional as F

# ### Components
class ImageModel(nn.Module):
    def __init__(self, num_hidden_units, num_classes, dropout_p):
        super(ImageModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5) # input_channels:3, output_channels:10 (aka num filters)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) 
        self.conv_dropout = nn.Dropout2d(dropout_p)
        self.fc1 = nn.Linear(20*5*5, num_hidden_units)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(num_hidden_units, num_classes)
    def forward(self, x, apply_softmax=False):
        # Conv pool
        z = self.conv1(x) # (N, 10, 28, 28)
        z = F.max_pool2d(z, 2) # (N, 10, 14, 14)
        z = F.relu(z)
        
        # Conv pool
        z = self.conv2(z) # (N, 20, 10, 10)
        z = self.conv_dropout(z) 
        z = F.max_pool2d(z, 2) # (N, 20, 5, 5)
        z = F.relu(z)
        
        # Flatten
        z = z.view(-1, 20*5*5)
        
        # FC
        z = F.relu(self.fc1(z))
        z = self.dropout(z)
        y_pred = self.fc2(z)
        
        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        return y_pred 
def initialize_model(config, vectorizer):
    """Initialize the model.
    """
    print ("\n==> 🚀 Initializing model:")
    model = ImageModel(
       num_hidden_units=config["fc"]["hidden_dim"], 
       num_classes=len(vectorizer.category_vocab),
       dropout_p=config["fc"]["dropout_p"])
    print (model.named_modules)
    return model

# ### Operations

# Initializing model
model = initialize_model(config=config, vectorizer=vectorizer)

# # Training

# Training operations for image classification.
import torch.optim as optim

# ### Components
def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100
def update_train_state(model, train_state):
    """ Update train state during training.
    """
    # Verbose
    print ("[EPOCH]: {0} | [LR]: {1} | [TRAIN LOSS]: {2:.2f} | [TRAIN ACC]: {3:.1f}% | [VAL LOSS]: {4:.2f} | [VAL ACC]: {5:.1f}%".format(
      train_state['epoch_index'], train_state['learning_rate'], 
        train_state['train_loss'][-1], train_state['train_acc'][-1], 
        train_state['val_loss'][-1], train_state['val_acc'][-1]))
    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False
    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state['val_loss'][-2:]
        # If loss worsened
        if loss_t >= train_state['early_stopping_best_val']:
            # Update step
            train_state['early_stopping_step'] += 1
        # Loss decreased
        else:
            # Save the best model
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])
            # Reset early stopping step
            train_state['early_stopping_step'] = 0
        # Stop early ?
        train_state['stop_early'] = train_state['early_stopping_step']           >= train_state['early_stopping_criteria']
    return train_state
class Trainer(object):
    def __init__(self, dataset, model, model_file, device, shuffle, 
               num_epochs, batch_size, learning_rate, early_stopping_criteria):
        self.dataset = dataset
        self.class_weights = dataset.class_weights.to(device)
        self.model = model.to(device)
        self.device = device
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.loss_func = nn.CrossEntropyLoss(self.class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer, mode='min', factor=0.5, patience=1)
        self.train_state = {
            'done_training': False,
            'stop_early': False, 
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'early_stopping_criteria': early_stopping_criteria,
            'learning_rate': learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': model_file}
  
    def run_train_loop(self):
        print ("==> 🏋 Training:")
        for epoch_index in range(self.num_epochs):
            self.train_state['epoch_index'] = epoch_index
      
            # Iterate over train dataset
            # initialize batch generator, set loss and acc to 0, set train mode on
            self.dataset.set_split('train')
            batch_generator = self.dataset.generate_batches(
                batch_size=self.batch_size, shuffle=self.shuffle, 
                device=self.device)
            running_loss = 0.0
            running_acc = 0.0
            self.model.train()
            for batch_index, batch_dict in enumerate(batch_generator):
                # zero the gradients
                self.optimizer.zero_grad()
                # compute the output
                y_pred = self.model(batch_dict['image'])
                # compute the loss
                loss = self.loss_func(y_pred, batch_dict['category'])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)
                # compute gradients using loss
                loss.backward()
                # use optimizer to take a gradient step
                self.optimizer.step()
                
                # compute the accuracy
                acc_t = compute_accuracy(y_pred, batch_dict['category'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)
            self.train_state['train_loss'].append(running_loss)
            self.train_state['train_acc'].append(running_acc)
            # Iterate over val dataset
            # initialize batch generator, set loss and acc to 0; set eval mode on
            self.dataset.set_split('val')
            batch_generator = self.dataset.generate_batches(
                batch_size=self.batch_size, shuffle=self.shuffle, device=self.device)
            running_loss = 0.
            running_acc = 0.
            self.model.eval()
            for batch_index, batch_dict in enumerate(batch_generator):
                # compute the output
                y_pred =  self.model(batch_dict['image'])
                # compute the loss
                loss = self.loss_func(y_pred, batch_dict['category'])
                loss_t = loss.to("cpu").item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)
                # compute the accuracy
                acc_t = compute_accuracy(y_pred, batch_dict['category'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)
            self.train_state['val_loss'].append(running_loss)
            self.train_state['val_acc'].append(running_acc)
            self.train_state = update_train_state(model=self.model, train_state=self.train_state)
            self.scheduler.step(self.train_state['val_loss'][-1])
            if self.train_state['stop_early']:
                break
          
    def run_test_loop(self):
        # initialize batch generator, set loss and acc to 0; set eval mode on
        self.dataset.set_split('test')
        batch_generator = self.dataset.generate_batches(
            batch_size=self.batch_size, shuffle=self.shuffle, device=self.device)
        running_loss = 0.0
        running_acc = 0.0
        self.model.eval()
        for batch_index, batch_dict in enumerate(batch_generator):
            # compute the output
            y_pred =  self.model(batch_dict['image'])
            # compute the loss
            loss = self.loss_func(y_pred, batch_dict['category'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            # compute the accuracy
            acc_t = compute_accuracy(y_pred, batch_dict['category'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)
        self.train_state['test_loss'] = running_loss
        self.train_state['test_acc'] = running_acc
        
        # Verbose
        print ("==> 💯 Test performance:")
        print ("Test loss: {0:.2f}".format(self.train_state['test_loss']))
        print ("Test Accuracy: {0:.1f}%".format(self.train_state['test_acc']))
def plot_performance(train_state, save_dir, show_plot=True):
    """ Plot loss and accuracy.
    """
    # Figure size
    plt.figure(figsize=(15,5))
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.plot(train_state["train_loss"], label="train")
    plt.plot(train_state["val_loss"], label="val")
    plt.legend(loc='upper right')
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plt.plot(train_state["train_acc"], label="train")
    plt.plot(train_state["val_acc"], label="val")
    plt.legend(loc='lower right')
    # Save figure
    plt.savefig(os.path.join(save_dir, "performance.png"))
    # Show plots
    if show_plot:
        print ("==> 📈 Metric plots:")
        plt.show()
def save_train_state(train_state, save_dir):
    train_state["done_training"] = True
    with open(os.path.join(save_dir, "train_state.json"), "w") as fp:
        json.dump(train_state, fp)
    print ("==> ✅ Training complete!")

# ### Operations

# Training
trainer = Trainer(
    dataset=dataset, model=model, model_file=config["model_file"],
    device=config["device"], shuffle=config["shuffle"], 
    num_epochs=config["num_epochs"], batch_size=config["batch_size"], 
    learning_rate=config["learning_rate"], 
    early_stopping_criteria=config["early_stopping_criteria"])
trainer.run_train_loop()

# Plot performance
plot_performance(train_state=trainer.train_state, 
                 save_dir=config["save_dir"], show_plot=True)

# Test performance
trainer.run_test_loop()

# Save all results
save_train_state(train_state=trainer.train_state, save_dir=config["save_dir"])

# ~60% test performance for our CIFAR10 dataset is not bad but we can do way better.

# # Transfer learning

# In this section, we're going to use a pretrained model that performs very well on a different dataset. We're going to take the architecture and the initial convolutional weights from the model to use on our data. We will freeze the initial convolutional weights and fine tune the later convolutional and fully-connected layers. 

# 

# Transfer learning works here because the initial convolution layers act as excellent feature extractors for common spatial features that are shared across images regardless of their class. We're going to leverage these large, pretrained models' feature extractors for our own dataset.

# get_ipython().system('pip install torchvision')
from torchvision import models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
print (model_names)
model_name = 'vgg19_bn'
vgg_19bn = models.__dict__[model_name](pretrained=True) # Set false to train from scratch
print (vgg_19bn.named_parameters)

# The VGG model we chose has a `features` and a `classifier` component. The `features` component is composed of convolution and pooling layers which act as feature extractors. The `classifier` component is composed on fully connected layers. We're going to freeze most of the `feature` component and design our own FC layers for our CIFAR10 task. You can access the default code for all models at `/usr/local/lib/python3.6/dist-packages/torchvision/models` if you prefer cloning and modifying that instead.

# ### Components
class ImageModel(nn.Module):
    def __init__(self, feature_extractor, num_hidden_units, 
                 num_classes, dropout_p):
        super(ImageModel, self).__init__()
        
        # Pretrained feature extractor
        self.feature_extractor = feature_extractor
        
        # FC weights
        self.classifier = nn.Sequential(
            nn.Linear(512, 250, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(250, 100, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 10, bias=True),
            )
    def forward(self, x, apply_softmax=False):
          
        # Feature extractor
        z = self.feature_extractor(x)
        z = z.view(x.size(0), -1)
        
        # FC
        y_pred = self.classifier(z)
        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        return y_pred 
def initialize_model(config, vectorizer, feature_extractor):
    """Initialize the model.
    """
    print ("\n==> 🚀 Initializing model:")
    model = ImageModel(
       feature_extractor=feature_extractor,
       num_hidden_units=config["fc"]["hidden_dim"], 
       num_classes=len(vectorizer.category_vocab),
       dropout_p=config["fc"]["dropout_p"])
    print (model.named_modules)
    return model

# ### Operations

# Initializing model
model = initialize_model(config=config, vectorizer=vectorizer, 
                         feature_extractor=vgg_19bn.features)

# Finetune last few conv layers and FC layers
for i, param in enumerate(model.feature_extractor.parameters()):
    if i < 36:
        param.requires_grad = False
    else:
        param.requires_grad = True

# Training
trainer = Trainer(
    dataset=dataset, model=model, model_file=config["model_file"],
    device=config["device"], shuffle=config["shuffle"], 
    num_epochs=config["num_epochs"], batch_size=config["batch_size"], 
    learning_rate=config["learning_rate"], 
    early_stopping_criteria=config["early_stopping_criteria"])
trainer.run_train_loop()

# Plot performance
plot_performance(train_state=trainer.train_state, 
                 save_dir=config["save_dir"], show_plot=True)

# Test performance
trainer.run_test_loop()

# Save all results
save_train_state(train_state=trainer.train_state, save_dir=config["save_dir"])

# Much better performance! If you let it train long enough, we'll actually reach ~95% accuracy :)

# ## Inference
from pylab import rcParams
rcParams['figure.figsize'] = 2, 2

# ### Components
class Inference(object):
    def __init__(self, model, vectorizer, device="cpu"):
        self.model = model.to(device)
        self.vectorizer = vectorizer
        self.device = device
  
    def predict_category(self, dataset):
        # Batch generator
        batch_generator = dataset.generate_batches(
            batch_size=len(dataset), shuffle=False, device=self.device)
        self.model.eval()
        
        # Predict
        for batch_index, batch_dict in enumerate(batch_generator):
            # compute the output
            y_pred =  self.model(batch_dict['image'], apply_softmax=True)
            # Top k categories
            y_prob, indices = torch.topk(y_pred, k=len(self.vectorizer.category_vocab))
            probabilities = y_prob.detach().to('cpu').numpy()[0]
            indices = indices.detach().to('cpu').numpy()[0]
            results = []
            for probability, index in zip(probabilities, indices):
                category = self.vectorizer.category_vocab.lookup_index(index)
                results.append({'category': category, 'probability': probability})
        return results

# ### Operations

# Load vectorizer
with open(config["vectorizer_file"]) as fp:
    vectorizer = ImageVectorizer.from_serializable(json.load(fp))

# Load the model
model = initialize_model(config=config, vectorizer=vectorizer, feature_extractor=vgg_19bn.features)
model.load_state_dict(torch.load(config["model_file"]))

# Initialize
inference = Inference(model=model, vectorizer=vectorizer, device=config["device"])

# Get a sample
sample = split_df[split_df.split=="test"].iloc[0]
plt.imshow(sample.image)
plt.axis("off")
print ("Actual:", sample.category)

# Inference
category = list(vectorizer.category_vocab.token_to_idx.keys())[0] # random filler category
infer_df = pd.DataFrame([[sample.image, category, "infer"]], columns=['image', 'category', 'split'])
infer_dataset = ImageDataset(df=infer_df, vectorizer=vectorizer, infer=True)
results = inference.predict_category(dataset=infer_dataset)
results

# # TODO

# - segmentation

# - interpretability via activation maps

# - processing images of different sizes
