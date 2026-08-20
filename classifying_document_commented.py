
# ============================================================
# TEXT CLASSIFICATION WITH AG_NEWS — FULLY COMMENTED VERSION
# ============================================================
# This is the "Classifying Document" notebook, annotated with
# extra explanations discussed in our conversation, especially
# around tokenization choices and the yield_tokens generator.
# ============================================================

from tqdm import tqdm
import numpy as np
import pandas as pd
from itertools import accumulate
import matplotlib.pyplot as plt
from torchtext.data.utils import get_tokenizer

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchtext.datasets import AG_NEWS
from IPython.display import Markdown as md
from tqdm import tqdm

from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from sklearn.manifold import TSNE
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split

# Suppress warnings for cleaner notebook output (cosmetic only)
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')


# ------------------------------------------------------------
# HELPER FUNCTION: plot cost and accuracy per epoch
# ------------------------------------------------------------
def plot(COST, ACC):
    """
    Plots training loss (COST) and validation accuracy (ACC) on
    the same figure using two different y-axes.

    Why this matters:
    - Loss almost always keeps dropping the longer you train.
    - Accuracy on validation data rises, then plateaus, then can
      start to *drop* once the model starts overfitting.
    - The "best epoch" is where validation accuracy peaks —
      not necessarily the last epoch, and not the epoch with the
      lowest loss.
    """
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(COST, color=color)
    ax1.set_xlabel('epoch', color=color)
    ax1.set_ylabel('total loss', color=color)
    ax1.tick_params(axis='y', color=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color=color)
    ax2.plot(ACC, color=color)
    ax2.tick_params(axis='y', color=color)
    fig.tight_layout()

    plt.show()


# ------------------------------------------------------------
# DATASET: AG_NEWS
# ------------------------------------------------------------
# AG_NEWS is a classic benchmark dataset for text classification,
# similar in spirit to MNIST for images. It contains ~120,000
# training articles and ~7,600 test articles, each labeled with
# one of four categories:
#   1 = World, 2 = Sports, 3 = Business, 4 = Sci/Tech
#
# The train/test split is FIXED by the dataset creators, not
# random. This is standard for benchmark datasets — it ensures
# everyone testing a model evaluates on the exact same held-out
# examples, making results comparable across papers/experiments.
train_iter = iter(AG_NEWS(split="train"))

# AG_NEWS is an ITERABLE dataset, not a list/tuple — you can't
# index into it directly (train_iter[0] won't work). You must
# pull examples one at a time with next(). This streaming style
# is common for large text datasets since it avoids loading
# everything into memory at once.
y, text = next(train_iter)
print(y, text)

# Map numeric label -> human-readable category name
ag_news_label = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tec"}
ag_news_label[y]

# Count how many distinct classes exist in the dataset (should be 4).
# This number directly determines the size of the model's final
# output layer — one score per possible class.
num_class = len(set([label for (label, text) in train_iter]))
num_class


# ------------------------------------------------------------
# TOKENIZATION AND VOCABULARY
# ------------------------------------------------------------
# Reinitialize the iterator since we already consumed some of it above.
train_iter = AG_NEWS(split="train")

# --- Tokenizer choice ---
# get_tokenizer("basic_english") is a lightweight, RULE-BASED
# tokenizer (not a machine learning model). It lowercases text
# and splits on whitespace/punctuation, handling simple cases
# like contractions.
#
# WHY "basic_english" HERE:
# - It's fast, has no external dependencies, and is "good enough"
#   for clean, formal text like news articles.
# - This model is a bag-of-embeddings model (EmbeddingBag), which
#   doesn't care about word order — so a simple, cheap tokenizer
#   is a totally reasonable choice.
#
# OTHER TOKENIZER OPTIONS (also pluggable via get_tokenizer):
# - "spacy"  -> proper linguistic tokenization, handles grammar,
#               hyphenated words, abbreviations more accurately.
#               Good choice if you need higher-quality splitting.
# - "moses"  -> rule-based tokenizer common in machine translation.
# - a custom function -> full control over splitting logic.
#
# WHAT IF YOUR DATA IS SOCIAL MEDIA / TWITTER TEXT?
# basic_english would actually HURT you here. It doesn't understand:
#   - hashtags (#something)
#   - @mentions
#   - emojis
#   - slang / abbreviations / elongated words ("sooooo good")
# It will chop these into meaningless fragments.
# For noisy, informal text like tweets, you'd want a tokenizer
# built for that domain, e.g. NLTK's TweetTokenizer, or spaCy
# with custom rules added for hashtags/mentions/emojis.
#
# GENERAL RULE:
# The tokenizer should match the "texture" of your data.
#   - Clean, formal text (news, articles, reports) -> basic_english is fine.
#   - Noisy, informal text (tweets, chat, reviews) -> need a specialized
#     tokenizer that understands that noise.
#   - Multiple languages -> spaCy (with the right language model) or
#     a multilingual tokenizer, not basic_english.
tokenizer = get_tokenizer("basic_english")


def yield_tokens(data_iter):
    """
    Generator function that tokenizes each text sample in the
    dataset, one at a time.

    WHY 'yield' INSTEAD OF 'return'?
    - 'return' would compute the ENTIRE result (all tokenized
      articles) up front, store it all in memory, and hand it
      back in one go. With ~120,000 articles, this can use a lot
      of memory unnecessarily.
    - 'yield' turns this function into a GENERATOR. Instead of
      computing everything at once, it produces ONE tokenized
      article, pauses, and waits. The next time something asks
      it for a value (e.g. the next loop iteration, or
      build_vocab_from_iterator pulling the next item), it
      resumes exactly where it left off and produces the next one.
    - This is called "lazy evaluation" — items are only computed
      when they're actually needed, not all upfront.
    - Practical benefit here: build_vocab_from_iterator can stream
      through all articles to build the vocabulary without ever
      holding the entire tokenized dataset in memory at once.
    - Also note: because this is a generator, once you've looped
      through it fully, it's "exhausted" — this is exactly why
      train_iter gets reinitialized (re-created) at various points
      in this notebook. A generator can't be reused/rewound like a
      list can.
    """
    for _, text in data_iter:
        yield tokenizer(text.lower())  # lowercase for consistency


# Build the vocabulary by streaming through every tokenized article.
# "<unk>" is a special token reserved for any word encountered later
# (e.g. during validation/test/inference) that never appeared during
# vocabulary building.
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

print(f"Vocabulary size: {len(vocab)}")
print(f"Sample tokens: {list(vocab.get_stoi().keys())[:10]}")

# Convert tokens to their integer indices in the vocabulary.
# Unknown words map to the <unk> index automatically.
vocab(["age", "hello"])


# ------------------------------------------------------------
# DATASET: converting to map-style + train/validation split
# ------------------------------------------------------------
# AG_NEWS (like most torchtext datasets) is an ITERABLE dataset.
# You can only walk through it once, front to back, using next().
# You CANNOT do dataset[5], and you CANNOT ask for len(dataset).
#
# WHY THIS IS A PROBLEM:
# random_split() (used to carve out a validation set) needs to:
#   1. know the total number of examples (to compute the split sizes)
#   2. be able to grab an item by index (to actually distribute
#      examples between the two resulting datasets)
# A raw iterable/streaming dataset supports neither of these.
#
# WHAT to_map_style_dataset() DOES:
# It walks through the iterable dataset ONE TIME, start to finish,
# and stores every item it sees (e.g. in an internal list-like
# structure). The result is a "map-style" dataset -- one that
# behaves like a normal list:
#   - dataset[i]      -> works (indexing)
#   - len(dataset)     -> works
#   - can be shuffled, split, sampled, etc.
#
# WHY IT MATTERS / HOW IMPORTANT IT IS:
# This conversion is what makes random_split possible at all.
# Without it, you could still train on the data in streaming
# order, but you could NOT randomly separate out a validation
# set the way this notebook does. It's a small one-line call,
# but it's a hard requirement for this train/validation split
# step to work.
#
# Trade-off to be aware of: because it loads everything into
# memory as a list-like structure, this only works well when the
# full dataset comfortably fits in memory. For AG_NEWS (~120k
# short articles) that's fine. For a truly massive corpus, you'd
# need a different strategy (e.g. sampling-based splitting while
# still streaming).
train_iter, test_iter = AG_NEWS()

train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)

# 95% train / 5% validation split
num_train = int(len(train_dataset) * 0.95)

split_train_, split_valid_ = random_split(
    train_dataset, [num_train, len(train_dataset) - num_train]
)

# Use GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# ------------------------------------------------------------
# DEVICE SELECTION: CUDA vs Apple MPS vs TPU
# ------------------------------------------------------------
# torch.cuda.is_available() ONLY detects NVIDIA GPUs (via CUDA).
# It does NOT detect AMD GPUs, Apple Silicon GPUs, or TPUs.
# If you're not on an NVIDIA machine, this line will silently
# fall back to "cpu" even if faster hardware is available.
#
# --- Option A: original code (NVIDIA GPU or CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# --- Option B: Apple Silicon Mac (M1/M2/M3/M4) ---
# PyTorch supports Apple's GPU cores through "MPS"
# (Metal Performance Shaders). Use this check instead:
#
#   device = torch.device(
#       "mps" if torch.backends.mps.is_available() else "cpu"
#   )
#
# You could combine all three checks into one function that
# picks the best available device on ANY machine:
#
#   def get_device():
#       if torch.cuda.is_available():
#           return torch.device("cuda")
#       elif torch.backends.mps.is_available():
#           return torch.device("mps")
#       else:
#           return torch.device("cpu")
#
#   device = get_device()
#
# Note: MPS support is good but not 100% identical to CUDA --
# a small number of ops may not be implemented for MPS yet, in
# which case PyTorch will raise an error for that specific op
# and you'd need to fall back to CPU for that operation only.

# --- Option C: Google Colab with a TPU runtime ---
# TPUs are NOT GPUs -- they're a completely different type of
# chip designed by Google specifically for ML workloads, and
# they don't work through torch.device("cuda") or "mps" at all.
#
# To use a TPU in PyTorch, you need a separate library called
# torch_xla. The setup looks fundamentally different, e.g.:
#
#   import torch_xla.core.xla_model as xm
#   device = xm.xla_device()
#
# and training loops often need small adjustments (e.g. calling
# xm.mark_step() or using a TPU-aware data loader wrapper) to run
# efficiently. It's a bigger structural change than swapping one
# line -- not a drop-in replacement like the MPS check above.
# If you're on Colab, check whether your runtime is a GPU or TPU
# runtime (Runtime -> Change runtime type) since the code path
# differs significantly between the two.


# ------------------------------------------------------------
# TEXT PIPELINE AND LABEL PIPELINE
# ------------------------------------------------------------
def text_pipeline(x):
    # Takes a raw text string, tokenizes it (splits into words),
    # then converts each word into its integer index using vocab.
    # Returns a plain Python list of integers, e.g. [45, 892, 12, ...]
    return vocab(tokenizer(x))


def label_pipeline(x):
    # AG_NEWS labels come as 1, 2, 3, 4 (World/Sports/Business/Sci-Tech).
    # PyTorch's loss functions expect class indices starting at 0.
    # Subtracting 1 converts labels to 0, 1, 2, 3.
    return int(x) - 1


# ------------------------------------------------------------
# COLLATE FUNCTION: builds one batch from individual samples
# ------------------------------------------------------------
# WHY THIS FUNCTION EXISTS AT ALL:
# Articles have different lengths (different numbers of tokens).
# Normally, PyTorch wants to stack a batch into one clean
# rectangular tensor, e.g. shape [batch_size, sequence_length].
# But that requires every item to be the SAME length, which text
# is not, unless you pad it. This model uses a different trick
# instead of padding: it concatenates ALL articles in the batch
# into ONE long flat tensor, and uses a separate "offsets" tensor
# to record where each individual article starts within that
# flat tensor. EmbeddingBag (used later in the model) is built to
# understand exactly this format.
def collate_batch(batch):
    # batch is a list of (label, text) tuples, e.g.:
    #   [(3, "Stocks rose today..."), (1, "War continues in..."), ...]

    label_list, text_list, offsets = [], [], [0]
    # offsets starts with [0] because the FIRST article in the
    # batch always starts at position 0 of the flattened tensor --
    # there's nothing before it yet.

    for _label, _text in batch:
        # Loop through every (label, text) pair in this batch, one at a time.

        label_list.append(label_pipeline(_label))
        # Shift the label to start at 0 and add it to our running list.

        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        # Convert this article's text into a tensor of token indices.
        # e.g. "the market fell" -> tensor([12, 4501, 209])

        text_list.append(processed_text)
        # Add this article's token tensor to our list. NOTE: at this
        # point every article's tensor is still a SEPARATE tensor,
        # possibly a different length than the others.

        offsets.append(processed_text.size(0))
        # Record HOW MANY tokens this article had (its length).
        # At this stage, offsets is just a list of individual
        # lengths, e.g. [0, 7, 5, 10, ...] -- NOT yet the actual
        # starting positions. That gets fixed below with cumsum.

    label_list = torch.tensor(label_list, dtype=torch.int64)
    # Convert the plain Python list of labels into a single 1D
    # PyTorch tensor of type int64 (PyTorch's expected integer type
    # for class labels used with CrossEntropyLoss). Before this line,
    # label_list was just a normal Python list like [2, 0, 3, 1].
    # After this line, it's a proper tensor: tensor([2, 0, 3, 1]).

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    # This is the key trick. offsets currently holds individual
    # article LENGTHS, e.g. [0, 7, 5, 10]. We drop the last value
    # ([:-1]) because we don't need a "start position" for
    # anything after the last article. Then cumsum (cumulative sum)
    # turns lengths into actual START POSITIONS:
    #   lengths:        [0, 7, 5]
    #   cumsum:          0, 0+7=7, 7+5=12
    #   result: offsets = [0, 7, 12]
    # This means: article 1 starts at position 0, article 2 starts
    # at position 7, article 3 starts at position 12, within the
    # big flattened text tensor built next.

    text_list = torch.cat(text_list)
    # Concatenate every article's separate token tensor into ONE
    # single long 1D tensor. This is what lets EmbeddingBag process
    # variable-length articles without any padding -- it just needs
    # this flat tensor plus the offsets to know where each article
    # begins and ends.

    return label_list.to(device), text_list.to(device), offsets.to(device)
    # Move all three tensors to the chosen device (GPU/MPS/CPU) and
    # return them. This is what the DataLoader will hand back on
    # each iteration when collate_fn=collate_batch is used.


# ============================================================
# BUILDING THE DATALOADERS -- BATCH_SIZE and what it controls
# ============================================================

# BATCH_SIZE = 64
#
# train_dataloader = DataLoader(
#     split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
# )
# valid_dataloader = DataLoader(
#     split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
# )
# test_dataloader = DataLoader(
#     test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
# )

# This is where the actual train/valid/test DataLoaders get built.
# collate_fn=collate_batch tells each DataLoader to use OUR custom
# function (defined above) to assemble each batch, instead of PyTorch's
# default collate function, which can't handle variable-length,
# un-padded sequences like these (it expects every sample to already be
# the same shape, fine for images/fixed-length numeric data, not for
# text like this). shuffle=True means the order of articles is
# randomized each epoch, which matters for the "stochastic" part of
# stochastic gradient descent -- each batch is a genuinely random slice
# of the dataset.
#
# BATCH_SIZE = 64: this single number controls how many articles get
# grouped into one batch, i.e. how many articles are processed together
# before the model's weights get updated once. Changing it has several
# effects at once, both practical and about how training behaves:
#
#   SMALLER batch size (e.g. 10): more frequent weight updates per epoch
#   (more batches = more steps), but each update is based on a noisier,
#   less representative sample of the data. Slower overall in wall-clock
#   time too, since you're doing more, but smaller, matrix operations --
#   less efficient, especially on a GPU.
#
#   LARGER batch size (e.g. 100,000): fewer but bigger updates, and more
#   stable/accurate gradient estimates since each one is averaged over
#   way more examples. But it demands a lot more memory, and past a
#   certain point can actually hurt generalization -- the model takes
#   fewer, coarser steps overall and can settle into a less flexible
#   solution.
#
#   64 here is a common middle-ground choice: small enough to fit
#   comfortably in memory and update frequently, big enough to get a
#   reasonably stable gradient estimate at each step.
#
# Concretely: with 1,000 articles and batch_size=64, that's about 16
# batches per epoch -- the model processes the first 64 articles,
# updates its weights based on those, moves to the next 64, updates
# again, and keeps going until all 1,000 have been seen once. That
# completes one epoch. (With the real dataset here, ~114k training
# articles, that's roughly 1,780 batches/weight-updates per epoch.)


# ============================================================
# THE MODEL: TextClassificationModel (EmbeddingBag + Linear)
# ============================================================
# This class defines the actual neural network. It inherits from
# nn.Module, PyTorch's base class for all neural network layers/models.
# Inheriting from nn.Module gives us, for free: automatic tracking of
# all learnable parameters, the ability to move everything to a device
# with .to(device), and built-in saving/loading of weights -- we don't
# have to write any of that bookkeeping ourselves.

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        # This runs the parent class's (nn.Module) setup code FIRST.
        # This has to happen before we define our own layers below,
        # otherwise PyTorch's internal parameter tracking won't work
        # correctly for this model.

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        # This layer turns raw token ID numbers into dense, meaningful
        # vectors, and then automatically AVERAGES those vectors together
        # per article using the offsets we built in collate_batch. That's
        # the "Bag" part of EmbeddingBag: it looks up + combines in one step.
        #   - vocab_size: how many unique tokens exist (needs one vector
        #     per token in the vocabulary).
        #   - embed_dim: how long each token's vector is, e.g. 64 numbers
        #     per token. This becomes the size of the averaged article
        #     representation too.

        self.fc = nn.Linear(embed_dim, num_class)
        # The actual classifier layer. Takes the embed_dim-sized averaged
        # article vector and maps it down to num_class numbers (4 here:
        # World, Sports, Business, Sci-Tech). Whichever of the 4 output
        # numbers is highest becomes the model's predicted category.

        self.init_weights()
        # Call our custom weight initialization method (defined below)
        # to set sensible starting values before training begins.

    def init_weights(self):
        initrange = 0.5
        # Defines a boundary: weights will be randomly initialized
        # somewhere between -0.5 and +0.5.

        self.embedding.weight.data.uniform_(-initrange, initrange)
        # Fills the embedding layer's weights with random numbers drawn
        # UNIFORMLY between -0.5 and +0.5 -- every value in that range is
        # equally likely. This matters because starting weights at exactly
        # zero, or with badly scaled random values, can make training slow
        # or unstable. A small, sensible random spread lets gradients flow
        # properly from the very first update.
        #
        # NOTE: Uniform is just one option, not a universal rule:
        #   - Normal/Gaussian init: draws from a bell curve centered at 0
        #     instead of a flat range.
        #   - Xavier/Glorot init: scales the random range based on a
        #     layer's number of inputs/outputs; pairs well with smoother
        #     activations like sigmoid/tanh.
        #   - He init: similar idea, but derived specifically to keep
        #     signal variance stable through ReLU activations -- pairs
        #     well with ReLU-based networks (common in CNNs).
        # Which one is "best" mainly depends on (1) the activation
        # function used in the network, and (2) how deep the network is
        # -- deeper networks are much more sensitive to bad initialization
        # since small errors compound across many layers. Shallow models
        # like this one can get away with simple uniform/normal init
        # without much practical difference. In practice, most frameworks
        # already default to a sensible scheme (He for ReLU-heavy CNNs,
        # Xavier for older/transformer-style nets), so you'd only override
        # it if you were seeing actual training instability.

        self.fc.weight.data.uniform_(-initrange, initrange)
        # Same uniform initialization applied to the final classifier
        # layer's weights.

        self.fc.bias.data.zero_()
        # The classifier's bias terms start at exactly zero (biases are
        # generally safe to start at zero, unlike weights).

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        # This is the forward pass: what actually happens when a batch of
        # data is fed into the model. text and offsets both come from
        # collate_batch. The embedding layer uses offsets to know exactly
        # where each article starts/ends inside the flattened text tensor,
        # so it can look up + average the right group of token vectors for
        # each article separately, even though nothing was padded. Only ONE
        # batch is processed per call here (e.g. 64 articles), not the
        # whole dataset at once -- that's why the training loop below calls
        # the model repeatedly, once per batch.
        return self.fc(embedded)
        # The averaged per-article vector then flows through the linear
        # layer, producing num_class (4) raw scores per article. Whichever
        # score is highest is the model's prediction for that article.


# Instantiate the model. This is OUR OWN custom class (not imported from
# any library) -- we built it above by inheriting nn.Module. vocab_size
# and num_class come from the data itself, embed_dim (64) is a
# hyperparameter we chose ourselves. Bigger embed_dim can capture richer
# word relationships but needs more memory/data; smaller is faster/lighter
# but less nuanced. 64 is a reasonable default for a small model like this.
# model = TextClassificationModel(vocab_size, embed_dim, num_class).to(device)


# ============================================================
# TRAINING SETUP: loss, optimizer, and LR scheduler
# ============================================================

# LR = 0.1
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=LR)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

# CRITERION: the "judge". CrossEntropyLoss compares the model's 4 raw
# predicted scores against the true label and produces a single number
# representing how wrong the prediction was -- bigger mistakes are
# penalized more heavily. It's the standard loss for multi-class
# classification. This number is what everything downstream (gradients,
# backward pass) gets calculated from.
#
# OPTIMIZER: the "mover". SGD (Stochastic Gradient Descent) is what
# actually updates the model's weights using the gradient (direction) and
# learning rate (step size). model.parameters() tells it exactly which
# numbers it's allowed to adjust (all learnable weights/biases in the
# embedding and linear layers). lr=LR passes in the step size, 0.1 here.
#
# Why "stochastic"? Not because the gradient math itself is random, but
# because of what data it's estimated from. TRUE gradient descent would
# calculate the exact, precise downhill direction using the ENTIRE
# training set (all ~114k articles) at once before taking even one step --
# painfully slow and memory-heavy. STOCHASTIC gradient descent instead
# estimates that direction from just one small random batch (64 articles)
# at a time. That estimate is noisy/approximate, it wobbles slightly batch
# to batch depending on which random articles got sampled -- that wobble
# IS the "stochastic" part. This noise is actually useful: it can nudge
# training out of small, shallow fake-bottom dips in the loss landscape
# (local minima) that aren't the true lowest point, rather than settling
# into the very first shallow valley found.
#
# Mental picture: imagine loss as a bumpy 3D landscape, height = loss, and
# you're trying to reach the lowest valley floor. At each batch, the
# GRADIENT tells you which direction is downhill from where you're
# standing (calculated fresh every single batch, from just that batch's
# data). The LEARNING RATE tells you how big a step to take in that
# downhill direction. Gradient = direction. Learning rate = step size.
# These are two separate jobs -- learning rate does NOT determine
# direction, and it does not react to whether loss is currently going up
# or down; that reacting/adjusting-over-time job belongs to the scheduler.
#
# LEARNING RATE SCHEDULER (StepLR): only ever SHRINKS the learning rate
# over time here, never increases it. With step_size=1.0 and gamma=0.1, it
# multiplies the learning rate by 0.1 after every epoch: epoch 1 starts at
# 0.1, epoch 2 drops to 0.01, epoch 3 to 0.001, and so on. Reasoning: early
# in training the model is far from a good answer, so big steps help it
# get there fast. But if you kept that same big step size the whole way
# through, once the model gets close to a good solution, big steps would
# keep overshooting past it (loss bounces around or even diverges/gets
# worse instead of converging). Shrinking the learning rate lets the model
# take smaller, more careful steps as it gets closer, settling precisely
# into a good solution instead of repeatedly overshooting it.
#
# Learning rate also affects under/overfitting, though indirectly:
#   - Too high: can cause loss to bounce erratically or diverge entirely,
#     so the model never converges to a good fit -- looks like
#     underfitting, but because training itself never settled, not
#     because the model lacked capacity.
#   - Too low: training is painfully slow; with a fixed epoch budget the
#     model may simply run out of time to get anywhere good -- another
#     indirect path to underfitting.
#   - Overfitting is more directly tied to training for too many epochs
#     (small, careful steps that keep inching the model closer to
#     memorizing training data specifics rather than generalizing).
#
# How would you actually find a good learning rate in practice? A
# "learning rate finder" gradually increases LR over a few mini-batches
# from tiny to large while tracking loss, then you look for the point
# where loss drops fastest just before exploding upward -- that's usually
# the sweet spot. Adaptive optimizers like Adam or RMSprop adjust the
# effective learning rate automatically per parameter during training,
# sidestepping a lot of this manual tuning (part of why they're often
# preferred over plain SGD in practice). Otherwise, a lot of real-world
# practice is starting from known-good defaults for the problem type, then
# doing a small grid search or automated hyperparameter tuning.


# ============================================================
# THE TRAINING LOOP
# ============================================================

# EPOCHS = 10
# cum_loss_list = []
# acc_epoch = []
# acc_old = 0
#
# for epoch in tqdm(range(1, EPOCHS + 1)):
#     model.train()
#     cum_loss = 0
#     for idx, (label, text, offsets) in enumerate(train_dataloader):
#         optimizer.zero_grad()
#         predicted_label = model(text, offsets)
#         loss = criterion(predicted_label, label)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
#         optimizer.step()
#         cum_loss += loss.item()
#
#     cum_loss_list.append(cum_loss)
#     accu_val = evaluate(valid_dataloader)
#     acc_epoch.append(accu_val)
#
#     if accu_val > acc_old:
#         acc_old = accu_val
#         torch.save(model.state_dict(), 'my_model.pth')

# EPOCHS = 10: one epoch means the model sees the ENTIRE training dataset
# once, all ~114k articles, via all the 64-article batches back to back.
# 10 epochs = 10 full passes through the whole training set.
#
# Trackers initialized before the loop: cum_loss_list will collect total
# loss per epoch; acc_epoch will collect validation accuracy per epoch;
# acc_old tracks the best validation accuracy seen so far, used to decide
# when to checkpoint (save) the model.
#
# tqdm: purely a visual progress bar library, has nothing to do with the
# training math itself. Wrapping range(1, EPOCHS+1) in tqdm just displays
# a live progress bar (percent done, time remaining) while training runs.
# Cosmetic only.
#
# Inside the epoch loop:
#   model.train() switches the model into training mode (opposite of
#   model.eval()) -- relevant for layers like dropout that behave
#   differently in train vs eval (this model doesn't use dropout, but the
#   pattern still matters generally).
#
#   cum_loss resets to 0 fresh for this epoch, about to accumulate loss
#   across all batches in it.
#
# Inside the INNER loop (one iteration per batch, e.g. 64 articles at a
# time, walking through every batch in the training set for this epoch):
#
#   optimizer.zero_grad() -- clears out any leftover gradients from the
#   PREVIOUS batch before calculating fresh ones. Without this, PyTorch
#   would just keep accumulating gradients on top of old ones batch after
#   batch, which is not what we want.
#
#   predicted_label = model(text, offsets) -- the forward pass for this
#   batch: feeds this batch's flattened text + offsets into the model,
#   returns 4 raw scores per article.
#
#   loss = criterion(predicted_label, label) -- CrossEntropyLoss compares
#   those raw scores against the true labels for these 64 articles and
#   produces one number for how wrong this batch's predictions were.
#
#   loss.backward() -- THE GRADIENT CALCULATION. Works backward through
#   the entire model (hence "backward"), starting from the loss number,
#   calculating exactly how much each individual weight (embedding
#   weights, linear layer weights, all of them) contributed to that error.
#   This is the gradient we've discussed throughout -- PyTorch computes
#   all of it automatically via automatic differentiation; we never derive
#   this math by hand.
#
#   torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1) -- GRADIENT
#   CLIPPING. Solves the problem of occasionally huge/"exploding"
#   gradients, which would cause a massive destabilizing weight update and
#   throw the model wildly off course. This caps the maximum gradient
#   magnitude at 0.1: if the actual gradient is bigger, it gets scaled down
#   proportionally (direction stays the same, magnitude reined in). A
#   safety net for training stability, especially useful with plain SGD
#   (no adaptive step sizing built in).
#
#   optimizer.step() -- THE ACTUAL WEIGHT UPDATE. Takes the calculated,
#   now safely-clipped gradients plus the current learning rate, and
#   adjusts every weight in the model one real step downhill toward lower
#   loss. This is the moment the model actually learns something from this
#   batch.
#
#   cum_loss += loss.item() -- loss.item() pulls the single loss number out
#   of the tensor as a plain Python number, and it keeps adding onto
#   cum_loss batch after batch. By the end of the epoch, cum_loss is the
#   total loss summed across every batch in that epoch.
#
# After all batches in the epoch are done:
#   cum_loss_list.append(cum_loss) -- stores this epoch's total loss.
#
#   accu_val = evaluate(valid_dataloader) -- runs the model against the
#   held-out validation set (the 5% split off earlier) that it was NOT
#   trained on, returning an accuracy score for how well it generalizes.
#
#   acc_epoch.append(accu_val) -- builds up the full record of validation
#   accuracy after every epoch.
#
#   THE CHECKPOINTING LOGIC (if accu_val > acc_old): acc_old started at 0
#   before the loop. This checks "did THIS epoch's validation accuracy beat
#   every previous epoch's accuracy". If yes, acc_old updates to the new
#   higher value, and the model's current weights get saved to
#   'my_model.pth' via torch.save(model.state_dict(), ...), OVERWRITING
#   whatever was saved before. It does NOT explicitly "detect" overfitting
#   -- it's simpler than that. It's just always comparing against the best
#   score so far. If a later epoch's validation accuracy is actually LOWER
#   than an earlier epoch's (a common sign of overfitting -- the model
#   starts memorizing training-specific patterns that don't generalize),
#   this if-statement stays false: no save, no acc_old update, effectively
#   skipping that epoch's version. Net effect: even if training continues
#   for more epochs and gets worse on validation data (classic overfitting
#   pattern -- training loss keeps dropping while validation accuracy
#   plateaus or declines), whatever ends up saved in 'my_model.pth' is the
#   best-performing version across ALL epochs, not necessarily the last
#   one. This protects you from accidentally ending up with an overfit
#   final-epoch model.
#
# Big picture takeaway from this whole loop: within one epoch, if you have
# N batches, SGD calculates a gradient and updates the weights N SEPARATE
# TIMES, once per batch -- not once per epoch. Gradient descent is
# happening at the batch level throughout.


# ============================================================
# EVALUATE FUNCTION
# ============================================================

# def evaluate(dataloader):
#     model.eval()
#     total_acc, total_count = 0, 0
#
#     with torch.no_grad():
#         for idx, (label, text, offsets) in enumerate(dataloader):
#             predicted_label = model(text, offsets)
#
#             total_acc += (predicted_label.argmax(1) == label).sum().item()
#             total_count += label.size(0)
#     return total_acc / total_count

# model.eval() -- switches the model into evaluation mode. Comes from the
# inherited nn.Module machinery. Matters because some layer types (like
# dropout) behave differently during training vs evaluation -- dropout
# randomly disables neurons during training to fight overfitting, but
# during evaluation you want the full model active with no randomness.
# This particular model doesn't use dropout, but it's still good practice
# to always call this before evaluating.
#
# total_acc, total_count = 0, 0 -- running counters: correct predictions
# and total predictions seen, added to as we loop through the data.
#
# with torch.no_grad(): -- disables gradient tracking for everything
# inside this block. Since we're only MEASURING performance here, not
# training/updating weights, there's no need to track gradients at all --
# this saves memory and speeds things up (gradients are only needed when
# you're about to call .backward() to learn from them).
#
# Inside the loop: predicted_label = model(text, offsets) runs the
# forward pass on this batch, same as training. Then:
#   total_acc += (predicted_label.argmax(1) == label).sum().item()
# -- argmax(1) picks whichever of the 4 output scores is highest per
# article (the model's predicted class), compares it against the true
# label, sums up how many matched in this batch, and adds that count to
# the running total.
#   total_count += label.size(0) -- adds how many articles were in this
# batch to the running total count.
#
# Finally, return total_acc / total_count -- overall fraction correct
# across the entire dataset passed in. On an UNTRAINED model (random
# weights), this comes out to roughly 0.25 (25%), matching pure chance on
# a 4-class problem -- a useful sanity check that everything is wired up
# correctly before training even starts. After training, this is the real
# measure of how well the model generalizes to unseen data (e.g. 81%, or
# 87% with more epochs in this notebook's actual run).


# ============================================================
# t-SNE VISUALIZATION OF LEARNED EMBEDDINGS
# ============================================================

# batch = next(iter(valid_dataloader))
# label, text, offsets = batch
# text = text.to(device)
# offsets = offsets.to(device)
# embedded = model.embedding(text, offsets)
# embeddings_numpy = embedded.detach().cpu().numpy()
# X_embedded_3d = TSNE(n_components=3).fit_transform(embeddings_numpy)
# ... (Plotly 3D scatter plot, colored by label)

# This section visualizes what the model actually LEARNED inside its
# embedding vectors, purely for inspection/understanding, not required for
# the model to function. Each article's embedded vector is embed_dim (64)
# dimensions -- far too many to plot or look at directly.
#
# t-SNE (t-distributed Stochastic Neighbor Embedding) squishes those 64
# dimensions down to just 3, while trying to preserve which articles were
# "similar" to each other in that original high-dimensional space. So
# articles the model learned to treat similarly (e.g. both about sports)
# end up plotted close together in the 3D scatter plot, and dissimilar
# ones end up far apart. It's a way to peek inside the model's learned
# understanding of article/word relationships visually -- coloring points
# by their true label (as this code does) lets you visually check whether
# the 4 categories form distinct, separated clusters, which would suggest
# the model learned genuinely meaningful, well-separated representations.


# ============================================================
# PREDICT FUNCTION -- classifying new, unseen text
# ============================================================

# def predict(text, text_pipeline):
#     with torch.no_grad():
#         text = torch.tensor(text_pipeline(text))
#         output = model(text, torch.tensor([0]))
#         return ag_news_label[output.argmax(1).item() + 1]

# Takes a raw text string plus the text_pipeline function (the same one
# used during training to convert text -> token IDs).
#
# torch.no_grad() -- same reasoning as evaluate(): we're just making a
# prediction, not training, so no need to track gradients. Saves memory
# and speeds things up.
#
# text = torch.tensor(text_pipeline(text)) -- runs the raw sentence
# through the text pipeline to turn it into a list of token ID numbers,
# then wraps it in a tensor.
#
# output = model(text, torch.tensor([0])) -- here's a neat detail worth
# noting: the offsets argument is just torch.tensor([0]), a single-element
# tensor. That's because we're predicting for ONE article, not a batch --
# there's only one starting position needed, position 0 (the article
# starts at the very beginning of this single flattened tensor). Compare
# this to training/evaluation, where offsets has one entry per article in
# the batch (e.g. 64 entries for a batch of 64).
#
# output.argmax(1).item() + 1 -- finds which of the 4 output scores is
# highest (argmax), converts it to a plain Python number (.item()), then
# adds 1 back, because the labels were shifted from PyTorch's original
# 1-4 numbering down to 0-3 during training (see label_pipeline earlier)
# -- this undoes that shift to look up the correct entry in ag_news_label.
#
# ag_news_label[...] -- looks up that number in a dictionary mapping
# 1->World, 2->Sports, 3->Business, 4->Sci-Tech, returning a readable
# category name.
#
# IMPORTANT CAVEAT DISCUSSED AT LENGTH: if you call predict() on an
# UNTRAINED model (before running the training loop), whatever it returns
# is essentially a coin flip between the 4 categories -- the embedding and
# linear layer weights are still just random numbers from init_weights(),
# there is no real "understanding" happening yet, even if it happens to
# guess correctly by chance. Only AFTER running the full training loop
# does a call to predict() reflect genuine learned patterns rather than
# luck. This was directly verified in this session: pre-training, the
# evaluate() function returned ~23% accuracy on the test set (matching the
# ~25% random-chance baseline for 4 classes); after training for 10
# epochs, accuracy jumped to 81%, and with 15 epochs it reached 87% --
# real, substantial learning, not chance.


# ============================================================
# EXERCISES 1-3: Load the trained model and classify new articles
# ============================================================

# --- Exercise 1: Load the pre-trained model ---
# model.load_state_dict(torch.load('my_model.pth'))
# model.eval()
#
# torch.load('my_model.pth') reads the saved weights file back into
# memory (the checkpoint saved during training, from whichever epoch had
# the best validation accuracy). model.load_state_dict(...) actually
# applies those loaded weights onto the existing model object. model.eval()
# switches it into evaluation mode since we're about to make predictions,
# not train further.

# --- Exercise 2: Define a list of new articles for classification ---
# articles = [
#     "the central bank raised interest rates to control inflation this week.",
#     "the striker scored a hat trick to lead his team to victory.",
#     "scientists unveiled a new quantum processor capable of record breaking speeds.",
#     "world leaders gathered at the summit to discuss the ongoing conflict.",
#     "shares of the tech giant surged after strong quarterly earnings.",
#     "the tennis champion announced her retirement after a legendary career.",
#     "researchers developed an artificial intelligence model that can detect disease earlier.",
# ]
# Just a plain Python list of strings, one sentence per article, spanning
# all 4 categories (Business, Sports, Sci-Tech, World) to give the model a
# good, varied test.

# --- Exercise 3: Classify each article and display the results ---
# for article in articles:
#     print(article, predict(article, text_pipeline))
#
# Loops through the articles list, calling predict() on each one (passing
# the article text and the same text_pipeline used during training), and
# prints each article alongside whatever category predict() returns.
#
# RESULT FROM THIS SESSION: all 7 out of 7 articles were classified
# correctly (Business, Sports, Sci-Tech, World, Business, Sports, Sci-Tech
# in that order) -- a genuinely strong result showing the model learned
# real, generalizable patterns rather than memorizing training data
# specifics, e.g. correctly connecting "quantum processor" and "artificial
# intelligence" to Sci-Tech, "summit" and "conflict" to World, "interest
# rates" and "quarterly earnings" to Business, and "hat trick" / "tennis
# champion" to Sports.
