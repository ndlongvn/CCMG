# CCMG

This repository contains the code and dataset for CCMG.

The Python code is tracked normally using Git, while the large dataset in the `data/` directory is tracked using [Git Large File Storage (LFS)](https://git-lfs.com/).

## Prerequisites

Before cloning the repository, make sure you have Git LFS installed on your system.

**Ubuntu/Debian:**
```bash
sudo apt-get install git-lfs
git lfs install
```

**macOS (using Homebrew):**
```bash
brew install git-lfs
git lfs install
```

**Windows:**
Download and run the installer from the [Git LFS website](https://git-lfs.com/), or install using winget:
```bash
winget install GitHub.GitLFS
git lfs install
```

## How to Clone

To clone this repository containing both the code and the large dataset, simply run:

```bash
git clone https://github.com/ndlongvn/CCMG.git
cd CCMG
```

Git LFS should automatically download the large files in the `data/` directory during the clone process.

If for some reason the `data/` files show up as small text pointer files, you can pull the actual data manually by running:

```bash
git lfs pull
```

## Running the Code

(Add instructions here regarding how to run your `.py` files)
