# cs224n-final-project

## Environment

### Setup

Create a conda environment with the dependencies in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate cs224n
```

### Installing New Packages

1. **Add the package to `environment.yaml`** and run:

    ```bash
    mamba env update -f environment.yaml
    ```

    - If you use `pip` to install packages, add them to the `pip` section in `environment.yaml`.
    - **Do not install new packages directly through `conda`, `mamba`, or `pip`.**
