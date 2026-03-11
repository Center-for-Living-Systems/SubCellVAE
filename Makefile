ENV_NAME := subcellvae

.PHONY: env env-cuda env-update notebook clean

env:
	conda env create -f environment.yml

env-cuda:
	conda env create -f environment-cuda.yml

env-update:
	conda env update -f environment.yml --prune

notebook:
	conda run -n $(ENV_NAME) jupyter lab --notebook-dir=notebook

clean:
	conda env remove -n $(ENV_NAME)
