RNFL:

    conda create -n rnfl_ocr_env python=3.8
    
    conda activate rnfl_ocr_env

    conda install -c conda-forge pillow numpy tqdm

    brew install swig  

    pip install paddlepaddle paddleocr

    pip install matplotlib IPython numpy pandas seaborn pyCompare scikit-learn statsmodels scipy opencv-python Pillow tqdm 