from setuptools import setup, find_packages

setup(
    name='handwriting-recognition-app',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A handwriting recognition application using deep learning techniques.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'torchvision',
        'opencv-python',
        'numpy',
        'matplotlib',
        'Pillow',
        'scikit-learn',
        'h5py',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)