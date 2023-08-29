from setuptools import setup, find_packages

setup(
    name='jet_leg_learn',
    version='0.1',
    packages=find_packages(),
    author='Siddhant Gangapurwala',
    author_email='siddhant@gangapurwala.com',
    python_requires='>=3.5.0',
    install_requires=[
        'torch==1.5.1',
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'tensorboard==2.4.1',
	'protobuf==3.19.6',
        'seaborn',
        'transforms3d==0.3.1',
        'psutil'
    ]
)
